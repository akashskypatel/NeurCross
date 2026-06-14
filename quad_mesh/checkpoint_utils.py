import glob
import json
import os
import random
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save_file as safetensors_save_file


@dataclass
class CheckpointMetadata:
    epoch: int
    global_step: int
    total_epochs: int
    total_steps: int
    best_smooth_loss: float
    best_step: int
    loss_history: list[float]
    args_dict: dict[str, Any]
    timestamp: str
    device: str
    batch_idx: int = -1


@dataclass
class TrainingCheckpoint:
    model_state_dict: dict[str, torch.Tensor]
    optimizer_state_dict: dict[str, Any]
    metadata: CheckpointMetadata
    early_stopper_state: Optional[dict[str, Any]] = None
    random_state: Optional[dict[str, Any]] = None


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def capture_random_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _cpu_rng_tensor(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.uint8)
    return value


def restore_random_state(state: Optional[dict[str, Any]]) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(_cpu_rng_tensor(state["torch"]))
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([_cpu_rng_tensor(item) for item in state["torch_cuda"]])


def _checkpoint_to_payload(checkpoint: TrainingCheckpoint) -> dict[str, Any]:
    metadata = asdict(checkpoint.metadata) if is_dataclass(checkpoint.metadata) else checkpoint.metadata
    return {
        "model_state_dict": checkpoint.model_state_dict,
        "optimizer_state_dict": checkpoint.optimizer_state_dict,
        "metadata": metadata,
        "early_stopper_state": checkpoint.early_stopper_state,
        "random_state": checkpoint.random_state,
    }


def _metadata_from_payload(metadata: Any) -> CheckpointMetadata:
    if isinstance(metadata, CheckpointMetadata):
        return metadata
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint metadata must be a dictionary")
    return CheckpointMetadata(**metadata)


def _torch_load(path: str, device: str) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _infer_format_from_path(path: str, default: str = "pt") -> str:
    return "safetensors" if path.endswith(".safetensors") else default


def _cpu_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu").contiguous()


def _encode_for_safetensors(value: Any, tensors: dict[str, torch.Tensor], prefix: str) -> Any:
    if isinstance(value, torch.Tensor):
        tensor_key = prefix or "tensor"
        tensors[tensor_key] = _cpu_tensor(value)
        return {"__kind__": "tensor", "key": tensor_key}
    if isinstance(value, np.ndarray):
        return {
            "__kind__": "ndarray",
            "dtype": str(value.dtype),
            "data": value.tolist(),
        }
    if isinstance(value, np.generic):
        return {
            "__kind__": "numpy_scalar",
            "dtype": str(value.dtype),
            "value": value.item(),
        }
    if isinstance(value, dict):
        items = []
        for idx, (key, item_value) in enumerate(value.items()):
            items.append(
                [
                    _encode_for_safetensors(key, tensors, f"{prefix}.key{idx}"),
                    _encode_for_safetensors(item_value, tensors, f"{prefix}.value{idx}"),
                ]
            )
        return {"__kind__": "dict", "items": items}
    if isinstance(value, list):
        return {
            "__kind__": "list",
            "items": [
                _encode_for_safetensors(item, tensors, f"{prefix}.item{idx}")
                for idx, item in enumerate(value)
            ],
        }
    if isinstance(value, tuple):
        return {
            "__kind__": "tuple",
            "items": [
                _encode_for_safetensors(item, tensors, f"{prefix}.item{idx}")
                for idx, item in enumerate(value)
            ],
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Unsupported checkpoint value type for safetensors export: {type(value)!r}")


def _decode_from_safetensors(value: Any, tensors: dict[str, torch.Tensor]) -> Any:
    if isinstance(value, dict) and "__kind__" in value:
        kind = value["__kind__"]
        if kind == "tensor":
            return tensors[value["key"]]
        if kind == "ndarray":
            return np.asarray(value["data"], dtype=np.dtype(value["dtype"]))
        if kind == "numpy_scalar":
            return np.asarray(value["value"], dtype=np.dtype(value["dtype"])).item()
        if kind == "dict":
            return {
                _decode_from_safetensors(key, tensors): _decode_from_safetensors(item_value, tensors)
                for key, item_value in value["items"]
            }
        if kind == "list":
            return [_decode_from_safetensors(item, tensors) for item in value["items"]]
        if kind == "tuple":
            return tuple(_decode_from_safetensors(item, tensors) for item in value["items"])
        raise ValueError(f"Unsupported safetensors payload node kind: {kind}")
    return value


def _save_safetensors_payload(payload: dict[str, Any], path: str) -> None:
    tensors: dict[str, torch.Tensor] = {}
    encoded_payload = _encode_for_safetensors(payload, tensors, "payload")
    metadata = {
        "neurcross_payload": json.dumps(encoded_payload, separators=(",", ":")),
        "neurcross_format": "training_checkpoint",
    }
    safetensors_save_file(tensors, path, metadata=metadata)


def _load_safetensors_payload(path: str, device: str) -> dict[str, Any]:
    with safe_open(path, framework="pt", device=device) as handle:
        metadata = handle.metadata()
        if metadata is None or "neurcross_payload" not in metadata:
            raise ValueError("Safetensors checkpoint missing embedded payload metadata")
        encoded_payload = json.loads(metadata["neurcross_payload"])
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
    payload = _decode_from_safetensors(encoded_payload, tensors)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a dictionary")
    return payload


def save_checkpoint(
    checkpoint: TrainingCheckpoint,
    output_dir: str,
    filename: str = "checkpoint.pt",
    save_best_only: bool = False,
    checkpoint_format: str | None = None,
) -> str:
    """Save training checkpoint to disk."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    if save_best_only and not filename.startswith("best"):
        path = os.path.join(output_dir, "best_" + filename)
    checkpoint_format = checkpoint_format or _infer_format_from_path(path)
    payload = _checkpoint_to_payload(checkpoint)
    if checkpoint_format == "safetensors":
        _save_safetensors_payload(payload, path)
    elif checkpoint_format == "pt":
        torch.save(payload, path)
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_format}")
    return path


def load_checkpoint(
    checkpoint_path: str,
    device: str = "cpu",
) -> TrainingCheckpoint:
    """Load training checkpoint from disk."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint_format = _infer_format_from_path(checkpoint_path)
    if checkpoint_format == "safetensors":
        payload = _load_safetensors_payload(checkpoint_path, device)
    else:
        payload = _torch_load(checkpoint_path, device)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a dictionary")

    try:
        return TrainingCheckpoint(
            model_state_dict=payload["model_state_dict"],
            optimizer_state_dict=payload["optimizer_state_dict"],
            metadata=_metadata_from_payload(payload["metadata"]),
            early_stopper_state=payload.get("early_stopper_state"),
            random_state=payload.get("random_state"),
        )
    except KeyError as exc:
        raise ValueError(f"Checkpoint missing required key: {exc.args[0]}") from exc


def save_model_weights_only(
    state_dict: dict[str, torch.Tensor],
    output_dir: str,
    filename: str = "model_weights.pt",
    checkpoint_format: str | None = None,
) -> str:
    """Export only model weights for inference."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    checkpoint_format = checkpoint_format or _infer_format_from_path(path)
    cpu_state_dict = {key: _cpu_tensor(value) for key, value in state_dict.items()}
    if checkpoint_format == "safetensors":
        safetensors_save_file(cpu_state_dict, path, metadata={"neurcross_format": "weights_only"})
    elif checkpoint_format == "pt":
        torch.save(cpu_state_dict, path)
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_format}")
    return path


def load_model_weights(path: str, device: str = "cpu") -> dict[str, torch.Tensor]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Weights file not found: {path}")
    checkpoint_format = _infer_format_from_path(path)
    if checkpoint_format == "safetensors":
        return safetensors_load_file(path, device=device)
    payload = _torch_load(path, device)
    if not isinstance(payload, dict):
        raise ValueError("Weights payload must be a dictionary")
    return payload


def prune_old_checkpoints(output_dir: str, pattern: str = "checkpoint_step_*", keep_last: int = 3) -> None:
    if keep_last <= 0:
        return
    paths = glob.glob(os.path.join(output_dir, pattern))
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    for path in paths[keep_last:]:
        os.remove(path)
