import glob
import os
import random
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import torch


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


def restore_random_state(state: Optional[dict[str, Any]]) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


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


def save_checkpoint(
    checkpoint: TrainingCheckpoint,
    output_dir: str,
    filename: str = "checkpoint.pt",
    save_best_only: bool = False,
) -> str:
    """Save training checkpoint to disk."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    if save_best_only and not filename.startswith("best"):
        path = os.path.join(output_dir, "best_" + filename)
    torch.save(_checkpoint_to_payload(checkpoint), path)
    return path


def load_checkpoint(
    checkpoint_path: str,
    device: str = "cpu",
) -> TrainingCheckpoint:
    """Load training checkpoint from disk."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

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
) -> str:
    """Export only model weights for inference."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    torch.save(state_dict, path)
    return path


def prune_old_checkpoints(output_dir: str, pattern: str = "checkpoint_step_*.pt", keep_last: int = 3) -> None:
    if keep_last <= 0:
        return
    paths = glob.glob(os.path.join(output_dir, pattern))
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    for path in paths[keep_last:]:
        os.remove(path)
