import os
import sys
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from models.loss_quad_mesh import MorseLoss_quad_mesh
from quad_mesh.checkpoint_utils import (
    CheckpointMetadata,
    TrainingCheckpoint,
    capture_random_state,
    load_checkpoint,
    load_model_weights,
    prune_old_checkpoints,
    restore_random_state,
    save_checkpoint,
    save_model_weights_only,
    utc_timestamp,
)
from quad_mesh.train_quad_mesh import _resolve_device
from quad_mesh.quad_mesh_args import get_args
from quad_mesh.inference import load_trained_model, predict_crossfield
from utils.utils import save_only_crossField


class TinyCrossfieldModel(torch.nn.Module):
    def __init__(self, **_kwargs):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)

    def forward(self, non_mnfld_pnts, mnfld_pnts=None, near_points=None, angle_features=None):
        output = {
            "manifold_pnts_pred": None if mnfld_pnts is None else self.linear(mnfld_pnts),
            "nonmanifold_pnts_pred": self.linear(non_mnfld_pnts),
            "near_points_pred": None if near_points is None else self.linear(near_points),
            "latent_reg": None,
        }
        theta = None if angle_features is None else angle_features[..., :1]
        return output, theta


def _tiny_training_step(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    inputs = torch.randn(4, 2)
    targets = torch.randn(4, 1)
    loss = torch.nn.functional.mse_loss(model(inputs), targets)
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())


def test_checkpoint_round_trip(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()

    metadata = CheckpointMetadata(
        epoch=0,
        batch_idx=3,
        global_step=4,
        total_epochs=10,
        total_steps=100,
        best_smooth_loss=0.25,
        best_step=4,
        loss_history=[0.5, 0.25],
        args_dict=vars(Namespace(num_epochs=10, lr=1e-3)),
        timestamp=utc_timestamp(),
        device="cpu",
    )
    checkpoint = TrainingCheckpoint(
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        metadata=metadata,
        early_stopper_state={"history": [0.5, 0.25], "best_step": 4},
    )

    path = save_checkpoint(checkpoint, str(tmp_path))
    loaded = load_checkpoint(path)

    assert loaded.metadata.global_step == 4
    assert loaded.metadata.batch_idx == 3
    assert loaded.metadata.args_dict["num_epochs"] == 10
    assert loaded.early_stopper_state["best_step"] == 4
    assert loaded.model_state_dict.keys() == checkpoint.model_state_dict.keys()
    assert loaded.optimizer_state_dict.keys() == checkpoint.optimizer_state_dict.keys()


def test_weights_only_export(tmp_path):
    model = torch.nn.Linear(2, 1)
    path = save_model_weights_only(model.state_dict(), str(tmp_path))
    loaded = torch.load(path, map_location="cpu")

    assert loaded.keys() == model.state_dict().keys()


def test_weights_only_export_safetensors(tmp_path):
    model = torch.nn.Linear(2, 1)
    path = save_model_weights_only(
        model.state_dict(),
        str(tmp_path),
        filename="model_weights.safetensors",
        checkpoint_format="safetensors",
    )
    loaded = load_model_weights(path, device="cpu")

    assert path.endswith(".safetensors")
    assert loaded.keys() == model.state_dict().keys()
    for key, value in loaded.items():
        assert torch.allclose(value, model.state_dict()[key])


def test_crossfield_export_uses_vec_extension(tmp_path):
    alpha = torch.tensor([[[1.0, 0.0, 0.0]]])
    beta = torch.tensor([[[0.0, 1.0, 0.0]]])

    path = save_only_crossField(
        alpha,
        beta,
        batch_idx=7,
        output_dir=str(tmp_path),
        shapename="mesh",
    )

    assert path.endswith("mesh_iter_7.vec")
    assert os.path.exists(path)


def test_device_and_topology_memory_args_parse():
    args = get_args(["--device", "cpu", "--max_topology_memory_gb", "0.25", "--checkpoint_format", "safetensors"])

    assert args.device == "cpu"
    assert args.max_topology_memory_gb == pytest.approx(0.25)
    assert args.checkpoint_format == "safetensors"


def test_explicit_cuda_device_arg_parses():
    args = get_args(["--device", "cuda:1"])

    assert args.device == "cuda:1"


def test_tensorboard_args_parse():
    args = get_args(["--tensorboard_dir", "runs/custom", "--no-tensorboard"])

    assert args.tensorboard is False
    assert args.tensorboard_dir == "runs/custom"


def test_resolve_device_accepts_explicit_cuda_index():
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 2,
        )
    )

    assert _resolve_device(fake_torch, "cuda:1") == "cuda:1"


def test_resolve_device_rejects_missing_cuda_index():
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
        )
    )

    with pytest.raises(RuntimeError, match="only 1 CUDA device"):
        _resolve_device(fake_torch, "cuda:1")


def test_topology_memory_guard_raises_before_large_allocation():
    vertex_neighbors_list = [list(range(20))]
    vertex_neighbors = {idx: list(range(20)) for idx in range(20)}

    with pytest.raises(MemoryError, match="Estimated cached topology tensor memory"):
        MorseLoss_quad_mesh(
            vertex_neighbors_list=vertex_neighbors_list,
            vertex_neighbors=vertex_neighbors,
            device="cpu",
            max_topology_memory_gb=1e-9,
        )


def test_missing_checkpoint_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(str(tmp_path / "missing.pt"))


def test_malformed_checkpoint_raises_value_error(tmp_path):
    path = tmp_path / "bad_checkpoint.pt"
    torch.save({"metadata": {}}, path)

    with pytest.raises(ValueError, match="missing required key"):
        load_checkpoint(str(path))


def test_prune_old_checkpoints_keeps_most_recent(tmp_path):
    for idx in range(5):
        path = tmp_path / f"checkpoint_step_{idx}.pt"
        path.write_text("checkpoint", encoding="utf-8")
        timestamp = 1000 + idx
        path.touch()
        os.utime(path, (timestamp, timestamp))

    prune_old_checkpoints(str(tmp_path), keep_last=2)

    remaining = sorted(path.name for path in tmp_path.glob("checkpoint_step_*.pt"))
    assert remaining == ["checkpoint_step_3.pt", "checkpoint_step_4.pt"]


def test_load_trained_model_and_predict_crossfield(monkeypatch, tmp_path):
    monkeypatch.setitem(
        sys.modules,
        "models",
        SimpleNamespace(Network_predict_angle=TinyCrossfieldModel),
    )
    model = TinyCrossfieldModel()
    checkpoint = TrainingCheckpoint(
        model_state_dict=model.state_dict(),
        optimizer_state_dict={},
        metadata=CheckpointMetadata(
            epoch=0,
            batch_idx=0,
            global_step=1,
            total_epochs=1,
            total_steps=1,
            best_smooth_loss=0.1,
            best_step=0,
            loss_history=[0.1],
            args_dict={
                "decoder_hidden_dim": 8,
                "decoder_n_hidden_layers": 1,
                "nl": "sine",
                "init_type": "siren",
                "sphere_init_params": [1.6, 0.1],
                "udf": False,
                "latent_size": 0,
            },
            timestamp=utc_timestamp(),
            device="cpu",
        ),
    )
    path = save_checkpoint(checkpoint, str(tmp_path), filename="trained.pt")

    loaded_model, metadata = load_trained_model(path, device="cpu")
    output, theta = predict_crossfield(
        loaded_model,
        torch.ones(1, 2, 3),
        torch.ones(1, 2, 3),
        angle_features=torch.full((1, 2, 12), 0.5),
        device="cpu",
    )

    assert loaded_model.training is False
    assert metadata["global_step"] == 1
    assert output["nonmanifold_pnts_pred"].shape == (1, 2, 1)
    assert theta.shape == (1, 2, 1)


def test_resume_state_matches_continuous_training_next_step(tmp_path):
    torch.manual_seed(1234)
    continuous_model = torch.nn.Linear(2, 1)
    continuous_optimizer = torch.optim.SGD(
        continuous_model.parameters(),
        lr=0.1,
        momentum=0.9,
    )
    loss_history = [_tiny_training_step(continuous_model, continuous_optimizer)]
    checkpoint = TrainingCheckpoint(
        model_state_dict=continuous_model.state_dict(),
        optimizer_state_dict=continuous_optimizer.state_dict(),
        metadata=CheckpointMetadata(
            epoch=0,
            batch_idx=0,
            global_step=1,
            total_epochs=1,
            total_steps=2,
            best_smooth_loss=loss_history[-1],
            best_step=0,
            loss_history=loss_history,
            args_dict={"lr": 0.1},
            timestamp=utc_timestamp(),
            device="cpu",
        ),
        random_state=capture_random_state(),
    )
    checkpoint_path = save_checkpoint(checkpoint, str(tmp_path), filename="resume.pt")

    continuous_next_loss = _tiny_training_step(continuous_model, continuous_optimizer)
    continuous_state = {
        key: value.detach().clone()
        for key, value in continuous_model.state_dict().items()
    }

    torch.manual_seed(9999)
    loaded = load_checkpoint(checkpoint_path)
    resumed_model = torch.nn.Linear(2, 1)
    resumed_optimizer = torch.optim.SGD(
        resumed_model.parameters(),
        lr=0.1,
        momentum=0.9,
    )
    resumed_model.load_state_dict(loaded.model_state_dict)
    resumed_optimizer.load_state_dict(loaded.optimizer_state_dict)
    restore_random_state(loaded.random_state)

    resumed_next_loss = _tiny_training_step(resumed_model, resumed_optimizer)

    assert resumed_next_loss == pytest.approx(continuous_next_loss)
    for key, value in resumed_model.state_dict().items():
        assert torch.allclose(value, continuous_state[key])


def test_checkpoint_round_trip_safetensors(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()

    metadata = CheckpointMetadata(
        epoch=1,
        batch_idx=2,
        global_step=3,
        total_epochs=5,
        total_steps=25,
        best_smooth_loss=0.125,
        best_step=3,
        loss_history=[0.25, 0.125],
        args_dict=vars(Namespace(num_epochs=5, lr=1e-3, checkpoint_format="safetensors")),
        timestamp=utc_timestamp(),
        device="cpu",
    )
    checkpoint = TrainingCheckpoint(
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        metadata=metadata,
        early_stopper_state={"history": [0.25, 0.125], "best_step": 3},
        random_state=capture_random_state(),
    )

    path = save_checkpoint(
        checkpoint,
        str(tmp_path),
        filename="resume.safetensors",
        checkpoint_format="safetensors",
    )
    loaded = load_checkpoint(path, device="cpu")

    assert path.endswith(".safetensors")
    assert loaded.metadata.global_step == 3
    assert loaded.metadata.batch_idx == 2
    assert loaded.metadata.args_dict["checkpoint_format"] == "safetensors"
    assert loaded.early_stopper_state["best_step"] == 3
    assert loaded.optimizer_state_dict.keys() == checkpoint.optimizer_state_dict.keys()
    for key, value in loaded.model_state_dict.items():
        assert torch.allclose(value, checkpoint.model_state_dict[key])
