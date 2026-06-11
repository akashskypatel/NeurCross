import os

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("NEURCROSS_RUN_SLOW_TESTS") != "1",
    reason="set NEURCROSS_RUN_SLOW_TESTS=1 to run slow mesh training integration tests",
)


def _require_cuda_torch():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for NeurCross training integration tests")
    return torch


def _cube_mesh_path():
    return os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "cube",
        "input",
        "cube.obj",
    )


def test_train_save_resume_and_export_weights(tmp_path):
    torch = _require_cuda_torch()
    import neurcross
    from models import Network_predict_angle
    from quad_mesh.checkpoint_utils import load_checkpoint

    mesh_path = _cube_mesh_path()
    if not os.path.exists(mesh_path):
        pytest.skip("sample cube mesh is not available")

    first = neurcross.train_crossfield(
        data_path=mesh_path,
        out_dir=str(tmp_path),
        device="cuda",
        num_epochs=1,
        n_samples=1,
        n_points=16,
        batch_size=1,
        num_workers=0,
        persistent_workers=False,
        log_interval=1,
        save_checkpoint_interval=1,
        keep_last_n_checkpoints=2,
        export_weights_only=True,
        fast_nondeterministic=True,
    )
    first_checkpoint = load_checkpoint(first.checkpoint_path)

    assert os.path.exists(first.checkpoint_path)
    assert os.path.exists(first.best_checkpoint_path)
    assert os.path.exists(first.weights_path)
    assert os.path.exists(os.path.join(first.output_dir, "mesh_quality_report.json"))
    assert os.path.exists(os.path.join(first.output_dir, "input", "normalized_mesh.obj"))
    assert os.path.exists(os.path.join(first.output_dir, "input", "normalized_mesh.ply"))
    assert first_checkpoint.metadata.global_step == 1
    crossfield_dir = os.path.join(first.output_dir, "save_crossField")
    assert os.path.exists(os.path.join(crossfield_dir, "cube_iter_0.vec"))
    assert os.path.exists(os.path.join(crossfield_dir, "cube_latest.vec"))
    assert os.path.exists(os.path.join(crossfield_dir, "cube_best.vec"))
    assert os.path.exists(os.path.join(crossfield_dir, "cube_final.vec"))

    resumed = neurcross.train_crossfield(
        data_path=mesh_path,
        out_dir=str(tmp_path),
        device="cuda",
        num_epochs=2,
        n_samples=1,
        n_points=16,
        batch_size=1,
        num_workers=0,
        persistent_workers=False,
        log_interval=1,
        save_checkpoint_interval=1,
        keep_last_n_checkpoints=2,
        load_checkpoint=first.checkpoint_path,
        export_weights_only=True,
        fast_nondeterministic=True,
    )
    resumed_checkpoint = load_checkpoint(resumed.checkpoint_path)
    periodic_paths = [
        path
        for path in os.listdir(os.path.join(resumed.output_dir, "checkpoints"))
        if path.startswith("checkpoint_step_") and path.endswith(".pt")
    ]

    assert os.path.exists(resumed.checkpoint_path)
    assert os.path.exists(resumed.weights_path)
    assert resumed_checkpoint.metadata.global_step == 2
    assert len(periodic_paths) <= 2

    checkpoint_model, _metadata = neurcross.load_trained_model(
        resumed.checkpoint_path,
        device="cuda",
    )
    args_dict = resumed_checkpoint.metadata.args_dict
    weights_model = Network_predict_angle(
        in_dim=3,
        angle_in_dim=12,
        decoder_hidden_dim=args_dict.get("decoder_hidden_dim", 256),
        nl=args_dict.get("nl", "sine"),
        decoder_n_hidden_layers=args_dict.get("decoder_n_hidden_layers", 4),
        init_type=args_dict.get("init_type", "siren"),
        sphere_init_params=args_dict.get("sphere_init_params", [1.6, 0.1]),
        udf=args_dict.get("udf", False),
        latent_size=args_dict.get("latent_size", 0),
    )
    weights_model.load_state_dict(torch.load(resumed.weights_path, map_location="cuda"))
    weights_model.to("cuda")
    weights_model.eval()

    torch.manual_seed(123)
    nonmanifold_points = torch.randn(1, 8, 3, device="cuda")
    manifold_points = torch.randn(1, 8, 3, device="cuda")
    angle_features = torch.randn(1, 8, 12, device="cuda")
    checkpoint_output, checkpoint_theta = neurcross.predict_crossfield(
        checkpoint_model,
        nonmanifold_points,
        manifold_points,
        angle_features=angle_features,
        device="cuda",
    )
    weights_output, weights_theta = neurcross.predict_crossfield(
        weights_model,
        nonmanifold_points,
        manifold_points,
        angle_features=angle_features,
        device="cuda",
    )

    assert torch.allclose(
        checkpoint_output["nonmanifold_pnts_pred"],
        weights_output["nonmanifold_pnts_pred"],
    )
    assert torch.allclose(checkpoint_theta, weights_theta)


def test_early_stop_writes_checkpoint(tmp_path):
    _require_cuda_torch()
    import neurcross
    from quad_mesh.checkpoint_utils import load_checkpoint

    mesh_path = _cube_mesh_path()
    if not os.path.exists(mesh_path):
        pytest.skip("sample cube mesh is not available")

    result = neurcross.train_crossfield(
        data_path=mesh_path,
        out_dir=str(tmp_path),
        device="cuda",
        num_epochs=2,
        n_samples=1,
        n_points=16,
        batch_size=1,
        num_workers=0,
        persistent_workers=False,
        log_interval=1,
        save_checkpoint_interval=0,
        keep_last_n_checkpoints=2,
        early_stop=True,
        early_stop_min_steps=0,
        early_stop_patience=1,
        early_stop_smooth_window=1,
        early_stop_check_interval=1,
        early_stop_target_loss=1e9,
        fast_nondeterministic=True,
    )
    checkpoint_dir = os.path.join(result.output_dir, "checkpoints")
    early_stop_path = os.path.join(checkpoint_dir, "early_stop_checkpoint.pt")
    final_checkpoint = load_checkpoint(result.checkpoint_path)
    early_stop_checkpoint = load_checkpoint(early_stop_path)

    assert result.stopped_early is True
    assert result.stop_summary["reason"] == "target_loss"
    assert os.path.exists(early_stop_path)
    assert os.path.exists(result.checkpoint_path)
    assert result.checkpoint_path.endswith("final_checkpoint.pt")
    assert os.path.exists(os.path.join(result.output_dir, "save_crossField", "cube_final.vec"))
    assert early_stop_checkpoint.metadata.global_step == 1
    assert final_checkpoint.metadata.global_step == 1


def test_generate_label_writes_manifest(tmp_path):
    _require_cuda_torch()
    import json
    from quad_mesh.generate_label import main as generate_label_main

    mesh_path = _cube_mesh_path()
    if not os.path.exists(mesh_path):
        pytest.skip("sample cube mesh is not available")

    dataset_root = tmp_path / "dataset"
    generate_label_main(
        [
            "--data_path",
            mesh_path,
            "--dataset_root",
            str(dataset_root),
            "--sample_id",
            "cube-sample",
            "--overwrite",
            "--device",
            "cuda",
            "--num_epochs",
            "1",
            "--n_samples",
            "1",
            "--n_points",
            "16",
            "--batch_size",
            "1",
            "--num_workers",
            "0",
            "--log_interval",
            "1",
            "--save_checkpoint_interval",
            "1",
            "--fast_nondeterministic",
        ]
    )

    manifest_path = dataset_root / "cube-sample" / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["neurcross_dataset_schema_version"] == "0.1"
    assert manifest["artifact_type"] == "neurcross_per_mesh_label"
    assert manifest["sample_id"] == "cube-sample"
    assert manifest["outputs"]["crossfield_best_vec"] == "fields/crossfield_best.vec"
