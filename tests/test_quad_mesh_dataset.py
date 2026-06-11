import numpy as np
import pytest
import trimesh


def _write_box_mesh(tmp_path, name="box.obj"):
    mesh = trimesh.creation.box()
    mesh_path = tmp_path / name
    mesh.export(mesh_path)
    return mesh_path


def test_dataset_uniform_sampling_respects_extent(tmp_path):
    from quad_mesh.quad_mesh_dataset import ReconDataset

    mesh_path = _write_box_mesh(tmp_path)
    dataset = ReconDataset(
        str(mesh_path),
        n_points=64,
        n_samples=2,
        nonmnfld_sample_type="uniform",
        uniform_extent=0.25,
        seed=7,
    )

    batch = dataset[0]
    assert batch["nonmnfld_points"].shape == (64, 3)
    assert np.all(batch["nonmnfld_points"] <= 0.25 + 1e-6)
    assert np.all(batch["nonmnfld_points"] >= -0.25 - 1e-6)
    assert np.all(batch["nonmnfld_sample_labels"] == 0)


def test_dataset_mixed_sampling_includes_all_configured_modes(tmp_path):
    from quad_mesh.quad_mesh_dataset import ReconDataset

    mesh_path = _write_box_mesh(tmp_path)
    dataset = ReconDataset(
        str(mesh_path),
        n_points=120,
        n_samples=1,
        nonmnfld_sample_type="mixed",
        uniform_ratio=0.4,
        near_surface_ratio=0.3,
        feature_ratio=0.3,
        seed=11,
    )

    batch = dataset[0]
    labels = set(batch["nonmnfld_sample_labels"].tolist())
    assert labels == {0, 1, 2}


def test_dataset_sampling_is_deterministic_per_seed_and_epoch(tmp_path):
    from quad_mesh.quad_mesh_dataset import ReconDataset

    mesh_path = _write_box_mesh(tmp_path)
    first = ReconDataset(str(mesh_path), n_points=32, n_samples=1, nonmnfld_sample_type="mixed", seed=123)
    second = ReconDataset(str(mesh_path), n_points=32, n_samples=1, nonmnfld_sample_type="mixed", seed=123)

    first_batch = first[0]
    second_batch = second[0]
    assert np.allclose(first_batch["nonmnfld_points"], second_batch["nonmnfld_points"])
    assert np.array_equal(first_batch["nonmnfld_sample_labels"], second_batch["nonmnfld_sample_labels"])

    first.set_epoch(1)
    changed_batch = first[0]
    assert not np.allclose(first_batch["nonmnfld_points"], changed_batch["nonmnfld_points"])


def test_dataset_validation_batch_is_fixed(tmp_path):
    from quad_mesh.quad_mesh_dataset import ReconDataset

    mesh_path = _write_box_mesh(tmp_path)
    dataset = ReconDataset(str(mesh_path), n_points=48, n_samples=1, nonmnfld_sample_type="feature_biased", seed=99)

    first = dataset.get_validation_batch()
    second = dataset.get_validation_batch()
    assert np.allclose(first["nonmnfld_points"], second["nonmnfld_points"])
    assert np.array_equal(first["nonmnfld_sample_labels"], second["nonmnfld_sample_labels"])
    assert "validation_face_indices" in first
    assert "validation_feature_indices" in first
    assert "validation_boundary_indices" in first


def test_dataset_parser_accepts_new_sampling_controls():
    from quad_mesh.generate_label import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "--data_path",
            "mesh.obj",
            "--nonmnfld_sample_type",
            "mixed",
            "--near_surface_ratio",
            "0.25",
            "--uniform_ratio",
            "0.5",
            "--feature_ratio",
            "0.25",
            "--boundary_ratio",
            "0.8",
            "--near_surface_sigma",
            "0.01",
            "--uniform_extent",
            "0.6",
        ]
    )

    assert args.nonmnfld_sample_type == "mixed"
    assert args.near_surface_ratio == pytest.approx(0.25)
    assert args.uniform_ratio == pytest.approx(0.5)
    assert args.feature_ratio == pytest.approx(0.25)
    assert args.boundary_ratio == pytest.approx(0.8)
    assert args.near_surface_sigma == pytest.approx(0.01)
    assert args.uniform_extent == pytest.approx(0.6)
    assert args.save_best_by == "val_field_score"
    assert args.eval_interval_steps == 0
    assert args.export_interval_steps == 500
    assert args.feature_mode == "auto"
    assert args.feature_angle_threshold == pytest.approx(35.0)
    assert args.feature_weight_scale == pytest.approx(1.0)
    assert args.export_features is True
    assert args.steps_per_epoch is None
    assert args.total_steps is None


def test_dataset_parser_defaults_to_mixed_sampling():
    from quad_mesh.generate_label import build_parser

    parser = build_parser()
    args = parser.parse_args(["--data_path", "mesh.obj"])

    assert args.nonmnfld_sample_type == "mixed"


def test_dataset_parser_accepts_save_best_by():
    from quad_mesh.generate_label import build_parser

    parser = build_parser()
    args = parser.parse_args(["--data_path", "mesh.obj", "--save_best_by", "train_field_score"])

    assert args.save_best_by == "train_field_score"


def test_dataset_parser_accepts_eval_and_export_interval_steps():
    from quad_mesh.generate_label import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "--data_path",
            "mesh.obj",
            "--eval_interval_steps",
            "2",
            "--export_interval_steps",
            "3",
        ]
    )

    assert args.eval_interval_steps == 2
    assert args.export_interval_steps == 3


def test_dataset_parser_accepts_step_based_schedule_args():
    from quad_mesh.generate_label import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "--data_path",
            "mesh.obj",
            "--steps_per_epoch",
            "4",
            "--total_steps",
            "10",
        ]
    )

    assert args.steps_per_epoch == 4
    assert args.total_steps == 10
