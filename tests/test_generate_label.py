import os
import json

import trimesh


def test_derive_sample_id_is_deterministic(tmp_path):
    from quad_mesh.generate_label import derive_sample_id

    mesh_path = tmp_path / "cube.obj"
    mesh_path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")

    first = derive_sample_id(str(mesh_path))
    second = derive_sample_id(str(mesh_path))

    assert first == second
    assert first.startswith("cube-")


def test_generate_label_parser_accepts_dataset_args():
    from quad_mesh.generate_label import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "--data_path",
            "mesh.obj",
            "--dataset_root",
            "labels",
            "--sample_id",
            "sample-001",
            "--overwrite",
            "--preflight_policy",
            "strict",
            "--quality_gate",
            "loose",
            "--no-export_geometry_npz",
            "--no-normalize_mesh",
            "--num_epochs",
            "1",
        ]
    )

    assert args.data_path == "mesh.obj"
    assert args.dataset_root == "labels"
    assert args.sample_id == "sample-001"
    assert args.overwrite is True
    assert args.preflight_policy == "strict"
    assert args.quality_gate == "loose"
    assert args.export_geometry_npz is False
    assert args.normalize_mesh is False


def test_cli_help_lists_generate_label():
    from neurcross.__main__ import build_parser

    parser = build_parser()
    help_text = parser.format_help()

    assert "generate-label" in help_text


def test_generate_label_strict_preflight_writes_skipped_manifest(tmp_path):
    from quad_mesh.generate_label import main as generate_label_main

    left = trimesh.creation.box()
    right = trimesh.creation.box()
    right.apply_translation((3.0, 0.0, 0.0))
    mesh = trimesh.util.concatenate((left, right))
    mesh_path = tmp_path / "two_boxes.obj"
    mesh.export(mesh_path)

    dataset_root = tmp_path / "dataset"
    generate_label_main(
        [
            "--data_path",
            str(mesh_path),
            "--dataset_root",
            str(dataset_root),
            "--sample_id",
            "skip-sample",
            "--overwrite",
            "--preflight_policy",
            "strict",
            "--device",
            "cpu",
        ]
    )

    manifest_path = dataset_root / "skip-sample" / "manifest.json"
    report_path = dataset_root / "skip-sample" / "mesh_quality_report.json"
    assert manifest_path.exists()
    assert report_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["sample_state"] == "skipped"
    assert manifest["outputs"]["selected_label"] == "none"
    assert manifest["outputs"]["crossfield_best_vec"] is None
    assert manifest["quality"]["accepted"] is False
    assert manifest["quality"]["failure_reason"]


def test_generate_label_setup_failure_captures_failed_artifacts(tmp_path, monkeypatch):
    from quad_mesh.generate_label import main as generate_label_main
    import quad_mesh.normalize as normalize_mod

    mesh = trimesh.creation.box()
    mesh_path = tmp_path / "box.obj"
    mesh.export(mesh_path)

    def _boom(*args, **kwargs):
        raise RuntimeError("forced-normalize-failure")

    monkeypatch.setattr(normalize_mod, "export_normalized_mesh", _boom)

    dataset_root = tmp_path / "dataset"
    try:
        generate_label_main(
            [
                "--data_path",
                str(mesh_path),
                "--dataset_root",
                str(dataset_root),
                "--sample_id",
                "failed-sample",
                "--overwrite",
                "--device",
                "cpu",
            ]
        )
    except RuntimeError as exc:
        assert "artifacts captured under" in str(exc)
    else:
        raise AssertionError("generate-label should raise on setup failure")

    manifest_path = dataset_root / "failed" / "failed-sample" / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["sample_state"] == "failed"
    assert manifest["quality"]["accepted"] is False
