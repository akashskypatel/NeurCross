import os
import json
from pathlib import Path

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
    assert args.fail_fast is False


def test_cli_help_lists_generate_label():
    from neurcross.__main__ import build_parser

    parser = build_parser()
    help_text = parser.format_help()

    assert "generate-label" in help_text


def test_resolve_input_meshes_accepts_directory_and_list(tmp_path):
    from quad_mesh.generate_label import resolve_input_meshes

    mesh_a = tmp_path / "a.obj"
    mesh_b = tmp_path / "nested" / "b.ply"
    mesh_b.parent.mkdir()
    mesh_a.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    mesh_b.write_text("ply\nformat ascii 1.0\ncomment test\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n", encoding="utf-8")

    meshes_from_dir, base_dir = resolve_input_meshes(str(tmp_path))
    assert meshes_from_dir == [str(mesh_a.resolve()), str(mesh_b.resolve())]
    assert base_dir == str(tmp_path.resolve())

    mesh_list = tmp_path / "meshes.txt"
    mesh_list.write_text("a.obj\nnested/b.ply\n", encoding="utf-8")
    meshes_from_list, list_base = resolve_input_meshes(str(mesh_list))
    assert meshes_from_list == [str(mesh_a.resolve()), str(mesh_b.resolve())]
    assert list_base == str(tmp_path.resolve())


def test_generate_label_batch_writes_summary_and_continues(tmp_path, monkeypatch):
    from quad_mesh.generate_label import main as generate_label_main

    mesh_a = tmp_path / "a.obj"
    mesh_b = tmp_path / "b.obj"
    for mesh_path in (mesh_a, mesh_b):
        mesh_path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")

    dataset_root = tmp_path / "dataset"

    def _fake_train(*, args, allow_multiprocessing_workers):
        sample_dir = Path(args.dataset_root) / args.sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "sample_state": "completed",
            "quality": {"accepted": True},
        }
        manifest_path = sample_dir / "manifest.json"
        if os.path.basename(args.data_path) == "b.obj":
            manifest["sample_state"] = "failed"
            manifest["quality"]["accepted"] = False
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            raise RuntimeError("forced-batch-failure")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        class _Result:
            pass

        result = _Result()
        result.output_dir = str(sample_dir)
        result.manifest_path = str(manifest_path)
        return result

    monkeypatch.setattr("quad_mesh.train_quad_mesh.train_crossfield", _fake_train)

    generate_label_main(
        [
            "--data_path",
            str(tmp_path),
            "--dataset_root",
            str(dataset_root),
        ]
    )

    summary_path = dataset_root / "dataset_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["total_samples"] == 2
    assert summary["counts"]["accepted"] == 1
    assert summary["counts"]["failed"] == 1


def test_generate_label_batch_fail_fast_stops_on_first_failure(tmp_path, monkeypatch):
    from quad_mesh.generate_label import main as generate_label_main

    mesh_a = tmp_path / "a.obj"
    mesh_b = tmp_path / "b.obj"
    for mesh_path in (mesh_a, mesh_b):
        mesh_path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")

    calls = []

    def _fake_train(*, args, allow_multiprocessing_workers):
        calls.append(os.path.basename(args.data_path))
        raise RuntimeError("forced-fail-fast")

    monkeypatch.setattr("quad_mesh.train_quad_mesh.train_crossfield", _fake_train)

    try:
        generate_label_main(
            [
                "--data_path",
                str(tmp_path),
                "--dataset_root",
                str(tmp_path / "dataset"),
                "--fail_fast",
            ]
        )
    except RuntimeError as exc:
        assert "forced-fail-fast" in str(exc)
    else:
        raise AssertionError("generate-label should stop on the first batch failure with --fail_fast")

    assert calls == ["a.obj"]


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


def test_generate_label_batch_outputs_can_build_dataset_index_and_splits(tmp_path, monkeypatch):
    from quad_mesh.dataset_splits import build_split_manifest, load_dataset_index, write_dataset_index
    from quad_mesh.generate_label import main as generate_label_main

    mesh_a = tmp_path / "a.obj"
    mesh_b = tmp_path / "b.obj"
    mesh_c = tmp_path / "c.obj"
    for mesh_path in (mesh_a, mesh_b, mesh_c):
        mesh_path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")

    dataset_root = tmp_path / "dataset"

    def _fake_train(*, args, allow_multiprocessing_workers):
        mesh_name = os.path.basename(args.data_path)
        if mesh_name == "a.obj":
            destination = "accepted"
            sample_state = "completed"
            accepted = True
            source_dataset = "family-a"
        elif mesh_name == "b.obj":
            destination = "quarantine"
            sample_state = "completed"
            accepted = False
            source_dataset = "family-b"
        else:
            destination = "failed"
            sample_state = "failed"
            accepted = False
            source_dataset = "family-c"

        sample_dir = Path(args.dataset_root) / destination / args.sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "sample_id": args.sample_id,
            "sample_state": sample_state,
            "source": {
                "source_mesh_sha256": f"hash-{mesh_name}",
                "source_dataset": source_dataset,
                "source_mesh_path": f"input/{mesh_name}",
            },
            "quality": {"accepted": accepted},
        }
        manifest_path = sample_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        class _Result:
            pass

        result = _Result()
        result.output_dir = str(sample_dir)
        result.manifest_path = str(manifest_path)
        if destination == "failed":
            raise RuntimeError("forced-generated-failure")
        return result

    monkeypatch.setattr("quad_mesh.train_quad_mesh.train_crossfield", _fake_train)

    generate_label_main(
        [
            "--data_path",
            str(tmp_path),
            "--dataset_root",
            str(dataset_root),
        ]
    )

    index_path = write_dataset_index(str(dataset_root))
    assert Path(index_path).exists()
    index_payload = json.loads(Path(index_path).read_text(encoding="utf-8"))
    assert len(index_payload["entries"]) == 3

    entries = load_dataset_index(str(dataset_root))
    assert {entry["destination"] for entry in entries} == {"accepted", "quarantine", "failed"}

    split_manifest = build_split_manifest(
        str(dataset_root),
        seed=5,
        train_ratio=1.0,
        validation_ratio=0.0,
        test_ratio=0.0,
        ood_source_datasets=["family-b"],
    )
    assert len(split_manifest["splits"]["train"]) == 1
    assert len(split_manifest["splits"]["quarantine"]) == 1
    assert len(split_manifest["splits"]["failed"]) == 1
    assert split_manifest["splits"]["ood_test"] == []
    assert split_manifest["counts"]["quarantine"] == 1
    assert split_manifest["counts"]["failed"] == 1
