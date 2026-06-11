import json
import os


def test_validate_manifest_accepts_minimum_schema(tmp_path):
    from quad_mesh.export_dataset_sample import validate_manifest, write_manifest

    input_dir = tmp_path / "input"
    fields_dir = tmp_path / "fields"
    geometry_dir = tmp_path / "geometry"
    metrics_dir = tmp_path / "metrics"
    logs_dir = tmp_path / "logs"
    input_dir.mkdir()
    fields_dir.mkdir()
    geometry_dir.mkdir()
    metrics_dir.mkdir()
    logs_dir.mkdir()

    source_mesh = input_dir / "original_mesh.obj"
    normalized_mesh = input_dir / "normalized_mesh.ply"
    best_vec = fields_dir / "crossfield_best.vec"
    geometry_npz = geometry_dir / "mesh_geometry.npz"
    best_metrics = metrics_dir / "train_metrics_best.json"
    acceptance_report = metrics_dir / "acceptance_report.json"
    train_log = logs_dir / "train.log"
    command_txt = logs_dir / "command.txt"

    source_mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    normalized_mesh.write_text("ply\nformat ascii 1.0\nend_header\n", encoding="utf-8")
    best_vec.write_text("1 0 0 0 1 0\n", encoding="utf-8")
    geometry_npz.write_bytes(b"npz")
    best_metrics.write_text("{}", encoding="utf-8")
    acceptance_report.write_text("{}", encoding="utf-8")
    train_log.write_text("log\n", encoding="utf-8")
    command_txt.write_text("python -m neurcross generate-label\n", encoding="utf-8")

    import hashlib

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        "neurcross_dataset_schema_version": "0.1",
        "artifact_type": "neurcross_per_mesh_label",
        "sample_id": "sample-001",
        "created_at_utc": "2026-06-11T00:00:00Z",
        "source": {
            "source_mesh_path": "input/original_mesh.obj",
            "source_mesh_sha256": sha(source_mesh),
            "source_format": "obj",
        },
        "mesh": {
            "normalized_mesh_path": "input/normalized_mesh.ply",
            "normalized_mesh_sha256": sha(normalized_mesh),
            "vertex_count": 3,
            "face_count": 1,
            "is_watertight": False,
        },
        "normalization": {
            "coordinate_space": "normalized",
            "target_bounds": "[-0.5, 0.5]^3",
            "center": [0.0, 0.0, 0.0],
            "scale": 1.0,
        },
        "training": {
            "tool": "neurcross",
            "neurcross_version": "0.1.0",
            "command": "python -m neurcross generate-label",
            "args": {},
            "seed": 1,
            "device": "cuda",
            "started_at_utc": "2026-06-11T00:00:00Z",
            "finished_at_utc": "2026-06-11T00:00:01Z",
            "elapsed_seconds": 1.0,
        },
        "outputs": {
            "selected_label": "best",
            "crossfield_best_vec": "fields/crossfield_best.vec",
            "metrics_best_json": "metrics/train_metrics_best.json",
            "geometry_npz": "geometry/mesh_geometry.npz",
            "log_path": "logs/train.log",
            "command_path": "logs/command.txt",
        },
        "quality": {
            "accepted": True,
            "quality_grade": "A",
            "quality_gate": "default_v0",
            "field_score": 0.1,
            "failure_reason": None,
            "acceptance_report_json": "metrics/acceptance_report.json",
        },
    }

    validate_manifest(manifest, str(tmp_path))
    path = write_manifest(manifest, str(tmp_path))
    persisted = json.loads(open(path, "r", encoding="utf-8").read())
    assert persisted["sample_id"] == "sample-001"


def test_validate_manifest_rejects_missing_required_field(tmp_path):
    from quad_mesh.export_dataset_sample import validate_manifest

    manifest = {
        "neurcross_dataset_schema_version": "0.1",
        "artifact_type": "neurcross_per_mesh_label",
        "sample_id": "sample-001",
        "created_at_utc": "2026-06-11T00:00:00Z",
        "source": {},
        "mesh": {},
        "normalization": {},
        "training": {},
        "outputs": {},
        "quality": {},
    }

    try:
        validate_manifest(manifest, str(tmp_path))
    except ValueError as exc:
        assert "source_mesh_path" in str(exc)
    else:
        raise AssertionError("validate_manifest should reject incomplete required sections")


def test_validate_manifest_rejects_crossfield_row_count_mismatch(tmp_path):
    from quad_mesh.export_dataset_sample import validate_manifest
    import hashlib

    (tmp_path / "input").mkdir()
    (tmp_path / "fields").mkdir()
    (tmp_path / "geometry").mkdir()
    (tmp_path / "metrics").mkdir()
    (tmp_path / "logs").mkdir()

    source_mesh = tmp_path / "input" / "original_mesh.obj"
    normalized_mesh = tmp_path / "input" / "normalized_mesh.ply"
    best_vec = tmp_path / "fields" / "crossfield_best.vec"
    metrics = tmp_path / "metrics" / "train_metrics_best.json"
    geometry = tmp_path / "geometry" / "mesh_geometry.npz"
    log_path = tmp_path / "logs" / "train.log"
    command_path = tmp_path / "logs" / "command.txt"
    acceptance_report = tmp_path / "metrics" / "acceptance_report.json"

    source_mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    normalized_mesh.write_text("ply\nformat ascii 1.0\nend_header\n", encoding="utf-8")
    best_vec.write_text("1 0 0 0 1 0\n2 0 0 0 2 0\n", encoding="utf-8")
    metrics.write_text("{}", encoding="utf-8")
    geometry.write_bytes(b"npz")
    log_path.write_text("log\n", encoding="utf-8")
    command_path.write_text("cmd\n", encoding="utf-8")
    acceptance_report.write_text("{}", encoding="utf-8")

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        "neurcross_dataset_schema_version": "0.1",
        "artifact_type": "neurcross_per_mesh_label",
        "sample_id": "sample-001",
        "created_at_utc": "2026-06-11T00:00:00Z",
        "source": {
            "source_mesh_path": "input/original_mesh.obj",
            "source_mesh_sha256": sha(source_mesh),
            "source_format": "obj",
        },
        "mesh": {
            "normalized_mesh_path": "input/normalized_mesh.ply",
            "normalized_mesh_sha256": sha(normalized_mesh),
            "vertex_count": 3,
            "face_count": 1,
            "is_watertight": False,
        },
        "normalization": {
            "coordinate_space": "normalized",
            "target_bounds": "[-0.5, 0.5]^3",
            "center": [0.0, 0.0, 0.0],
            "scale": 1.0,
        },
        "training": {
            "tool": "neurcross",
            "neurcross_version": "0.1.0",
            "command": "python -m neurcross generate-label",
            "args": {},
            "seed": 1,
            "device": "cuda",
            "started_at_utc": "2026-06-11T00:00:00Z",
            "finished_at_utc": "2026-06-11T00:00:01Z",
            "elapsed_seconds": 1.0,
        },
        "outputs": {
            "selected_label": "best",
            "crossfield_best_vec": "fields/crossfield_best.vec",
            "metrics_best_json": "metrics/train_metrics_best.json",
            "geometry_npz": "geometry/mesh_geometry.npz",
            "log_path": "logs/train.log",
            "command_path": "logs/command.txt",
        },
        "quality": {
            "accepted": True,
            "quality_grade": "A",
            "quality_gate": "default_v0",
            "field_score": 0.1,
            "failure_reason": None,
            "acceptance_report_json": "metrics/acceptance_report.json",
        },
    }

    try:
        validate_manifest(manifest, str(tmp_path))
    except ValueError as exc:
        assert "row count mismatch" in str(exc)
    else:
        raise AssertionError("validate_manifest should reject cross-field row count mismatches")


def test_validate_manifest_allows_missing_optional_geometry_path(tmp_path):
    from quad_mesh.export_dataset_sample import validate_manifest
    import hashlib

    (tmp_path / "input").mkdir()
    (tmp_path / "fields").mkdir()
    (tmp_path / "metrics").mkdir()
    (tmp_path / "logs").mkdir()

    source_mesh = tmp_path / "input" / "original_mesh.obj"
    normalized_mesh = tmp_path / "input" / "normalized_mesh.ply"
    best_vec = tmp_path / "fields" / "crossfield_best.vec"
    metrics = tmp_path / "metrics" / "train_metrics_best.json"
    log_path = tmp_path / "logs" / "train.log"
    command_path = tmp_path / "logs" / "command.txt"
    acceptance_report = tmp_path / "metrics" / "acceptance_report.json"

    source_mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    normalized_mesh.write_text("ply\nformat ascii 1.0\nend_header\n", encoding="utf-8")
    best_vec.write_text("1 0 0 0 1 0\n", encoding="utf-8")
    metrics.write_text("{}", encoding="utf-8")
    log_path.write_text("log\n", encoding="utf-8")
    command_path.write_text("cmd\n", encoding="utf-8")
    acceptance_report.write_text("{}", encoding="utf-8")

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        "neurcross_dataset_schema_version": "0.1",
        "artifact_type": "neurcross_per_mesh_label",
        "sample_id": "sample-001",
        "created_at_utc": "2026-06-11T00:00:00Z",
        "source": {
            "source_mesh_path": "input/original_mesh.obj",
            "source_mesh_sha256": sha(source_mesh),
            "source_format": "obj",
        },
        "mesh": {
            "normalized_mesh_path": "input/normalized_mesh.ply",
            "normalized_mesh_sha256": sha(normalized_mesh),
            "vertex_count": 3,
            "face_count": 1,
            "is_watertight": False,
        },
        "normalization": {
            "coordinate_space": "normalized",
            "target_bounds": "[-0.5, 0.5]^3",
            "center": [0.0, 0.0, 0.0],
            "scale": 1.0,
        },
        "training": {
            "tool": "neurcross",
            "neurcross_version": "0.1.0",
            "command": "python -m neurcross generate-label",
            "args": {},
            "seed": 1,
            "device": "cuda",
            "started_at_utc": "2026-06-11T00:00:00Z",
            "finished_at_utc": "2026-06-11T00:00:01Z",
            "elapsed_seconds": 1.0,
        },
        "outputs": {
            "selected_label": "best",
            "crossfield_best_vec": "fields/crossfield_best.vec",
            "metrics_best_json": "metrics/train_metrics_best.json",
            "geometry_npz": None,
            "log_path": "logs/train.log",
            "command_path": "logs/command.txt",
        },
        "quality": {
            "accepted": True,
            "quality_grade": "A",
            "quality_gate": "strict",
            "field_score": 0.1,
            "failure_reason": None,
            "acceptance_report_json": "metrics/acceptance_report.json",
        },
    }

    validate_manifest(manifest, str(tmp_path))


def test_validate_manifest_accepts_skipped_sample(tmp_path):
    from quad_mesh.export_dataset_sample import validate_manifest
    import hashlib

    (tmp_path / "input").mkdir()
    (tmp_path / "metrics").mkdir()
    (tmp_path / "logs").mkdir()

    source_mesh = tmp_path / "input" / "original_mesh.obj"
    acceptance_report = tmp_path / "metrics" / "acceptance_report.json"
    command_path = tmp_path / "logs" / "command.txt"

    source_mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    acceptance_report.write_text("{}", encoding="utf-8")
    command_path.write_text("cmd\n", encoding="utf-8")

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        "neurcross_dataset_schema_version": "0.1",
        "artifact_type": "neurcross_per_mesh_label",
        "sample_state": "skipped",
        "sample_id": "sample-001",
        "created_at_utc": "2026-06-11T00:00:00Z",
        "source": {
            "source_mesh_path": "input/original_mesh.obj",
            "source_mesh_sha256": sha(source_mesh),
            "source_format": "obj",
        },
        "mesh": {
            "normalized_mesh_path": None,
            "normalized_mesh_sha256": None,
            "vertex_count": 0,
            "face_count": 0,
            "is_watertight": False,
        },
        "normalization": {
            "coordinate_space": "source",
            "target_bounds": None,
            "center": [0.0, 0.0, 0.0],
            "scale": 1.0,
        },
        "training": {
            "tool": "neurcross",
            "neurcross_version": "0.1.0",
            "command": "python -m neurcross generate-label",
            "args": {},
            "seed": 1,
            "device": "cpu",
            "started_at_utc": "2026-06-11T00:00:00Z",
            "finished_at_utc": "2026-06-11T00:00:01Z",
            "elapsed_seconds": 1.0,
        },
        "outputs": {
            "selected_label": "none",
            "crossfield_best_vec": None,
            "metrics_best_json": None,
            "command_path": "logs/command.txt",
        },
        "quality": {
            "accepted": False,
            "quality_grade": "D",
            "quality_gate": "strict",
            "field_score": float("inf"),
            "failure_reason": "mesh_preflight_rejected",
            "acceptance_report_json": "metrics/acceptance_report.json",
        },
    }

    validate_manifest(manifest, str(tmp_path))
