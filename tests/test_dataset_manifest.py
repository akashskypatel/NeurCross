import json
import os


def test_validate_manifest_accepts_minimum_schema(tmp_path):
    from quad_mesh.export_dataset_sample import validate_manifest, write_manifest

    input_dir = tmp_path / "input"
    fields_dir = tmp_path / "fields"
    metrics_dir = tmp_path / "metrics"
    logs_dir = tmp_path / "logs"
    input_dir.mkdir()
    fields_dir.mkdir()
    metrics_dir.mkdir()
    logs_dir.mkdir()

    source_mesh = input_dir / "original_mesh.obj"
    normalized_mesh = input_dir / "normalized_mesh.ply"
    best_vec = fields_dir / "crossfield_best.vec"
    best_metrics = metrics_dir / "train_metrics_best.json"
    train_log = logs_dir / "train.log"

    source_mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    normalized_mesh.write_text("ply\nformat ascii 1.0\nend_header\n", encoding="utf-8")
    best_vec.write_text("1 0 0 0 1 0\n", encoding="utf-8")
    best_metrics.write_text("{}", encoding="utf-8")
    train_log.write_text("log\n", encoding="utf-8")

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
            "log_path": "logs/train.log",
        },
        "quality": {
            "accepted": True,
            "quality_grade": "A",
            "quality_gate": "default_v0",
            "field_score": 0.1,
            "failure_reason": None,
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
