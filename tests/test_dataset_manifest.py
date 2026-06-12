import json
import os

import pytest


COMPLETED_MANIFEST_TOP_LEVEL_KEYS = {
    "artifact_type",
    "created_at_utc",
    "features",
    "mesh",
    "neurcross_dataset_schema_version",
    "normalization",
    "outputs",
    "quality",
    "sample_id",
    "source",
    "training",
}

COMPLETED_MANIFEST_SECTION_KEYS = {
    "source": {
        "author",
        "license",
        "original_filename",
        "source_dataset",
        "source_format",
        "source_mesh_path",
        "source_mesh_sha256",
        "source_url",
    },
    "mesh": {
        "boundary_edge_count",
        "connected_component_count",
        "face_count",
        "is_watertight",
        "nonmanifold_edge_count",
        "normalized_mesh_path",
        "normalized_mesh_sha256",
        "repair_actions",
        "vertex_count",
    },
    "normalization": {
        "center",
        "coordinate_space",
        "normalized_bounds_max",
        "normalized_bounds_min",
        "original_bounds_max",
        "original_bounds_min",
        "scale",
        "target_bounds",
    },
    "features": {
        "feature_angle_threshold",
        "feature_constrained",
        "feature_edge_count",
        "feature_mode",
        "feature_vertex_count",
        "feature_weight_scale",
    },
    "training": {
        "args",
        "command",
        "curriculum",
        "cuda_version",
        "device",
        "elapsed_seconds",
        "finished_at_utc",
        "git_commit",
        "neurcross_version",
        "platform",
        "python_version",
        "seed",
        "started_at_utc",
        "stop_summary",
        "stopped_early",
        "tool",
        "torch_version",
    },
    "outputs": {
        "command_path",
        "crossfield_best_vec",
        "crossfield_final_vec",
        "face_feature_distance_npy",
        "feature_lines_json",
        "feature_vertices_npy",
        "geometry_npz",
        "log_path",
        "metrics_best_json",
        "metrics_final_json",
        "sdf_samples_npz",
        "selected_label",
        "sharp_edges_npy",
        "validation_samples_npz",
    },
    "quality": {
        "accepted",
        "acceptance_report_json",
        "failure_reason",
        "field_score",
        "quality_gate",
        "quality_grade",
        "recommended_destination",
        "validation_history_json",
        "validation_metrics_json",
        "warnings",
    },
}

CURRICULUM_KEYS = {
    "alignment_stage_ratio",
    "final_stage",
    "final_stage_index",
    "geometry_stage_ratio",
    "mode",
    "schedule_unit",
    "smooth_stage_ratio",
    "stage_bounds",
}

COMPLETED_ACCEPTANCE_REPORT_KEYS = {
    "accepted",
    "failed_threshold_checks",
    "failure_reason",
    "field_score",
    "field_smoothness",
    "field_validity",
    "preflight_status",
    "quality_gate",
    "quality_grade",
    "recommended_destination",
    "repair_actions",
    "singularity_proxy",
    "training",
    "warning_threshold_checks",
    "warnings",
}

SKIPPED_ACCEPTANCE_REPORT_KEYS = {
    "accepted",
    "failed_threshold_checks",
    "failure_reason",
    "field_score",
    "preflight_status",
    "quality_gate",
    "quality_grade",
    "recommended_destination",
    "repair_actions",
    "training_skipped",
    "warning_threshold_checks",
    "warnings",
}


def _assert_completed_manifest_schema(manifest):
    assert set(manifest) == COMPLETED_MANIFEST_TOP_LEVEL_KEYS
    for section_name, expected_keys in COMPLETED_MANIFEST_SECTION_KEYS.items():
        assert set(manifest[section_name]) == expected_keys
    assert set(manifest["training"]["curriculum"]) == CURRICULUM_KEYS


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
            "curriculum": {
                "mode": "none",
                "schedule_unit": "step",
                "geometry_stage_ratio": 0.2,
                "alignment_stage_ratio": 0.6,
                "smooth_stage_ratio": 0.2,
                "final_stage": None,
                "final_stage_index": None,
                "stage_bounds": None,
            },
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
            "curriculum": {
                "mode": "none",
                "schedule_unit": "step",
                "geometry_stage_ratio": 0.2,
                "alignment_stage_ratio": 0.6,
                "smooth_stage_ratio": 0.2,
                "final_stage": None,
                "final_stage_index": None,
                "stage_bounds": None,
            },
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
            "curriculum": {
                "mode": "none",
                "schedule_unit": "step",
                "geometry_stage_ratio": 0.2,
                "alignment_stage_ratio": 0.6,
                "smooth_stage_ratio": 0.2,
                "final_stage": None,
                "final_stage_index": None,
                "stage_bounds": None,
            },
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
            "curriculum": {
                "mode": "none",
                "schedule_unit": "step",
                "geometry_stage_ratio": 0.2,
                "alignment_stage_ratio": 0.6,
                "smooth_stage_ratio": 0.2,
                "final_stage": None,
                "final_stage_index": None,
                "stage_bounds": None,
            },
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


def test_validate_manifest_requires_curriculum_section(tmp_path):
    from quad_mesh.export_dataset_sample import validate_manifest

    manifest = {
        "neurcross_dataset_schema_version": "0.1",
        "artifact_type": "neurcross_per_mesh_label",
        "sample_id": "sample-001",
        "created_at_utc": "2026-06-11T00:00:00Z",
        "source": {"source_mesh_path": "input/original_mesh.obj", "source_mesh_sha256": "x", "source_format": "obj"},
        "mesh": {"vertex_count": 3, "face_count": 1, "is_watertight": False},
        "normalization": {"coordinate_space": "normalized", "target_bounds": "[-0.5, 0.5]^3", "center": [0, 0, 0], "scale": 1.0},
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
        "outputs": {"selected_label": "best"},
        "quality": {"accepted": True, "quality_grade": "A", "quality_gate": "default_v0", "field_score": 0.1, "failure_reason": None},
    }

    with pytest.raises(ValueError, match="curriculum"):
        validate_manifest(manifest, str(tmp_path))


def test_golden_completed_manifest_schema_and_acceptance_report_keys():
    manifest = {
        "neurcross_dataset_schema_version": "0.1",
        "artifact_type": "neurcross_per_mesh_label",
        "sample_id": "sample-001",
        "created_at_utc": "2026-06-11T00:00:00Z",
        "source": {
            "source_mesh_path": "input/original_mesh.obj",
            "source_mesh_sha256": "abc",
            "source_format": "obj",
            "source_dataset": None,
            "source_url": None,
            "license": None,
            "author": None,
            "original_filename": "original_mesh.obj",
        },
        "mesh": {
            "normalized_mesh_path": "input/normalized_mesh.ply",
            "normalized_mesh_sha256": "def",
            "vertex_count": 3,
            "face_count": 1,
            "is_watertight": False,
            "connected_component_count": 1,
            "nonmanifold_edge_count": 0,
            "boundary_edge_count": 3,
            "repair_actions": [],
        },
        "normalization": {
            "coordinate_space": "normalized",
            "target_bounds": "[-0.5, 0.5]^3",
            "center": [0.0, 0.0, 0.0],
            "scale": 1.0,
            "original_bounds_min": None,
            "original_bounds_max": None,
            "normalized_bounds_min": None,
            "normalized_bounds_max": None,
        },
        "features": {
            "feature_mode": "auto",
            "feature_angle_threshold": 35.0,
            "feature_weight_scale": 1.0,
            "feature_constrained": False,
            "feature_edge_count": 0,
            "feature_vertex_count": 0,
        },
        "training": {
            "tool": "neurcross",
            "neurcross_version": "0.1.0",
            "command": "python -m neurcross generate-label",
            "args": {},
            "curriculum": {
                "mode": "none",
                "schedule_unit": "step",
                "geometry_stage_ratio": 0.2,
                "alignment_stage_ratio": 0.6,
                "smooth_stage_ratio": 0.2,
                "final_stage": None,
                "final_stage_index": None,
                "stage_bounds": None,
            },
            "seed": 1,
            "device": "cuda",
            "started_at_utc": "2026-06-11T00:00:00Z",
            "finished_at_utc": "2026-06-11T00:00:01Z",
            "elapsed_seconds": 1.0,
            "git_commit": None,
            "python_version": None,
            "torch_version": None,
            "cuda_version": None,
            "platform": None,
            "stopped_early": False,
            "stop_summary": None,
        },
        "outputs": {
            "selected_label": "best",
            "crossfield_best_vec": "fields/crossfield_best.vec",
            "metrics_best_json": "metrics/train_metrics_best.json",
            "crossfield_final_vec": None,
            "metrics_final_json": None,
            "geometry_npz": None,
            "sdf_samples_npz": None,
            "validation_samples_npz": None,
            "sharp_edges_npy": None,
            "feature_vertices_npy": None,
            "feature_lines_json": None,
            "face_feature_distance_npy": None,
            "log_path": "logs/train.log",
            "command_path": "logs/command.txt",
        },
        "quality": {
            "accepted": True,
            "quality_grade": "A",
            "quality_gate": "default_v0",
            "field_score": 0.1,
            "failure_reason": None,
            "recommended_destination": "accepted",
            "warnings": [],
            "validation_metrics_json": None,
            "validation_history_json": None,
            "acceptance_report_json": "metrics/acceptance_report.json",
        },
    }
    acceptance_report = {
        "accepted": True,
        "quality_grade": "A",
        "quality_gate": "default_v0",
        "field_score": 0.1,
        "failure_reason": None,
        "failed_threshold_checks": [],
        "warning_threshold_checks": [],
        "recommended_destination": "accepted",
        "warnings": [],
        "preflight_status": "accepted_for_training",
        "repair_actions": [],
        "field_validity": None,
        "field_smoothness": None,
        "singularity_proxy": None,
        "training": None,
    }

    _assert_completed_manifest_schema(manifest)
    assert set(acceptance_report) == COMPLETED_ACCEPTANCE_REPORT_KEYS


def test_golden_skipped_acceptance_report_schema():
    acceptance_report = {
        "accepted": False,
        "quality_grade": "D",
        "quality_gate": "default",
        "field_score": None,
        "failure_reason": "mesh_preflight_rejected",
        "failed_threshold_checks": ["mesh_preflight_rejected"],
        "warning_threshold_checks": [],
        "recommended_destination": "quarantine",
        "warnings": [],
        "preflight_status": "skip",
        "repair_actions": [],
        "training_skipped": True,
    }

    assert set(acceptance_report) == SKIPPED_ACCEPTANCE_REPORT_KEYS
