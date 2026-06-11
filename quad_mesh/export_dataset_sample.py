from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone

import numpy as np


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _count_crossfield_rows(path: str) -> int:
    count = 0
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.split("#", 1)[0].strip()
            if stripped:
                count += 1
    return count


def _export_geometry_npz(normalized_mesh_obj_path: str, geometry_dir: str, normalization: dict) -> str:
    import trimesh

    os.makedirs(geometry_dir, exist_ok=True)
    mesh = trimesh.load(normalized_mesh_obj_path, force="mesh", process=False)
    path = os.path.join(geometry_dir, "mesh_geometry.npz")
    np.savez(
        path,
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces, dtype=np.int32),
        face_normals=np.asarray(mesh.face_normals, dtype=np.float32),
        vertex_normals=np.asarray(mesh.vertex_normals, dtype=np.float32),
        face_centers=np.asarray(mesh.triangles_center, dtype=np.float32),
        normalization_center=np.asarray(normalization["center"], dtype=np.float32),
        normalization_scale=np.asarray([normalization["scale"]], dtype=np.float32),
        original_bounds_min=np.asarray(normalization.get("bounds_before_min", []), dtype=np.float32),
        original_bounds_max=np.asarray(normalization.get("bounds_before_max", []), dtype=np.float32),
        normalized_bounds_min=np.asarray(normalization.get("bounds_after_min", []), dtype=np.float32),
        normalized_bounds_max=np.asarray(normalization.get("bounds_after_max", []), dtype=np.float32),
    )
    return path


def _copy_if_exists(source_path: str | None, destination_path: str) -> str | None:
    if not source_path or not os.path.exists(source_path):
        return None
    if os.path.abspath(source_path) == os.path.abspath(destination_path):
        return destination_path
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    shutil.copyfile(source_path, destination_path)
    return destination_path


def _copy_or_keep(source_path: str | None, destination_path: str) -> str | None:
    if not source_path or not os.path.exists(source_path):
        return None
    if os.path.abspath(source_path) == os.path.abspath(destination_path):
        return destination_path
    return _copy_if_exists(source_path, destination_path)


def _copy_required(source_path: str, destination_path: str) -> str:
    if not os.path.exists(source_path):
        raise FileNotFoundError(source_path)
    if os.path.abspath(source_path) == os.path.abspath(destination_path):
        return destination_path
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    shutil.copyfile(source_path, destination_path)
    return destination_path


def _write_json(path: str, payload: dict[str, object]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def _rel(root: str, path: str | None) -> str | None:
    if path is None:
        return None
    return os.path.relpath(path, root).replace("\\", "/")


def _quality_from_metrics(best_metrics: dict, *, gate_name: str = "default") -> dict[str, object]:
    field_validity = best_metrics.get("field_validity", {})
    score = float(best_metrics.get("score", best_metrics.get("field_score", 0.0)))
    nan_count = int(field_validity.get("nan_count", 0))
    flipped_frame_ratio = float(field_validity.get("flipped_frame_ratio", 0.0))
    gate_profiles = {
        "none": {"grade_a_score": float("inf"), "grade_b_score": float("inf"), "grade_b_flip": float("inf")},
        "default": {"grade_a_score": 1.0, "grade_b_score": 5.0, "grade_b_flip": 0.05},
        "strict": {"grade_a_score": 0.5, "grade_b_score": 2.5, "grade_b_flip": 0.02},
        "loose": {"grade_a_score": 2.0, "grade_b_score": 8.0, "grade_b_flip": 0.1},
    }
    profile = gate_profiles[gate_name]
    if nan_count > 0:
        grade = "D"
        failure_reason = "field_contains_nonfinite_vectors"
    elif score <= profile["grade_a_score"] and flipped_frame_ratio <= min(0.01, profile["grade_b_flip"]):
        grade = "A"
        failure_reason = None
    elif score <= profile["grade_b_score"] and flipped_frame_ratio <= profile["grade_b_flip"]:
        grade = "B"
        failure_reason = None
    else:
        grade = "C"
        failure_reason = None
    accepted = nan_count == 0
    if gate_name == "strict":
        accepted = accepted and grade in {"A", "B"}
    return {
        "accepted": accepted,
        "quality_grade": grade,
        "quality_gate": gate_name,
        "field_score": score,
        "failure_reason": failure_reason,
    }


def _build_acceptance_report(*, preflight_report: dict, best_metrics: dict, quality: dict[str, object]) -> dict[str, object]:
    return {
        "accepted": bool(quality["accepted"]),
        "quality_grade": quality["quality_grade"],
        "quality_gate": quality["quality_gate"],
        "field_score": float(quality["field_score"]),
        "failure_reason": quality["failure_reason"],
        "warnings": list(preflight_report.get("warnings", [])),
        "preflight_status": preflight_report.get("status"),
        "repair_actions": list(preflight_report.get("repair_actions", [])),
        "field_validity": best_metrics.get("field_validity"),
        "field_smoothness": best_metrics.get("field_smoothness"),
        "singularity_proxy": best_metrics.get("singularity_proxy"),
        "training": best_metrics.get("training"),
    }


def build_manifest(
    *,
    output_dir: str,
    sample_id: str,
    mesh_name: str,
    source_mesh_path: str,
    normalized_mesh_ply_path: str,
    preflight_report: dict,
    normalization: dict,
    args_dict: dict,
    device: str,
    log_path: str,
    created_at_utc: str,
    started_at_utc: str,
    finished_at_utc: str,
    elapsed_seconds: float,
    neurcross_version: str,
    training_command: str,
    stopped_early: bool,
    stop_summary: dict | None,
    runtime_info: dict[str, object] | None = None,
    sdf_samples_path: str | None = None,
    validation_samples_path: str | None = None,
    validation_metrics_path: str | None = None,
    export_geometry_npz: bool = True,
    quality_gate: str = "default",
) -> dict[str, object]:
    runtime_info = runtime_info or {}
    source_mesh_name = os.path.basename(source_mesh_path)
    source_copy_path = os.path.join(output_dir, "input", source_mesh_name)
    normalized_mesh_copy = os.path.join(output_dir, "input", "normalized_mesh.ply")
    normalized_mesh_obj_copy = os.path.join(output_dir, "input", "normalized_mesh.obj")
    fields_dir = os.path.join(output_dir, "fields")
    geometry_dir = os.path.join(output_dir, "geometry")
    metrics_dir = os.path.join(output_dir, "metrics")
    logs_dir = os.path.join(output_dir, "logs")

    original_mesh_path = _copy_required(source_mesh_path, source_copy_path)
    normalized_ply_path = _copy_required(normalized_mesh_ply_path, normalized_mesh_copy)
    normalized_obj_source = preflight_report["artifacts"]["normalized_mesh_obj"]
    _copy_required(normalized_obj_source, normalized_mesh_obj_copy)

    source_format = os.path.splitext(source_mesh_name)[1].lstrip(".").lower()
    best_vec_source = os.path.join(output_dir, "save_crossField", f"{mesh_name}_best.vec")
    final_vec_source = os.path.join(output_dir, "save_crossField", f"{mesh_name}_final.vec")
    best_metrics_source = os.path.join(output_dir, "metrics", f"{mesh_name}_best.json")
    final_metrics_source = os.path.join(output_dir, "metrics", f"{mesh_name}_final.json")

    best_vec_path = _copy_required(best_vec_source, os.path.join(fields_dir, "crossfield_best.vec"))
    final_vec_path = _copy_if_exists(final_vec_source, os.path.join(fields_dir, "crossfield_final.vec"))
    geometry_npz_path = None
    if export_geometry_npz:
        geometry_npz_path = _export_geometry_npz(normalized_obj_source, geometry_dir, normalization)
    best_metrics_path = _copy_required(best_metrics_source, os.path.join(metrics_dir, "train_metrics_best.json"))
    final_metrics_path = _copy_if_exists(final_metrics_source, os.path.join(metrics_dir, "train_metrics_final.json"))
    validation_samples_copy_path = _copy_or_keep(
        validation_samples_path,
        os.path.join(metrics_dir, "validation_samples.npz"),
    )
    validation_metrics_copy_path = _copy_or_keep(
        validation_metrics_path,
        os.path.join(metrics_dir, "validation_metrics.json"),
    )
    log_copy_path = _copy_required(log_path, os.path.join(logs_dir, "train.log"))
    command_txt_path = os.path.join(logs_dir, "command.txt")
    with open(command_txt_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(training_command)
        handle.write("\n")

    best_metrics = _load_json(best_metrics_path)
    quality = _quality_from_metrics(best_metrics, gate_name=quality_gate)
    acceptance_report = _build_acceptance_report(
        preflight_report=preflight_report,
        best_metrics=best_metrics,
        quality=quality,
    )
    acceptance_report_path = os.path.join(metrics_dir, "acceptance_report.json")
    with open(acceptance_report_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(acceptance_report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    manifest = {
        "neurcross_dataset_schema_version": "0.1",
        "artifact_type": "neurcross_per_mesh_label",
        "sample_state": "completed",
        "sample_id": sample_id,
        "created_at_utc": created_at_utc,
        "source": {
            "source_mesh_path": _rel(output_dir, original_mesh_path),
            "source_mesh_sha256": _sha256_file(original_mesh_path),
            "source_format": source_format,
            "source_dataset": None,
            "source_url": None,
            "license": None,
            "author": None,
            "original_filename": source_mesh_name,
        },
        "mesh": {
            "normalized_mesh_path": _rel(output_dir, normalized_ply_path),
            "normalized_mesh_sha256": _sha256_file(normalized_ply_path),
            "vertex_count": preflight_report["metrics"]["vertex_count"],
            "face_count": preflight_report["metrics"]["face_count"],
            "is_watertight": preflight_report["metrics"]["watertight"],
            "connected_component_count": preflight_report["metrics"]["connected_components"],
            "nonmanifold_edge_count": preflight_report["metrics"]["nonmanifold_edges"],
            "boundary_edge_count": preflight_report["metrics"]["boundary_edges"],
            "repair_actions": preflight_report.get("repair_actions", []),
        },
        "normalization": {
            "coordinate_space": "normalized",
            "target_bounds": "[-0.5, 0.5]^3",
            "center": normalization["center"],
            "scale": normalization["scale"],
            "original_bounds_min": normalization.get("bounds_before_min"),
            "original_bounds_max": normalization.get("bounds_before_max"),
            "normalized_bounds_min": normalization.get("bounds_after_min"),
            "normalized_bounds_max": normalization.get("bounds_after_max"),
        },
        "training": {
            "tool": "neurcross",
            "neurcross_version": neurcross_version,
            "command": training_command,
            "args": args_dict,
            "seed": args_dict["seed"],
            "device": device,
            "started_at_utc": started_at_utc,
            "finished_at_utc": finished_at_utc,
            "elapsed_seconds": float(elapsed_seconds),
            "git_commit": runtime_info.get("git_commit"),
            "python_version": runtime_info.get("python_version"),
            "torch_version": runtime_info.get("torch_version"),
            "cuda_version": runtime_info.get("cuda_version"),
            "platform": runtime_info.get("platform"),
            "stopped_early": bool(stopped_early),
            "stop_summary": stop_summary,
        },
        "outputs": {
            "selected_label": "best",
            "crossfield_best_vec": _rel(output_dir, best_vec_path),
            "metrics_best_json": _rel(output_dir, best_metrics_path),
            "crossfield_final_vec": _rel(output_dir, final_vec_path),
            "metrics_final_json": _rel(output_dir, final_metrics_path),
            "geometry_npz": _rel(output_dir, geometry_npz_path),
            "sdf_samples_npz": _rel(output_dir, sdf_samples_path),
            "validation_samples_npz": _rel(output_dir, validation_samples_copy_path),
            "log_path": _rel(output_dir, log_copy_path),
            "command_path": _rel(output_dir, command_txt_path),
        },
        "quality": {
            **quality,
            "warnings": preflight_report.get("warnings", []),
            "validation_metrics_json": _rel(output_dir, validation_metrics_copy_path),
            "acceptance_report_json": _rel(output_dir, acceptance_report_path),
        },
    }
    return manifest


def build_skipped_manifest(
    *,
    output_dir: str,
    sample_id: str,
    source_mesh_path: str,
    preflight_report: dict,
    args_dict: dict,
    device: str,
    created_at_utc: str,
    started_at_utc: str,
    finished_at_utc: str,
    elapsed_seconds: float,
    neurcross_version: str,
    training_command: str,
    quality_gate: str,
    log_path: str | None = None,
    sample_state: str = "skipped",
    failure_reason: str | None = None,
) -> dict[str, object]:
    source_mesh_name = os.path.basename(source_mesh_path)
    source_copy_path = os.path.join(output_dir, "input", source_mesh_name)
    original_mesh_path = _copy_required(source_mesh_path, source_copy_path)
    command_txt_path = os.path.join(output_dir, "logs", "command.txt")
    os.makedirs(os.path.dirname(command_txt_path), exist_ok=True)
    with open(command_txt_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(training_command)
        handle.write("\n")

    log_copy_path = None
    if log_path and os.path.exists(log_path):
        log_copy_path = _copy_required(log_path, os.path.join(output_dir, "logs", "train.log"))

    normalized_mesh_path = None
    normalized_mesh_sha256 = None
    normalization = preflight_report.get("normalization") or {
        "center": [0.0, 0.0, 0.0],
        "scale": 1.0,
        "bounds_before_min": [],
        "bounds_before_max": [],
        "bounds_after_min": [],
        "bounds_after_max": [],
    }
    normalized_ply_source = (preflight_report.get("artifacts") or {}).get("normalized_mesh_ply")
    if normalized_ply_source and os.path.exists(normalized_ply_source):
        normalized_mesh_path = _copy_required(
            normalized_ply_source,
            os.path.join(output_dir, "input", "normalized_mesh.ply"),
        )
        normalized_mesh_sha256 = _sha256_file(normalized_mesh_path)

    quality = {
        "accepted": False,
        "quality_grade": "D",
        "quality_gate": quality_gate,
        "field_score": float("inf"),
        "failure_reason": failure_reason or preflight_report.get("skip_reason") or "mesh_preflight_rejected",
    }
    acceptance_report = {
        "accepted": False,
        "quality_grade": "D",
        "quality_gate": quality_gate,
        "field_score": None,
        "failure_reason": quality["failure_reason"],
        "warnings": list(preflight_report.get("warnings", [])),
        "preflight_status": preflight_report.get("status"),
        "repair_actions": list(preflight_report.get("repair_actions", [])),
        "training_skipped": sample_state == "skipped",
    }
    acceptance_report_path = _write_json(
        os.path.join(output_dir, "metrics", "acceptance_report.json"),
        acceptance_report,
    )

    manifest = {
        "neurcross_dataset_schema_version": "0.1",
        "artifact_type": "neurcross_per_mesh_label",
        "sample_state": sample_state,
        "sample_id": sample_id,
        "created_at_utc": created_at_utc,
        "source": {
            "source_mesh_path": _rel(output_dir, original_mesh_path),
            "source_mesh_sha256": _sha256_file(original_mesh_path),
            "source_format": os.path.splitext(source_mesh_name)[1].lstrip(".").lower(),
            "source_dataset": None,
            "source_url": None,
            "license": None,
            "author": None,
            "original_filename": source_mesh_name,
        },
        "mesh": {
            "normalized_mesh_path": _rel(output_dir, normalized_mesh_path),
            "normalized_mesh_sha256": normalized_mesh_sha256,
            "vertex_count": int(preflight_report.get("metrics", {}).get("vertex_count", 0)),
            "face_count": int(preflight_report.get("metrics", {}).get("face_count", 0)),
            "is_watertight": bool(preflight_report.get("metrics", {}).get("watertight", False)),
            "connected_component_count": int(preflight_report.get("metrics", {}).get("connected_components", 0)),
            "nonmanifold_edge_count": int(preflight_report.get("metrics", {}).get("nonmanifold_edges", 0)),
            "boundary_edge_count": int(preflight_report.get("metrics", {}).get("boundary_edges", 0)),
            "repair_actions": list(preflight_report.get("repair_actions", [])),
        },
        "normalization": {
            "coordinate_space": "normalized" if normalized_mesh_path else "source",
            "target_bounds": "[-0.5, 0.5]^3" if normalized_mesh_path else None,
            "center": normalization.get("center", [0.0, 0.0, 0.0]),
            "scale": normalization.get("scale", 1.0),
            "original_bounds_min": normalization.get("bounds_before_min"),
            "original_bounds_max": normalization.get("bounds_before_max"),
            "normalized_bounds_min": normalization.get("bounds_after_min"),
            "normalized_bounds_max": normalization.get("bounds_after_max"),
        },
        "training": {
            "tool": "neurcross",
            "neurcross_version": neurcross_version,
            "command": training_command,
            "args": args_dict,
            "seed": args_dict["seed"],
            "device": device,
            "started_at_utc": started_at_utc,
            "finished_at_utc": finished_at_utc,
            "elapsed_seconds": float(elapsed_seconds),
            "git_commit": None,
            "python_version": None,
            "torch_version": None,
            "cuda_version": None,
            "platform": None,
            "stopped_early": False,
            "stop_summary": None,
        },
        "outputs": {
            "selected_label": "none",
            "crossfield_best_vec": None,
            "metrics_best_json": None,
            "crossfield_final_vec": None,
            "metrics_final_json": None,
            "geometry_npz": None,
            "sdf_samples_npz": None,
            "validation_samples_npz": None,
            "log_path": _rel(output_dir, log_copy_path),
            "command_path": _rel(output_dir, command_txt_path),
        },
        "quality": {
            **quality,
            "warnings": list(preflight_report.get("warnings", [])),
            "validation_metrics_json": None,
            "acceptance_report_json": _rel(output_dir, acceptance_report_path),
        },
    }
    return manifest


def validate_manifest(manifest: dict, output_dir: str) -> None:
    required_top_level = (
        "neurcross_dataset_schema_version",
        "artifact_type",
        "sample_id",
        "created_at_utc",
        "source",
        "mesh",
        "normalization",
        "training",
        "outputs",
        "quality",
    )
    for key in required_top_level:
        if key not in manifest:
            raise ValueError(f"manifest missing required top-level field: {key}")
    if manifest["neurcross_dataset_schema_version"] != "0.1":
        raise ValueError(f"unsupported schema version: {manifest['neurcross_dataset_schema_version']}")

    required_section_fields = {
        "source": ("source_mesh_path", "source_mesh_sha256", "source_format"),
        "mesh": ("vertex_count", "face_count", "is_watertight"),
        "normalization": ("coordinate_space", "target_bounds", "center", "scale"),
        "training": ("tool", "neurcross_version", "command", "args", "seed", "device", "started_at_utc", "finished_at_utc", "elapsed_seconds"),
        "outputs": ("selected_label",),
        "quality": ("accepted", "quality_grade", "quality_gate", "field_score", "failure_reason"),
    }
    for section_name, fields in required_section_fields.items():
        section = manifest.get(section_name)
        if not isinstance(section, dict):
            raise ValueError(f"manifest section must be an object: {section_name}")
        for field in fields:
            if field not in section:
                raise ValueError(f"manifest section '{section_name}' missing field: {field}")

    sample_state = manifest.get("sample_state", "completed")

    path_fields = (
        ("source", "source_mesh_path"),
        ("mesh", "normalized_mesh_path"),
        ("outputs", "crossfield_best_vec"),
        ("outputs", "metrics_best_json"),
        ("outputs", "geometry_npz"),
        ("outputs", "validation_samples_npz"),
        ("outputs", "log_path"),
        ("outputs", "command_path"),
        ("quality", "validation_metrics_json"),
        ("quality", "acceptance_report_json"),
    )
    for section_name, field_name in path_fields:
        value = manifest[section_name].get(field_name)
        if value:
            path = os.path.join(output_dir, value)
            if not os.path.exists(path):
                raise ValueError(f"manifest path does not exist: {section_name}.{field_name} -> {value}")

    source_path = os.path.join(output_dir, manifest["source"]["source_mesh_path"])
    if _sha256_file(source_path) != manifest["source"]["source_mesh_sha256"]:
        raise ValueError("source mesh sha256 does not match manifest")
    if sample_state == "completed":
        normalized_path = os.path.join(output_dir, manifest["mesh"]["normalized_mesh_path"])
        if _sha256_file(normalized_path) != manifest["mesh"]["normalized_mesh_sha256"]:
            raise ValueError("normalized mesh sha256 does not match manifest")
        expected_rows = int(manifest["mesh"]["face_count"])
        actual_rows = _count_crossfield_rows(os.path.join(output_dir, manifest["outputs"]["crossfield_best_vec"]))
        if actual_rows != expected_rows:
            raise ValueError(
                f"cross-field row count mismatch: expected {expected_rows} rows from face_count, got {actual_rows}"
            )


def write_manifest(manifest: dict, output_dir: str) -> str:
    validate_manifest(manifest, output_dir)
    path = os.path.join(output_dir, "manifest.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def package_dataset_sample(
    *,
    output_dir: str,
    sample_id: str,
    mesh_name: str,
    source_mesh_path: str,
    normalized_mesh_ply_path: str,
    preflight_report: dict,
    normalization: dict,
    args_dict: dict,
    device: str,
    log_path: str,
    started_at_utc: str,
    finished_at_utc: str,
    elapsed_seconds: float,
    neurcross_version: str,
    training_command: str,
    stopped_early: bool,
    stop_summary: dict | None,
    runtime_info: dict[str, object] | None = None,
    sdf_samples_path: str | None = None,
    validation_samples_path: str | None = None,
    validation_metrics_path: str | None = None,
    export_geometry_npz: bool = True,
    quality_gate: str = "default",
) -> str:
    created_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    manifest = build_manifest(
        output_dir=output_dir,
        sample_id=sample_id,
        mesh_name=mesh_name,
        source_mesh_path=source_mesh_path,
        normalized_mesh_ply_path=normalized_mesh_ply_path,
        preflight_report=preflight_report,
        normalization=normalization,
        args_dict=args_dict,
        device=device,
        log_path=log_path,
        created_at_utc=created_at_utc,
        started_at_utc=started_at_utc,
        finished_at_utc=finished_at_utc,
        elapsed_seconds=elapsed_seconds,
        neurcross_version=neurcross_version,
        training_command=training_command,
        stopped_early=stopped_early,
        stop_summary=stop_summary,
        runtime_info=runtime_info,
        sdf_samples_path=sdf_samples_path,
        validation_samples_path=validation_samples_path,
        validation_metrics_path=validation_metrics_path,
        export_geometry_npz=export_geometry_npz,
        quality_gate=quality_gate,
    )
    return write_manifest(manifest, output_dir)


def package_skipped_dataset_sample(
    *,
    output_dir: str,
    sample_id: str,
    source_mesh_path: str,
    preflight_report: dict,
    args_dict: dict,
    device: str,
    started_at_utc: str,
    finished_at_utc: str,
    elapsed_seconds: float,
    neurcross_version: str,
    training_command: str,
    quality_gate: str,
    log_path: str | None = None,
) -> str:
    created_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    manifest = build_skipped_manifest(
        output_dir=output_dir,
        sample_id=sample_id,
        source_mesh_path=source_mesh_path,
        preflight_report=preflight_report,
        args_dict=args_dict,
        device=device,
        created_at_utc=created_at_utc,
        started_at_utc=started_at_utc,
        finished_at_utc=finished_at_utc,
        elapsed_seconds=elapsed_seconds,
        neurcross_version=neurcross_version,
        training_command=training_command,
        quality_gate=quality_gate,
        log_path=log_path,
    )
    return write_manifest(manifest, output_dir)


def package_failed_dataset_sample(
    *,
    output_dir: str,
    sample_id: str,
    source_mesh_path: str,
    preflight_report: dict,
    args_dict: dict,
    device: str,
    started_at_utc: str,
    finished_at_utc: str,
    elapsed_seconds: float,
    neurcross_version: str,
    training_command: str,
    quality_gate: str,
    failure_reason: str,
    log_path: str | None = None,
) -> str:
    created_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    manifest = build_skipped_manifest(
        output_dir=output_dir,
        sample_id=sample_id,
        source_mesh_path=source_mesh_path,
        preflight_report=preflight_report,
        args_dict=args_dict,
        device=device,
        created_at_utc=created_at_utc,
        started_at_utc=started_at_utc,
        finished_at_utc=finished_at_utc,
        elapsed_seconds=elapsed_seconds,
        neurcross_version=neurcross_version,
        training_command=training_command,
        quality_gate=quality_gate,
        log_path=log_path,
        sample_state="failed",
        failure_reason=failure_reason,
    )
    return write_manifest(manifest, output_dir)
