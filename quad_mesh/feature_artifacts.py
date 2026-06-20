from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class EdgeQueryTeacherConfig:
    angle_start_degrees: float = 25.0
    angle_full_degrees: float = 75.0
    high_threshold: float = 0.65
    low_threshold: float = 0.30
    min_component_edges: int = 4
    keep_boundaries: bool = True

    # Heuristic quality gate for pseudo-label acceptance.
    min_positive_ratio: float = 0.0005
    max_positive_ratio: float = 0.35
    max_branch_vertex_ratio: float = 0.20


def _smoothstep_scalar(x: float, edge0: float, edge1: float) -> float:
    if edge1 <= edge0:
        return 1.0 if x >= edge1 else 0.0
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return float(t * t * (3.0 - 2.0 * t))


def _boundary_edges(mesh: trimesh.Trimesh) -> np.ndarray:
    edges = np.asarray(mesh.edges_sorted, dtype=np.int64)
    if edges.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    return unique_edges[counts == 1]


def _sharp_edges(mesh: trimesh.Trimesh, angle_threshold_degrees: float) -> np.ndarray:
    adjacency_edges = np.asarray(mesh.face_adjacency_edges, dtype=np.int64)
    adjacency_angles = np.asarray(mesh.face_adjacency_angles, dtype=np.float64)
    if adjacency_edges.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    threshold_radians = np.deg2rad(float(angle_threshold_degrees))
    keep_mask = adjacency_angles >= threshold_radians
    return np.sort(adjacency_edges[keep_mask], axis=1)


def _feature_lines(feature_edges: np.ndarray) -> list[dict[str, object]]:
    if feature_edges.size == 0:
        return []
    adjacency: dict[int, set[int]] = {}
    for a, b in feature_edges.tolist():
        adjacency.setdefault(int(a), set()).add(int(b))
        adjacency.setdefault(int(b), set()).add(int(a))

    lines = []
    visited: set[int] = set()
    for start in sorted(adjacency):
        if start in visited:
            continue
        stack = [start]
        component_vertices = []
        component_set = set()
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component_set.add(node)
            component_vertices.append(node)
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    stack.append(neighbor)
        component_edges = [
            [int(a), int(b)]
            for a, b in feature_edges.tolist()
            if int(a) in component_set and int(b) in component_set
        ]
        lines.append(
            {
                "vertex_indices": sorted(component_vertices),
                "edge_count": len(component_edges),
                "edges": component_edges,
            }
        )
    return lines


def _build_unique_edge_incident_faces(
    faces_unique_edges: np.ndarray,
    num_edges: int,
) -> list[list[int]]:
    incident_faces_per_edge: list[list[int]] = [[] for _ in range(num_edges)]

    for face_idx, face_edge_ids in enumerate(faces_unique_edges):
        for edge_idx in face_edge_ids:
            incident_faces_per_edge[int(edge_idx)].append(int(face_idx))

    return incident_faces_per_edge


def _compute_edge_dihedral_angles(
    *,
    face_normals_np: np.ndarray,
    incident_faces_per_edge: list[list[int]],
) -> np.ndarray:
    """
    Returns per-unique-edge dihedral angle in radians.

    Boundary edges and edges with fewer than two incident faces are assigned NaN,
    because they do not have a well-defined two-face dihedral angle.
    """
    dihedral_angles = np.full((len(incident_faces_per_edge),), np.nan, dtype=np.float32)

    for edge_idx, incident_faces in enumerate(incident_faces_per_edge):
        if len(incident_faces) < 2:
            continue

        normals = face_normals_np[np.asarray(incident_faces, dtype=np.int64)]
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)

        # Manifold meshes have two incident faces. Non-manifold edges use the
        # largest pairwise dihedral angle as a conservative sharpness signal.
        dots = np.clip(normals @ normals.T, -1.0, 1.0)
        upper = dots[np.triu_indices(len(incident_faces), k=1)]
        min_dot = float(np.min(upper)) if upper.size else 1.0
        dihedral_angles[edge_idx] = float(np.arccos(min_dot))

    return dihedral_angles


def _compute_edge_sharpness(
    *,
    face_normals_np: np.ndarray,
    incident_faces_per_edge: list[list[int]],
    config: EdgeQueryTeacherConfig,
) -> np.ndarray:
    start = np.deg2rad(float(config.angle_start_degrees))
    full = np.deg2rad(float(config.angle_full_degrees))

    dihedral_angles = _compute_edge_dihedral_angles(
        face_normals_np=face_normals_np,
        incident_faces_per_edge=incident_faces_per_edge,
    )
    sharpness = np.zeros((len(incident_faces_per_edge),), dtype=np.float32)

    for edge_idx, incident_faces in enumerate(incident_faces_per_edge):
        if len(incident_faces) == 1:
            sharpness[edge_idx] = 1.0 if config.keep_boundaries else 0.0
            continue

        if len(incident_faces) < 2:
            sharpness[edge_idx] = 0.0
            continue

        sharpness[edge_idx] = _smoothstep_scalar(
            float(dihedral_angles[edge_idx]),
            start,
            full,
        )

    return sharpness


def _edge_adjacency(edges_unique_np: np.ndarray) -> list[list[int]]:
    vertex_to_edges: dict[int, list[int]] = {}

    for edge_idx, (a, b) in enumerate(edges_unique_np):
        vertex_to_edges.setdefault(int(a), []).append(edge_idx)
        vertex_to_edges.setdefault(int(b), []).append(edge_idx)

    adjacency = [set() for _ in range(edges_unique_np.shape[0])]

    for edge_ids in vertex_to_edges.values():
        for edge_idx in edge_ids:
            adjacency[edge_idx].update(
                other for other in edge_ids if other != edge_idx
            )

    return [sorted(items) for items in adjacency]


def _hysteresis_select_edges(
    edge_sharpness: np.ndarray,
    adjacency: list[list[int]],
    config: EdgeQueryTeacherConfig,
) -> np.ndarray:
    strong = edge_sharpness >= float(config.high_threshold)
    weak = edge_sharpness >= float(config.low_threshold)

    selected = np.zeros(edge_sharpness.shape[0], dtype=bool)
    stack = [int(i) for i in np.flatnonzero(strong)]

    while stack:
        edge_idx = stack.pop()
        if selected[edge_idx]:
            continue

        selected[edge_idx] = True

        for neighbor_idx in adjacency[edge_idx]:
            if weak[neighbor_idx] and not selected[neighbor_idx]:
                stack.append(int(neighbor_idx))

    return selected


def _connected_edge_components(
    selected_edges: np.ndarray,
    adjacency: list[list[int]],
) -> list[list[int]]:
    visited = np.zeros_like(selected_edges, dtype=bool)
    components: list[list[int]] = []

    for start in np.flatnonzero(selected_edges):
        if visited[start]:
            continue

        stack = [int(start)]
        component: list[int] = []

        while stack:
            edge_idx = stack.pop()
            if visited[edge_idx] or not selected_edges[edge_idx]:
                continue

            visited[edge_idx] = True
            component.append(edge_idx)

            for neighbor_idx in adjacency[edge_idx]:
                if selected_edges[neighbor_idx] and not visited[neighbor_idx]:
                    stack.append(int(neighbor_idx))

        components.append(component)

    return components


def _filter_components(
    components: list[list[int]],
    *,
    min_component_edges: int,
) -> tuple[list[list[int]], int]:
    kept: list[list[int]] = []
    tiny_count = 0

    for component in components:
        if len(component) >= min_component_edges:
            kept.append(component)
        else:
            tiny_count += 1

    return kept, tiny_count


def _components_to_structure_lines(
    *,
    edges_unique_np: np.ndarray,
    edge_sharpness: np.ndarray,
    components: list[list[int]],
) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []

    for line_idx, component in enumerate(components):
        edge_indices = np.asarray(component, dtype=np.int64)
        component_edges = edges_unique_np[edge_indices]
        vertex_indices = sorted(set(component_edges.reshape(-1).tolist()))
        component_sharpness = edge_sharpness[edge_indices]

        lines.append(
            {
                "line_index": int(line_idx),
                "edge_indices": edge_indices.astype(int).tolist(),
                "vertex_indices": [int(v) for v in vertex_indices],
                "edges": component_edges.astype(int).tolist(),
                "edge_count": int(edge_indices.shape[0]),
                "mean_sharpness": float(component_sharpness.mean()),
                "max_sharpness": float(component_sharpness.max()),
            }
        )

    return lines


def _selected_vertex_degrees(
    *,
    edges_unique_np: np.ndarray,
    selected_edges: np.ndarray,
    vertex_count: int,
) -> np.ndarray:
    degrees = np.zeros((vertex_count,), dtype=np.int32)

    for a, b in edges_unique_np[selected_edges]:
        degrees[int(a)] += 1
        degrees[int(b)] += 1

    return degrees


def _build_quality_metrics(
    *,
    edge_count: int,
    vertex_count: int,
    selected_edges: np.ndarray,
    raw_component_count: int,
    kept_component_count: int,
    tiny_component_count: int,
    vertex_degrees: np.ndarray,
    edge_is_boundary: np.ndarray | None = None,
    edge_dihedral_angles: np.ndarray | None = None,
    config: EdgeQueryTeacherConfig,
) -> dict[str, Any]:
    selected_edge_count = int(np.count_nonzero(selected_edges))
    positive_ratio = selected_edge_count / max(edge_count, 1)

    endpoint_count = int(np.count_nonzero(vertex_degrees == 1))
    branch_vertex_count = int(np.count_nonzero(vertex_degrees > 2))
    used_vertex_count = int(np.count_nonzero(vertex_degrees > 0))
    branch_vertex_ratio = branch_vertex_count / max(used_vertex_count, 1)

    boundary_edge_count = int(np.count_nonzero(edge_is_boundary)) if edge_is_boundary is not None else 0
    interior_dihedral = np.asarray([], dtype=np.float32)
    if edge_dihedral_angles is not None:
        finite = np.isfinite(edge_dihedral_angles)
        interior_dihedral = edge_dihedral_angles[finite]

    failed_checks: list[str] = []
    warning_checks: list[str] = []

    if selected_edge_count == 0:
        failed_checks.append("no_structure_edges_selected")

    if positive_ratio < config.min_positive_ratio:
        warning_checks.append("structure_positive_ratio_below_minimum")

    if positive_ratio > config.max_positive_ratio:
        failed_checks.append("structure_positive_ratio_above_maximum")

    if branch_vertex_ratio > config.max_branch_vertex_ratio:
        warning_checks.append("branch_vertex_ratio_high")

    if kept_component_count == 0 and selected_edge_count > 0:
        failed_checks.append("all_structure_components_filtered")

    if tiny_component_count > 0:
        warning_checks.append("tiny_structure_components_removed")

    accepted = len(failed_checks) == 0

    if not accepted:
        grade = "D"
        recommended_destination = "quarantine"
    elif not warning_checks:
        grade = "A"
        recommended_destination = "accepted"
    else:
        grade = "B"
        recommended_destination = "accepted"

    return {
        "accepted": accepted,
        "quality_grade": grade,
        "recommended_destination": recommended_destination,
        "failed_checks": failed_checks,
        "warning_checks": warning_checks,
        "edge_count": int(edge_count),
        "vertex_count": int(vertex_count),
        "selected_edge_count": selected_edge_count,
        "positive_ratio": float(positive_ratio),
        "raw_component_count": int(raw_component_count),
        "kept_component_count": int(kept_component_count),
        "tiny_component_count": int(tiny_component_count),
        "endpoint_count": endpoint_count,
        "branch_vertex_count": branch_vertex_count,
        "used_vertex_count": used_vertex_count,
        "branch_vertex_ratio": float(branch_vertex_ratio),
        "boundary_edge_count": boundary_edge_count,
        "interior_dihedral_angle_mean_degrees": float(np.rad2deg(interior_dihedral).mean()) if interior_dihedral.size else None,
        "interior_dihedral_angle_max_degrees": float(np.rad2deg(interior_dihedral).max()) if interior_dihedral.size else None,
        "config": {
            "angle_start_degrees": float(config.angle_start_degrees),
            "angle_full_degrees": float(config.angle_full_degrees),
            "high_threshold": float(config.high_threshold),
            "low_threshold": float(config.low_threshold),
            "min_component_edges": int(config.min_component_edges),
            "keep_boundaries": bool(config.keep_boundaries),
            "min_positive_ratio": float(config.min_positive_ratio),
            "max_positive_ratio": float(config.max_positive_ratio),
            "max_branch_vertex_ratio": float(config.max_branch_vertex_ratio),
        },
    }


def _structure_from_sharpness(
    *,
    mesh: trimesh.Trimesh,
    edges_unique_np: np.ndarray,
    face_normals_np: np.ndarray,
    incident_faces_per_edge: list[list[int]],
    config: EdgeQueryTeacherConfig,
) -> dict[str, Any]:
    edge_sharpness = _compute_edge_sharpness(
        face_normals_np=face_normals_np,
        incident_faces_per_edge=incident_faces_per_edge,
        config=config,
    )
    edge_dihedral_angles = _compute_edge_dihedral_angles(
        face_normals_np=face_normals_np,
        incident_faces_per_edge=incident_faces_per_edge,
    )
    edge_is_boundary = np.asarray(
        [len(incident_faces) == 1 for incident_faces in incident_faces_per_edge],
        dtype=bool,
    )

    adjacency = _edge_adjacency(edges_unique_np)
    selected_edges_raw = _hysteresis_select_edges(edge_sharpness, adjacency, config)
    raw_components = _connected_edge_components(selected_edges_raw, adjacency)
    kept_components, tiny_component_count = _filter_components(
        raw_components,
        min_component_edges=int(config.min_component_edges),
    )

    selected_edges_clean = np.zeros((edges_unique_np.shape[0],), dtype=bool)
    for component in kept_components:
        selected_edges_clean[np.asarray(component, dtype=np.int64)] = True

    structure_lines = _components_to_structure_lines(
        edges_unique_np=edges_unique_np,
        edge_sharpness=edge_sharpness,
        components=kept_components,
    )
    vertex_degrees = _selected_vertex_degrees(
        edges_unique_np=edges_unique_np,
        selected_edges=selected_edges_clean,
        vertex_count=int(mesh.vertices.shape[0]),
    )
    quality_metrics = _build_quality_metrics(
        edge_count=int(edges_unique_np.shape[0]),
        vertex_count=int(mesh.vertices.shape[0]),
        selected_edges=selected_edges_clean,
        raw_component_count=len(raw_components),
        kept_component_count=len(kept_components),
        tiny_component_count=tiny_component_count,
        vertex_degrees=vertex_degrees,
        edge_is_boundary=edge_is_boundary,
        edge_dihedral_angles=edge_dihedral_angles,
        config=config,
    )

    return {
        "edge_sharpness": edge_sharpness.astype(np.float32),
        "edge_dihedral_angles": edge_dihedral_angles.astype(np.float32),
        "edge_dihedral_degrees": np.rad2deg(edge_dihedral_angles).astype(np.float32),
        "edge_is_boundary": edge_is_boundary,
        "salient_edge_mask": (edge_sharpness >= float(config.low_threshold)),
        "structure_edge_labels": selected_edges_clean.astype(np.float32),
        "structure_lines": structure_lines,
        "structure_quality_metrics": quality_metrics,
    }


def _edge_set_to_unique_mask(edges_unique_np: np.ndarray, edges: np.ndarray) -> np.ndarray:
    mask = np.zeros((edges_unique_np.shape[0],), dtype=bool)
    if edges.size == 0:
        return mask

    edge_to_index = {
        tuple(edge.tolist()): int(idx)
        for idx, edge in enumerate(np.sort(edges_unique_np, axis=1))
    }
    for edge in np.sort(edges.astype(np.int64), axis=1):
        idx = edge_to_index.get(tuple(edge.tolist()))
        if idx is not None:
            mask[idx] = True
    return mask


def detect_mesh_features(
    mesh: trimesh.Trimesh,
    *,
    angle_threshold_degrees: float,
    sharpness_config: EdgeQueryTeacherConfig | None = None,
) -> dict[str, object]:
    """
    Detect hard feature edges and continuous dihedral/boundary sharpness.

    Backward-compatible outputs are preserved:
        sharp_edges, boundary_edges, feature_edges, feature_vertices,
        feature_lines, face_feature_distance

    New training-oriented outputs are added:
        edges_unique, edge_sharpness, edge_dihedral_degrees,
        edge_is_boundary, salient_edge_mask, structure_edge_labels,
        structure_lines, structure_quality_metrics
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("mesh must be a trimesh.Trimesh")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("mesh must contain vertices and faces")

    sharpness_config = sharpness_config or EdgeQueryTeacherConfig()

    edges_unique = np.asarray(mesh.edges_unique, dtype=np.int64).copy()
    faces_unique_edges = np.asarray(mesh.faces_unique_edges, dtype=np.int64).copy()
    face_normals = np.asarray(mesh.face_normals, dtype=np.float32)
    incident_faces_per_edge = _build_unique_edge_incident_faces(
        faces_unique_edges,
        int(edges_unique.shape[0]),
    )

    structure = _structure_from_sharpness(
        mesh=mesh,
        edges_unique_np=edges_unique,
        face_normals_np=face_normals,
        incident_faces_per_edge=incident_faces_per_edge,
        config=sharpness_config,
    )

    sharp_edges = _sharp_edges(mesh, angle_threshold_degrees)
    boundary_edges = _boundary_edges(mesh)
    if sharp_edges.size and boundary_edges.size:
        feature_edges = np.unique(np.concatenate((sharp_edges, boundary_edges), axis=0), axis=0)
    elif sharp_edges.size:
        feature_edges = sharp_edges
    elif boundary_edges.size:
        feature_edges = boundary_edges
    else:
        feature_edges = np.zeros((0, 2), dtype=np.int64)

    feature_vertices = (
        np.unique(feature_edges.reshape(-1)).astype(np.int64)
        if feature_edges.size
        else np.zeros((0,), dtype=np.int64)
    )

    hard_feature_edge_mask = _edge_set_to_unique_mask(edges_unique, feature_edges)
    sharp_edge_mask = _edge_set_to_unique_mask(edges_unique, sharp_edges)
    boundary_edge_mask = _edge_set_to_unique_mask(edges_unique, boundary_edges)

    face_centers = np.asarray(mesh.triangles_center, dtype=np.float32)
    if feature_edges.size:
        edge_midpoints = mesh.vertices[feature_edges].mean(axis=1)
        tree = cKDTree(edge_midpoints)
        distances, _ = tree.query(face_centers, k=1)
        face_feature_distance = np.asarray(distances, dtype=np.float32)
    else:
        face_feature_distance = np.full((face_centers.shape[0],), np.inf, dtype=np.float32)

    return {
        "sharp_edges": sharp_edges.astype(np.int64),
        "boundary_edges": boundary_edges.astype(np.int64),
        "feature_edges": feature_edges.astype(np.int64),
        "feature_vertices": feature_vertices,
        "feature_lines": _feature_lines(feature_edges),
        "face_feature_distance": face_feature_distance,
        "edges_unique": edges_unique.astype(np.int64),
        "sharp_edge_mask": sharp_edge_mask,
        "boundary_edge_mask": boundary_edge_mask,
        "hard_feature_edge_mask": hard_feature_edge_mask,
        "edge_sharpness": structure["edge_sharpness"],
        "edge_dihedral_angles": structure["edge_dihedral_angles"],
        "edge_dihedral_degrees": structure["edge_dihedral_degrees"],
        "edge_is_boundary": structure["edge_is_boundary"],
        "salient_edge_mask": structure["salient_edge_mask"],
        "structure_edge_labels": structure["structure_edge_labels"],
        "structure_lines": structure["structure_lines"],
        "structure_quality_metrics": structure["structure_quality_metrics"],
    }


def export_feature_artifacts(
    *,
    mesh: trimesh.Trimesh,
    output_dir: str,
    feature_mode: str,
    angle_threshold_degrees: float,
    sharpness_config: EdgeQueryTeacherConfig | None = None,
) -> dict[str, object]:
    mode = str(feature_mode).lower()
    if mode == "none":
        return {
            "feature_mode": mode,
            "sharp_edges_path": None,
            "feature_vertices_path": None,
            "feature_lines_path": None,
            "face_feature_distance_path": None,
            "edges_unique_path": None,
            "edge_sharpness_path": None,
            "edge_dihedral_degrees_path": None,
            "edge_is_boundary_path": None,
            "salient_edge_mask_path": None,
            "structure_edge_labels_path": None,
            "structure_lines_path": None,
            "structure_quality_metrics_path": None,
            "feature_edge_count": 0,
            "feature_vertex_count": 0,
            "boundary_edge_count": 0,
            "sharp_edge_count": 0,
            "structure_edge_count": 0,
            "edge_sharpness_mean": 0.0,
            "edge_sharpness_max": 0.0,
            "feature_constrained": False,
        }
    if mode == "file":
        raise NotImplementedError("--feature_mode file is not implemented yet")
    if mode != "auto":
        raise ValueError(f"unsupported feature mode: {feature_mode}")

    os.makedirs(output_dir, exist_ok=True)
    detected = detect_mesh_features(
        mesh,
        angle_threshold_degrees=angle_threshold_degrees,
        sharpness_config=sharpness_config,
    )

    sharp_edges_path = os.path.join(output_dir, "sharp_edges.npy")
    feature_vertices_path = os.path.join(output_dir, "feature_vertices.npy")
    feature_lines_path = os.path.join(output_dir, "feature_lines.json")
    face_feature_distance_path = os.path.join(output_dir, "face_feature_distance.npy")

    edges_unique_path = os.path.join(output_dir, "edges_unique.npy")
    edge_sharpness_path = os.path.join(output_dir, "edge_sharpness.npy")
    edge_dihedral_degrees_path = os.path.join(output_dir, "edge_dihedral_degrees.npy")
    edge_is_boundary_path = os.path.join(output_dir, "edge_is_boundary.npy")
    salient_edge_mask_path = os.path.join(output_dir, "salient_edge_mask.npy")
    structure_edge_labels_path = os.path.join(output_dir, "structure_edge_labels.npy")
    structure_lines_path = os.path.join(output_dir, "structure_lines.json")
    structure_quality_metrics_path = os.path.join(output_dir, "structure_quality_metrics.json")

    # Backward-compatible feature artifacts. sharp_edges.npy intentionally keeps
    # storing the combined feature edge set, matching the previous behavior.
    np.save(sharp_edges_path, detected["feature_edges"])
    np.save(feature_vertices_path, detected["feature_vertices"])
    np.save(face_feature_distance_path, detected["face_feature_distance"])
    with open(feature_lines_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(detected["feature_lines"], handle, indent=2, sort_keys=True)
        handle.write("\n")

    # New structural-training artifacts.
    np.save(edges_unique_path, detected["edges_unique"])
    np.save(edge_sharpness_path, detected["edge_sharpness"])
    np.save(edge_dihedral_degrees_path, detected["edge_dihedral_degrees"])
    np.save(edge_is_boundary_path, detected["edge_is_boundary"])
    np.save(salient_edge_mask_path, detected["salient_edge_mask"])
    np.save(structure_edge_labels_path, detected["structure_edge_labels"])
    with open(structure_lines_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(detected["structure_lines"], handle, indent=2, sort_keys=True)
        handle.write("\n")
    with open(structure_quality_metrics_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(detected["structure_quality_metrics"], handle, indent=2, sort_keys=True)
        handle.write("\n")

    edge_sharpness = np.asarray(detected["edge_sharpness"], dtype=np.float32)
    structure_edge_labels = np.asarray(detected["structure_edge_labels"], dtype=np.float32)

    return {
        "feature_mode": mode,
        "sharp_edges_path": sharp_edges_path,
        "feature_vertices_path": feature_vertices_path,
        "feature_lines_path": feature_lines_path,
        "face_feature_distance_path": face_feature_distance_path,
        "edges_unique_path": edges_unique_path,
        "edge_sharpness_path": edge_sharpness_path,
        "edge_dihedral_degrees_path": edge_dihedral_degrees_path,
        "edge_is_boundary_path": edge_is_boundary_path,
        "salient_edge_mask_path": salient_edge_mask_path,
        "structure_edge_labels_path": structure_edge_labels_path,
        "structure_lines_path": structure_lines_path,
        "structure_quality_metrics_path": structure_quality_metrics_path,
        "feature_edge_count": int(detected["feature_edges"].shape[0]),
        "feature_vertex_count": int(detected["feature_vertices"].shape[0]),
        "boundary_edge_count": int(detected["boundary_edges"].shape[0]),
        "sharp_edge_count": int(detected["sharp_edges"].shape[0]),
        "structure_edge_count": int(np.count_nonzero(structure_edge_labels > 0.5)),
        "edge_sharpness_mean": float(edge_sharpness.mean()) if edge_sharpness.size else 0.0,
        "edge_sharpness_max": float(edge_sharpness.max()) if edge_sharpness.size else 0.0,
        "feature_constrained": bool(detected["feature_edges"].shape[0] > 0),
    }


def _compute_edge_normals(
    edges_unique: torch.Tensor,
    vertex_normals: torch.Tensor,
    face_normals: torch.Tensor,
    incident_faces_per_edge: list[list[int]],
) -> torch.Tensor:
    edge_normals = torch.zeros((edges_unique.shape[0], 3), dtype=torch.float32)

    for edge_idx, incident_faces in enumerate(incident_faces_per_edge):
        if incident_faces:
            face_ids = torch.as_tensor(incident_faces, dtype=torch.long)
            normal = face_normals[face_ids].mean(dim=0)
        else:
            # Defensive fallback. For normal trimesh edges this should not happen.
            v0, v1 = edges_unique[edge_idx]
            normal = 0.5 * (vertex_normals[v0] + vertex_normals[v1])

        edge_normals[edge_idx] = F.normalize(normal, dim=0, eps=1e-12)

    return edge_normals


def construct_mesh_aligned_edge_queries(
    mesh: trimesh.Trimesh,
    *,
    config: EdgeQueryTeacherConfig = EdgeQueryTeacherConfig(),
) -> dict[str, Any]:
    """
    TopGen-style mesh-aligned edge query construction plus per-mesh
    sharp-feature pseudo-label export.

    Returns:
        edge_queries:              FloatTensor [E, 6]
        neighborhood_features:     FloatTensor [E, K, 6]
        neighborhood_masks:        BoolTensor  [E, K], True = valid
        edges_unique:              LongTensor  [E, 2]
        edge_to_faces:             LongTensor  [E, M], -1 = padding

        edge_sharpness:            FloatTensor [E], continuous [0, 1]
        edge_dihedral_degrees:     FloatTensor [E], NaN for boundary edges
        edge_is_boundary:          BoolTensor  [E]
        salient_edge_mask:         BoolTensor  [E]
        structure_edge_labels:     FloatTensor [E], binary pseudo-labels {0, 1}
        structure_lines:           list[dict]
        quality_metrics:           dict
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("mesh must be a trimesh.Trimesh")

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("mesh must contain vertices and faces")

    vertices_np = np.asarray(mesh.vertices, dtype=np.float32)
    faces_np = np.asarray(mesh.faces, dtype=np.int64)
    vertex_normals_np = np.asarray(mesh.vertex_normals, dtype=np.float32)
    face_normals_np = np.asarray(mesh.face_normals, dtype=np.float32)

    edges_unique_np = np.asarray(mesh.edges_unique, dtype=np.int64).copy()
    faces_unique_edges_np = np.asarray(mesh.faces_unique_edges, dtype=np.int64).copy()

    if edges_unique_np.shape[0] == 0:
        raise ValueError("mesh has no unique edges")

    vertices = torch.as_tensor(vertices_np, dtype=torch.float32)
    vertex_normals = F.normalize(
        torch.as_tensor(vertex_normals_np, dtype=torch.float32),
        dim=-1,
        eps=1e-12,
    )
    faces = torch.as_tensor(faces_np, dtype=torch.long)
    face_normals = F.normalize(
        torch.as_tensor(face_normals_np, dtype=torch.float32),
        dim=-1,
        eps=1e-12,
    )
    edges_unique = torch.as_tensor(edges_unique_np, dtype=torch.long)
    faces_unique_edges = torch.as_tensor(faces_unique_edges_np, dtype=torch.long)

    num_edges = int(edges_unique.shape[0])

    incident_faces_per_edge = _build_unique_edge_incident_faces(
        faces_unique_edges_np,
        num_edges,
    )

    edge_v0 = vertices[edges_unique[:, 0]]
    edge_v1 = vertices[edges_unique[:, 1]]
    edge_midpoints = 0.5 * (edge_v0 + edge_v1)

    edge_normals = _compute_edge_normals(
        edges_unique=edges_unique,
        vertex_normals=vertex_normals,
        face_normals=face_normals,
        incident_faces_per_edge=incident_faces_per_edge,
    )

    edge_queries = torch.cat([edge_midpoints, edge_normals], dim=-1)

    structure = _structure_from_sharpness(
        mesh=mesh,
        edges_unique_np=edges_unique_np,
        face_normals_np=face_normals_np,
        incident_faces_per_edge=incident_faces_per_edge,
        config=config,
    )

    # Topology-aware local context: for each edge, collect vertices and edge
    # midpoints belonging to faces incident to that edge.
    neighborhood_list: list[torch.Tensor] = []
    max_neighborhood_size = 0

    for incident_faces in incident_faces_per_edge:
        neighbor_vertex_ids: set[int] = set()
        neighbor_edge_ids: set[int] = set()

        for face_idx in incident_faces:
            for vertex_id in faces[face_idx].tolist():
                neighbor_vertex_ids.add(int(vertex_id))

            for edge_id in faces_unique_edges[face_idx].tolist():
                neighbor_edge_ids.add(int(edge_id))

        elements: list[torch.Tensor] = []

        for vertex_id in sorted(neighbor_vertex_ids):
            elements.append(
                torch.cat(
                    [
                        vertices[vertex_id],
                        vertex_normals[vertex_id],
                    ],
                    dim=-1,
                )
            )

        for edge_id in sorted(neighbor_edge_ids):
            elements.append(
                torch.cat(
                    [
                        edge_midpoints[edge_id],
                        edge_normals[edge_id],
                    ],
                    dim=-1,
                )
            )

        if not elements:
            raise RuntimeError("encountered edge with empty topology neighborhood")

        neighborhood = torch.stack(elements, dim=0)
        neighborhood_list.append(neighborhood)
        max_neighborhood_size = max(max_neighborhood_size, neighborhood.shape[0])

    padded_neighborhoods = torch.zeros(
        (num_edges, max_neighborhood_size, 6),
        dtype=torch.float32,
    )
    neighborhood_masks = torch.zeros(
        (num_edges, max_neighborhood_size),
        dtype=torch.bool,
    )

    for edge_idx, neighborhood in enumerate(neighborhood_list):
        count = neighborhood.shape[0]
        padded_neighborhoods[edge_idx, :count] = neighborhood
        neighborhood_masks[edge_idx, :count] = True

    max_incident_faces = max((len(x) for x in incident_faces_per_edge), default=1)
    edge_to_faces = torch.full(
        (num_edges, max_incident_faces),
        -1,
        dtype=torch.long,
    )

    for edge_idx, incident_faces in enumerate(incident_faces_per_edge):
        if incident_faces:
            edge_to_faces[edge_idx, :len(incident_faces)] = torch.as_tensor(
                incident_faces,
                dtype=torch.long,
            )

    return {
        "edge_queries": edge_queries,
        "neighborhood_features": padded_neighborhoods,
        "neighborhood_masks": neighborhood_masks,
        "edges_unique": edges_unique,
        "edge_to_faces": edge_to_faces,
        # Pseudo-teacher outputs.
        "edge_sharpness": torch.as_tensor(structure["edge_sharpness"], dtype=torch.float32),
        "edge_dihedral_degrees": torch.as_tensor(structure["edge_dihedral_degrees"], dtype=torch.float32),
        "edge_is_boundary": torch.as_tensor(structure["edge_is_boundary"], dtype=torch.bool),
        "salient_edge_mask": torch.as_tensor(structure["salient_edge_mask"], dtype=torch.bool),
        "structure_edge_labels": torch.as_tensor(
            structure["structure_edge_labels"],
            dtype=torch.float32,
        ),
        "structure_lines": structure["structure_lines"],
        "quality_metrics": structure["structure_quality_metrics"],
    }


def export_mesh_aligned_edge_query_artifacts(
    outputs: dict[str, Any],
    output_dir: str,
) -> dict[str, str]:
    """
    Optional exporter for your per-mesh dataset package.

    Suggested output_dir:
        sample_id/structure_query/
    """
    os.makedirs(output_dir, exist_ok=True)

    paths = {
        "edge_queries": os.path.join(output_dir, "edge_queries.npy"),
        "neighborhood_features": os.path.join(output_dir, "edge_neighborhood_features.npy"),
        "neighborhood_masks": os.path.join(output_dir, "edge_neighborhood_masks.npy"),
        "edges_unique": os.path.join(output_dir, "edges_unique.npy"),
        "edge_to_faces": os.path.join(output_dir, "edge_to_faces.npy"),
        "edge_sharpness": os.path.join(output_dir, "edge_sharpness.npy"),
        "edge_dihedral_degrees": os.path.join(output_dir, "edge_dihedral_degrees.npy"),
        "edge_is_boundary": os.path.join(output_dir, "edge_is_boundary.npy"),
        "salient_edge_mask": os.path.join(output_dir, "salient_edge_mask.npy"),
        "structure_edge_labels": os.path.join(output_dir, "structure_edge_labels.npy"),
        "structure_lines": os.path.join(output_dir, "structure_lines.json"),
        "quality_metrics": os.path.join(output_dir, "structure_quality_metrics.json"),
    }

    tensor_keys = [
        "edge_queries",
        "neighborhood_features",
        "neighborhood_masks",
        "edges_unique",
        "edge_to_faces",
        "edge_sharpness",
        "edge_dihedral_degrees",
        "edge_is_boundary",
        "salient_edge_mask",
        "structure_edge_labels",
    ]

    for key in tensor_keys:
        value = outputs[key]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        np.save(paths[key], value)

    with open(paths["structure_lines"], "w", encoding="utf-8", newline="\n") as handle:
        json.dump(outputs["structure_lines"], handle, indent=2, sort_keys=True)
        handle.write("\n")

    with open(paths["quality_metrics"], "w", encoding="utf-8", newline="\n") as handle:
        json.dump(outputs["quality_metrics"], handle, indent=2, sort_keys=True)
        handle.write("\n")

    return paths
