from __future__ import annotations

import json
import os

import numpy as np
import trimesh
from scipy.spatial import cKDTree


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


def detect_mesh_features(
    mesh: trimesh.Trimesh,
    *,
    angle_threshold_degrees: float,
) -> dict[str, object]:
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
    }


def export_feature_artifacts(
    *,
    mesh: trimesh.Trimesh,
    output_dir: str,
    feature_mode: str,
    angle_threshold_degrees: float,
) -> dict[str, object]:
    mode = str(feature_mode).lower()
    if mode == "none":
        return {
            "feature_mode": mode,
            "sharp_edges_path": None,
            "feature_vertices_path": None,
            "feature_lines_path": None,
            "face_feature_distance_path": None,
            "feature_edge_count": 0,
            "feature_vertex_count": 0,
            "boundary_edge_count": 0,
            "sharp_edge_count": 0,
            "feature_constrained": False,
        }
    if mode == "file":
        raise NotImplementedError("--feature_mode file is not implemented yet")
    if mode != "auto":
        raise ValueError(f"unsupported feature mode: {feature_mode}")

    os.makedirs(output_dir, exist_ok=True)
    detected = detect_mesh_features(mesh, angle_threshold_degrees=angle_threshold_degrees)

    sharp_edges_path = os.path.join(output_dir, "sharp_edges.npy")
    feature_vertices_path = os.path.join(output_dir, "feature_vertices.npy")
    feature_lines_path = os.path.join(output_dir, "feature_lines.json")
    face_feature_distance_path = os.path.join(output_dir, "face_feature_distance.npy")

    np.save(sharp_edges_path, detected["feature_edges"])
    np.save(feature_vertices_path, detected["feature_vertices"])
    np.save(face_feature_distance_path, detected["face_feature_distance"])
    with open(feature_lines_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(detected["feature_lines"], handle, indent=2, sort_keys=True)
        handle.write("\n")

    return {
        "feature_mode": mode,
        "sharp_edges_path": sharp_edges_path,
        "feature_vertices_path": feature_vertices_path,
        "feature_lines_path": feature_lines_path,
        "face_feature_distance_path": face_feature_distance_path,
        "feature_edge_count": int(detected["feature_edges"].shape[0]),
        "feature_vertex_count": int(detected["feature_vertices"].shape[0]),
        "boundary_edge_count": int(detected["boundary_edges"].shape[0]),
        "sharp_edge_count": int(detected["sharp_edges"].shape[0]),
        "feature_constrained": bool(detected["feature_edges"].shape[0] > 0),
    }
