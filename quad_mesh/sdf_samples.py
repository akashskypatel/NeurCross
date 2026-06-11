from __future__ import annotations

import os

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def _sample_surface(mesh: trimesh.Trimesh, count: int, rng: np.random.Generator) -> np.ndarray:
    if count <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    points, _face_index = trimesh.sample.sample_surface(mesh, count, seed=rng)
    return np.asarray(points, dtype=np.float32)


def _surface_kdtree_distance(mesh: trimesh.Trimesh, query_points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    surface_count = max(8192, query_points.shape[0] * 4)
    surface_points = _sample_surface(mesh, surface_count, rng)
    tree = cKDTree(surface_points)
    distances, _indices = tree.query(query_points, k=1)
    return np.asarray(distances, dtype=np.float32)


def _compute_signed_or_unsigned_distance(
    mesh: trimesh.Trimesh,
    query_points: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        signed = trimesh.proximity.signed_distance(mesh, query_points)
        signed = np.asarray(signed, dtype=np.float32)
        reliability = np.ones(signed.shape[0], dtype=bool)
        if not np.isfinite(signed).all():
            raise ValueError("non-finite signed distances")
        return signed, reliability
    except Exception:
        try:
            _closest_points, distances, _triangle_id = trimesh.proximity.closest_point(mesh, query_points)
            unsigned = np.asarray(distances, dtype=np.float32)
        except Exception:
            unsigned = _surface_kdtree_distance(mesh, query_points, rng)
        reliability = np.zeros(unsigned.shape[0], dtype=bool)
        return unsigned, reliability


def export_sdf_samples(
    *,
    mesh_path: str,
    output_dir: str,
    normalization: dict,
    seed: int,
    n_surface: int,
    n_near: int,
    n_uniform: int,
    near_sigma: float,
    uniform_extent: float,
    tsdf_truncation: float,
) -> str:
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"mesh did not load as a single triangle mesh: {mesh_path}")

    rng = np.random.default_rng(int(seed))
    surface_points = _sample_surface(mesh, int(n_surface), rng)
    near_surface_anchor = _sample_surface(mesh, int(n_near), rng)
    near_points = near_surface_anchor + rng.normal(
        loc=0.0,
        scale=float(near_sigma),
        size=near_surface_anchor.shape,
    ).astype(np.float32)
    uniform_points = rng.uniform(
        low=-float(uniform_extent),
        high=float(uniform_extent),
        size=(int(n_uniform), 3),
    ).astype(np.float32)

    query_points = np.concatenate((surface_points, near_points, uniform_points), axis=0)
    sample_type = np.concatenate(
        (
            np.zeros(surface_points.shape[0], dtype=np.int32),
            np.ones(near_points.shape[0], dtype=np.int32),
            np.full(uniform_points.shape[0], 2, dtype=np.int32),
        ),
        axis=0,
    )
    sdf_values, sign_reliability = _compute_signed_or_unsigned_distance(mesh, query_points, rng)
    truncation = max(float(tsdf_truncation), 1e-6)
    tsdf_values = np.clip(sdf_values / truncation, -1.0, 1.0).astype(np.float32)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "sdf_samples.npz")
    np.savez(
        path,
        query_points=np.asarray(query_points, dtype=np.float32),
        sdf_values=np.asarray(sdf_values, dtype=np.float32),
        tsdf_values=tsdf_values,
        sample_type=sample_type,
        sign_reliability=np.asarray(sign_reliability, dtype=bool),
        normalization_center=np.asarray(normalization["center"], dtype=np.float32),
        normalization_scale=np.asarray([normalization["scale"]], dtype=np.float32),
        mesh_is_watertight=np.asarray([bool(mesh.is_watertight)], dtype=bool),
    )
    return path
