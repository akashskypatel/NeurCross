from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import numpy as np
import trimesh


@dataclass
class NormalizationMetadata:
    center: list[float]
    scale: float
    bounds_before_min: list[float]
    bounds_before_max: list[float]
    bounds_after_min: list[float]
    bounds_after_max: list[float]
    max_extent_before: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class NormalizedMeshExport:
    mesh: trimesh.Trimesh
    metadata: NormalizationMetadata
    obj_path: str
    ply_path: str


def normalize_mesh(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, NormalizationMetadata]:
    normalized_mesh = mesh.copy()
    bounds_before = normalized_mesh.bounds.astype(np.float64)
    center = (bounds_before[0] + bounds_before[1]) * 0.5
    extents = bounds_before[1] - bounds_before[0]
    max_extent = float(np.max(extents))
    if not np.isfinite(extents).all() or max_extent <= 0.0:
        raise ValueError("cannot normalize a mesh with zero or invalid extent")

    normalized_mesh.apply_translation(-center)
    scale = 1.0 / max_extent
    normalized_mesh.apply_scale(scale)
    bounds_after = normalized_mesh.bounds.astype(np.float64)

    metadata = NormalizationMetadata(
        center=center.astype(float).tolist(),
        scale=float(scale),
        bounds_before_min=bounds_before[0].astype(float).tolist(),
        bounds_before_max=bounds_before[1].astype(float).tolist(),
        bounds_after_min=bounds_after[0].astype(float).tolist(),
        bounds_after_max=bounds_after[1].astype(float).tolist(),
        max_extent_before=max_extent,
    )
    return normalized_mesh, metadata


def export_normalized_mesh(mesh: trimesh.Trimesh, output_dir: str) -> NormalizedMeshExport:
    os.makedirs(output_dir, exist_ok=True)
    normalized_mesh, metadata = normalize_mesh(mesh)
    obj_path = os.path.join(output_dir, "normalized_mesh.obj")
    ply_path = os.path.join(output_dir, "normalized_mesh.ply")
    normalized_mesh.export(obj_path)
    normalized_mesh.export(ply_path)
    return NormalizedMeshExport(
        mesh=normalized_mesh,
        metadata=metadata,
        obj_path=obj_path,
        ply_path=ply_path,
    )
