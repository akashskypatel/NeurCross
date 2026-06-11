from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field

import numpy as np
import trimesh


def _sha256_file(path: str) -> str | None:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _edge_metrics(faces: np.ndarray) -> tuple[int, int]:
    if faces.size == 0:
        return 0, 0
    edges = np.concatenate(
        (
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ),
        axis=0,
    )
    edges = np.sort(edges.astype(np.int64, copy=False), axis=1)
    _unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    return int(np.count_nonzero(counts == 1)), int(np.count_nonzero(counts > 2))


def _triangle_keep_mask(vertices: np.ndarray, faces: np.ndarray, *, area_epsilon: float = 1e-12) -> tuple[np.ndarray, int]:
    if faces.size == 0:
        return np.zeros((0,), dtype=bool), 0
    distinct_vertices = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 0] != faces[:, 2])
    )
    triangles = vertices[faces]
    double_area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    keep_mask = distinct_vertices & np.isfinite(double_area) & (double_area > area_epsilon)
    return keep_mask, int(np.count_nonzero(~keep_mask))


@dataclass
class MeshMetrics:
    vertex_count: int = 0
    face_count: int = 0
    bounds_min: list[float] = field(default_factory=list)
    bounds_max: list[float] = field(default_factory=list)
    extents: list[float] = field(default_factory=list)
    max_extent: float = 0.0
    surface_area: float | None = None
    volume: float | None = None
    connected_components: int = 0
    boundary_edges: int = 0
    nonmanifold_edges: int = 0
    watertight: bool = False
    winding_consistent: bool | None = None


@dataclass
class MeshPreflightReport:
    source_path: str | None
    source_sha256: str | None
    status: str
    metrics: MeshMetrics
    repair_actions: list[dict[str, object]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skip_reason: str | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    normalization: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _load_mesh(mesh_path: str, *, allow_scene_merge: bool = False) -> trimesh.Trimesh:
    loaded = trimesh.load(mesh_path, process=False)
    if isinstance(loaded, trimesh.Scene):
        if not allow_scene_merge:
            raise ValueError("input resolved to a trimesh.Scene; pass a mesh file or implement scene merging explicitly")
        if not loaded.geometry:
            raise ValueError("input scene does not contain any mesh geometry")
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"unsupported mesh type: {type(loaded)!r}")
    return loaded.copy()


def inspect_mesh_path(mesh_path: str, *, allow_scene_merge: bool = False) -> tuple[MeshPreflightReport, trimesh.Trimesh | None]:
    source_hash = _sha256_file(mesh_path)
    try:
        mesh = _load_mesh(mesh_path, allow_scene_merge=allow_scene_merge)
    except Exception as exc:
        report = MeshPreflightReport(
            source_path=mesh_path,
            source_sha256=source_hash,
            status="skip",
            metrics=MeshMetrics(),
            skip_reason=str(exc),
        )
        return report, None
    report, prepared_mesh = inspect_mesh(mesh, source_path=mesh_path, source_sha256=source_hash)
    return report, prepared_mesh


def inspect_mesh(
    mesh: trimesh.Trimesh,
    *,
    source_path: str | None = None,
    source_sha256: str | None = None,
) -> tuple[MeshPreflightReport, trimesh.Trimesh | None]:
    report = MeshPreflightReport(
        source_path=source_path,
        source_sha256=source_sha256,
        status="accepted_for_training",
        metrics=MeshMetrics(),
    )

    prepared_mesh = mesh.copy()
    vertices = np.asarray(prepared_mesh.vertices, dtype=np.float64)
    faces = np.asarray(prepared_mesh.faces, dtype=np.int64)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        report.status = "skip"
        report.skip_reason = "mesh vertices are not a valid Nx3 array"
        return report, None
    if faces.ndim != 2 or faces.shape[1] != 3:
        report.status = "skip"
        report.skip_reason = "mesh faces are not a valid Nx3 triangle array"
        return report, None
    if len(vertices) == 0 or len(faces) == 0:
        report.status = "skip"
        report.skip_reason = "mesh has no vertices or faces"
        return report, None
    if not np.isfinite(vertices).all():
        report.status = "skip"
        report.skip_reason = "mesh contains non-finite vertex coordinates"
        return report, None

    invalid_index_mask = (faces < 0).any(axis=1) | (faces >= len(vertices)).any(axis=1)
    if np.any(invalid_index_mask):
        prepared_mesh.update_faces(~invalid_index_mask)
        report.repair_actions.append(
            {"action": "remove_invalid_faces", "count": int(np.count_nonzero(invalid_index_mask))}
        )

    faces = np.asarray(prepared_mesh.faces, dtype=np.int64)
    keep_mask, removed_degenerate = _triangle_keep_mask(vertices, faces)
    if removed_degenerate:
        prepared_mesh.update_faces(keep_mask)
        report.repair_actions.append({"action": "remove_degenerate_faces", "count": removed_degenerate})

    faces = np.asarray(prepared_mesh.faces, dtype=np.int64)
    sorted_faces = np.sort(faces, axis=1)
    _unique_faces, unique_indices = np.unique(sorted_faces, axis=0, return_index=True)
    keep_mask = np.zeros(len(faces), dtype=bool)
    keep_mask[np.sort(unique_indices)] = True
    removed_duplicates = int(np.count_nonzero(~keep_mask))
    if removed_duplicates:
        prepared_mesh.update_faces(keep_mask)
        report.repair_actions.append({"action": "remove_duplicate_faces", "count": removed_duplicates})

    vertices_before_cleanup = len(prepared_mesh.vertices)
    prepared_mesh.remove_unreferenced_vertices()
    removed_vertices = int(vertices_before_cleanup - len(prepared_mesh.vertices))
    if removed_vertices:
        report.repair_actions.append({"action": "remove_unreferenced_vertices", "count": removed_vertices})

    vertices = np.asarray(prepared_mesh.vertices, dtype=np.float64)
    faces = np.asarray(prepared_mesh.faces, dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        report.status = "skip"
        report.skip_reason = "mesh became empty after conservative repairs"
        return report, None

    bounds = prepared_mesh.bounds.astype(np.float64)
    extents = bounds[1] - bounds[0]
    max_extent = float(np.max(extents))
    if not np.isfinite(extents).all() or max_extent <= 0.0:
        report.status = "skip"
        report.skip_reason = "mesh has zero or invalid spatial extent"
        return report, None

    components = prepared_mesh.split(only_watertight=False)
    boundary_edges, nonmanifold_edges = _edge_metrics(faces)
    try:
        winding_consistent = bool(prepared_mesh.is_winding_consistent)
    except Exception:
        winding_consistent = None

    report.metrics = MeshMetrics(
        vertex_count=int(len(vertices)),
        face_count=int(len(faces)),
        bounds_min=bounds[0].astype(float).tolist(),
        bounds_max=bounds[1].astype(float).tolist(),
        extents=extents.astype(float).tolist(),
        max_extent=max_extent,
        surface_area=float(prepared_mesh.area) if np.isfinite(prepared_mesh.area) else None,
        volume=float(prepared_mesh.volume) if prepared_mesh.is_watertight and np.isfinite(prepared_mesh.volume) else None,
        connected_components=int(len(components)),
        boundary_edges=boundary_edges,
        nonmanifold_edges=nonmanifold_edges,
        watertight=bool(prepared_mesh.is_watertight),
        winding_consistent=winding_consistent,
    )

    if report.metrics.connected_components != 1:
        report.warnings.append(f"mesh has {report.metrics.connected_components} connected components")
    if boundary_edges:
        report.warnings.append(f"mesh has {boundary_edges} boundary edges")
    if nonmanifold_edges:
        report.warnings.append(f"mesh has {nonmanifold_edges} non-manifold edges")
    if not report.metrics.watertight:
        report.warnings.append("mesh is not watertight")
    if winding_consistent is False:
        report.warnings.append("mesh winding is not consistent")

    if report.repair_actions or report.warnings:
        report.status = "needs_repair"
    return report, prepared_mesh
