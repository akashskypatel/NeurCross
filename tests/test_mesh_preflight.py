import numpy as np
import pytest
import trimesh


def test_preflight_accepts_clean_mesh(tmp_path):
    from quad_mesh.preflight import inspect_mesh_path

    mesh = trimesh.creation.box(extents=(2.0, 4.0, 6.0))
    mesh_path = tmp_path / "box.obj"
    mesh.export(mesh_path)

    report, prepared_mesh = inspect_mesh_path(str(mesh_path))

    assert prepared_mesh is not None
    assert report.status == "accepted_for_training"
    assert report.metrics.vertex_count > 0
    assert report.metrics.face_count > 0
    assert report.metrics.connected_components == 1
    assert report.repair_actions == []
    assert report.source_sha256


def test_preflight_repairs_degenerate_and_duplicate_faces():
    from quad_mesh.preflight import inspect_mesh

    mesh = trimesh.Trimesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        faces=np.array(
            [
                [0, 1, 2],
                [0, 1, 2],
                [0, 0, 3],
            ],
            dtype=np.int64,
        ),
        process=False,
        validate=False,
    )

    report, prepared_mesh = inspect_mesh(mesh, source_path="in-memory")

    assert prepared_mesh is not None
    assert report.status == "needs_repair"
    assert any(action["action"] == "remove_degenerate_faces" for action in report.repair_actions)
    assert any(action["action"] == "remove_duplicate_faces" for action in report.repair_actions)
    assert prepared_mesh.faces.shape[0] == 1


def test_preflight_skips_nonfinite_mesh():
    from quad_mesh.preflight import inspect_mesh

    mesh = trimesh.Trimesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [np.nan, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        process=False,
        validate=False,
    )

    report, prepared_mesh = inspect_mesh(mesh, source_path="nan.obj")

    assert prepared_mesh is None
    assert report.status == "skip"
    assert report.skip_reason == "mesh contains non-finite vertex coordinates"


def test_preflight_marks_multicomponent_mesh_for_review():
    from quad_mesh.preflight import inspect_mesh

    left = trimesh.creation.box()
    right = trimesh.creation.box()
    right.apply_translation((3.0, 0.0, 0.0))
    mesh = trimesh.util.concatenate((left, right))

    report, prepared_mesh = inspect_mesh(mesh, source_path="two_boxes.obj")

    assert prepared_mesh is not None
    assert report.status == "needs_repair"
    assert report.metrics.connected_components == 2


def test_preflight_detects_nonmanifold_edges():
    from quad_mesh.preflight import inspect_mesh

    mesh = trimesh.Trimesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        faces=np.array(
            [
                [0, 1, 2],
                [0, 1, 3],
                [0, 1, 4],
            ],
            dtype=np.int64,
        ),
        process=False,
        validate=False,
    )

    report, prepared_mesh = inspect_mesh(mesh, source_path="nonmanifold.obj")

    assert prepared_mesh is not None
    assert report.status == "needs_repair"
    assert report.metrics.nonmanifold_edges >= 1


def test_normalize_mesh_exports_unit_box(tmp_path):
    from quad_mesh.normalize import export_normalized_mesh

    mesh = trimesh.creation.box(extents=(4.0, 2.0, 8.0))
    mesh.apply_translation((10.0, -5.0, 3.0))

    exported = export_normalized_mesh(mesh, str(tmp_path))
    bounds = np.array([exported.metadata.bounds_after_min, exported.metadata.bounds_after_max], dtype=np.float64)

    assert exported.obj_path.endswith("normalized_mesh.obj")
    assert exported.ply_path.endswith("normalized_mesh.ply")
    assert np.max(bounds[1] - bounds[0]) == pytest.approx(1.0)
    assert np.all(bounds[0] >= -0.5 - 1e-6)
    assert np.all(bounds[1] <= 0.5 + 1e-6)
