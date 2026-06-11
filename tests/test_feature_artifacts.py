import json

import numpy as np
import trimesh


def test_feature_detection_finds_cube_edges(tmp_path):
    from quad_mesh.feature_artifacts import export_feature_artifacts

    mesh = trimesh.creation.box()
    exported = export_feature_artifacts(
        mesh=mesh,
        output_dir=str(tmp_path),
        feature_mode="auto",
        angle_threshold_degrees=35.0,
    )

    sharp_edges = np.load(exported["sharp_edges_path"])
    feature_vertices = np.load(exported["feature_vertices_path"])

    assert exported["feature_edge_count"] == 12
    assert exported["feature_vertex_count"] == 8
    assert sharp_edges.shape == (12, 2)
    assert feature_vertices.shape == (8,)


def test_feature_detection_finds_few_or_no_sharp_edges_on_smooth_mesh(tmp_path):
    from quad_mesh.feature_artifacts import export_feature_artifacts

    mesh = trimesh.creation.icosphere(subdivisions=2)
    exported = export_feature_artifacts(
        mesh=mesh,
        output_dir=str(tmp_path),
        feature_mode="auto",
        angle_threshold_degrees=80.0,
    )

    sharp_edges = np.load(exported["sharp_edges_path"])
    assert sharp_edges.shape[0] == 0
    assert exported["feature_edge_count"] == 0


def test_feature_detection_writes_feature_lines_and_distances(tmp_path):
    from quad_mesh.feature_artifacts import export_feature_artifacts

    mesh = trimesh.creation.box()
    exported = export_feature_artifacts(
        mesh=mesh,
        output_dir=str(tmp_path),
        feature_mode="auto",
        angle_threshold_degrees=35.0,
    )

    lines = json.loads(open(exported["feature_lines_path"], "r", encoding="utf-8").read())
    distances = np.load(exported["face_feature_distance_path"])

    assert len(lines) >= 1
    assert distances.shape[0] == len(mesh.faces)
