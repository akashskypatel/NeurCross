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
    edges_unique = np.load(exported["edges_unique_path"])
    edge_sharpness = np.load(exported["edge_sharpness_path"])
    edge_dihedral_degrees = np.load(exported["edge_dihedral_degrees_path"])
    edge_is_boundary = np.load(exported["edge_is_boundary_path"])
    salient_edge_mask = np.load(exported["salient_edge_mask_path"])
    structure_edge_labels = np.load(exported["structure_edge_labels_path"])
    structure_lines = json.loads(open(exported["structure_lines_path"], "r", encoding="utf-8").read())
    structure_quality_metrics = json.loads(
        open(exported["structure_quality_metrics_path"], "r", encoding="utf-8").read()
    )

    assert exported["feature_edge_count"] == 12
    assert exported["feature_vertex_count"] == 8
    assert exported["boundary_edge_count"] == 0
    assert exported["sharp_edge_count"] == 12
    assert exported["structure_edge_count"] > 0
    assert exported["edge_sharpness_mean"] > 0.0
    assert exported["edge_sharpness_max"] == 1.0
    assert exported["feature_constrained"] is True
    assert sharp_edges.shape == (12, 2)
    assert feature_vertices.shape == (8,)
    assert edges_unique.shape == (18, 2)
    assert edge_sharpness.shape == (18,)
    assert edge_dihedral_degrees.shape == (18,)
    assert edge_is_boundary.shape == (18,)
    assert salient_edge_mask.shape == (18,)
    assert structure_edge_labels.shape == (18,)
    assert edge_is_boundary.dtype == np.bool_
    assert salient_edge_mask.dtype == np.bool_
    assert len(structure_lines) >= 1
    assert "accepted" in structure_quality_metrics
    assert "quality_grade" in structure_quality_metrics
    assert structure_quality_metrics["selected_edge_count"] == exported["structure_edge_count"]


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
    structure_edge_labels = np.load(exported["structure_edge_labels_path"])
    structure_quality_metrics = json.loads(
        open(exported["structure_quality_metrics_path"], "r", encoding="utf-8").read()
    )
    assert sharp_edges.shape[0] == 0
    assert exported["feature_edge_count"] == 0
    assert exported["feature_constrained"] is False
    assert exported["structure_edge_count"] >= 0
    assert structure_edge_labels.ndim == 1
    assert "accepted" in structure_quality_metrics


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


def test_feature_detection_none_mode_returns_extended_empty_contract(tmp_path):
    from quad_mesh.feature_artifacts import export_feature_artifacts

    mesh = trimesh.creation.box()
    exported = export_feature_artifacts(
        mesh=mesh,
        output_dir=str(tmp_path),
        feature_mode="none",
        angle_threshold_degrees=35.0,
    )

    assert exported["sharp_edges_path"] is None
    assert exported["edges_unique_path"] is None
    assert exported["edge_sharpness_path"] is None
    assert exported["edge_dihedral_degrees_path"] is None
    assert exported["edge_is_boundary_path"] is None
    assert exported["salient_edge_mask_path"] is None
    assert exported["structure_edge_labels_path"] is None
    assert exported["structure_lines_path"] is None
    assert exported["structure_quality_metrics_path"] is None
    assert exported["feature_edge_count"] == 0
    assert exported["feature_vertex_count"] == 0
    assert exported["boundary_edge_count"] == 0
    assert exported["sharp_edge_count"] == 0
    assert exported["structure_edge_count"] == 0
    assert exported["edge_sharpness_mean"] == 0.0
    assert exported["edge_sharpness_max"] == 0.0
    assert exported["feature_constrained"] is False
