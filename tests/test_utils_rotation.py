import numpy as np
import trimesh


def test_get_rotation_matrix_handles_faces_with_no_neighbors(tmp_path):
    from utils import utils

    mesh = trimesh.Trimesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        faces=np.array(
            [
                [0, 1, 2],
                [3, 4, 5],
            ],
            dtype=np.int64,
        ),
        process=False,
        validate=False,
    )
    mesh_path = tmp_path / "disconnected.obj"
    mesh.export(mesh_path)

    vertex_neighbors = utils.get_sample_vers_neighbors_for_face_center_points_or_vertices(str(mesh_path))
    vertex_neighbors_list = utils.calculate_same_neighbors_verts(vertex_neighbors)
    axis_angle_rotations = utils.get_rotation_matrix(vertex_neighbors_list, vertex_neighbors, str(mesh_path))

    assert vertex_neighbors == [[], []]
    assert vertex_neighbors_list == [[0, 1]]
    assert len(axis_angle_rotations) == 1
    assert axis_angle_rotations[0].shape == (2, 0, 3, 3)
    assert axis_angle_rotations[0].dtype == np.float32
