import numpy as np
import trimesh


def _normalization_dict():
    return {
        "center": [0.0, 0.0, 0.0],
        "scale": 1.0,
    }


def test_export_sdf_samples_marks_watertight_signs_reliable(tmp_path):
    from quad_mesh.sdf_samples import export_sdf_samples

    mesh = trimesh.creation.box()
    mesh_path = tmp_path / "box.obj"
    mesh.export(mesh_path)

    output_path = export_sdf_samples(
        mesh_path=str(mesh_path),
        output_dir=str(tmp_path / "sdf"),
        normalization=_normalization_dict(),
        seed=123,
        n_surface=8,
        n_near=8,
        n_uniform=16,
        near_sigma=0.02,
        uniform_extent=0.5,
        tsdf_truncation=0.1,
    )

    data = np.load(output_path)
    assert data["mesh_is_watertight"].shape == (1,)
    assert bool(data["mesh_is_watertight"][0]) is True
    assert data["sign_reliability"].dtype == np.bool_
    assert bool(np.all(data["sign_reliability"])) is True
    assert data["query_points"].shape == (32, 3)
    assert data["sdf_values"].shape == (32,)
    assert data["tsdf_values"].shape == (32,)
    assert data["sample_type"].shape == (32,)


def test_export_sdf_samples_marks_nonwatertight_signs_unreliable(tmp_path):
    from quad_mesh.sdf_samples import export_sdf_samples

    mesh = trimesh.creation.box()
    mesh.update_faces(np.arange(len(mesh.faces) - 1))
    mesh.remove_unreferenced_vertices()
    mesh_path = tmp_path / "open_box.obj"
    mesh.export(mesh_path)

    output_path = export_sdf_samples(
        mesh_path=str(mesh_path),
        output_dir=str(tmp_path / "sdf"),
        normalization=_normalization_dict(),
        seed=123,
        n_surface=8,
        n_near=8,
        n_uniform=16,
        near_sigma=0.02,
        uniform_extent=0.5,
        tsdf_truncation=0.1,
    )

    data = np.load(output_path)
    assert bool(data["mesh_is_watertight"][0]) is False
    assert bool(np.any(data["sign_reliability"])) is False
    assert np.all(data["sdf_values"] >= 0.0)


def test_export_sdf_samples_is_deterministic_for_fixed_seed(tmp_path):
    from quad_mesh.sdf_samples import export_sdf_samples

    mesh = trimesh.creation.icosphere(subdivisions=1)
    mesh_path = tmp_path / "sphere.obj"
    mesh.export(mesh_path)

    output_a = export_sdf_samples(
        mesh_path=str(mesh_path),
        output_dir=str(tmp_path / "sdf_a"),
        normalization=_normalization_dict(),
        seed=42,
        n_surface=8,
        n_near=8,
        n_uniform=16,
        near_sigma=0.02,
        uniform_extent=0.5,
        tsdf_truncation=0.1,
    )
    output_b = export_sdf_samples(
        mesh_path=str(mesh_path),
        output_dir=str(tmp_path / "sdf_b"),
        normalization=_normalization_dict(),
        seed=42,
        n_surface=8,
        n_near=8,
        n_uniform=16,
        near_sigma=0.02,
        uniform_extent=0.5,
        tsdf_truncation=0.1,
    )

    data_a = np.load(output_a)
    data_b = np.load(output_b)
    for key in ("query_points", "sdf_values", "tsdf_values", "sample_type", "sign_reliability"):
        assert np.array_equal(data_a[key], data_b[key])
