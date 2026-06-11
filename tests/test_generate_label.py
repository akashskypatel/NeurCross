import os


def test_derive_sample_id_is_deterministic(tmp_path):
    from quad_mesh.generate_label import derive_sample_id

    mesh_path = tmp_path / "cube.obj"
    mesh_path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")

    first = derive_sample_id(str(mesh_path))
    second = derive_sample_id(str(mesh_path))

    assert first == second
    assert first.startswith("cube-")


def test_generate_label_parser_accepts_dataset_args():
    from quad_mesh.generate_label import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "--data_path",
            "mesh.obj",
            "--dataset_root",
            "labels",
            "--sample_id",
            "sample-001",
            "--overwrite",
            "--preflight_policy",
            "strict",
            "--quality_gate",
            "loose",
            "--no-export_geometry_npz",
            "--no-normalize_mesh",
            "--num_epochs",
            "1",
        ]
    )

    assert args.data_path == "mesh.obj"
    assert args.dataset_root == "labels"
    assert args.sample_id == "sample-001"
    assert args.overwrite is True
    assert args.preflight_policy == "strict"
    assert args.quality_gate == "loose"
    assert args.export_geometry_npz is False
    assert args.normalize_mesh is False


def test_cli_help_lists_generate_label():
    from neurcross.__main__ import build_parser

    parser = build_parser()
    help_text = parser.format_help()

    assert "generate-label" in help_text
