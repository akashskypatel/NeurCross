from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m neurcross",
        description=(
            "NeurCross trains neural cross fields on triangle meshes and converts saved "
            "cross-field snapshots into .rosy files for downstream quad extraction."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command")

    train = subparsers.add_parser(
        "train-quad-mesh",
        help="Train a cross field on an input mesh.",
    )
    train.add_argument("args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)

    convert = subparsers.add_parser(
        "convert",
        help="Convert a saved cross-field snapshot to a .rosy file.",
    )
    convert.add_argument("args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)

    generate = subparsers.add_parser(
        "generate-label",
        help="Generate a dataset sample package for one mesh.",
    )
    generate.add_argument("args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)

    parser.epilog = (
        "High-level functionality:\n"
        "  train-quad-mesh     Train NeurCross on a mesh to produce cross-field snapshots.\n"
        "  train               Alias for train-quad-mesh.\n"
        "  convert             Convert cross-field, .rosy, or .rawfield files.\n"
        "  generate-label      Generate a dataset sample package from one mesh.\n\n"
        "Examples:\n"
        "  python -m neurcross --help\n"
        "  python -m neurcross train-quad-mesh --help\n"
        "  python -m neurcross train --help\n"
        "  python -m neurcross convert --help\n"
        "  python -m neurcross generate-label --help"
    )
    return parser


def main() -> None:
    parser = build_parser()
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        return

    command, command_args = argv[0], argv[1:]

    if command == "train-quad-mesh" or command == "train":
        from quad_mesh.train_quad_mesh import main as train_main

        sys.argv = ["neurcross-train-quad-mesh", *command_args]
        train_main()
        return

    if command == "convert":
        from quad_mesh.convert_crossfield import main as convert_main

        sys.argv = ["neurcross-convert", *command_args]
        convert_main()
        return

    if command == "generate-label":
        from quad_mesh.generate_label import main as generate_label_main

        sys.argv = ["neurcross-generate-label", *command_args]
        generate_label_main(command_args)
        return

    parser.error(f"unknown command: {command}")


if __name__ == "__main__":
    main()
