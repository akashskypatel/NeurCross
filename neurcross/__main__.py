from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m neurcross",
        description=(
            "NeurCross trains neural cross fields on triangle meshes and can package "
            "per-mesh dataset labels around the resulting .vec field artifacts."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command")

    train = subparsers.add_parser(
        "train-quad-mesh",
        help="Train a cross field on an input mesh.",
    )
    train.add_argument("args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)

    generate = subparsers.add_parser(
        "generate-label",
        help="Generate a dataset sample package for one mesh.",
    )
    generate.add_argument("args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)

    build_index = subparsers.add_parser(
        "build-dataset-index",
        help="Scan generated dataset samples and write a dataset index.",
    )
    build_index.add_argument("args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)

    split_dataset = subparsers.add_parser(
        "split-dataset",
        help="Build deterministic shape-level dataset splits from accepted samples.",
    )
    split_dataset.add_argument("args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)

    parser.epilog = (
        "High-level functionality:\n"
        "  train-quad-mesh     Train NeurCross on a mesh to produce cross-field snapshots.\n"
        "  train               Alias for train-quad-mesh.\n"
        "  generate-label      Generate a dataset sample package from one mesh.\n"
        "  build-dataset-index Scan generated dataset samples and write an index.\n"
        "  split-dataset       Build deterministic shape-level dataset splits.\n\n"
        "Examples:\n"
        "  python -m neurcross --help\n"
        "  python -m neurcross train-quad-mesh --help\n"
        "  python -m neurcross train --help\n"
        "  python -m neurcross generate-label --help\n"
        "  python -m neurcross build-dataset-index --help\n"
        "  python -m neurcross split-dataset --help"
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

    if command == "generate-label":
        from quad_mesh.generate_label import main as generate_label_main

        sys.argv = ["neurcross-generate-label", *command_args]
        generate_label_main(command_args)
        return

    if command == "build-dataset-index":
        from quad_mesh.dataset_splits import build_dataset_index_main

        sys.argv = ["neurcross-build-dataset-index", *command_args]
        build_dataset_index_main(command_args)
        return

    if command == "split-dataset":
        from quad_mesh.dataset_splits import split_dataset_main

        sys.argv = ["neurcross-split-dataset", *command_args]
        split_dataset_main(command_args)
        return

    parser.error(f"unknown command: {command}")


if __name__ == "__main__":
    main()
