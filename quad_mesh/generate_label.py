from __future__ import annotations

import argparse
import hashlib
import os
import sys

from . import quad_mesh_args


def _source_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_sample_id(mesh_path: str) -> str:
    normalized_path = os.path.normcase(os.path.abspath(mesh_path)).replace("\\", "/")
    path_slug = hashlib.sha256(normalized_path.encode("utf-8")).hexdigest()[:12]
    source_hash = _source_sha256(mesh_path)[:12]
    stem = os.path.splitext(os.path.basename(mesh_path))[0]
    return f"{stem}-{path_slug}-{source_hash}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m neurcross generate-label",
        description="Run NeurCross as an offline label generator and write one sample package per mesh.",
    )
    quad_mesh_args.add_args(parser)
    return parser


def _resolve_dataset_root(mesh_path: str, dataset_root: str | None) -> str:
    mesh_dir = os.path.dirname(os.path.abspath(mesh_path))
    if dataset_root is None:
        return os.path.join(mesh_dir, "generated_labels")
    if os.path.isabs(dataset_root):
        return dataset_root
    return os.path.join(mesh_dir, dataset_root)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args([] if argv is None else argv)
    if not args.data_path:
        parser.error("--data_path is required")

    args.dataset_root = _resolve_dataset_root(args.data_path, args.dataset_root)
    if args.sample_id is None:
        args.sample_id = derive_sample_id(args.data_path)

    from .train_quad_mesh import train_crossfield

    train_crossfield(args=args, allow_multiprocessing_workers=True)


if __name__ == "__main__":
    main(sys.argv[1:])
