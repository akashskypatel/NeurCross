from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

from . import quad_mesh_args


_MESH_EXTENSIONS = {".obj", ".ply", ".off", ".stl"}


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


def _resolve_dataset_root(mesh_path: str, dataset_root: str | None, *, base_dir: str | None = None) -> str:
    if dataset_root is None:
        mesh_dir = os.path.dirname(os.path.abspath(mesh_path))
        return os.path.join(mesh_dir, "generated_labels")
    if os.path.isabs(dataset_root):
        return dataset_root
    if base_dir is not None:
        return os.path.join(base_dir, dataset_root)
    mesh_dir = os.path.dirname(os.path.abspath(mesh_path))
    return os.path.join(mesh_dir, dataset_root)


def _is_mesh_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in _MESH_EXTENSIONS


def _iter_meshes_from_directory(path: Path) -> list[str]:
    return sorted(
        str(candidate.resolve())
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in _MESH_EXTENSIONS
    )


def _parse_list_entry(raw_line: str, *, list_path: Path) -> str | None:
    line = raw_line.strip()
    if not line:
        return None
    if list_path.suffix.lower() == ".jsonl":
        payload = json.loads(line)
        if isinstance(payload, str):
            entry = payload
        elif isinstance(payload, dict):
            entry = payload.get("mesh_path") or payload.get("data_path") or payload.get("path")
            if not entry:
                raise ValueError(f"JSONL entry in {list_path} is missing mesh_path/data_path/path")
        else:
            raise ValueError(f"Unsupported JSONL entry in {list_path}: {type(payload).__name__}")
    else:
        entry = line
    candidate = Path(entry)
    if not candidate.is_absolute():
        candidate = (list_path.parent / candidate).resolve()
    return str(candidate)


def resolve_input_meshes(data_path: str) -> tuple[list[str], str | None]:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {data_path}")
    if path.is_dir():
        meshes = _iter_meshes_from_directory(path)
        return meshes, str(path.resolve())
    if _is_mesh_path(path):
        return [str(path.resolve())], None
    if path.suffix.lower() in {".txt", ".jsonl"}:
        meshes: list[str] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                mesh_path = _parse_list_entry(line, list_path=path)
                if mesh_path is not None:
                    meshes.append(mesh_path)
        return meshes, str(path.parent.resolve())
    raise ValueError(
        "--data_path must be a mesh file, a directory containing meshes, or a .txt/.jsonl mesh list"
    )


def _find_manifest_path(dataset_root: str, sample_id: str) -> str | None:
    candidates = (
        os.path.join(dataset_root, sample_id, "manifest.json"),
        os.path.join(dataset_root, "quarantine", sample_id, "manifest.json"),
        os.path.join(dataset_root, "failed", sample_id, "manifest.json"),
    )
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _classify_sample(manifest: dict, *, output_dir: str | None) -> str:
    sample_state = manifest.get("sample_state")
    if sample_state == "failed":
        return "failed"
    if sample_state == "skipped":
        return "skipped"
    if output_dir:
        normalized = output_dir.replace("\\", "/")
        if "/quarantine/" in normalized:
            return "quarantined"
        if "/failed/" in normalized:
            return "failed"
    if not manifest.get("quality", {}).get("accepted", True):
        return "quarantined"
    return "accepted"


def _write_dataset_summary(dataset_root: str, entries: list[dict]) -> str:
    counts = {
        "accepted": 0,
        "quarantined": 0,
        "failed": 0,
        "skipped": 0,
    }
    for entry in entries:
        counts[entry["status"]] += 1
    summary = {
        "total_samples": len(entries),
        "counts": counts,
        "samples": entries,
    }
    os.makedirs(dataset_root, exist_ok=True)
    summary_path = os.path.join(dataset_root, "dataset_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary_path


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args([] if argv is None else argv)
    if not args.data_path:
        parser.error("--data_path is required")

    mesh_paths, input_base_dir = resolve_input_meshes(args.data_path)
    if not mesh_paths:
        parser.error("No mesh files were found for --data_path")
    if len(mesh_paths) > 1 and args.sample_id is not None:
        parser.error("--sample_id can only be used with a single input mesh")

    from .train_quad_mesh import train_crossfield

    summary_entries: list[dict] = []
    summary_root: str | None = None
    last_error: Exception | None = None

    for mesh_path in mesh_paths:
        sample_args = copy.deepcopy(args)
        sample_args.data_path = mesh_path
        sample_args.dataset_root = _resolve_dataset_root(
            mesh_path,
            args.dataset_root,
            base_dir=input_base_dir,
        )
        sample_args.sample_id = args.sample_id or derive_sample_id(mesh_path)
        summary_root = sample_args.dataset_root

        try:
            result = train_crossfield(args=sample_args, allow_multiprocessing_workers=True)
            manifest_path = result.manifest_path or _find_manifest_path(sample_args.dataset_root, sample_args.sample_id)
            manifest = {}
            if manifest_path and os.path.exists(manifest_path):
                with open(manifest_path, "r", encoding="utf-8") as handle:
                    manifest = json.load(handle)
            summary_entries.append(
                {
                    "sample_id": sample_args.sample_id,
                    "source_mesh_path": mesh_path,
                    "status": _classify_sample(manifest, output_dir=result.output_dir),
                    "output_dir": result.output_dir,
                    "manifest_path": manifest_path,
                    "error": None,
                }
            )
        except Exception as exc:
            last_error = exc
            manifest_path = _find_manifest_path(sample_args.dataset_root, sample_args.sample_id)
            manifest = {}
            output_dir = None
            if manifest_path and os.path.exists(manifest_path):
                with open(manifest_path, "r", encoding="utf-8") as handle:
                    manifest = json.load(handle)
                output_dir = os.path.dirname(manifest_path)
            summary_entries.append(
                {
                    "sample_id": sample_args.sample_id,
                    "source_mesh_path": mesh_path,
                    "status": _classify_sample(manifest, output_dir=output_dir) if manifest else "failed",
                    "output_dir": output_dir,
                    "manifest_path": manifest_path,
                    "error": str(exc),
                }
            )
            if args.fail_fast:
                break

    if summary_root is not None and len(mesh_paths) > 1:
        _write_dataset_summary(summary_root, summary_entries)

    if last_error is not None and (args.fail_fast or len(mesh_paths) == 1):
        raise last_error


if __name__ == "__main__":
    main(sys.argv[1:])
