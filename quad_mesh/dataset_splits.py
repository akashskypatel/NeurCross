from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime, timezone


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _rel(root: str, path: str) -> str:
    return os.path.relpath(path, root).replace("\\", "/")


def iter_manifest_paths(dataset_root: str):
    for current_root, _dirs, files in os.walk(dataset_root):
        if "manifest.json" in files:
            yield os.path.join(current_root, "manifest.json")


def load_dataset_index(dataset_root: str, *, validate_artifacts: bool = False) -> list[dict[str, object]]:
    entries = []
    if validate_artifacts:
        from .export_dataset_sample import validate_manifest

    for manifest_path in sorted(iter_manifest_paths(dataset_root)):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if validate_artifacts:
            validate_manifest(manifest, os.path.dirname(manifest_path))

        sample_dir = os.path.dirname(manifest_path)
        destination = os.path.basename(os.path.dirname(sample_dir))
        source = manifest.get("source", {})
        entry = {
            "sample_id": manifest["sample_id"],
            "sample_state": manifest.get("sample_state", "completed"),
            "destination": destination,
            "manifest_path": _rel(dataset_root, manifest_path),
            "sample_dir": _rel(dataset_root, sample_dir),
            "source_shape_id": source.get("source_mesh_sha256"),
            "source_dataset": source.get("source_dataset"),
            "source_mesh_path": source.get("source_mesh_path"),
        }
        entries.append(entry)
    return entries


def group_index_by_shape_identity(entries: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for entry in entries:
        shape_id = str(entry.get("source_shape_id") or entry["sample_id"])
        grouped.setdefault(shape_id, []).append(entry)
    return grouped


def _resolve_ood_entries(
    entries: list[dict[str, object]],
    *,
    seed: int,
    ood_ratio: float,
    ood_source_datasets: list[str] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    explicit_datasets = [name for name in (ood_source_datasets or []) if name]
    accepted_entries = [
        entry for entry in entries
        if entry.get("destination") == "accepted" and entry.get("sample_state", "completed") == "completed"
    ]
    accepted_by_dataset: dict[str, list[dict[str, object]]] = {}
    for entry in accepted_entries:
        dataset_name = entry.get("source_dataset")
        if dataset_name:
            accepted_by_dataset.setdefault(str(dataset_name), []).append(entry)

    if explicit_datasets:
        selected_dataset_names = sorted({name for name in explicit_datasets if name in accepted_by_dataset})
        policy = "explicit_source_dataset"
    elif ood_ratio > 0.0 and len(accepted_by_dataset) >= 2:
        dataset_names = sorted(accepted_by_dataset)
        rng = random.Random(int(seed))
        rng.shuffle(dataset_names)
        holdout_count = max(1, int(round(len(dataset_names) * float(ood_ratio))))
        holdout_count = min(holdout_count, max(len(dataset_names) - 1, 0))
        selected_dataset_names = sorted(dataset_names[:holdout_count])
        policy = "ratio_source_dataset"
    else:
        selected_dataset_names = []
        policy = "disabled"

    ood_entries = []
    if selected_dataset_names:
        selected_set = set(selected_dataset_names)
        ood_entries = [
            entry for entry in accepted_entries
            if entry.get("source_dataset") in selected_set
        ]
    return ood_entries, {
        "policy": policy,
        "requested_ratio": float(ood_ratio),
        "selected_source_datasets": selected_dataset_names,
    }


def assign_grouped_splits(
    entries: list[dict[str, object]],
    *,
    seed: int = 0,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    excluded_sample_ids: set[str] | None = None,
) -> dict[str, list[dict[str, object]]]:
    ratios = {
        "train": float(train_ratio),
        "validation": float(validation_ratio),
        "test": float(test_ratio),
    }
    if any(value < 0.0 for value in ratios.values()):
        raise ValueError("split ratios must be non-negative")
    ratio_sum = sum(ratios.values())
    if ratio_sum <= 0.0:
        raise ValueError("split ratios must sum to a positive value")

    accepted_entries = [
        entry for entry in entries
        if entry.get("destination") == "accepted" and entry.get("sample_state", "completed") == "completed"
    ]
    if excluded_sample_ids:
        accepted_entries = [
            entry for entry in accepted_entries
            if entry["sample_id"] not in excluded_sample_ids
        ]
    grouped = list(group_index_by_shape_identity(accepted_entries).items())
    rng = random.Random(int(seed))
    rng.shuffle(grouped)

    group_count = len(grouped)
    train_cutoff = ratios["train"] / ratio_sum
    validation_cutoff = (ratios["train"] + ratios["validation"]) / ratio_sum

    assignments = {"train": [], "validation": [], "test": []}
    for index, (_shape_id, group_entries) in enumerate(grouped):
        progress = 0.0 if group_count == 0 else index / float(group_count)
        if progress < train_cutoff:
            split_name = "train"
        elif progress < validation_cutoff:
            split_name = "validation"
        else:
            split_name = "test"
        assignments[split_name].extend(group_entries)
    return assignments


def build_split_manifest(
    dataset_root: str,
    *,
    seed: int = 0,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    validate_artifacts: bool = False,
    ood_ratio: float = 0.0,
    ood_source_datasets: list[str] | None = None,
) -> dict[str, object]:
    entries = load_dataset_index(dataset_root, validate_artifacts=validate_artifacts)
    ood_entries, ood_policy = _resolve_ood_entries(
        entries,
        seed=seed,
        ood_ratio=ood_ratio,
        ood_source_datasets=ood_source_datasets,
    )
    accepted_splits = assign_grouped_splits(
        entries,
        seed=seed,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        excluded_sample_ids={entry["sample_id"] for entry in ood_entries},
    )
    quarantine_entries = [entry for entry in entries if entry.get("destination") == "quarantine"]
    failed_entries = [entry for entry in entries if entry.get("destination") == "failed"]
    excluded_entries = [
        entry for entry in entries
        if entry.get("destination") not in {"accepted", "quarantine", "failed"}
    ]

    source_datasets = sorted(
        {
            entry["source_dataset"]
            for entry in entries
            if entry.get("source_dataset")
        }
    )
    return {
        "artifact_type": "neurcross_dataset_splits",
        "created_at_utc": _utc_now(),
        "seed": int(seed),
        "split_ratios": {
            "train": float(train_ratio),
            "validation": float(validation_ratio),
            "test": float(test_ratio),
        },
        "ood_policy": ood_policy,
        "source_datasets": source_datasets,
        "splits": {
            "train": [entry["sample_id"] for entry in accepted_splits["train"]],
            "validation": [entry["sample_id"] for entry in accepted_splits["validation"]],
            "test": [entry["sample_id"] for entry in accepted_splits["test"]],
            "ood_test": [entry["sample_id"] for entry in ood_entries],
            "quarantine": [entry["sample_id"] for entry in quarantine_entries],
            "failed": [entry["sample_id"] for entry in failed_entries],
        },
        "counts": {
            "train": len(accepted_splits["train"]),
            "validation": len(accepted_splits["validation"]),
            "test": len(accepted_splits["test"]),
            "ood_test": len(ood_entries),
            "quarantine": len(quarantine_entries),
            "failed": len(failed_entries),
            "excluded": len(excluded_entries),
        },
        "excluded_samples": [entry["sample_id"] for entry in excluded_entries],
    }


def write_dataset_index(
    dataset_root: str,
    *,
    output_path: str | None = None,
    validate_artifacts: bool = False,
) -> str:
    output_path = output_path or os.path.join(dataset_root, "dataset_index.json")
    payload = {
        "artifact_type": "neurcross_dataset_index",
        "created_at_utc": _utc_now(),
        "dataset_root": os.path.abspath(dataset_root),
        "entries": load_dataset_index(dataset_root, validate_artifacts=validate_artifacts),
    }
    with open(output_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def write_split_manifest(
    dataset_root: str,
    *,
    output_path: str | None = None,
    seed: int = 0,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    validate_artifacts: bool = False,
    ood_ratio: float = 0.0,
    ood_source_datasets: list[str] | None = None,
) -> str:
    output_path = output_path or os.path.join(dataset_root, "dataset_splits.json")
    payload = build_split_manifest(
        dataset_root,
        seed=seed,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        validate_artifacts=validate_artifacts,
        ood_ratio=ood_ratio,
        ood_source_datasets=ood_source_datasets,
    )
    with open(output_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def build_index_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neurcross-build-dataset-index")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--validate_artifacts", action="store_true")
    return parser


def split_dataset_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neurcross-split-dataset")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--validation_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument(
        "--ood_ratio",
        type=float,
        default=0.0,
        help="fraction of distinct source_dataset families to hold out as OOD; ignored when --ood_source_dataset is provided",
    )
    parser.add_argument(
        "--ood_source_dataset",
        action="append",
        default=[],
        help="explicit source_dataset family to place into ood_test; can be passed multiple times",
    )
    parser.add_argument("--validate_artifacts", action="store_true")
    return parser


def build_dataset_index_main(argv: list[str] | None = None) -> str:
    args = build_index_parser().parse_args([] if argv is None else argv)
    return write_dataset_index(
        args.dataset_root,
        output_path=args.output_path,
        validate_artifacts=args.validate_artifacts,
    )


def split_dataset_main(argv: list[str] | None = None) -> str:
    args = split_dataset_parser().parse_args([] if argv is None else argv)
    return write_split_manifest(
        args.dataset_root,
        output_path=args.output_path,
        seed=args.seed,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        validate_artifacts=args.validate_artifacts,
        ood_ratio=args.ood_ratio,
        ood_source_datasets=args.ood_source_dataset,
    )
