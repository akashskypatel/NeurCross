import json


def _write_manifest(dataset_root, destination, sample_id, source_hash, *, sample_state="completed", source_dataset=None):
    sample_dir = dataset_root / destination / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "sample_id": sample_id,
        "sample_state": sample_state,
        "source": {
            "source_mesh_sha256": source_hash,
            "source_dataset": source_dataset,
            "source_mesh_path": f"input/{sample_id}.obj",
        },
    }
    path = sample_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def test_load_dataset_index_scans_manifest_tree(tmp_path):
    from quad_mesh.dataset_splits import load_dataset_index

    _write_manifest(tmp_path, "accepted", "sample-a", "hash-a")
    _write_manifest(tmp_path, "quarantine", "sample-b", "hash-b", source_dataset="abc")

    entries = load_dataset_index(str(tmp_path))

    assert [entry["sample_id"] for entry in entries] == ["sample-a", "sample-b"]
    assert entries[0]["destination"] == "accepted"
    assert entries[1]["source_dataset"] == "abc"


def test_assign_grouped_splits_keeps_same_shape_identity_together(tmp_path):
    from quad_mesh.dataset_splits import assign_grouped_splits, load_dataset_index

    _write_manifest(tmp_path, "accepted", "sample-a1", "shared")
    _write_manifest(tmp_path, "accepted", "sample-a2", "shared")
    _write_manifest(tmp_path, "accepted", "sample-b1", "other")

    entries = load_dataset_index(str(tmp_path))
    first = assign_grouped_splits(entries, seed=17, train_ratio=0.5, validation_ratio=0.0, test_ratio=0.5)
    second = assign_grouped_splits(entries, seed=17, train_ratio=0.5, validation_ratio=0.0, test_ratio=0.5)

    assert first == second
    train_ids = set(entry["sample_id"] for entry in first["train"])
    test_ids = set(entry["sample_id"] for entry in first["test"])
    assert {"sample-a1", "sample-a2"} <= train_ids or {"sample-a1", "sample-a2"} <= test_ids
    assert not ({"sample-a1", "sample-a2"} & train_ids and {"sample-a1", "sample-a2"} & test_ids)


def test_build_split_manifest_tracks_quarantine_and_failed(tmp_path):
    from quad_mesh.dataset_splits import build_split_manifest

    _write_manifest(tmp_path, "accepted", "sample-a", "hash-a", source_dataset="set-1")
    _write_manifest(tmp_path, "accepted", "sample-b", "hash-b", source_dataset="set-1")
    _write_manifest(tmp_path, "quarantine", "sample-q", "hash-q", source_dataset="set-2")
    _write_manifest(tmp_path, "failed", "sample-f", "hash-f")

    manifest = build_split_manifest(
        str(tmp_path),
        seed=3,
        train_ratio=0.5,
        validation_ratio=0.5,
        test_ratio=0.0,
    )

    assert manifest["seed"] == 3
    assert manifest["source_datasets"] == ["set-1", "set-2"]
    assert manifest["splits"]["quarantine"] == ["sample-q"]
    assert manifest["splits"]["failed"] == ["sample-f"]
    assert manifest["counts"]["quarantine"] == 1
    assert manifest["counts"]["failed"] == 1
    assert manifest["splits"]["ood_test"] == []


def test_neurcross_main_parser_includes_dataset_split_commands():
    from neurcross.__main__ import build_parser

    parser = build_parser()
    help_text = parser.format_help()

    assert "build-dataset-index" in help_text
    assert "split-dataset" in help_text
