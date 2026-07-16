import errno
import os
from types import SimpleNamespace


def test_relocate_dataset_sample_recovers_from_partial_destination(monkeypatch, tmp_path):
    from quad_mesh.train_quad_mesh import _relocate_dataset_sample

    dataset_root = tmp_path / "dataset_labels"
    sample_id = "sample-001"
    out_dir = dataset_root / sample_id
    out_dir.mkdir(parents=True)
    (out_dir / "checkpoints").mkdir()
    (out_dir / "checkpoints" / "final_checkpoint.safetensors").write_bytes(b"checkpoint")
    (out_dir / "metrics").mkdir()
    (out_dir / "metrics" / "metrics_final.json").write_text("{}", encoding="utf-8")

    original_rename = os.rename

    def fail_after_partial_destination(src, dst):
        target = os.fspath(dst)
        os.makedirs(os.path.join(target, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(target, "metrics"), exist_ok=True)
        raise OSError(errno.EIO, "Input/output error")

    monkeypatch.setattr(os, "rename", fail_after_partial_destination)

    source_mesh = tmp_path / "source.obj"
    source_mesh.write_text("v 0 0 0\n", encoding="utf-8")
    args = SimpleNamespace(dataset_root=str(dataset_root), data_path=str(source_mesh), sample_id=sample_id)
    relocated = _relocate_dataset_sample(args, str(out_dir), "accepted")

    monkeypatch.setattr(os, "rename", original_rename)

    target_dir = dataset_root / "accepted" / sample_id
    assert relocated == str(target_dir)
    assert not out_dir.exists()
    assert (target_dir / "checkpoints" / "final_checkpoint.safetensors").read_bytes() == b"checkpoint"
    assert (target_dir / "metrics" / "metrics_final.json").read_text(encoding="utf-8") == "{}"
