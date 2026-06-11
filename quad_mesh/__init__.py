def train_crossfield(*, argv=None, args=None, allow_multiprocessing_workers=False, **overrides):
    from .train_quad_mesh import train_crossfield as _train_crossfield

    return _train_crossfield(
        argv=argv,
        args=args,
        allow_multiprocessing_workers=allow_multiprocessing_workers,
        **overrides,
    )

def __getattr__(name):
    if name == "TrainingResult":
        from .train_quad_mesh import TrainingResult

        return TrainingResult
    if name in {
        "CheckpointMetadata",
        "TrainingCheckpoint",
        "load_checkpoint",
        "save_checkpoint",
        "save_model_weights_only",
    }:
        from . import checkpoint_utils

        return getattr(checkpoint_utils, name)
    if name in {"load_trained_model", "predict_crossfield"}:
        from . import inference

        return getattr(inference, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CheckpointMetadata",
    "TrainingResult",
    "TrainingCheckpoint",
    "load_checkpoint",
    "load_trained_model",
    "predict_crossfield",
    "save_checkpoint",
    "save_model_weights_only",
    "train_crossfield",
]
