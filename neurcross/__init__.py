__version__ = "0.1.0"

from quad_mesh.train_quad_mesh import TrainingResult


def train_crossfield(*, argv=None, args=None, allow_multiprocessing_workers=False, **overrides):
    from quad_mesh.train_quad_mesh import train_crossfield as _train_crossfield

    return _train_crossfield(
        argv=argv,
        args=args,
        allow_multiprocessing_workers=allow_multiprocessing_workers,
        **overrides,
    )

def load_checkpoint(checkpoint_path, device="cpu"):
    from quad_mesh.checkpoint_utils import load_checkpoint as _load_checkpoint

    return _load_checkpoint(checkpoint_path, device=device)


def save_checkpoint(checkpoint, output_dir, filename="checkpoint.pt", save_best_only=False):
    from quad_mesh.checkpoint_utils import save_checkpoint as _save_checkpoint

    return _save_checkpoint(
        checkpoint,
        output_dir,
        filename=filename,
        save_best_only=save_best_only,
    )


def save_model_weights_only(state_dict, output_dir, filename="model_weights.pt"):
    from quad_mesh.checkpoint_utils import save_model_weights_only as _save_model_weights_only

    return _save_model_weights_only(state_dict, output_dir, filename=filename)


def load_trained_model(checkpoint_path, device="auto"):
    from quad_mesh.inference import load_trained_model as _load_trained_model

    return _load_trained_model(checkpoint_path, device=device)


def predict_crossfield(model, nonmanifold_points, manifold_points=None, **kwargs):
    from quad_mesh.inference import predict_crossfield as _predict_crossfield

    return _predict_crossfield(
        model,
        nonmanifold_points,
        manifold_points,
        **kwargs,
    )


def __getattr__(name):
    if name in {"CheckpointMetadata", "TrainingCheckpoint"}:
        from quad_mesh import checkpoint_utils

        return getattr(checkpoint_utils, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CheckpointMetadata",
    "TrainingResult",
    "TrainingCheckpoint",
    "__version__",
    "load_checkpoint",
    "load_trained_model",
    "predict_crossfield",
    "save_checkpoint",
    "save_model_weights_only",
    "train_crossfield",
]
