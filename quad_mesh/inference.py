from dataclasses import asdict

import torch

from .checkpoint_utils import load_checkpoint


def _resolve_device(device: str):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_trained_model(
    checkpoint_path: str,
    device: str = "auto",
) -> tuple[torch.nn.Module, dict]:
    """
    Load a trained model checkpoint for inference.

    Returns:
        - model: loaded network in eval mode
        - metadata: checkpoint metadata dictionary
    """
    from models import Network_predict_angle

    device = _resolve_device(device)
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    args_dict = checkpoint.metadata.args_dict

    model = Network_predict_angle(
        in_dim=3,
        angle_in_dim=12,
        decoder_hidden_dim=args_dict.get("decoder_hidden_dim", 256),
        nl=args_dict.get("nl", "sine"),
        decoder_n_hidden_layers=args_dict.get("decoder_n_hidden_layers", 4),
        init_type=args_dict.get("init_type", "siren"),
        sphere_init_params=args_dict.get("sphere_init_params", [1.6, 0.1]),
        udf=args_dict.get("udf", False),
        latent_size=args_dict.get("latent_size", 0),
    )
    model.load_state_dict(checkpoint.model_state_dict)
    model.to(device)
    model.eval()
    return model, asdict(checkpoint.metadata)


def predict_crossfield(
    model: torch.nn.Module,
    nonmanifold_points: torch.Tensor,
    manifold_points: torch.Tensor | None = None,
    *,
    near_points: torch.Tensor | None = None,
    angle_features: torch.Tensor | None = None,
    device: str = "auto",
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """
    Run cross-field model inference.
    """
    if device == "auto":
        device = next(model.parameters()).device

    nonmanifold_points = nonmanifold_points.to(device)
    manifold_points = None if manifold_points is None else manifold_points.to(device)
    near_points = None if near_points is None else near_points.to(device)
    angle_features = None if angle_features is None else angle_features.to(device)

    model.eval()
    with torch.no_grad():
        output_pred, theta_output = model(
            nonmanifold_points,
            manifold_points,
            near_points=near_points,
            angle_features=angle_features,
        )
    return output_pred, theta_output
