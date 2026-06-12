import json
import os
import platform
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from multiprocessing import freeze_support

from . import quad_mesh_args

warnings.filterwarnings(
    'ignore',
    message=r'Attempting to run cuBLAS, but there was no current CUDA context!',
    category=UserWarning,
)


def _load_runtime_dependencies():
    import torch
    import torch.optim as optim

    try:
        from torchinfo import summary
    except ImportError:
        summary = None

    from models import Network_predict_angle
    from models import MorseLoss_quad_mesh as MorseLoss
    import utils.utils as utils
    from . import quad_mesh_dataset as dataset

    return {
        'torch': torch,
        'optim': optim,
        'summary': summary,
        'Network_predict_angle': Network_predict_angle,
        'MorseLoss': MorseLoss,
        'utils': utils,
        'dataset': dataset,
    }


def _format_duration(seconds: float) -> str:
    minutes, secs = divmod(seconds, 60.0)
    hours, minutes = divmod(minutes, 60.0)
    if hours >= 1:
        return f"{int(hours)}h {int(minutes)}m {secs:.2f}s"
    if minutes >= 1:
        return f"{int(minutes)}m {secs:.2f}s"
    return f"{seconds:.3f}s"


def _resolve_device(torch, requested_device):
    if requested_device == 'cpu':
        return 'cpu'
    if requested_device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available.")
        return 'cuda'
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def _raise_cuda_oom_guidance(exc, *, context):
    message = (
        f"CUDA out of memory during {context}. "
        "Try --device cpu, lower mesh complexity, or lower --max_topology_memory_gb to fail earlier. "
        "The topology/neighbor tensors are driven by mesh connectivity and are not reduced by --n_points."
    )
    raise RuntimeError(message) from exc


@dataclass
class TrainingResult:
    args: object
    output_dir: str
    log_path: str
    mesh_name: str
    total_elapsed_seconds: float
    stopped_early: bool
    stop_summary: dict | None
    checkpoint_path: str | None = None
    best_checkpoint_path: str | None = None
    weights_path: str | None = None
    manifest_path: str | None = None


def _build_training_args(argv=None, args=None, **overrides):
    if args is None:
        resolved = quad_mesh_args.get_args([] if argv is None else argv)
    else:
        resolved = args
    for key, value in overrides.items():
        setattr(resolved, key, value)
    return resolved


def _configure_programmatic_dataloader_workers(args, *, allow_multiprocessing_workers):
    if allow_multiprocessing_workers:
        return
    if os.name != 'nt' or int(getattr(args, 'num_workers', 0)) <= 0:
        return
    args.num_workers = 0
    args.persistent_workers = False
    print(
        'NeurCross API note: forcing num_workers=0 on Windows for embedded training. '
        'Use allow_multiprocessing_workers=True only from a script guarded by '
        'if __name__ == "__main__".',
        file=sys.stderr,
    )


def _resolve_output_directory(args):
    mesh_dir = os.path.dirname(os.path.abspath(args.data_path))
    file_name = os.path.splitext(os.path.basename(args.data_path))[0]
    dataset_root = getattr(args, "dataset_root", None)
    sample_id = getattr(args, "sample_id", None)
    if dataset_root:
        if not sample_id:
            raise ValueError("--sample_id is required when --dataset_root is set")
        if os.path.isabs(dataset_root):
            dataset_root_path = dataset_root
        else:
            dataset_root_path = os.path.join(mesh_dir, dataset_root)
        out_dir = os.path.join(dataset_root_path, sample_id)
        if os.path.exists(out_dir) and not getattr(args, "overwrite", False):
            raise FileExistsError(
                "Dataset sample directory already exists: {}. Pass --overwrite to reuse it.".format(out_dir)
            )
        os.makedirs(out_dir, exist_ok=True)
        return mesh_dir, file_name, out_dir

    if args.out_dir is None:
        out_dir_root = mesh_dir
    elif os.path.isabs(args.out_dir):
        out_dir_root = args.out_dir
    else:
        out_dir_root = os.path.join(mesh_dir, args.out_dir)
    out_dir = os.path.join(out_dir_root, file_name)
    os.makedirs(out_dir, exist_ok=True)
    return mesh_dir, file_name, out_dir


def _dataset_root_path(args):
    dataset_root = getattr(args, "dataset_root", None)
    if not dataset_root:
        return None
    mesh_dir = os.path.dirname(os.path.abspath(args.data_path))
    if os.path.isabs(dataset_root):
        return dataset_root
    return os.path.join(mesh_dir, dataset_root)


def _build_training_command(args, argv):
    command_name = "generate-label" if getattr(args, "sample_id", None) else "train-quad-mesh"
    return "python -m neurcross {} {}".format(
        command_name,
        " ".join(argv or []),
    ).strip()


def _detect_git_commit() -> str | None:
    repo_hint = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        completed = subprocess.run(
            ["git", "-C", repo_hint, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    return commit or None


def _relocate_dataset_sample(args, out_dir: str, destination: str) -> str:
    dataset_root = _dataset_root_path(args)
    sample_id = getattr(args, "sample_id", None)
    if not dataset_root or not sample_id:
        return out_dir
    target_dir = os.path.join(dataset_root, destination, sample_id)
    if os.path.abspath(target_dir) == os.path.abspath(out_dir):
        return out_dir
    os.makedirs(os.path.dirname(target_dir), exist_ok=True)
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    shutil.move(out_dir, target_dir)
    return target_dir


def _remap_path(path: str | None, old_root: str, new_root: str) -> str | None:
    if not path:
        return path
    rel_path = os.path.relpath(path, old_root)
    return os.path.join(new_root, rel_path)


def _capture_failed_dataset_run(
    *,
    args,
    argv,
    out_dir: str,
    preflight_report,
    run_started_utc: str,
    run_start_time: float,
    log_file,
    failure_message: str,
):
    from .export_dataset_sample import package_failed_dataset_sample
    from .checkpoint_utils import utc_timestamp
    from neurcross import __version__ as neurcross_version
    log_path = log_file.name
    log_file.close()
    run_finished_utc = utc_timestamp()
    elapsed_seconds = time.perf_counter() - run_start_time
    training_command = _build_training_command(args, argv)
    manifest_path = package_failed_dataset_sample(
        output_dir=out_dir,
        sample_id=args.sample_id,
        source_mesh_path=args.data_path,
        preflight_report=preflight_report.to_dict(),
        args_dict=dict(vars(args)),
        device=getattr(args, "device", "unknown"),
        started_at_utc=run_started_utc,
        finished_at_utc=run_finished_utc,
        elapsed_seconds=elapsed_seconds,
        neurcross_version=neurcross_version,
        training_command=training_command,
        quality_gate=getattr(args, "quality_gate", "default"),
        failure_reason=failure_message,
        log_path=log_path,
    )
    failed_out_dir = _relocate_dataset_sample(args, out_dir, "failed")
    return (
        failed_out_dir,
        _remap_path(log_path, out_dir, failed_out_dir),
        _remap_path(manifest_path, out_dir, failed_out_dir),
    )


def _write_validation_batch_artifact(out_dir: str, validation_batch: dict) -> str:
    import numpy as np

    metrics_dir = os.path.join(out_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    path = os.path.join(metrics_dir, "validation_samples.npz")
    payload = {
        key: np.asarray(value)
        for key, value in validation_batch.items()
    }
    np.savez(path, **payload)
    return path


def _write_validation_metrics_artifact(out_dir: str, metrics: dict, *, filename: str = "validation_metrics.json") -> str:
    metrics_dir = os.path.join(out_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    path = os.path.join(metrics_dir, filename)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def _write_validation_history_artifact(out_dir: str, history: list[dict]) -> str:
    metrics_dir = os.path.join(out_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    path = os.path.join(metrics_dir, "validation_history.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(history, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def _evaluate_validation_metrics(
    *,
    net,
    criterion,
    validation_batch: dict,
    mnfld_points_base,
    mnfld_n_gt,
    local_coord_u,
    local_coord_v,
    angle_feature_static,
    args,
    device: str,
    out_dir: str,
    file_name: str,
    global_step: int,
):
    torch = __import__("torch")
    non_blocking = device == "cuda"
    was_training = net.training
    net.eval()
    try:
        mnfld_points = mnfld_points_base.detach().clone().requires_grad_()
        nonmnfld_points = (
            torch.from_numpy(validation_batch["nonmnfld_points"])
            .unsqueeze(0)
            .to(device, non_blocking=non_blocking)
            .requires_grad_()
        )
        near_points = (
            torch.from_numpy(validation_batch["near_points"])
            .unsqueeze(0)
            .to(device, non_blocking=non_blocking)
            .requires_grad_()
        )
        features = torch.cat((mnfld_points, angle_feature_static), dim=-1)
        output_pred, mnfld_pts_theta_output_pred = net(
            nonmnfld_points,
            mnfld_points,
            near_points=near_points if args.morse_near else None,
            angle_features=features,
        )
        loss_dict = criterion(
            output_pred,
            mnfld_points,
            nonmnfld_points,
            mnfld_n_gt,
            near_points=near_points if args.morse_near else None,
            batch_idx=global_step,
            out_dir=out_dir,
            filename=file_name,
            save_best=False,
            mnfld_pts_theta_output_pred=mnfld_pts_theta_output_pred,
            local_coord_u=local_coord_u,
            local_coord_v=local_coord_v,
            is_final_export=False,
            collect_metrics=True,
        )
        metrics = dict(loss_dict["metrics"] or {})
        metrics["evaluation"] = {
            "kind": "fixed_validation_batch",
            "global_step": int(global_step),
            "nonmnfld_sample_count": int(validation_batch["nonmnfld_points"].shape[0]),
            "near_point_count": int(validation_batch["near_points"].shape[0]),
        }
        return metrics
    finally:
        if was_training:
            net.train()


def _copy_json_file(source_path: str, destination_path: str) -> str:
    with open(source_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    with open(destination_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return destination_path


def _resolve_training_schedule(args) -> tuple[int, int, int]:
    requested_steps_per_epoch = getattr(args, "steps_per_epoch", None)
    requested_total_steps = getattr(args, "total_steps", None)

    if requested_steps_per_epoch is not None and int(requested_steps_per_epoch) <= 0:
        raise ValueError("--steps_per_epoch must be positive when provided")
    if requested_total_steps is not None and int(requested_total_steps) <= 0:
        raise ValueError("--total_steps must be positive when provided")

    steps_per_epoch = int(requested_steps_per_epoch) if requested_steps_per_epoch is not None else int(args.n_samples)
    if requested_total_steps is None:
        num_epochs = int(args.num_epochs)
        total_steps = steps_per_epoch * num_epochs
    else:
        total_steps = int(requested_total_steps)
        num_epochs = max(1, (total_steps + steps_per_epoch - 1) // steps_per_epoch)
    return steps_per_epoch, num_epochs, total_steps


_CURRICULUM_STAGE_ORDER = ("geometry", "alignment", "smooth")
_CURRICULUM_STAGE_MULTIPLIERS = {
    "default": {
        "geometry": (1.25, 1.25, 0.65, 1.10, 0.60),
        "alignment": (1.00, 1.00, 1.10, 1.00, 1.00),
        "smooth": (0.85, 0.85, 1.20, 0.90, 1.35),
    },
    "cad": {
        "geometry": (1.20, 1.20, 0.70, 1.10, 0.55),
        "alignment": (1.00, 1.00, 1.20, 1.00, 0.95),
        "smooth": (0.80, 0.80, 1.30, 0.90, 1.45),
    },
    "organic": {
        "geometry": (1.15, 1.15, 0.75, 1.05, 0.65),
        "alignment": (1.00, 1.00, 1.05, 1.00, 1.00),
        "smooth": (0.90, 0.90, 1.10, 0.95, 1.30),
    },
}


def _build_curriculum_state(args, *, global_step: int, total_steps: int, base_weights) -> dict[str, object]:
    mode = getattr(args, "curriculum", "none")
    ratios = {
        "geometry": float(getattr(args, "geometry_stage_ratio", 0.2)),
        "alignment": float(getattr(args, "alignment_stage_ratio", 0.6)),
        "smooth": float(getattr(args, "smooth_stage_ratio", 0.2)),
    }
    if any(value < 0.0 for value in ratios.values()):
        raise ValueError("curriculum stage ratios must be non-negative")
    ratio_sum = sum(ratios.values())
    if ratio_sum <= 0.0:
        raise ValueError("curriculum stage ratios must sum to a positive value")

    normalized_ratios = {name: value / ratio_sum for name, value in ratios.items()}
    total_steps = max(int(total_steps), 1)
    progress = min(max(int(global_step), 0), total_steps - 1) / float(total_steps)

    cumulative = 0.0
    stage_bounds = {}
    stage_name = "smooth"
    stage_index = len(_CURRICULUM_STAGE_ORDER) - 1
    for index, name in enumerate(_CURRICULUM_STAGE_ORDER):
        start = cumulative
        cumulative += normalized_ratios[name]
        end = cumulative if index < len(_CURRICULUM_STAGE_ORDER) - 1 else 1.0
        stage_bounds[name] = {"start_ratio": start, "end_ratio": end}
        if progress < end or index == len(_CURRICULUM_STAGE_ORDER) - 1:
            stage_name = name
            stage_index = index
            break

    base_weight_list = [float(value) for value in base_weights]
    if mode == "none":
        active_weights = list(base_weight_list)
    else:
        multipliers = _CURRICULUM_STAGE_MULTIPLIERS[mode][stage_name]
        active_weights = [
            base_weight_list[idx] * multipliers[idx]
            for idx in range(min(len(base_weight_list), len(multipliers)))
        ]

    return {
        "mode": mode,
        "schedule_unit": getattr(args, "schedule_unit", "step"),
        "stage_name": stage_name,
        "stage_index": stage_index,
        "stage_bounds": stage_bounds,
        "normalized_ratios": normalized_ratios,
        "active_weights": active_weights,
        "enabled": mode != "none",
    }


def _apply_curriculum_weights(criterion, *, base_weights, curriculum_state: dict[str, object]) -> None:
    criterion.weights = list(criterion.weights)
    active_weights = curriculum_state["active_weights"]
    limit = min(len(active_weights), len(criterion.weights))
    for idx in range(limit):
        criterion.weights[idx] = float(active_weights[idx])


def _enforce_preflight_policy(preflight_report, policy: str):
    if preflight_report.status == "skip":
        return preflight_report
    if policy == "strict" and preflight_report.status != "accepted_for_training":
        preflight_report.status = "skip"
        preflight_report.skip_reason = (
            "preflight_policy strict rejected mesh because warnings or conservative repairs were required"
        )
    return preflight_report


class EarlyStopper:
    def __init__(
        self,
        *,
        min_steps,
        patience,
        min_delta,
        smooth_window,
        check_interval,
        target_loss=None,
        theta_neighbor_threshold=None,
        theta_hessian_threshold=None,
    ):
        self.min_steps = max(0, int(min_steps))
        self.patience = max(1, int(patience))
        self.min_delta = float(min_delta)
        self.smooth_window = max(1, int(smooth_window))
        self.check_interval = max(1, int(check_interval))
        self.target_loss = None if target_loss is None else float(target_loss)
        self.theta_neighbor_threshold = (
            None if theta_neighbor_threshold is None else float(theta_neighbor_threshold)
        )
        self.theta_hessian_threshold = (
            None if theta_hessian_threshold is None else float(theta_hessian_threshold)
        )
        self.history = []
        self.best_smooth_loss = float('inf')
        self.best_step = -1

    def update(self, step, loss_value, theta_neighbor_value, theta_hessian_value):
        self.history.append(float(loss_value))
        if (step + 1) % self.check_interval != 0:
            return None
        if len(self.history) < self.smooth_window:
            return None

        smooth_loss = sum(self.history[-self.smooth_window:]) / self.smooth_window
        if smooth_loss < self.best_smooth_loss - self.min_delta:
            self.best_smooth_loss = smooth_loss
            self.best_step = step

        if step < self.min_steps:
            return None

        theta_neighbor_ok = (
            self.theta_neighbor_threshold is None
            or float(theta_neighbor_value) <= self.theta_neighbor_threshold
        )
        theta_hessian_ok = (
            self.theta_hessian_threshold is None
            or float(theta_hessian_value) <= self.theta_hessian_threshold
        )
        thresholds_ok = theta_neighbor_ok and theta_hessian_ok

        if self.target_loss is not None and smooth_loss <= self.target_loss and thresholds_ok:
            return {
                'reason': 'target_loss',
                'smooth_loss': smooth_loss,
                'best_smooth_loss': self.best_smooth_loss,
                'best_step': self.best_step,
            }

        if self.best_step >= 0 and step - self.best_step >= self.patience and thresholds_ok:
            return {
                'reason': 'plateau',
                'smooth_loss': smooth_loss,
                'best_smooth_loss': self.best_smooth_loss,
                'best_step': self.best_step,
            }

        return None


def train_crossfield(*, argv=None, args=None, allow_multiprocessing_workers=False, **overrides):
    from .checkpoint_utils import (
        CheckpointMetadata,
        TrainingCheckpoint,
        capture_random_state,
        load_checkpoint,
        prune_old_checkpoints,
        restore_random_state,
        save_checkpoint,
        save_model_weights_only,
        utc_timestamp,
    )
    from .normalize import export_normalized_mesh
    from .preflight import inspect_mesh_path
    from .export_dataset_sample import (
        package_dataset_sample,
        package_failed_dataset_sample,
        package_skipped_dataset_sample,
    )
    from .feature_artifacts import export_feature_artifacts
    from .sdf_samples import export_sdf_samples

    # get training parameters
    args = _build_training_args(argv=argv, args=args, **overrides)
    _configure_programmatic_dataloader_workers(
        args,
        allow_multiprocessing_workers=allow_multiprocessing_workers,
    )
    if not args.data_path:
        raise ValueError('No default training mesh is bundled in the wheel build. Pass --data_path to a mesh file.')

    deps = _load_runtime_dependencies()
    torch = deps['torch']
    optim = deps['optim']
    summary = deps['summary']
    Network_predict_angle = deps['Network_predict_angle']
    MorseLoss = deps['MorseLoss']
    utils = deps['utils']
    dataset = deps['dataset']
    steps_per_epoch_target, derived_num_epochs, total_steps = _resolve_training_schedule(args)

    mesh_dir, file_name, out_dir = _resolve_output_directory(args)
    run_started_utc = utc_timestamp()
    manifest_path = None
    validation_samples_path = None
    validation_metrics_path = None
    validation_history_path = None
    feature_artifacts = None
    selected_label = "best"
    validation_history = []
    curriculum_state = None
    last_logged_curriculum_stage = None
    if args.checkpoint_dir is None:
        checkpoint_dir = os.path.join(out_dir, "checkpoints")
    elif os.path.isabs(args.checkpoint_dir):
        checkpoint_dir = args.checkpoint_dir
    else:
        checkpoint_dir = os.path.join(out_dir, args.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    run_start_time = time.perf_counter()

    # set up logging
    log_file = utils.setup_out_dir_only_log(out_dir, args)
    input_dir = os.path.join(out_dir, "input")
    mesh_report_path = os.path.join(out_dir, "mesh_quality_report.json")

    preflight_report, prepared_mesh = inspect_mesh_path(args.data_path)
    preflight_report = _enforce_preflight_policy(preflight_report, getattr(args, "preflight_policy", "repair"))
    if prepared_mesh is None or preflight_report.status == "skip":
        with open(mesh_report_path, "w", encoding="utf-8") as handle:
            json.dump(preflight_report.to_dict(), handle, indent=2, sort_keys=True)
        if getattr(args, "dataset_root", None):
            from neurcross import __version__ as neurcross_version

            run_finished_utc = utc_timestamp()
            elapsed_seconds = time.perf_counter() - run_start_time
            training_command = _build_training_command(args, argv)
            manifest_path = package_skipped_dataset_sample(
                output_dir=out_dir,
                sample_id=args.sample_id,
                source_mesh_path=args.data_path,
                preflight_report=preflight_report.to_dict(),
                args_dict=dict(vars(args)),
                device=args.device,
                started_at_utc=run_started_utc,
                finished_at_utc=run_finished_utc,
                elapsed_seconds=elapsed_seconds,
                neurcross_version=neurcross_version,
                training_command=training_command,
                quality_gate=getattr(args, "quality_gate", "default"),
                log_path=log_file.name,
            )
        log_file.close()
        return TrainingResult(
            args=args,
            output_dir=out_dir,
            log_path=log_file.name,
            mesh_name=file_name,
            total_elapsed_seconds=time.perf_counter() - run_start_time,
            stopped_early=False,
            stop_summary={
                "reason": "preflight_skip",
                "message": preflight_report.skip_reason or "input mesh is not trainable",
            },
            manifest_path=manifest_path,
        )

    try:
        if getattr(args, "normalize_mesh", True):
            normalized_export = export_normalized_mesh(prepared_mesh, input_dir)
            training_mesh_path = normalized_export.obj_path
        else:
            import numpy as np
            from .normalize import NormalizationMetadata, NormalizedMeshExport

            normalized_obj_path = os.path.join(input_dir, "normalized_mesh.obj")
            normalized_ply_path = os.path.join(input_dir, "normalized_mesh.ply")
            prepared_mesh.export(normalized_obj_path)
            prepared_mesh.export(normalized_ply_path)
            passthrough_metadata = NormalizationMetadata(
                center=[0.0, 0.0, 0.0],
                scale=1.0,
                bounds_before_min=np.asarray(prepared_mesh.bounds[0], dtype=np.float64).astype(float).tolist(),
                bounds_before_max=np.asarray(prepared_mesh.bounds[1], dtype=np.float64).astype(float).tolist(),
                bounds_after_min=np.asarray(prepared_mesh.bounds[0], dtype=np.float64).astype(float).tolist(),
                bounds_after_max=np.asarray(prepared_mesh.bounds[1], dtype=np.float64).astype(float).tolist(),
            )
            normalized_export = NormalizedMeshExport(
                mesh=prepared_mesh.copy(),
                obj_path=normalized_obj_path,
                ply_path=normalized_ply_path,
                metadata=passthrough_metadata,
            )
            training_mesh_path = args.data_path
        preflight_report.artifacts = {
            "normalized_mesh_obj": normalized_export.obj_path,
            "normalized_mesh_ply": normalized_export.ply_path,
        }
        if getattr(args, "export_features", True):
            feature_artifacts = export_feature_artifacts(
                mesh=normalized_export.mesh,
                output_dir=os.path.join(out_dir, "features"),
                feature_mode=getattr(args, "feature_mode", "auto"),
                angle_threshold_degrees=getattr(args, "feature_angle_threshold", 35.0),
            )
        preflight_report.normalization = normalized_export.metadata.to_dict()
        with open(mesh_report_path, "w", encoding="utf-8") as handle:
            json.dump(preflight_report.to_dict(), handle, indent=2, sort_keys=True)
        utils.log_string(
            "Mesh preflight: status={} faces={} vertices={} normalized_mesh={}".format(
                preflight_report.status,
                preflight_report.metrics.face_count,
                preflight_report.metrics.vertex_count,
                normalized_export.obj_path,
            ),
            log_file,
        )
        device = _resolve_device(torch, args.device)
        if device == 'cuda':
            torch.cuda.set_device(0)
            torch.cuda.init()
        utils.log_string("Training device: {}".format(device), log_file)

        # get data loaders
        utils.same_seed(args.seed, deterministic=not args.fast_nondeterministic)
        train_set = dataset.ReconDataset(
            training_mesh_path,
            args.n_points,
            steps_per_epoch_target,
            args.grid_res,
            nonmnfld_sample_type=args.nonmnfld_sample_type,
            near_surface_ratio=args.near_surface_ratio,
            uniform_ratio=args.uniform_ratio,
            feature_ratio=args.feature_ratio,
            boundary_ratio=args.boundary_ratio,
            near_surface_sigma=args.near_surface_sigma,
            uniform_extent=args.uniform_extent,
            seed=args.seed,
        )
        static_batch = train_set.get_static_batch()
        validation_batch = train_set.get_validation_batch()
        validation_samples_path = _write_validation_batch_artifact(out_dir, validation_batch)

        train_dataloader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device == 'cuda'),
            persistent_workers=(args.persistent_workers and args.num_workers > 0),
        )
    except Exception as exc:
        if getattr(args, "dataset_root", None):
            failure_message = "{}: {}".format(type(exc).__name__, exc)
            utils.log_string("Dataset-label setup failed: {}".format(failure_message), log_file)
            failed_out_dir, _failed_log_path, _failed_manifest_path = _capture_failed_dataset_run(
                args=args,
                argv=argv,
                out_dir=out_dir,
                preflight_report=preflight_report,
                run_started_utc=run_started_utc,
                run_start_time=run_start_time,
                log_file=log_file,
                failure_message=failure_message,
            )
            raise RuntimeError(
                "Dataset-label setup failed; artifacts captured under {}".format(failed_out_dir)
            ) from exc
        log_file.close()
        raise
    steps_per_epoch = len(train_dataloader)
    if steps_per_epoch != steps_per_epoch_target:
        raise RuntimeError(
            "training schedule mismatch: expected steps_per_epoch={}, got {}".format(
                steps_per_epoch_target,
                steps_per_epoch,
            )
        )
    utils.log_string(
        "Training schedule: requested_n_samples={} derived_steps_per_epoch={} requested_num_epochs={} "
        "derived_num_epochs={} requested_total_steps={} n_points_per_batch={} batch_size={} total_steps={}".format(
            args.n_samples,
            steps_per_epoch,
            args.num_epochs,
            derived_num_epochs,
            getattr(args, "total_steps", None),
            args.n_points,
            args.batch_size,
            total_steps,
        ),
        log_file,
    )
    if args.early_stop and args.early_stop_min_steps > total_steps:
        utils.log_string(
            "Early-stop note: early_stop_min_steps={} exceeds total_steps={}, so early stopping cannot trigger "
            "unless you increase --n_samples/--num_epochs or lower --early_stop_min_steps.".format(
                args.early_stop_min_steps,
                total_steps,
            ),
            log_file,
        )

    non_blocking = (device == 'cuda')
    mnfld_points_base = torch.from_numpy(static_batch['points']).unsqueeze(0).to(device, non_blocking=non_blocking)
    mnfld_n_gt = torch.from_numpy(static_batch['mnfld_n']).unsqueeze(0).to(device, non_blocking=non_blocking)
    local_coord_u = torch.from_numpy(static_batch['local_coordinates_u']).unsqueeze(0).to(device, non_blocking=non_blocking)
    local_coord_v = torch.from_numpy(static_batch['local_coordinates_v']).unsqueeze(0).to(device, non_blocking=non_blocking)
    angle_feature_static = torch.cat((mnfld_n_gt, local_coord_u, local_coord_v), dim=-1)

    # get model
    net = Network_predict_angle(in_dim=3, angle_in_dim=12, decoder_hidden_dim=args.decoder_hidden_dim, nl=args.nl,
                                decoder_n_hidden_layers=args.decoder_n_hidden_layers, init_type=args.init_type,
                                sphere_init_params=args.sphere_init_params, udf=args.udf)

    net.to(device)
    if args.load_path is not None and args.load_checkpoint is not None:
        raise ValueError("Use either --load_path for weights-only loading or --load_checkpoint for training resume, not both.")
    if args.load_path is not None:
        net.load_state_dict(torch.load(args.load_path))
        print('Loaded model from %s' % args.load_path)
    if summary is not None:
        try:
            summary(net.decoder, (1, 1024, 3))
        except UnicodeEncodeError:
            print('torchinfo summary could not be printed with the current console encoding; skipping model summary.')
    else:
        print('torchinfo is not installed; skipping model summary.')

    n_parameters = utils.count_parameters(net)
    utils.log_string("Number of parameters in the current model:{}".format(n_parameters), log_file)

    # Setup Adam optimizers
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.0)
    start_epoch = 0
    start_batch_idx = -1
    initial_global_step = 0
    resumed_checkpoint = None
    if args.load_checkpoint is not None:
        resumed_checkpoint = load_checkpoint(args.load_checkpoint, device=device)
        net.load_state_dict(resumed_checkpoint.model_state_dict)
        optimizer.load_state_dict(resumed_checkpoint.optimizer_state_dict)
        restore_random_state(resumed_checkpoint.random_state)
        start_epoch = resumed_checkpoint.metadata.epoch
        start_batch_idx = resumed_checkpoint.metadata.batch_idx
        initial_global_step = resumed_checkpoint.metadata.global_step
        utils.log_string(
            "Resumed training from checkpoint: epoch={} batch_idx={} next_global_step={}".format(
                start_epoch,
                start_batch_idx,
                initial_global_step,
            ),
            log_file,
        )
    print('steps_per_epoch: ', steps_per_epoch)
    print('n_iterations: ', total_steps)

    net.to(device)

    SAVE_BEST = False

    ##################################################################################
    # get the vertices neighbors of the mesh
    vertex_neighbors = utils.get_sample_vers_neighbors_for_face_center_points_or_vertices(training_mesh_path)
    vertex_neighbors_list = utils.calculate_same_neighbors_verts(vertex_neighbors)
    ###################################################################################
    axis_angle_R_mat_list = utils.get_rotation_matrix(vertex_neighbors_list, vertex_neighbors, training_mesh_path)

    try:
        criterion = MorseLoss(weights=args.loss_weights, loss_type=args.loss_type, div_decay=args.morse_decay,
                              div_type=args.morse_type,
                              vertex_neighbors_list=vertex_neighbors_list,
                              vertex_neighbors=vertex_neighbors, axis_angle_R_mat_list=axis_angle_R_mat_list,
                              device=device, max_topology_memory_gb=args.max_topology_memory_gb
                              )
    except torch.cuda.OutOfMemoryError as exc:
        _raise_cuda_oom_guidance(exc, context="topology cache setup")
    criterion_base_weights = [float(value) for value in criterion.weights[:5]]
    from models.loss_quad_mesh import export_crossfield_snapshot
    curriculum_state = _build_curriculum_state(
        args,
        global_step=0,
        total_steps=total_steps,
        base_weights=criterion_base_weights,
    )
    if curriculum_state["enabled"]:
        utils.log_string(
            "Curriculum enabled: mode={} schedule_unit={} stage_ratios={} initial_stage={}".format(
                curriculum_state["mode"],
                curriculum_state["schedule_unit"],
                curriculum_state["normalized_ratios"],
                curriculum_state["stage_name"],
            ),
            log_file,
        )
    else:
        utils.log_string("Curriculum disabled; using base loss weights.", log_file)

    early_stopper = None
    loss_history = []
    if args.early_stop:
        early_stopper = EarlyStopper(
            min_steps=args.early_stop_min_steps,
            patience=args.early_stop_patience,
            min_delta=args.early_stop_min_delta,
            smooth_window=args.early_stop_smooth_window,
            check_interval=args.early_stop_check_interval,
            target_loss=args.early_stop_target_loss,
            theta_neighbor_threshold=args.early_stop_theta_neighbor_threshold,
            theta_hessian_threshold=args.early_stop_theta_hessian_threshold,
        )
        utils.log_string(
            "Early stopping enabled: min_steps={} patience={} min_delta={} smooth_window={} "
            "check_interval={} target_loss={} theta_neighbor_threshold={} theta_hessian_threshold={}".format(
                early_stopper.min_steps,
                early_stopper.patience,
                early_stopper.min_delta,
                early_stopper.smooth_window,
                early_stopper.check_interval,
                early_stopper.target_loss,
                early_stopper.theta_neighbor_threshold,
                early_stopper.theta_hessian_threshold,
            ),
            log_file,
        )
    if resumed_checkpoint is not None:
        loss_history = list(resumed_checkpoint.metadata.loss_history)
        if early_stopper is not None and resumed_checkpoint.early_stopper_state:
            early_stopper.history = list(resumed_checkpoint.early_stopper_state.get('history', []))
            early_stopper.best_smooth_loss = resumed_checkpoint.early_stopper_state.get('best_smooth_loss', float('inf'))
            early_stopper.best_step = resumed_checkpoint.early_stopper_state.get('best_step', -1)

    net.train()
    stopped_early = False
    stop_summary = None
    checkpoint_path = None
    best_checkpoint_path = None
    weights_path = None
    manifest_path = None
    sdf_samples_path = None
    best_loss = min(loss_history) if loss_history else float('inf')
    last_epoch = start_epoch
    last_batch_idx = start_batch_idx
    next_global_step = initial_global_step
    current_global_step = initial_global_step

    def _save_training_checkpoint(filename, epoch, batch_idx, next_global_step):
        early_stopper_state = None
        best_smooth_loss = best_loss
        best_step = next_global_step - 1
        if early_stopper is not None:
            early_stopper_state = {
                'history': list(early_stopper.history),
                'best_smooth_loss': early_stopper.best_smooth_loss,
                'best_step': early_stopper.best_step,
            }
            best_smooth_loss = early_stopper.best_smooth_loss
            best_step = early_stopper.best_step
        metadata = CheckpointMetadata(
            epoch=epoch,
            batch_idx=batch_idx,
            global_step=next_global_step,
            total_epochs=derived_num_epochs,
            total_steps=total_steps,
            best_smooth_loss=float(best_smooth_loss),
            best_step=int(best_step),
            loss_history=list(loss_history),
            args_dict=dict(vars(args)),
            timestamp=utc_timestamp(),
            device=device,
        )
        checkpoint = TrainingCheckpoint(
            model_state_dict=net.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            metadata=metadata,
            early_stopper_state=early_stopper_state,
            random_state=capture_random_state(),
        )
        return save_checkpoint(checkpoint, checkpoint_dir, filename=filename)

    # For each epoch
    for epoch in range(start_epoch, derived_num_epochs):
        if hasattr(train_set, "set_epoch"):
            train_set.set_epoch(epoch)
        for batch_idx, data in enumerate(train_dataloader):
            if current_global_step >= total_steps:
                break
            if epoch == start_epoch and start_batch_idx >= 0 and batch_idx <= start_batch_idx:
                continue
            batch_start_time = time.perf_counter()
            global_step = current_global_step
            last_epoch = epoch
            last_batch_idx = batch_idx
            next_global_step = global_step + 1
            current_global_step = next_global_step
            curriculum_state = _build_curriculum_state(
                args,
                global_step=global_step,
                total_steps=total_steps,
                base_weights=criterion_base_weights,
            )
            _apply_curriculum_weights(
                criterion,
                base_weights=criterion_base_weights,
                curriculum_state=curriculum_state,
            )
            if curriculum_state["stage_name"] != last_logged_curriculum_stage:
                utils.log_string(
                    "Curriculum stage: global_step={} stage={} weights={}".format(
                        global_step,
                        curriculum_state["stage_name"],
                        curriculum_state["active_weights"],
                    ),
                    log_file,
                )
                last_logged_curriculum_stage = curriculum_state["stage_name"]
            export_interval_steps = max(int(getattr(args, "export_interval_steps", 500)), 0)
            if batch_idx != 0 and (
                (export_interval_steps > 0 and global_step > 0 and global_step % export_interval_steps == 0)
                or batch_idx == steps_per_epoch - 1
            ):
                SAVE_BEST = True
            is_final_batch = next_global_step >= total_steps

            optimizer.zero_grad(set_to_none=True)

            mnfld_points = mnfld_points_base.detach().clone()
            nonmnfld_points = data['nonmnfld_points'].to(device, non_blocking=non_blocking)
            near_points = data['near_points'].to(device, non_blocking=non_blocking)
            mnfld_points.requires_grad_()
            nonmnfld_points.requires_grad_()
            near_points.requires_grad_()

            features = torch.cat((mnfld_points, angle_feature_static), dim=-1)

            try:
                output_pred, mnfld_pts_theta_output_pred = net(nonmnfld_points, mnfld_points,
                                                               near_points=near_points if args.morse_near else None,
                                                               angle_features=features)

                loss_dict = criterion(output_pred, mnfld_points, nonmnfld_points, mnfld_n_gt,
                                      near_points=near_points if args.morse_near else None, batch_idx=global_step,
                                      out_dir=out_dir, filename=file_name, save_best=SAVE_BEST,
                                      mnfld_pts_theta_output_pred=mnfld_pts_theta_output_pred,
                                      local_coord_u=local_coord_u, local_coord_v=local_coord_v,
                                      is_final_export=is_final_batch)
            except torch.cuda.OutOfMemoryError as exc:
                _raise_cuda_oom_guidance(exc, context="forward/loss computation")

            lr = optimizer.param_groups[0]['lr']

            try:
                loss_dict["loss"].backward()
            except torch.cuda.OutOfMemoryError as exc:
                _raise_cuda_oom_guidance(exc, context="backward pass")

            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip_norm)

            SAVE_BEST = False
            try:
                optimizer.step()
            except Exception as exc:
                if getattr(args, "dataset_root", None):
                    failure_message = "training_step_failed: {}: {}".format(type(exc).__name__, exc)
                    utils.log_string("Dataset-label training failed: {}".format(failure_message), log_file)
                    failed_out_dir, _failed_log_path, _failed_manifest_path = _capture_failed_dataset_run(
                        args=args,
                        argv=argv,
                        out_dir=out_dir,
                        preflight_report=preflight_report,
                        run_started_utc=run_started_utc,
                        run_start_time=run_start_time,
                        log_file=log_file,
                        failure_message=failure_message,
                    )
                    raise RuntimeError(
                        "Dataset-label training failed; artifacts captured under {}".format(failed_out_dir)
                    ) from exc
                raise
            batch_elapsed = time.perf_counter() - batch_start_time
            current_loss_value = float(loss_dict["loss"].detach().cpu().item())
            loss_history.append(current_loss_value)
            improved_loss = current_loss_value < best_loss
            if improved_loss:
                best_loss = current_loss_value
                best_checkpoint_path = _save_training_checkpoint(
                    "best_checkpoint.pt",
                    epoch,
                    batch_idx,
                    global_step + 1,
                )
            should_save_periodic = (
                args.save_checkpoint_interval > 0
                and (global_step + 1) % args.save_checkpoint_interval == 0
                and (not args.save_best_only or improved_loss)
            )
            if should_save_periodic:
                checkpoint_path = _save_training_checkpoint(
                    "checkpoint_step_{}.pt".format(global_step + 1),
                    epoch,
                    batch_idx,
                    global_step + 1,
                )
                prune_old_checkpoints(
                    checkpoint_dir,
                    keep_last=args.keep_last_n_checkpoints,
                )

            # Output training stats
            if batch_idx % args.log_interval == 0:
                weights = criterion.weights
                scalar_names = (
                    "loss",
                    "sdf_term",
                    "inter_term",
                    "eikonal_term",
                    "morse_term",
                    "theta_hessian_term",
                    "theta_neighbors_term",
                )
                scalar_values = torch.stack([loss_dict[name].reshape(()) for name in scalar_names]).detach().cpu().tolist()
                (
                    loss_value,
                    sdf_term_value,
                    inter_term_value,
                    eikonal_term_value,
                    morse_term_value,
                    theta_hessian_term_value,
                    theta_neighbors_term_value,
                ) = scalar_values
                elapsed_total = time.perf_counter() - run_start_time
                utils.log_string("Weights: {}, lr={:.3e}".format(weights, lr), log_file)
                utils.log_string(
                    "Timing: batch={} total_elapsed={}".format(
                        _format_duration(batch_elapsed),
                        _format_duration(elapsed_total),
                    ),
                    log_file,
                )
                utils.log_string('Epoch: {} [{:4d}/{} ({:.0f}%)] Loss: {:.5f} = L_Mnfld: {:.5f} + '
                                 'L_NonMnfld: {:.5f} + L_Eknl: {:.5f} + L_Morse: {:.5f} + L_thetaHessian: {:.5f} + L_thetaNeighbor: {:.5f}'.format(
                    epoch, batch_idx * args.batch_size, len(train_set), 100. * batch_idx / max(steps_per_epoch, 1),
                    loss_value, weights[0] * sdf_term_value,
                           weights[1] * inter_term_value,
                           weights[3] * eikonal_term_value, weights[5] * morse_term_value,
                           weights[2] * theta_hessian_term_value,
                           weights[4] * theta_neighbors_term_value
                ),
                    log_file)
                utils.log_string('Epoch: {} [{:4d}/{} ({:.0f}%)] Unweighted L_s : L_Mnfld: {:.5f} + '
                                 'L_NonMnfld: {:.5f} + L_Eknl: {:.5f} + L_Morse: {:.5f} + L_thetaHessian: {:.5f} + L_thetaNeighbor: {:.5f}'.format(
                    epoch, batch_idx * args.batch_size, len(train_set), 100. * batch_idx / max(steps_per_epoch, 1),
                    sdf_term_value, inter_term_value,
                    eikonal_term_value, morse_term_value,
                    theta_hessian_term_value, theta_neighbors_term_value),
                    log_file)
                utils.log_string('', log_file)

            if early_stopper is not None:
                stop_summary = early_stopper.update(
                    global_step,
                    loss_dict["loss"].detach().cpu().item(),
                    loss_dict["theta_neighbors_term"].detach().cpu().item(),
                    loss_dict["theta_hessian_term"].detach().cpu().item(),
                )
                if stop_summary is not None:
                    features_detached = features.detach()
                    with torch.no_grad():
                        _output_pred, theta_output = net(
                            nonmnfld_points.detach(),
                            mnfld_points.detach(),
                            near_points=near_points.detach() if args.morse_near else None,
                            angle_features=features_detached,
                        )
                        theta_output = theta_output.squeeze(0)
                        vector_alpha = local_coord_u.squeeze(0) * torch.cos(theta_output) + local_coord_v.squeeze(0) * torch.sin(theta_output)
                        vector_alpha = vector_alpha / (vector_alpha.norm(dim=-1, keepdim=True) + 1e-12)
                        vector_beta = -local_coord_u.squeeze(0) * torch.sin(theta_output) + local_coord_v.squeeze(0) * torch.cos(theta_output)
                        vector_beta = vector_beta / (vector_beta.norm(dim=-1, keepdim=True) + 1e-12)
                        export_crossfield_snapshot(
                            vector_alpha,
                            vector_beta,
                            out_dir=out_dir,
                            filename=file_name,
                            batch_idx=global_step,
                            manager=criterion._export_manager,
                            total_loss=loss_dict["loss"].detach().cpu().item(),
                            field_score=(0.5 * (loss_dict["theta_hessian_term"] + loss_dict["theta_neighbors_term"])).detach().cpu().item(),
                            metrics={
                                "training": {
                                    "loss_total": float(loss_dict["loss"].detach().cpu().item()),
                                    "loss_mnfld": float(loss_dict["sdf_term"].detach().cpu().item()),
                                    "loss_nonmnfld": float(loss_dict["inter_term"].detach().cpu().item()),
                                    "loss_eikonal": float(loss_dict["eikonal_term"].detach().cpu().item()),
                                    "loss_morse": float(loss_dict["morse_term"].detach().cpu().item()),
                                    "loss_theta_hessian": float(loss_dict["theta_hessian_term"].detach().cpu().item()),
                                    "loss_theta_neighbor": float(loss_dict["theta_neighbors_term"].detach().cpu().item()),
                                }
                            },
                            is_final=True,
                        )
                    utils.log_string(
                        "Early stopping triggered at global_step={} epoch={} batch_idx={} reason={} "
                        "smooth_loss={:.6f} best_smooth_loss={:.6f} best_step={}".format(
                            global_step,
                            epoch,
                            batch_idx,
                            stop_summary['reason'],
                            stop_summary['smooth_loss'],
                            stop_summary['best_smooth_loss'],
                            stop_summary['best_step'],
                        ),
                        log_file,
                    )
                    checkpoint_path = _save_training_checkpoint(
                        "early_stop_checkpoint.pt",
                        epoch,
                        batch_idx,
                        global_step + 1,
                    )
                    stopped_early = True
                    break

            eval_interval_steps = max(int(getattr(args, "eval_interval_steps", 0)), 0)
            if eval_interval_steps > 0 and next_global_step % eval_interval_steps == 0:
                validation_metrics_step = _evaluate_validation_metrics(
                    net=net,
                    criterion=criterion,
                    validation_batch=validation_batch,
                    mnfld_points_base=mnfld_points_base,
                    mnfld_n_gt=mnfld_n_gt,
                    local_coord_u=local_coord_u,
                    local_coord_v=local_coord_v,
                    angle_feature_static=angle_feature_static,
                    args=args,
                    device=device,
                    out_dir=out_dir,
                    file_name=file_name,
                    global_step=next_global_step,
                )
                validation_history.append(validation_metrics_step)
                utils.log_string(
                    "Validation: global_step={} score={:.6f}".format(
                        next_global_step,
                        float(validation_metrics_step["score"]),
                    ),
                    log_file,
                )

            criterion.update_morse_weight(global_step, total_steps,
                                          args.decay_params)  # assumes batch size of 1
        if stopped_early:
            break
        if current_global_step >= total_steps:
            break

    try:
        total_elapsed = time.perf_counter() - run_start_time
        run_finished_utc = utc_timestamp()
        if stopped_early:
            utils.log_string("Training stopped early in {}".format(_format_duration(total_elapsed)), log_file)
        else:
            utils.log_string("Training finished in {}".format(_format_duration(total_elapsed)), log_file)
        final_checkpoint_path = _save_training_checkpoint(
            "final_checkpoint.pt",
            last_epoch,
            last_batch_idx,
            next_global_step,
        )
        checkpoint_path = final_checkpoint_path
        if args.export_weights_only:
            weights_path = save_model_weights_only(
                net.state_dict(),
                checkpoint_dir,
                filename="model_weights.pt",
            )
        log_path = log_file.name
        log_file.close()
        if getattr(args, "export_sdf_samples", False):
            sdf_samples_path = export_sdf_samples(
                mesh_path=normalized_export.obj_path,
                output_dir=os.path.join(out_dir, "sdf"),
                normalization=normalized_export.metadata.to_dict(),
                seed=args.seed,
                n_surface=args.sdf_n_surface,
                n_near=args.sdf_n_near,
                n_uniform=args.sdf_n_uniform,
                near_sigma=args.sdf_near_sigma,
                uniform_extent=args.sdf_uniform_extent,
                tsdf_truncation=args.tsdf_truncation,
            )
        validation_metrics = _evaluate_validation_metrics(
            net=net,
            criterion=criterion,
            validation_batch=validation_batch,
            mnfld_points_base=mnfld_points_base,
            mnfld_n_gt=mnfld_n_gt,
            local_coord_u=local_coord_u,
            local_coord_v=local_coord_v,
            angle_feature_static=angle_feature_static,
            args=args,
            device=device,
            out_dir=out_dir,
            file_name=file_name,
            global_step=max(next_global_step - 1, 0),
        )
        validation_metrics_final_path = _write_validation_metrics_artifact(
            out_dir,
            validation_metrics,
            filename="validation_metrics_final.json",
        )
        validation_history.append(validation_metrics)
        validation_history_path = _write_validation_history_artifact(out_dir, validation_history)
        validation_metrics_path = validation_metrics_final_path
        if args.save_best_by == "quad_score":
            raise NotImplementedError("--save_best_by quad_score is not implemented yet")
        if args.save_best_by == "val_field_score" and best_checkpoint_path:
            best_checkpoint = load_checkpoint(best_checkpoint_path, device=device)
            final_state_dict = net.state_dict()
            net.load_state_dict(best_checkpoint.model_state_dict)
            best_validation_metrics = _evaluate_validation_metrics(
                net=net,
                criterion=criterion,
                validation_batch=validation_batch,
                mnfld_points_base=mnfld_points_base,
                mnfld_n_gt=mnfld_n_gt,
                local_coord_u=local_coord_u,
                local_coord_v=local_coord_v,
                angle_feature_static=angle_feature_static,
                args=args,
                device=device,
                out_dir=out_dir,
                file_name=file_name,
                global_step=int(best_checkpoint.metadata.global_step),
            )
            net.load_state_dict(final_state_dict)
            best_validation_metrics_path = _write_validation_metrics_artifact(
                out_dir,
                best_validation_metrics,
                filename="validation_metrics_best.json",
            )
            if float(best_validation_metrics["score"]) <= float(validation_metrics["score"]):
                selected_label = "best"
                validation_metrics_path = _copy_json_file(
                    best_validation_metrics_path,
                    os.path.join(out_dir, "metrics", "validation_metrics.json"),
                )
            else:
                selected_label = "final"
                validation_metrics_path = _copy_json_file(
                    validation_metrics_final_path,
                    os.path.join(out_dir, "metrics", "validation_metrics.json"),
                )
        elif args.save_best_by == "train_field_score":
            selected_label = "best"
            validation_metrics_path = _copy_json_file(
                validation_metrics_final_path,
                os.path.join(out_dir, "metrics", "validation_metrics.json"),
            )
        else:
            selected_label = "final" if args.save_best_by == "val_field_score" else "best"
            validation_metrics_path = _copy_json_file(
                validation_metrics_final_path,
                os.path.join(out_dir, "metrics", "validation_metrics.json"),
            )
        if getattr(args, "dataset_root", None):
            from neurcross import __version__ as neurcross_version

            training_command = _build_training_command(args, argv)
            runtime_info = {
                "git_commit": _detect_git_commit(),
                "python_version": sys.version.split()[0],
                "torch_version": getattr(torch, "__version__", None),
                "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
                "platform": platform.platform(),
            }
            manifest_path = package_dataset_sample(
                output_dir=out_dir,
                sample_id=args.sample_id,
                mesh_name=file_name,
                source_mesh_path=args.data_path,
                normalized_mesh_ply_path=normalized_export.ply_path,
                preflight_report=preflight_report.to_dict(),
                normalization=normalized_export.metadata.to_dict(),
                args_dict=dict(vars(args)),
                device=device,
                log_path=log_path,
                started_at_utc=run_started_utc,
                finished_at_utc=run_finished_utc,
                elapsed_seconds=total_elapsed,
                neurcross_version=neurcross_version,
                training_command=training_command,
                stopped_early=stopped_early,
                stop_summary=stop_summary,
                runtime_info=runtime_info,
                sdf_samples_path=sdf_samples_path,
                validation_samples_path=validation_samples_path,
                validation_metrics_path=validation_metrics_path,
                validation_history_path=validation_history_path,
                feature_artifacts=feature_artifacts,
                selected_label=selected_label,
                export_geometry_npz=getattr(args, "export_geometry_npz", True),
                quality_gate=getattr(args, "quality_gate", "default"),
                curriculum_state=curriculum_state,
            )
            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            recommended_destination = manifest["quality"].get("recommended_destination")
            if recommended_destination == "quarantine":
                quarantine_out_dir = _relocate_dataset_sample(args, out_dir, "quarantine")
                log_path = _remap_path(log_path, out_dir, quarantine_out_dir)
                checkpoint_path = _remap_path(checkpoint_path, out_dir, quarantine_out_dir)
                best_checkpoint_path = _remap_path(best_checkpoint_path, out_dir, quarantine_out_dir)
                weights_path = _remap_path(weights_path, out_dir, quarantine_out_dir)
                manifest_path = _remap_path(manifest_path, out_dir, quarantine_out_dir)
                out_dir = quarantine_out_dir
            elif recommended_destination == "accepted":
                accepted_out_dir = _relocate_dataset_sample(args, out_dir, "accepted")
                log_path = _remap_path(log_path, out_dir, accepted_out_dir)
                checkpoint_path = _remap_path(checkpoint_path, out_dir, accepted_out_dir)
                best_checkpoint_path = _remap_path(best_checkpoint_path, out_dir, accepted_out_dir)
                weights_path = _remap_path(weights_path, out_dir, accepted_out_dir)
                manifest_path = _remap_path(manifest_path, out_dir, accepted_out_dir)
                out_dir = accepted_out_dir
    except Exception as exc:
        if getattr(args, "dataset_root", None):
            failure_message = "training_finalize_failed: {}: {}".format(type(exc).__name__, exc)
            failed_out_dir, _failed_log_path, _failed_manifest_path = _capture_failed_dataset_run(
                args=args,
                argv=argv,
                out_dir=out_dir,
                preflight_report=preflight_report,
                run_started_utc=run_started_utc,
                run_start_time=run_start_time,
                log_file=log_file,
                failure_message=failure_message,
            )
            raise RuntimeError(
                "Dataset-label finalization failed; artifacts captured under {}".format(failed_out_dir)
            ) from exc
        raise
    return TrainingResult(
        args=args,
        output_dir=out_dir,
        log_path=log_path,
        mesh_name=file_name,
        total_elapsed_seconds=total_elapsed,
        stopped_early=stopped_early,
        stop_summary=stop_summary,
        checkpoint_path=checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        weights_path=weights_path,
        manifest_path=manifest_path,
    )


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    train_crossfield(argv=argv, allow_multiprocessing_workers=True)


if __name__ == '__main__':
    freeze_support()
    main()
