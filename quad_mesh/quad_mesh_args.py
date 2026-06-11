import argparse
import os


def _default_data_path():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate = os.path.join(repo_root, 'data', 'doubleTorus', 'input', 'doubleTorus.ply')
    return candidate if os.path.exists(candidate) else None


def add_args(parser):
    parser.add_argument(
        '--out_dir',
        type=str,
        default=None,
        help='optional output directory; if omitted, outputs are written beside the input mesh',
    )
    parser.add_argument(
        '--dataset_root',
        type=str,
        default=None,
        help='optional dataset output root used by generate-label; when set, outputs are written under dataset_root/sample_id',
    )
    parser.add_argument(
        '--sample_id',
        type=str,
        default=None,
        help='optional dataset sample identifier used by generate-label',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='allow writing into an existing dataset sample directory when using generate-label',
    )
    parser.add_argument(
        '--fail_fast',
        action='store_true',
        help='stop batch generate-label runs on the first sample failure instead of continuing',
    )
    parser.add_argument(
        '--normalize_mesh',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='normalize the prepared mesh into [-0.5, 0.5]^3 before training and packaging',
    )
    parser.add_argument(
        '--preflight_policy',
        choices=('strict', 'repair', 'report_only'),
        default='repair',
        help='how generate-label handles preflight findings before training',
    )
    parser.add_argument(
        '--export_geometry_npz',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='export geometry/mesh_geometry.npz for dataset-label runs',
    )
    parser.add_argument(
        '--quality_gate',
        choices=('none', 'default', 'strict', 'loose'),
        default='default',
        help='quality acceptance profile recorded and applied during dataset packaging',
    )
    parser.add_argument('--model_name', type=str, default='model', help='trained model name')
    parser.add_argument('--seed', type=int, default=3627473, help='random seed')
    parser.add_argument(
        '--data_path',
        type=str,
        default=_default_data_path(),
        help='path to input mesh; required when running from an installed wheel without repo data files',
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=10000,
        help='number of training samples (batches) per epoch; this controls steps-per-epoch when batch_size=1; recommended value is 10000',
    )
    parser.add_argument(
        '--n_points',
        type=int,
        default=15000,
        help='number of sampled points generated inside each training batch; this does not control epoch length',
    )
    parser.add_argument('--grid_res', type=int, default=256, help='uniform grid resolution')
    parser.add_argument(
        '--nonmnfld_sample_type',
        choices=('uniform', 'near_surface', 'mixed', 'feature_biased', 'grid', 'gaussian', 'combined'),
        default='uniform',
        help='how to sample off-manifold points for training',
    )
    parser.add_argument('--near_surface_ratio', type=float, default=None, help='mixed-mode ratio allocated to near-surface off-manifold samples')
    parser.add_argument('--uniform_ratio', type=float, default=None, help='mixed-mode ratio allocated to uniform volume off-manifold samples')
    parser.add_argument('--feature_ratio', type=float, default=None, help='mixed-mode ratio allocated to feature-biased off-manifold samples')
    parser.add_argument('--boundary_ratio', type=float, default=0.5, help='feature-biased blend between boundary emphasis and normal-variation emphasis')
    parser.add_argument('--near_surface_sigma', type=float, default=None, help='optional fixed sigma for near-surface sampling in normalized coordinates')
    parser.add_argument('--uniform_extent', type=float, default=None, help='uniform off-manifold sampling extent in normalized coordinates')

    # training parameters
    parser.add_argument('--num_epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='initial learning rate')
    parser.add_argument('--grad_clip_norm', type=float, default=10.0, help='Value to clip gradients to')
    parser.add_argument('--batch_size', type=int, default=1, help='number of samples in a minibatch')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument(
        '--save_checkpoint_interval',
        type=int,
        default=50,
        help='save checkpoint every N batches (0 to disable periodic checkpoints)',
    )
    parser.add_argument(
        '--save_best_only',
        action='store_true',
        help='only save periodic checkpoints when loss improves',
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default=None,
        help='directory for checkpoints (default: output_dir/checkpoints)',
    )
    parser.add_argument(
        '--load_checkpoint',
        type=str,
        default=None,
        help='path to checkpoint file to resume training',
    )
    parser.add_argument(
        '--export_weights_only',
        action='store_true',
        help='also export model_weights.pt for inference-only use',
    )
    parser.add_argument(
        '--keep_last_n_checkpoints',
        type=int,
        default=3,
        help='number of recent periodic checkpoints to keep (0 to keep all)',
    )
    parser.add_argument('--num_workers', type=int, default=4, help='number of DataLoader worker processes')
    parser.add_argument('--persistent_workers', action='store_true',
                        help='keep DataLoader workers alive across epochs to reduce worker startup overhead')
    parser.add_argument('--fast_nondeterministic', action='store_true',
                        help='allow faster nondeterministic CUDA/cuDNN behavior instead of fully deterministic seeding')
    parser.add_argument(
        '--device',
        choices=('auto', 'cpu', 'cuda'),
        default='auto',
        help='training device: auto chooses CUDA when available, otherwise CPU',
    )
    parser.add_argument(
        '--max_topology_memory_gb',
        type=float,
        default=8.0,
        help='maximum estimated memory for cached mesh topology tensors; set <=0 to disable the preflight guard',
    )
    parser.add_argument(
        '--export_sdf_samples',
        action='store_true',
        help='export sdf/sdf_samples.npz for dataset-label runs',
    )
    parser.add_argument('--sdf_n_surface', type=int, default=512, help='number of exact surface samples for SDF export')
    parser.add_argument('--sdf_n_near', type=int, default=512, help='number of near-surface samples for SDF export')
    parser.add_argument('--sdf_n_uniform', type=int, default=1024, help='number of uniform volume samples for SDF export')
    parser.add_argument('--sdf_near_sigma', type=float, default=0.02, help='gaussian sigma for near-surface SDF samples')
    parser.add_argument('--tsdf_truncation', type=float, default=0.1, help='truncation distance used for TSDF export')
    parser.add_argument('--sdf_uniform_extent', type=float, default=0.5, help='uniform query extent in normalized space for SDF export')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='number of batches between training log updates')
    parser.add_argument(
        '--early_stop',
        action='store_true',
        help='enable early stopping based on smoothed loss plateau and optional theta thresholds',
    )
    parser.add_argument(
        '--early_stop_min_steps',
        type=int,
        default=1000,
        help='minimum training steps before early stopping can trigger',
    )
    parser.add_argument(
        '--early_stop_patience',
        type=int,
        default=500,
        help='number of steps without sufficient smoothed-loss improvement before stopping',
    )
    parser.add_argument(
        '--early_stop_min_delta',
        type=float,
        default=1e-3,
        help='minimum smoothed-loss improvement required to reset early-stop patience',
    )
    parser.add_argument(
        '--early_stop_smooth_window',
        type=int,
        default=50,
        help='moving-average window size used for early stopping',
    )
    parser.add_argument(
        '--early_stop_check_interval',
        type=int,
        default=10,
        help='evaluate early stopping every N training steps',
    )
    parser.add_argument(
        '--early_stop_target_loss',
        type=float,
        default=None,
        help='optional smoothed total-loss target that can trigger early stopping once minimum steps are reached',
    )
    parser.add_argument(
        '--early_stop_theta_neighbor_threshold',
        type=float,
        default=None,
        help='optional maximum unweighted theta-neighbor term required before early stopping is allowed',
    )
    parser.add_argument(
        '--early_stop_theta_hessian_threshold',
        type=float,
        default=None,
        help='optional maximum unweighted theta-hessian term required before early stopping is allowed',
    )

    # Network architecture and loss
    parser.add_argument('--init_type', type=str, default='siren',
                        help='initialization type siren | geometric_sine | geometric_relu | mfgi')
    parser.add_argument('--decoder_hidden_dim', type=int, default=256, help='length of decoder hidden dim')
    parser.add_argument('--decoder_n_hidden_layers', type=int, default=4, help='number of decoder hidden layers')
    parser.add_argument('--latent_size', type=int, default=0)
    parser.add_argument('--nl', type=str, default='sine', help='type of non linearity sine | relu | softplus')
    parser.add_argument('--sphere_init_params', nargs='+', type=float, default=[1.6, 0.1],
                        help='radius and scaling')
    parser.add_argument('--udf', action='store_true')
    parser.add_argument('--output_any', action='store_true')

    parser.add_argument('--loss_type', type=str, default='siren_wo_n_w_morse_w_theta')
    parser.add_argument('--decay_params', nargs='+', type=float, default=[3, 0.2, 3, 0.4, 0.001, 0],
                        help='epoch number to evaluate')
    parser.add_argument('--morse_type', type=str, default='l1', help='divergence term norm l1 | l2')
    parser.add_argument('--morse_decay', type=str, default='linear',
                        help='divergence term importance decay none | step | linear')
    parser.add_argument('--loss_weights', nargs='+', type=float, default=[7e3, 6e2, 10, 5e1, 30, 3],
                        help='loss terms weights sdf | inter | normal | eikonal | div | morse')
    parser.add_argument('--morse_near', action='store_true')
    parser.add_argument('--weight_for_morse', action='store_true',
                        help='if true, Weighting A according to the distance of the sampling point')
    parser.add_argument('--use_morse_nonmnfld_grad', type=bool, default=True, help='if True, use morse loss on nonmnfld')
    parser.add_argument('--relax_morse', type=float, default=0.5, help='the max value of relax Morse')
    parser.add_argument('--use_vertices', type=bool, default=False, help='if False, sample points to overfitting')
    parser.add_argument('--featureLine_threshold', type=float, default=1.0)


    return parser


def build_parser():
    parser = argparse.ArgumentParser()
    return add_args(parser)


def get_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args
