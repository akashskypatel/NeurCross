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
        default=10,
        help='number of training samples (batches) per epoch; this controls steps-per-epoch when batch_size=1',
    )
    parser.add_argument(
        '--n_points',
        type=int,
        default=15000,
        help='number of sampled points generated inside each training batch; this does not control epoch length',
    )
    parser.add_argument('--grid_res', type=int, default=256, help='uniform grid resolution')
    parser.add_argument('--nonmnfld_sample_type', type=str, default='gaussian',
                        help='how to sample points off the manifold - grid | gaussian | combined')

    # training parameters
    parser.add_argument('--num_epochs', type=int, default=1, help='always be 1')
    parser.add_argument('--lr', type=float, default=5e-5, help='initial learning rate')
    parser.add_argument('--grad_clip_norm', type=float, default=10.0, help='Value to clip gradients to')
    parser.add_argument('--batch_size', type=int, default=1, help='number of samples in a minibatch')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=4, help='number of DataLoader worker processes')
    parser.add_argument('--persistent_workers', action='store_true',
                        help='keep DataLoader workers alive across epochs to reduce worker startup overhead')
    parser.add_argument('--fast_nondeterministic', action='store_true',
                        help='allow faster nondeterministic CUDA/cuDNN behavior instead of fully deterministic seeding')
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
    parser.add_argument(
        '--convert_crossfield_to_rosy',
        action='store_true',
        help='if set, also convert each saved cross-field snapshot to a QuadWild-compatible .rosy sidecar file',
    )


    return parser


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    return args
