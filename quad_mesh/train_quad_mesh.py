import os
import time
import warnings
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


def main():
    # get training parameters
    args = quad_mesh_args.get_args()
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

    mesh_dir = os.path.dirname(os.path.abspath(args.data_path))
    file_name = os.path.splitext(os.path.basename(args.data_path))[0]
    if args.out_dir is None:
        out_dir_root = mesh_dir
    elif os.path.isabs(args.out_dir):
        out_dir_root = args.out_dir
    else:
        out_dir_root = os.path.join(mesh_dir, args.out_dir)

    out_dir = os.path.join(out_dir_root, file_name)
    os.makedirs(out_dir, exist_ok=True)
    run_start_time = time.perf_counter()

    # set up logging
    log_file = utils.setup_out_dir_only_log(out_dir, args)

    device = 'cpu' if not torch.cuda.is_available() else 'cuda'
    if device == 'cuda':
        torch.cuda.set_device(0)
        torch.cuda.init()

    # get data loaders
    utils.same_seed(args.seed, deterministic=not args.fast_nondeterministic)
    train_set = dataset.ReconDataset(args.data_path, args.n_points, args.n_samples, args.grid_res)
    static_batch = train_set.get_static_batch()

    train_dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == 'cuda'),
        persistent_workers=(args.persistent_workers and args.num_workers > 0),
    )
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * args.num_epochs
    utils.log_string(
        "Training schedule: n_samples_per_epoch={} n_points_per_batch={} batch_size={} "
        "steps_per_epoch={} num_epochs={} total_steps={}".format(
            args.n_samples,
            args.n_points,
            args.batch_size,
            steps_per_epoch,
            args.num_epochs,
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
    print('steps_per_epoch: ', steps_per_epoch)
    print('n_iterations: ', total_steps)

    net.to(device)

    SAVE_BEST = False

    ##################################################################################
    # get the vertices neighbors of the mesh
    vertex_neighbors = utils.get_sample_vers_neighbors_for_face_center_points_or_vertices(args.data_path)
    vertex_neighbors_list = utils.calculate_same_neighbors_verts(vertex_neighbors)
    ###################################################################################
    axis_angle_R_mat_list = utils.get_rotation_matrix(vertex_neighbors_list, vertex_neighbors, args.data_path)

    criterion = MorseLoss(weights=args.loss_weights, loss_type=args.loss_type, div_decay=args.morse_decay,
                          div_type=args.morse_type,
                          vertex_neighbors_list=vertex_neighbors_list,
                          vertex_neighbors=vertex_neighbors, axis_angle_R_mat_list=axis_angle_R_mat_list,
                          device=device, convert_crossfield_to_rosy=args.convert_crossfield_to_rosy
                          )
    from models.loss_quad_mesh import export_crossfield_snapshot

    early_stopper = None
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

    net.train()
    stopped_early = False
    stop_summary = None
    # For each epoch
    for epoch in range(args.num_epochs):
        for batch_idx, data in enumerate(train_dataloader):
            batch_start_time = time.perf_counter()
            global_step = epoch * len(train_dataloader) + batch_idx
            if batch_idx != 0 and (batch_idx % 500 == 0 or batch_idx == steps_per_epoch - 1):
                SAVE_BEST = True
            is_final_batch = epoch == args.num_epochs - 1 and batch_idx == steps_per_epoch - 1

            optimizer.zero_grad(set_to_none=True)

            mnfld_points = mnfld_points_base.detach().clone()
            nonmnfld_points = data['nonmnfld_points'].to(device, non_blocking=non_blocking)
            near_points = data['near_points'].to(device, non_blocking=non_blocking)
            mnfld_points.requires_grad_()
            nonmnfld_points.requires_grad_()
            near_points.requires_grad_()

            features = torch.cat((mnfld_points, angle_feature_static), dim=-1)

            output_pred, mnfld_pts_theta_output_pred = net(nonmnfld_points, mnfld_points,
                                                           near_points=near_points if args.morse_near else None,
                                                           angle_features=features)

            loss_dict = criterion(output_pred, mnfld_points, nonmnfld_points, mnfld_n_gt,
                                  near_points=near_points if args.morse_near else None, batch_idx=global_step,
                                  out_dir=out_dir, filename=file_name, save_best=SAVE_BEST,
                                  mnfld_pts_theta_output_pred=mnfld_pts_theta_output_pred,
                                  local_coord_u=local_coord_u, local_coord_v=local_coord_v,
                                  is_final_export=is_final_batch)

            lr = optimizer.param_groups[0]['lr']

            loss_dict["loss"].backward()

            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip_norm)

            SAVE_BEST = False
            optimizer.step()
            batch_elapsed = time.perf_counter() - batch_start_time

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
                            convert_crossfield_to_rosy=args.convert_crossfield_to_rosy,
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
                    stopped_early = True
                    break

            criterion.update_morse_weight(global_step, total_steps,
                                          args.decay_params)  # assumes batch size of 1
        if stopped_early:
            break

    total_elapsed = time.perf_counter() - run_start_time
    if stopped_early:
        utils.log_string("Training stopped early in {}".format(_format_duration(total_elapsed)), log_file)
    else:
        utils.log_string("Training finished in {}".format(_format_duration(total_elapsed)), log_file)
    log_file.close()


if __name__ == '__main__':
    freeze_support()
    main()
