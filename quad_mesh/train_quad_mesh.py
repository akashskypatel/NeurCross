import os
import warnings
from multiprocessing import freeze_support

from . import quad_mesh_args

warnings.filterwarnings(
    'ignore',
    message=r'Attempting to run cuBLAS, but there was no current CUDA context!',
    category=UserWarning,
)


def _load_runtime_dependencies():
    import numpy as np
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
        'np': np,
        'torch': torch,
        'optim': optim,
        'summary': summary,
        'Network_predict_angle': Network_predict_angle,
        'MorseLoss': MorseLoss,
        'utils': utils,
        'dataset': dataset,
    }


def main():
    # get training parameters
    args = quad_mesh_args.get_args()
    if not args.data_path:
        raise ValueError('No default training mesh is bundled in the wheel build. Pass --data_path to a mesh file.')

    deps = _load_runtime_dependencies()
    np = deps['np']
    torch = deps['torch']
    optim = deps['optim']
    summary = deps['summary']
    Network_predict_angle = deps['Network_predict_angle']
    MorseLoss = deps['MorseLoss']
    utils = deps['utils']
    dataset = deps['dataset']

    mesh_dir = os.path.dirname(os.path.abspath(args.data_path))
    file_name = os.path.splitext(os.path.basename(args.data_path))[0]
    if args.logdir is None:
        logdir_root = mesh_dir
    elif os.path.isabs(args.logdir):
        logdir_root = args.logdir
    else:
        logdir_root = os.path.join(mesh_dir, args.logdir)

    logdir = os.path.join(logdir_root, file_name)
    os.makedirs(logdir, exist_ok=True)

    # set up logging
    log_file = utils.setup_logdir_only_log(logdir, args)

    device = 'cpu' if not torch.cuda.is_available() else 'cuda'
    if device == 'cuda':
        torch.cuda.set_device(0)
        torch.cuda.init()

    # get data loaders
    utils.same_seed(args.seed)
    train_set = dataset.ReconDataset(args.data_path, args.n_points, args.n_samples, args.grid_res)

    train_dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device == 'cuda'),
    )
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
    n_iterations = args.n_samples * (args.num_epochs)
    print('n_iterations: ', n_iterations)

    net.to(device)

    num_batches = len(train_dataloader)
    refine_flag = True
    min_cd = np.inf
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

    # For each epoch
    for epoch in range(args.num_epochs):
        for batch_idx, data in enumerate(train_dataloader):
            if batch_idx != 0 and (batch_idx % 500 == 0 or batch_idx == len(train_dataloader) - 1):
                SAVE_BEST = True

            net.zero_grad()
            net.train()

            mnfld_points, mnfld_n_gt, nonmnfld_points, near_points, local_coord_u, local_coord_v = data[
                'points'].to(device), data['mnfld_n'].to(device), data['nonmnfld_points'].to(device), data[
                'near_points'].to(device), data['local_coordinates_u'].to(device), data['local_coordinates_v'].to(device)

            mnfld_points.requires_grad_()
            nonmnfld_points.requires_grad_()
            near_points.requires_grad_()

            features = torch.cat((mnfld_points, mnfld_n_gt, local_coord_u, local_coord_v), dim=-1)

            output_pred, mnfld_pts_theta_output_pred = net(nonmnfld_points, mnfld_points,
                                                           near_points=near_points if args.morse_near else None,
                                                           angle_features=features)

            loss_dict = criterion(output_pred, mnfld_points, nonmnfld_points, mnfld_n_gt,
                                  near_points=near_points if args.morse_near else None, batch_idx=batch_idx,
                                  logdir=logdir, filename=file_name, save_best=SAVE_BEST,
                                  mnfld_pts_theta_output_pred=mnfld_pts_theta_output_pred,
                                  local_coord_u=local_coord_u, local_coord_v=local_coord_v)

            lr = torch.tensor(optimizer.param_groups[0]['lr'])
            loss_dict["lr"] = lr

            loss_dict["loss"].backward()

            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip_norm)

            SAVE_BEST = False
            optimizer.step()

            # Output training stats
            if batch_idx % 10 == 0:
                weights = criterion.weights
                utils.log_string("Weights: {}, lr={:.3e}".format(weights, lr), log_file)
                utils.log_string('Epoch: {} [{:4d}/{} ({:.0f}%)] Loss: {:.5f} = L_Mnfld: {:.5f} + '
                                 'L_NonMnfld: {:.5f} + L_Eknl: {:.5f} + L_Morse: {:.5f} + L_thetaHessian: {:.5f} + L_thetaNeighbor: {:.5f}'.format(
                    epoch, batch_idx * args.batch_size, len(train_set), 100. * batch_idx / args.n_samples,
                    loss_dict["loss"].item(), weights[0] * loss_dict["sdf_term"].item(),
                           weights[1] * loss_dict["inter_term"].item(),
                           weights[3] * loss_dict["eikonal_term"].item(), weights[5] * loss_dict["morse_term"].item(),
                           weights[2] * loss_dict["theta_hessian_term"].item(),
                           weights[4] * loss_dict['theta_neighbors_term'].item()
                ),
                    log_file)
                utils.log_string('Epoch: {} [{:4d}/{} ({:.0f}%)] Unweighted L_s : L_Mnfld: {:.5f} + '
                                 'L_NonMnfld: {:.5f} + L_Eknl: {:.5f} + L_Morse: {:.5f} + L_thetaHessian: {:.5f} + L_thetaNeighbor: {:.5f}'.format(
                    epoch, batch_idx * args.batch_size, len(train_set), 100. * batch_idx / args.n_samples,
                    loss_dict["sdf_term"].item(), loss_dict["inter_term"].item(),
                    loss_dict["eikonal_term"].item(), loss_dict["morse_term"].item(),
                    loss_dict['theta_hessian_term'].item(), loss_dict['theta_neighbors_term'].item()),
                    log_file)
                utils.log_string('', log_file)

            criterion.update_morse_weight(epoch * args.n_samples + batch_idx, args.num_epochs * args.n_samples,
                                          args.decay_params)  # assumes batch size of 1


if __name__ == '__main__':
    freeze_support()
    main()
