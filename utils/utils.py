import os
import random
import re
import sys
import warnings
from datetime import datetime

import trimesh

import numpy as np
import torch
from torch.autograd import grad
import torch.backends.cudnn as cudnn

_TRAINING_SCHEDULE_RE = re.compile(
    r"\bderived_steps_per_epoch=(?P<steps_per_epoch>\d+).*?"
    r"\bbatch_size=(?P<batch_size>\d+).*?"
    r"\btotal_steps=(?P<total_steps>\d+)"
)
_TRAINING_LOSS_RE = re.compile(
    r"^Epoch:\s+(?P<epoch>\d+)\s+"
    r"\[\s*(?P<n_sample>\d+)/\d+\s+\([^)]*\)\]\s+"
    r"Loss:\s+(?P<loss>[-+0-9.eE]+)\b"
)
_PROGRESS_DETAIL_PREFIXES = ("Weights:", "Timing:")
_PROGRESS_TERMINAL_PREFIXES = (
    "Dataset-label training failed:",
    "Early stopping triggered",
    "Training stopped early",
    "Training finished",
)


class _TrainingProgressRenderer:
    def __init__(self, stream):
        self.stream = stream
        self.steps_per_epoch = None
        self.batch_size = None
        self.total_steps = None
        self.header = None
        self.progress_line = None
        self.rendered = False
        self.last_width = 0
        try:
            is_tty = bool(stream.isatty())
        except (AttributeError, OSError):
            is_tty = False
        self.use_ansi = is_tty and os.environ.get("TERM", "") != "dumb"

    @property
    def configured(self):
        return (
            self.steps_per_epoch is not None
            and self.batch_size is not None
            and self.total_steps is not None
        )

    def configure(self, *, steps_per_epoch, batch_size, total_steps):
        self.steps_per_epoch = max(1, int(steps_per_epoch))
        self.batch_size = max(1, int(batch_size))
        self.total_steps = max(1, int(total_steps))

    def _safe_write(self, text):
        try:
            self.stream.write(text)
        except UnicodeEncodeError:
            self.stream.write(text.replace("█", "#").replace("░", "-"))
        self.stream.flush()

    def _clear_rendered_block(self):
        if not self.rendered:
            return
        if self.use_ansi:
            self._safe_write("\r\x1b[2K\x1b[1A\r\x1b[2K")
        else:
            self._safe_write("\r" + (" " * self.last_width) + "\r")
        self.rendered = False

    def _draw_saved_block(self):
        if self.header is None or self.progress_line is None:
            return
        if self.use_ansi:
            self._safe_write(self.header + "\n" + self.progress_line)
        else:
            self._safe_write(self.header + "\n\r" + self.progress_line)
        self.last_width = max(self.last_width, len(self.progress_line))
        self.rendered = True

    def update(self, *, identity, epoch, n_sample, loss):
        if not self.configured:
            return False
        batch_idx = int(n_sample) // self.batch_size
        current_step = min(
            self.total_steps,
            int(epoch) * self.steps_per_epoch + batch_idx + 1,
        )
        ratio = current_step / self.total_steps
        percent = min(100, max(1 if current_step > 0 else 0, int(ratio * 100.0 + 0.5)))
        bar_width = 52
        filled = int(ratio * bar_width + 0.5)
        if current_step > 0:
            filled = max(1, filled)
        filled = min(bar_width, filled)
        bar = "█" * filled + "░" * (bar_width - filled)
        self.header = "[{}] [{}]".format(
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            identity,
        )
        self.progress_line = (
            "[{}] {:3d}% ( {}/{}) | Epoch: {} Loss: {:.5f}".format(
                bar,
                percent,
                current_step,
                self.total_steps,
                int(epoch),
                float(loss),
            )
        )
        if self.rendered:
            self._clear_rendered_block()
        self._draw_saved_block()
        return True

    def before_message(self):
        was_rendered = self.rendered
        if was_rendered:
            self._clear_rendered_block()
        return was_rendered

    def after_message(self, should_redraw):
        if should_redraw:
            self._draw_saved_block()

    def finish(self):
        if self.rendered:
            self._safe_write("\n")
            self.rendered = False


def same_seed(seed, deterministic=True):
    """

    :param seed:
    :return:
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = not deterministic
    cudnn.deterministic = deterministic


def normalize_mesh_export(mesh, file_out=None):
    # unit to [-0.5, 0.5]
    bounds = mesh.extents.astype(np.float32)
    if bounds.min() == 0.0:
        return

    # translate to origin
    translation = (mesh.bounds[0] + mesh.bounds[1]) * 0.5
    translation = trimesh.transformations.translation_matrix(direction=-translation)
    mesh.apply_transform(translation)

    # scale to unit cube
    scale = 1.0 / bounds.max()
    scale_trafo = trimesh.transformations.scale_matrix(factor=scale)
    mesh.apply_transform(scale_trafo)
    if file_out is not None:
        mesh.export(file_out)
    return mesh


def _log_identity_from_handle(log_file):
    identity = getattr(log_file, "_neurcross_log_identity", None)
    if identity:
        return identity
    return "pid={}".format(os.getpid())


def _progress_renderer_from_handle(log_file):
    renderer = getattr(log_file, "_neurcross_progress_renderer", None)
    if renderer is None:
        renderer = _TrainingProgressRenderer(sys.stdout)
        log_file._neurcross_progress_renderer = renderer
    return renderer


def log_string(out_str, log_file):
    # helper function to log a string to file and print it
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    identity = _log_identity_from_handle(log_file)
    lines = out_str.splitlines() or ['']
    formatted = '\n'.join(f'[{timestamp}] [{identity}] {line}' for line in lines)
    log_file.write(formatted + '\n')
    log_file.flush()

    progress = _progress_renderer_from_handle(log_file)
    schedule_match = _TRAINING_SCHEDULE_RE.search(out_str)
    if schedule_match:
        progress.configure(
            steps_per_epoch=schedule_match.group("steps_per_epoch"),
            batch_size=schedule_match.group("batch_size"),
            total_steps=schedule_match.group("total_steps"),
        )

    loss_match = _TRAINING_LOSS_RE.match(out_str)
    if loss_match and progress.update(
        identity=identity,
        epoch=loss_match.group("epoch"),
        n_sample=loss_match.group("n_sample"),
        loss=loss_match.group("loss"),
    ):
        return

    if progress.configured and (
        out_str.startswith(_PROGRESS_DETAIL_PREFIXES)
        or "Unweighted L_s" in out_str
        or out_str == ''
    ):
        return

    is_terminal_message = out_str.startswith(_PROGRESS_TERMINAL_PREFIXES)
    if is_terminal_message:
        progress.finish()
        redraw_progress = False
    else:
        redraw_progress = progress.before_message()
    try:
        print(formatted)
    except (OSError, UnicodeEncodeError):
        print(formatted.encode('utf-8', errors='replace').decode('utf-8'))
    progress.after_message(redraw_progress)


def setup_out_dir_only_log(out_dir, args=None):
    # helper function to set up logging directory

    os.makedirs(out_dir, exist_ok=True)
    log_filename = os.path.join(out_dir, 'out.log')
    log_file = open(log_filename, 'w')
    identity_parts = ["pid={}".format(os.getpid())]
    device = getattr(args, "device", None) if args is not None else None
    if device:
        identity_parts.append("device={}".format(device))
    sample_id = getattr(args, "sample_id", None) if args is not None else None
    if sample_id:
        identity_parts.append("sample={}".format(sample_id))
    else:
        data_path = getattr(args, "data_path", None) if args is not None else None
        if data_path:
            identity_parts.append("mesh={}".format(os.path.basename(data_path)))
    log_file._neurcross_log_identity = " ".join(identity_parts)

    if args is not None:
        log_string("input params: \n" + str(args), log_file)
    else:
        warnings.warn("Training options not provided. Not saving training options...")

    return log_file


def gradient(inputs, outputs, create_graph=True, retain_graph=True):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=create_graph,
        retain_graph=retain_graph,
        only_inputs=True)[0]  # [:, -3:]
    return points_grad


def save_only_crossField(vector_alpha, vector_beta, batch_idx=None, output_dir=None, shapename=None):
    os.makedirs(output_dir, exist_ok=True)

    pts_vector_alpha = vector_alpha.squeeze(1).detach().cpu().numpy()
    pts_vector_beta = vector_beta.squeeze(1).detach().cpu().numpy()

    cross_field = np.concatenate((pts_vector_alpha, pts_vector_beta), axis=-1)

    cross_field_save_path = os.path.join(output_dir, shapename + '_iter_{}.vec'.format(batch_idx))
    np.savetxt(cross_field_save_path, cross_field)
    return cross_field_save_path


def count_parameters(model):
    # count the number of parameters in a given model
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_same_neighbors_verts(vertex_neighbors):
    lens = list(map(len, vertex_neighbors))
    lens = np.asarray(lens)
    lens_unique, lens_inverse = np.unique(lens, return_inverse=True)

    vertex_neighbors_list = []
    for i in range(lens_unique.shape[0]):
        vertex_neighbors_list.append(np.argwhere(lens_inverse == i).squeeze(-1).tolist())

    return vertex_neighbors_list



def get_sample_vers_neighbors_for_face_center_points_or_vertices(mesh_path):
    # the face adjacency, shape=(len(faces), 3)
    mesh = trimesh.load_mesh(mesh_path, process=False)

    face_adj = mesh.face_adjacency.astype(np.int32)
    vertex_neighbors = [[] for _ in range(len(mesh.faces))]
    for face_a, face_b in face_adj:
        vertex_neighbors[int(face_a)].append(int(face_b))
        vertex_neighbors[int(face_b)].append(int(face_a))

    return vertex_neighbors


def transform_vectors_only_rotation(verct, trans):

    assert verct.shape[1] == trans.shape[1], "Error!"

    verct_flat = verct.view(-1, 3)  # reshape to [bs * num, 3]

    trans_flat = trans.view(-1, 3, 3)

    transformed_flat = torch.bmm(trans_flat, verct_flat.unsqueeze(2))  # [bs * num, 3, 1]

    transformed_vectors = transformed_flat.view(verct.shape[0], verct.shape[1], 1, 3)

    return transformed_vectors


def original_the_edge_information_to_face_neighbor_list(face_adjacency, edge_info, neighbors_each_face, face_indices=None):
    info_map = {
        (min(f1, f2), max(f1, f2)): angle
        for (f1, f2), angle in zip(face_adjacency, edge_info)
    }

    if face_indices is None:
        face_indices = range(len(neighbors_each_face))

    R = [[info_map[(min(int(f), int(n)), max(int(f), int(n)))] for n in neighbors]
         for f, neighbors in zip(face_indices, neighbors_each_face)]
    R = np.array(R)

    return R


def batch_axis_angle_to_rotation_matrix_only_rotation(axes, thetas):
    axes = axes / np.linalg.norm(axes, axis=-1, keepdims=True)


    vx = axes[..., 0]
    vy = axes[..., 1]
    vz = axes[..., 2]


    cos_theta = np.cos(thetas)
    sin_theta = np.sin(thetas)
    one_minus_cos = 1 - cos_theta


    K = np.zeros((axes.shape[0], axes.shape[1], 3, 3), dtype=np.float32)
    K[..., 0, 1] = -vz
    K[..., 0, 2] = vy
    K[..., 1, 0] = vz
    K[..., 1, 2] = -vx
    K[..., 2, 0] = -vy
    K[..., 2, 1] = vx

    #
    I = np.eye(3, dtype=np.float32)[np.newaxis, np.newaxis, :, :]  #
    R = I + sin_theta[..., np.newaxis, np.newaxis] * K + one_minus_cos[..., np.newaxis, np.newaxis] * np.einsum(
        '...ij,...jk->...ik', K, K)

    return R

def get_rotation_matrix(vertex_neighbors_list, vertex_neighbors, mesh_path):
    mesh = trimesh.load_mesh(mesh_path, process=False)
    face_normals = mesh.face_normals.astype(np.float32)
    face_adjacency = mesh.face_adjacency.astype(np.int32)
    face_adjacency_angles = mesh.face_adjacency_angles.astype(np.float32)
    face_adjacency_edges = mesh.face_adjacency_edges.astype(np.int32)
    verts = mesh.vertices.astype(np.float32)
    rota_axis = verts[face_adjacency_edges[:, 0]] - verts[face_adjacency_edges[:, 1]]

    axis_angle_R_mat_list = list()

    for i in range(len(vertex_neighbors_list)):
        idx = np.asarray(vertex_neighbors_list[i], dtype=np.int32)

        face_normals_i = np.expand_dims(face_normals[idx], axis=1)  # n x 1 x 3

        vertex_neighbors_i = [vertex_neighbors[z] for z in idx]
        neighbor_count = len(vertex_neighbors_i[0]) if vertex_neighbors_i else 0
        if neighbor_count == 0:
            axis_angle_R_mat_list.append(np.zeros((idx.shape[0], 0, 3, 3), dtype=np.float32))
            continue

        vertex_neighbors_i = np.asarray(vertex_neighbors_i, dtype=np.int32)  # n x neighbors_size
        face_normals_i_neighbor = face_normals[vertex_neighbors_i]
        #
        desired_rota_axis_direction = np.cross(face_normals_i_neighbor, face_normals_i)

        # map the rota_axis on the edge to the face
        rota_axis_map_to_face = original_the_edge_information_to_face_neighbor_list(face_adjacency, rota_axis,
                                                                                    vertex_neighbors_i,
                                                                                    idx)
        desired_axis_dot_axis = np.einsum('ijk,ijk->ij', desired_rota_axis_direction, rota_axis_map_to_face)
        flag = (desired_axis_dot_axis < 0)
        rota_axis_map_to_face[flag] = -rota_axis_map_to_face[flag]

        # map the dihedral angles on the edge to the face
        dihedral_angle_map_to_face = original_the_edge_information_to_face_neighbor_list(face_adjacency,
                                                                                         face_adjacency_angles,
                                                                                         vertex_neighbors_i,
                                                                                         idx)

        axis_angle_R_mat = batch_axis_angle_to_rotation_matrix_only_rotation(rota_axis_map_to_face,
                                                                             dihedral_angle_map_to_face)

        axis_angle_R_mat_list.append(axis_angle_R_mat)

    return axis_angle_R_mat_list
