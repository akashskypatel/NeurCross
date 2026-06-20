import json
import os
import shutil
import math
import torch
import torch.nn as nn

import utils.utils as utils


def _detached_cpu_tensor(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu")
    return value


def _cpu_chunk_tensor(value, start: int, end: int):
    if isinstance(value, torch.Tensor):
        return value[start:end].detach().to(device="cpu")
    return value[start:end]


class CrossFieldExportManager:
    def __init__(self, out_dir, filename):
        self.output_dir = os.path.join(out_dir, 'save_crossField')
        self.metrics_dir = os.path.join(out_dir, 'metrics')
        self.filename = filename
        self.best_score = float('inf')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

    def _sidecar_path(self, suffix):
        return os.path.join(self.output_dir, f'{self.filename}_{suffix}.vec')

    def _meta_path(self, suffix):
        return os.path.join(self.output_dir, f'{self.filename}_{suffix}.meta.txt')

    def _metrics_path(self, suffix):
        return os.path.join(self.metrics_dir, f'{self.filename}_{suffix}.json')

    def _history_metrics_path(self, step):
        return os.path.join(self.metrics_dir, f'{self.filename}_iter_{step}.json')

    def _copy_snapshot(self, source_path, target_path):
        shutil.copyfile(source_path, target_path)

    def _write_meta(self, suffix, *, step, total_loss, field_score):
        with open(self._meta_path(suffix), 'w', encoding='utf-8', newline='\n') as handle:
            handle.write(f"step={step}\n")
            handle.write(f"total_loss={total_loss:.10f}\n")
            handle.write(f"field_score={field_score:.10f}\n")

    def _write_metrics_json(self, path, metrics):
        with open(path, 'w', encoding='utf-8', newline='\n') as handle:
            json.dump(metrics, handle, indent=2, sort_keys=True)
            handle.write('\n')

    def export(
        self,
        vector_alpha,
        vector_beta,
        *,
        step,
        total_loss,
        field_score,
        metrics,
        is_final=False,
    ):
        report = dict(metrics)
        report["step"] = int(step)
        report["total_loss"] = float(total_loss)
        report["field_score"] = float(field_score)
        report["history_crossfield_path"] = None
        report["latest_crossfield_path"] = self._sidecar_path('latest')
        report["best_crossfield_path"] = self._sidecar_path('best')
        report["final_crossfield_path"] = self._sidecar_path('final')

        history_path = utils.save_only_crossField(
            vector_alpha.unsqueeze(1),
            vector_beta.unsqueeze(1),
            batch_idx=step,
            output_dir=self.output_dir,
            shapename=self.filename,
        )
        report["history_crossfield_path"] = history_path

        latest_path = self._sidecar_path('latest')
        self._copy_snapshot(history_path, latest_path)
        self._write_meta('latest', step=step, total_loss=total_loss, field_score=field_score)
        self._write_metrics_json(self._history_metrics_path(step), report)
        self._write_metrics_json(self._metrics_path('latest'), report)

        if field_score < self.best_score:
            self.best_score = field_score
            best_path = self._sidecar_path('best')
            self._copy_snapshot(history_path, best_path)
            self._write_meta('best', step=step, total_loss=total_loss, field_score=field_score)
            self._write_metrics_json(self._metrics_path('best'), report)

        if is_final:
            final_path = self._sidecar_path('final')
            self._copy_snapshot(history_path, final_path)
            self._write_meta('final', step=step, total_loss=total_loss, field_score=field_score)
            self._write_metrics_json(self._metrics_path('final'), report)

        return history_path


def export_crossfield_snapshot(
    vector_alpha,
    vector_beta,
    out_dir,
    filename,
    batch_idx,
    manager=None,
    total_loss=None,
    field_score=None,
    metrics=None,
    is_final=False,
):
    if manager is None:
        manager = CrossFieldExportManager(out_dir, filename)
    if total_loss is None:
        total_loss = 0.0
    if field_score is None:
        field_score = total_loss
    if metrics is None:
        metrics = {}
    return manager.export(
        vector_alpha,
        vector_beta,
        step=batch_idx,
        total_loss=float(total_loss),
        field_score=float(field_score),
        metrics=metrics,
        is_final=is_final,
    )


def eikonal_loss(nonmnfld_grad, mnfld_grad, eikonal_type='abs'):
    # Compute the eikonal loss that penalises when ||grad(f)|| != 1 for points on and off the manifold
    # shape is (bs, num_points, dim=3) for both grads
    # Eikonal
    if nonmnfld_grad is not None and mnfld_grad is not None:
        all_grads = torch.cat([nonmnfld_grad, mnfld_grad], dim=-2)
    elif nonmnfld_grad is not None:
        all_grads = nonmnfld_grad
    elif mnfld_grad is not None:
        all_grads = mnfld_grad

    if eikonal_type == 'abs':
        eikonal_term = ((all_grads.norm(2, dim=2) - 1).abs()).mean()
    else:
        eikonal_term = ((all_grads.norm(2, dim=2) - 1).square()).mean()
    # eikonal_term = (-torch.log(all_grads.norm(2, dim=2))).mean()
    return eikonal_term

class MorseLoss_quad_mesh(nn.Module):
    def __init__(self, weights=None, loss_type='siren_wo_n_w_morse', div_decay='none',
                 div_type='l1', vertex_neighbors_list=None,
                 vertex_neighbors=None, axis_angle_R_mat_list=None, device=None,
                 max_topology_memory_gb=8.0):
        super().__init__()
        if weights is None:
            weights = [7e3, 6e2, 10, 5e1, 30, 3]
        self.weights = weights  # sdf, intern, normal, eikonal, div
        self.loss_type = loss_type
        self.div_decay = div_decay
        self.div_type = div_type
        self.use_morse = True if 'morse' in self.loss_type else False
        self.device = device
        self.max_topology_memory_gb = float(max_topology_memory_gb)
        self._export_manager = None

        # Cache padded topology tensors once so theta-loss evaluation can run
        # in a single batched pass instead of looping over groups in Python.
        self._group_face_indices = None
        self._group_face_mask = None
        self._group_neighbor_indices = None
        self._group_neighbor_mask = None
        self._group_rotation_mats = None
        self._num_vert_neigh = 0
        if vertex_neighbors_list is not None and vertex_neighbors is not None and device is not None:
            num_groups = len(vertex_neighbors_list)
            max_faces = max(len(group) for group in vertex_neighbors_list)
            max_neighbors = max(len(vertex_neighbors[group[0]]) for group in vertex_neighbors_list)
            padded_entries = num_groups * max_faces * max_neighbors
            estimated_bytes = (
                num_groups * max_faces * 8
                + num_groups * max_faces
                + padded_entries * 8
                + padded_entries
            )
            if axis_angle_R_mat_list is not None:
                estimated_bytes += padded_entries * 9 * 4
            estimated_gb = estimated_bytes / (1024 ** 3)
            if self.max_topology_memory_gb > 0 and estimated_gb > self.max_topology_memory_gb:
                raise MemoryError(
                    "Estimated cached topology tensor memory is {:.2f} GiB, above "
                    "--max_topology_memory_gb={:.2f}. Use --device cpu, simplify the mesh, "
                    "or raise --max_topology_memory_gb if this allocation is intentional.".format(
                        estimated_gb,
                        self.max_topology_memory_gb,
                    )
                )

            face_indices = torch.zeros((num_groups, max_faces), dtype=torch.long, device=device)
            face_mask = torch.zeros((num_groups, max_faces), dtype=torch.bool, device=device)
            neighbor_indices = torch.zeros((num_groups, max_faces, max_neighbors), dtype=torch.long, device=device)
            neighbor_mask = torch.zeros((num_groups, max_faces, max_neighbors), dtype=torch.bool, device=device)
            rotation_mats = None
            if axis_angle_R_mat_list is not None:
                rotation_mats = torch.eye(3, dtype=torch.float32, device=device).view(1, 1, 1, 3, 3).repeat(
                    num_groups, max_faces, max_neighbors, 1, 1
                )

            for group_idx, group in enumerate(vertex_neighbors_list):
                group_faces = torch.as_tensor(group, dtype=torch.long, device=device)
                group_face_count = group_faces.shape[0]
                group_neighbor_count = len(vertex_neighbors[group[0]])

                face_indices[group_idx, :group_face_count] = group_faces
                face_mask[group_idx, :group_face_count] = True

                neighbors = torch.as_tensor(
                    [vertex_neighbors[z] for z in group],
                    dtype=torch.long,
                    device=device,
                )
                neighbor_indices[group_idx, :group_face_count, :group_neighbor_count] = neighbors
                neighbor_mask[group_idx, :group_face_count, :group_neighbor_count] = True

                if rotation_mats is not None:
                    group_rotations = torch.as_tensor(axis_angle_R_mat_list[group_idx], dtype=torch.float32, device=device)
                    rotation_mats[group_idx, :group_face_count, :group_neighbor_count] = group_rotations

            self._group_face_indices = face_indices
            self._group_face_mask = face_mask
            self._group_neighbor_indices = neighbor_indices
            self._group_neighbor_mask = neighbor_mask
            self._group_rotation_mats = rotation_mats
            self._num_vert_neigh = num_groups

    def _build_metrics(self, loss, sdf_term, inter_term, eikonal_term, morse_loss,
                       theta_hessian_term, theta_neighbors_term, vector_alpha, vector_beta,
                       mnfld_n_gt, neighbors_term, neighbor_mask):
        metrics_device = vector_alpha.device
        normals = mnfld_n_gt.squeeze(0)
        alpha_norm = vector_alpha.norm(dim=-1)
        beta_norm = vector_beta.norm(dim=-1)
        alpha_unit = vector_alpha / (alpha_norm.unsqueeze(-1) + 1e-12)
        beta_unit = vector_beta / (beta_norm.unsqueeze(-1) + 1e-12)

        alpha_beta_dot = (alpha_unit * beta_unit).sum(dim=-1).abs()
        alpha_tangent = (alpha_unit * normals).sum(dim=-1).abs()
        beta_tangent = (beta_unit * normals).sum(dim=-1).abs()
        handedness = torch.linalg.cross(alpha_unit, beta_unit).mul(normals).sum(dim=-1)

        valid_neighbor_errors = neighbors_term[neighbor_mask]
        if valid_neighbor_errors.numel() > 0:
            smooth_mean = valid_neighbor_errors.mean().item()
            smooth_median = valid_neighbor_errors.median().item()
            smooth_p95 = torch.quantile(valid_neighbor_errors, 0.95).item()
            smooth_max = valid_neighbor_errors.max().item()
        else:
            smooth_mean = smooth_median = smooth_p95 = smooth_max = 0.0

        face_indices = self._group_face_indices.to(metrics_device)
        face_mask = self._group_face_mask.to(metrics_device)
        expanded_neighbor_errors = neighbors_term * neighbor_mask.to(neighbors_term.dtype)
        face_error_count = neighbor_mask.to(expanded_neighbor_errors.dtype).sum(dim=2).clamp_min(1.0)
        face_avg_error_group = expanded_neighbor_errors.sum(dim=2) / face_error_count
        vertex_proxy_badness = (
            (face_avg_error_group * face_mask.to(face_avg_error_group.dtype)).sum(dim=1)
            / face_mask.to(face_avg_error_group.dtype).sum(dim=1).clamp_min(1.0)
        )
        singularity_threshold = 0.25
        singularity_proxy = vertex_proxy_badness > singularity_threshold

        nan_count = int(
            torch.isnan(vector_alpha).sum().item()
            + torch.isnan(vector_beta).sum().item()
            + torch.isinf(vector_alpha).sum().item()
            + torch.isinf(vector_beta).sum().item()
        )

        training_metrics = {
            "loss_total": float(loss.detach().cpu().item()),
            "loss_mnfld": float(sdf_term.detach().cpu().item()),
            "loss_nonmnfld": float(inter_term.detach().cpu().item()),
            "loss_eikonal": float(eikonal_term.detach().cpu().item()),
            "loss_morse": float(morse_loss.detach().cpu().item()),
            "loss_theta_hessian": float(theta_hessian_term.detach().cpu().item()),
            "loss_theta_neighbor": float(theta_neighbors_term.detach().cpu().item()),
        }
        field_validity = {
            "num_faces": int(vector_alpha.shape[0]),
            "nan_count": nan_count,
            "alpha_norm_error_mean": float((alpha_norm - 1.0).abs().mean().item()),
            "alpha_norm_error_max": float((alpha_norm - 1.0).abs().max().item()),
            "beta_norm_error_mean": float((beta_norm - 1.0).abs().mean().item()),
            "beta_norm_error_max": float((beta_norm - 1.0).abs().max().item()),
            "alpha_beta_dot_mean": float(alpha_beta_dot.mean().item()),
            "alpha_beta_dot_max": float(alpha_beta_dot.max().item()),
            "alpha_tangent_error_mean": float(alpha_tangent.mean().item()),
            "alpha_tangent_error_max": float(alpha_tangent.max().item()),
            "beta_tangent_error_mean": float(beta_tangent.mean().item()),
            "beta_tangent_error_max": float(beta_tangent.max().item()),
            "handedness_mean": float(handedness.mean().item()),
            "handedness_min": float(handedness.min().item()),
            "flipped_frame_ratio": float((handedness < 0.0).to(torch.float32).mean().item()),
        }
        field_smoothness = {
            "adjacent_cross_error_mean": float(smooth_mean),
            "adjacent_cross_error_median": float(smooth_median),
            "adjacent_cross_error_p95": float(smooth_p95),
            "adjacent_cross_error_max": float(smooth_max),
        }
        singularity_metrics = {
            "singularity_proxy_threshold": singularity_threshold,
            "singularity_proxy_count": int(singularity_proxy.sum().item()),
            "singularity_proxy_ratio": float(singularity_proxy.to(torch.float32).mean().item()),
            "singularity_proxy_badness_mean": float(vertex_proxy_badness.mean().item()),
            "singularity_proxy_badness_p95": float(torch.quantile(vertex_proxy_badness, 0.95).item()),
            "singularity_proxy_badness_max": float(vertex_proxy_badness.max().item()),
        }
        score = (
            training_metrics["loss_theta_hessian"]
            + training_metrics["loss_theta_neighbor"]
            + 10.0 * field_smoothness["adjacent_cross_error_mean"]
            + 50.0 * field_validity["flipped_frame_ratio"]
            + 10.0 * field_validity["alpha_tangent_error_mean"]
            + 10.0 * field_validity["beta_tangent_error_mean"]
        )
        return {
            "training": training_metrics,
            "field_validity": field_validity,
            "field_smoothness": field_smoothness,
            "singularity_proxy": singularity_metrics,
            "score": float(score),
        }

    def _build_metrics_chunked_cpu(
        self,
        loss,
        sdf_term,
        inter_term,
        eikonal_term,
        morse_loss,
        theta_hessian_term,
        theta_neighbors_term,
        vector_alpha,
        vector_beta,
        mnfld_n_gt,
        neighbors_term,
        neighbor_mask,
        *,
        face_chunk_size: int = 8192,
        group_chunk_size: int = 256,
        log_label: str | None = None,
    ):
        num_faces = int(vector_alpha.shape[0])
        num_groups = int(self._group_face_mask.shape[0])
        max_neighbors = int(neighbor_mask.shape[-1])
        if log_label:
            print(
                f"[metrics-cpu] start {log_label} faces={num_faces} groups={num_groups} max_neighbors={max_neighbors} "
                f"face_chunk_size={face_chunk_size} group_chunk_size={group_chunk_size}",
                flush=True,
            )

        alpha_norm_err_sum = 0.0
        beta_norm_err_sum = 0.0
        alpha_beta_dot_sum = 0.0
        alpha_tangent_sum = 0.0
        beta_tangent_sum = 0.0
        handedness_sum = 0.0
        flipped_count = 0.0
        alpha_norm_err_max = 0.0
        beta_norm_err_max = 0.0
        alpha_beta_dot_max = 0.0
        alpha_tangent_max = 0.0
        beta_tangent_max = 0.0
        handedness_min = float("inf")
        nan_count = 0

        normals_all = mnfld_n_gt.squeeze(0)
        for start in range(0, num_faces, face_chunk_size):
            end = min(start + face_chunk_size, num_faces)
            alpha_chunk = _cpu_chunk_tensor(vector_alpha, start, end)
            beta_chunk = _cpu_chunk_tensor(vector_beta, start, end)
            normals_chunk = _cpu_chunk_tensor(normals_all, start, end)

            alpha_norm = alpha_chunk.norm(dim=-1)
            beta_norm = beta_chunk.norm(dim=-1)
            alpha_unit = alpha_chunk / (alpha_norm.unsqueeze(-1) + 1e-12)
            beta_unit = beta_chunk / (beta_norm.unsqueeze(-1) + 1e-12)

            alpha_beta_dot = (alpha_unit * beta_unit).sum(dim=-1).abs()
            alpha_tangent = (alpha_unit * normals_chunk).sum(dim=-1).abs()
            beta_tangent = (beta_unit * normals_chunk).sum(dim=-1).abs()
            handedness = torch.linalg.cross(alpha_unit, beta_unit).mul(normals_chunk).sum(dim=-1)

            alpha_norm_err = (alpha_norm - 1.0).abs()
            beta_norm_err = (beta_norm - 1.0).abs()

            alpha_norm_err_sum += float(alpha_norm_err.sum().item())
            beta_norm_err_sum += float(beta_norm_err.sum().item())
            alpha_beta_dot_sum += float(alpha_beta_dot.sum().item())
            alpha_tangent_sum += float(alpha_tangent.sum().item())
            beta_tangent_sum += float(beta_tangent.sum().item())
            handedness_sum += float(handedness.sum().item())
            flipped_count += float((handedness < 0.0).to(torch.float32).sum().item())

            alpha_norm_err_max = max(alpha_norm_err_max, float(alpha_norm_err.max().item()))
            beta_norm_err_max = max(beta_norm_err_max, float(beta_norm_err.max().item()))
            alpha_beta_dot_max = max(alpha_beta_dot_max, float(alpha_beta_dot.max().item()))
            alpha_tangent_max = max(alpha_tangent_max, float(alpha_tangent.max().item()))
            beta_tangent_max = max(beta_tangent_max, float(beta_tangent.max().item()))
            handedness_min = min(handedness_min, float(handedness.min().item()))

            nan_count += int(
                torch.isnan(alpha_chunk).sum().item()
                + torch.isnan(beta_chunk).sum().item()
                + torch.isinf(alpha_chunk).sum().item()
                + torch.isinf(beta_chunk).sum().item()
            )

        valid_neighbor_chunks = []
        singularity_proxy_count = 0
        singularity_proxy_badness_sum = 0.0
        singularity_proxy_badness_max = 0.0
        vertex_proxy_badness_chunks = []
        singularity_threshold = 0.25

        for start in range(0, num_groups, group_chunk_size):
            end = min(start + group_chunk_size, num_groups)
            neighbors_chunk = _cpu_chunk_tensor(neighbors_term, start, end)
            neighbor_mask_chunk = _cpu_chunk_tensor(neighbor_mask, start, end)
            valid_chunk = neighbors_chunk[neighbor_mask_chunk]
            if valid_chunk.numel() > 0:
                valid_neighbor_chunks.append(valid_chunk.reshape(-1))

            chunk_face_mask = self._group_face_mask[start:end].detach().to(device="cpu")
            expanded_neighbor_errors = neighbors_chunk * neighbor_mask_chunk.to(neighbors_chunk.dtype)
            face_error_count = neighbor_mask_chunk.to(expanded_neighbor_errors.dtype).sum(dim=2).clamp_min(1.0)
            face_avg_error_group = expanded_neighbor_errors.sum(dim=2) / face_error_count
            face_mask_float = chunk_face_mask.to(face_avg_error_group.dtype)
            vertex_proxy_badness_chunk = (
                (face_avg_error_group * face_mask_float).sum(dim=1)
                / face_mask_float.sum(dim=1).clamp_min(1.0)
            )
            singularity_proxy_chunk = vertex_proxy_badness_chunk > singularity_threshold
            singularity_proxy_count += int(singularity_proxy_chunk.sum().item())
            singularity_proxy_badness_sum += float(vertex_proxy_badness_chunk.sum().item())
            singularity_proxy_badness_max = max(
                singularity_proxy_badness_max,
                float(vertex_proxy_badness_chunk.max().item()) if vertex_proxy_badness_chunk.numel() else 0.0,
            )
            if vertex_proxy_badness_chunk.numel() > 0:
                vertex_proxy_badness_chunks.append(vertex_proxy_badness_chunk.reshape(-1))

        if valid_neighbor_chunks:
            valid_neighbor_errors = torch.cat(valid_neighbor_chunks, dim=0)
            smooth_mean = float(valid_neighbor_errors.mean().item())
            smooth_median = float(valid_neighbor_errors.median().item())
            smooth_p95 = float(torch.quantile(valid_neighbor_errors, 0.95).item())
            smooth_max = float(valid_neighbor_errors.max().item())
        else:
            smooth_mean = smooth_median = smooth_p95 = smooth_max = 0.0

        if vertex_proxy_badness_chunks:
            vertex_proxy_badness_all = torch.cat(vertex_proxy_badness_chunks, dim=0)
            singularity_proxy_ratio = float(singularity_proxy_count / max(int(vertex_proxy_badness_all.numel()), 1))
            singularity_proxy_badness_mean = float(
                singularity_proxy_badness_sum / max(int(vertex_proxy_badness_all.numel()), 1)
            )
            singularity_proxy_badness_p95 = float(torch.quantile(vertex_proxy_badness_all, 0.95).item())
        else:
            singularity_proxy_ratio = 0.0
            singularity_proxy_badness_mean = 0.0
            singularity_proxy_badness_p95 = 0.0

        training_metrics = {
            "loss_total": float(loss.detach().cpu().item()),
            "loss_mnfld": float(sdf_term.detach().cpu().item()),
            "loss_nonmnfld": float(inter_term.detach().cpu().item()),
            "loss_eikonal": float(eikonal_term.detach().cpu().item()),
            "loss_morse": float(morse_loss.detach().cpu().item()),
            "loss_theta_hessian": float(theta_hessian_term.detach().cpu().item()),
            "loss_theta_neighbor": float(theta_neighbors_term.detach().cpu().item()),
        }
        denom_faces = max(num_faces, 1)
        field_validity = {
            "num_faces": num_faces,
            "nan_count": nan_count,
            "alpha_norm_error_mean": alpha_norm_err_sum / denom_faces,
            "alpha_norm_error_max": alpha_norm_err_max,
            "beta_norm_error_mean": beta_norm_err_sum / denom_faces,
            "beta_norm_error_max": beta_norm_err_max,
            "alpha_beta_dot_mean": alpha_beta_dot_sum / denom_faces,
            "alpha_beta_dot_max": alpha_beta_dot_max,
            "alpha_tangent_error_mean": alpha_tangent_sum / denom_faces,
            "alpha_tangent_error_max": alpha_tangent_max,
            "beta_tangent_error_mean": beta_tangent_sum / denom_faces,
            "beta_tangent_error_max": beta_tangent_max,
            "handedness_mean": handedness_sum / denom_faces,
            "handedness_min": handedness_min if math.isfinite(handedness_min) else 0.0,
            "flipped_frame_ratio": flipped_count / denom_faces,
        }
        field_smoothness = {
            "adjacent_cross_error_mean": smooth_mean,
            "adjacent_cross_error_median": smooth_median,
            "adjacent_cross_error_p95": smooth_p95,
            "adjacent_cross_error_max": smooth_max,
        }
        singularity_metrics = {
            "singularity_proxy_threshold": singularity_threshold,
            "singularity_proxy_count": singularity_proxy_count,
            "singularity_proxy_ratio": singularity_proxy_ratio,
            "singularity_proxy_badness_mean": singularity_proxy_badness_mean,
            "singularity_proxy_badness_p95": singularity_proxy_badness_p95,
            "singularity_proxy_badness_max": singularity_proxy_badness_max,
        }
        score = (
            training_metrics["loss_theta_hessian"]
            + training_metrics["loss_theta_neighbor"]
            + 10.0 * field_smoothness["adjacent_cross_error_mean"]
            + 50.0 * field_validity["flipped_frame_ratio"]
            + 10.0 * field_validity["alpha_tangent_error_mean"]
            + 10.0 * field_validity["beta_tangent_error_mean"]
        )
        if log_label:
            print(
                f"[metrics-cpu] complete {log_label} score={float(score):.6f} "
                f"faces={num_faces} groups={num_groups} valid_neighbor_values="
                f"{sum(int(chunk.numel()) for chunk in valid_neighbor_chunks)}",
                flush=True,
            )
        return {
            "training": training_metrics,
            "field_validity": field_validity,
            "field_smoothness": field_smoothness,
            "singularity_proxy": singularity_metrics,
            "score": float(score),
        }



    def forward(self, output_pred, mnfld_points, nonmnfld_points, mnfld_n_gt=None, near_points=None, batch_idx=0,
                out_dir=None, filename=None, save_best=False, mnfld_pts_theta_output_pred=None, local_coord_u=None,
                local_coord_v=None, is_final_export=False, collect_metrics=False):

        dims = mnfld_points.shape[-1]

        #########################################
        # Compute required terms
        #########################################

        non_manifold_pred = output_pred["nonmanifold_pnts_pred"]
        manifold_pred = output_pred["manifold_pnts_pred"]

        zero = mnfld_points.new_zeros(())
        morse_loss = zero
        normal_term = zero
        eikonal_term = zero
        mnfld_hessian_term = zero

        # compute gradients for div (divergence), curl and curv (curvature)
        if manifold_pred is not None:
            mnfld_grad = utils.gradient(mnfld_points, manifold_pred)
        else:
            mnfld_grad = None

        nonmnfld_grad = utils.gradient(nonmnfld_points, non_manifold_pred)

        morse_nonmnfld_grad = None
        if self.use_morse and near_points is not None:
            morse_nonmnfld_grad = utils.gradient(near_points, output_pred['near_points_pred'])
        elif self.use_morse and near_points is None:
            morse_nonmnfld_grad = nonmnfld_grad

        if self.use_morse:
            mnfld_dx = utils.gradient(mnfld_points, mnfld_grad[:, :, 0])
            mnfld_dy = utils.gradient(mnfld_points, mnfld_grad[:, :, 1])
            if dims == 3:
                mnfld_dz = utils.gradient(mnfld_points, mnfld_grad[:, :, 2])
                mnfld_hessian_term = torch.stack((mnfld_dx, mnfld_dy, mnfld_dz), dim=-1)
            else:
                mnfld_hessian_term = torch.stack((mnfld_dx, mnfld_dy), dim=-1)

            morse_mnfld = zero
            if self.div_type == 'l1':
                mnfld_n_gt = mnfld_n_gt.permute(1, 0, 2)
                mnfld_hessian_term = mnfld_hessian_term.squeeze(0)

                morse_mnfld = torch.bmm(mnfld_n_gt, mnfld_hessian_term)
                morse_mnfld = morse_mnfld.abs().mean()

            morse_loss = 0.5 * morse_mnfld

        sdf_term = torch.abs(manifold_pred).mean()

        # eikonal term
        eikonal_term = eikonal_loss(morse_nonmnfld_grad, mnfld_grad=mnfld_grad, eikonal_type='abs')

        # inter term
        inter_term = torch.exp(-1e2 * torch.abs(non_manifold_pred)).mean()

        # theta_term
        local_coord_u = local_coord_u.squeeze(0)
        local_coord_v = local_coord_v.squeeze(0)

        mnfld_pts_theta_output_pred = mnfld_pts_theta_output_pred.squeeze(0)

        vector_alpha = local_coord_u * torch.cos(mnfld_pts_theta_output_pred) + local_coord_v * torch.sin(
            mnfld_pts_theta_output_pred)
        vector_alpha = vector_alpha / (vector_alpha.norm(dim=-1, keepdim=True) + 1e-12)

        vector_beta = -local_coord_u * torch.sin(mnfld_pts_theta_output_pred) + local_coord_v * torch.cos(
            mnfld_pts_theta_output_pred)
        vector_beta = vector_beta / (vector_beta.norm(dim=-1, keepdim=True) + 1e-12)

        face_indices = self._group_face_indices
        face_mask = self._group_face_mask
        neighbor_indices = self._group_neighbor_indices
        neighbor_mask = self._group_neighbor_mask

        hessian_group = mnfld_hessian_term[face_indices]  # G x M x 3 x 3
        vector_alpha_i = vector_alpha[face_indices]  # G x M x 3
        vector_beta_i = vector_beta[face_indices]  # G x M x 3

        alpha_h = torch.matmul(vector_alpha_i.unsqueeze(-2), hessian_group).squeeze(-2)
        beta_h = torch.matmul(vector_beta_i.unsqueeze(-2), hessian_group).squeeze(-2)
        alpha_cross = torch.linalg.cross(alpha_h, vector_alpha_i).abs()
        beta_cross = torch.linalg.cross(beta_h, vector_beta_i).abs()

        face_mask_float = face_mask.to(alpha_cross.dtype)
        face_counts = face_mask_float.sum(dim=1).clamp_min(1.0)
        alpha_group_mean = (alpha_cross * face_mask_float.unsqueeze(-1)).sum(dim=(1, 2)) / (face_counts * alpha_cross.shape[-1])
        beta_group_mean = (beta_cross * face_mask_float.unsqueeze(-1)).sum(dim=(1, 2)) / (face_counts * beta_cross.shape[-1])
        theta_hessian_term = (0.5 * (alpha_group_mean + beta_group_mean)).mean()

        vector_alpha_j = vector_alpha[neighbor_indices]  # G x M x K x 3
        vector_beta_j = vector_beta[neighbor_indices]  # G x M x K x 3

        if self._group_rotation_mats is not None:
            vector_alpha_j = torch.matmul(self._group_rotation_mats, vector_alpha_j.unsqueeze(-1)).squeeze(-1)
            vector_beta_j = torch.matmul(self._group_rotation_mats, vector_beta_j.unsqueeze(-1)).squeeze(-1)

        vector_alpha_i_neighbors = vector_alpha_i.unsqueeze(2)  # G x M x 1 x 3
        vector_beta_i_neighbors = vector_beta_i.unsqueeze(2)  # G x M x 1 x 3
        neighbors_term = (
            (vector_alpha_i_neighbors * vector_alpha_j).sum(dim=-1).abs()
            + (vector_alpha_i_neighbors * vector_beta_j).sum(dim=-1).abs()
            + (vector_beta_i_neighbors * vector_alpha_j).sum(dim=-1).abs()
            + (vector_beta_i_neighbors * vector_beta_j).sum(dim=-1).abs()
            - 2
        )
        neighbor_mask_float = neighbor_mask.to(neighbors_term.dtype)
        neighbor_counts = neighbor_mask_float.sum(dim=(1, 2)).clamp_min(1.0)
        theta_neighbors_term = ((neighbors_term * neighbor_mask_float).sum(dim=(1, 2)) / neighbor_counts).mean()

        # losses used in the paper
        if self.loss_type == 'siren_wo_n_w_morse_w_theta':
            loss = self.weights[0] * sdf_term + self.weights[1] * inter_term + self.weights[3] * eikonal_term + \
                   self.weights[5] * morse_loss + self.weights[2] * theta_hessian_term + self.weights[
                       4] * theta_neighbors_term
        else:
            print(self.loss_type)
            raise Warning("unrecognized loss type")

        metrics = None

        if save_best or is_final_export or collect_metrics:
            if is_final_export:
                metrics_log_label = "final_export"
            elif save_best:
                metrics_log_label = f"save_best_step_{batch_idx}"
            else:
                metrics_log_label = f"collect_metrics_step_{batch_idx}"
            metrics = self._build_metrics_chunked_cpu(
                loss,
                sdf_term,
                inter_term,
                eikonal_term,
                morse_loss,
                theta_hessian_term,
                theta_neighbors_term,
                vector_alpha,
                vector_beta,
                mnfld_n_gt,
                neighbors_term,
                neighbor_mask,
                log_label=metrics_log_label,
            )
            if (save_best or is_final_export) and self._export_manager is None:
                self._export_manager = CrossFieldExportManager(
                    out_dir,
                    filename,
                )
            if save_best or is_final_export:
                vector_alpha_export = _detached_cpu_tensor(vector_alpha)
                vector_beta_export = _detached_cpu_tensor(vector_beta)
                export_crossfield_snapshot(
                    vector_alpha_export,
                    vector_beta_export,
                    out_dir=out_dir,
                    filename=filename,
                    batch_idx=batch_idx,
                    manager=self._export_manager,
                    total_loss=loss.detach().cpu().item(),
                    field_score=metrics["score"],
                    metrics=metrics,
                    is_final=is_final_export,
                )


        return {"loss": loss, 'sdf_term': sdf_term, 'inter_term': inter_term,
                'eikonal_term': eikonal_term, 'normals_loss': normal_term, 'morse_term': morse_loss,
                'theta_hessian_term': theta_hessian_term, 'theta_neighbors_term': theta_neighbors_term,
                'metrics': metrics}

    def update_morse_weight(self, current_iteration, n_iterations, params=None):
        # `params`` should be (start_weight, *optional middle, end_weight) where optional middle is of the form [percent, value]*
        # Thus (1e2, 0.5, 1e2 0.7 0.0, 0.0) means that the weight at [0, 0.5, 0.7, 1] of the training process, the weight should
        #   be [1e2,1e2,0.0,0.0]. Between these points, the weights change as per the div_decay parameter, e.g. linearly, quintic, step etc.
        #   Thus the weight stays at 1e2 from 0-0.5, decay from 1e2 to 0.0 from 0.5-0.75, and then stays at 0.0 from 0.75-1.

        if not hasattr(self, 'decay_params_list'):
            assert len(params) >= 2, params
            assert len(params[1:-1]) % 2 == 0
            self.decay_params_list = list(zip([params[0], *params[1:-1][1::2], params[-1]], [0, *params[1:-1][::2], 1]))

        curr = current_iteration / n_iterations
        we, e = min([tup for tup in self.decay_params_list if tup[1] >= curr], key=lambda tup: tup[1])
        w0, s = max([tup for tup in self.decay_params_list if tup[1] <= curr], key=lambda tup: tup[1])

        # Divergence term anealing functions
        if self.div_decay == 'linear':  # linearly decrease weight from iter s to iter e
            if current_iteration < s * n_iterations:
                self.weights[5] = w0
            elif current_iteration >= s * n_iterations and current_iteration < e * n_iterations:
                self.weights[5] = w0 + (we - w0) * (current_iteration / n_iterations - s) / (e - s)
            else:
                self.weights[5] = we
        elif self.div_decay == 'quintic':  # linearly decrease weight from iter s to iter e
            if current_iteration < s * n_iterations:
                self.weights[5] = w0
            elif current_iteration >= s * n_iterations and current_iteration < e * n_iterations:
                self.weights[5] = w0 + (we - w0) * (1 - (1 - (current_iteration / n_iterations - s) / (e - s)) ** 5)
            else:
                self.weights[5] = we
        elif self.div_decay == 'step':  # change weight at s
            if current_iteration < s * n_iterations:
                self.weights[5] = w0
            else:
                self.weights[5] = we
        elif self.div_decay == 'none':
            pass
        else:
            raise Warning("unsupported div decay value")
