import numpy as np
import scipy.spatial as spatial
import torch.utils.data as data
import trimesh


_SAMPLE_LABEL_UNIFORM = 0
_SAMPLE_LABEL_NEAR_SURFACE = 1
_SAMPLE_LABEL_FEATURE = 2
_DEFAULT_MIXED_RATIOS = {
    "uniform": 0.5,
    "near_surface": 0.35,
    "feature": 0.15,
}


class ReconDataset(data.Dataset):
    def __init__(
        self,
        file_path,
        n_points,
        n_samples=128,
        grid_range=1.1,
        *,
        nonmnfld_sample_type="uniform",
        near_surface_ratio=None,
        uniform_ratio=None,
        feature_ratio=None,
        boundary_ratio=0.5,
        near_surface_sigma=None,
        uniform_extent=None,
        seed=0,
    ):
        self.file_path = file_path
        self.n_points = int(n_points)
        self.n_samples = int(n_samples)
        self.seed = int(seed)
        self.epoch = 0

        self.mesh = trimesh.load_mesh(self.file_path, process=False)
        self.grid_range = float(grid_range)
        self.uniform_extent = float(uniform_extent if uniform_extent is not None else grid_range)
        self.near_surface_sigma_scale = None if near_surface_sigma is None else float(near_surface_sigma)
        self.nonmnfld_sample_type = self._normalize_sample_type(nonmnfld_sample_type)
        self.boundary_ratio = float(np.clip(boundary_ratio, 0.0, 1.0))

        self.points, self.mnfld_n = self.get_face_center_points()
        self.mnfld_n, self.vector_u, self.vector_v = self.vector_u_v(self.mnfld_n)
        self.change_u_v_FLAG = 0.0
        self.change_theta_hessian_term = 0.0
        self.bbox = np.array([np.min(self.points, axis=0), np.max(self.points, axis=0)]).transpose()

        self._build_face_sampling_state()
        self.sample_gaussian_noise_around_shape()
        self.uniform_ratio, self.near_surface_ratio, self.feature_ratio = self._resolve_mixed_ratios(
            uniform_ratio=uniform_ratio,
            near_surface_ratio=near_surface_ratio,
            feature_ratio=feature_ratio,
        )
        self.validation_batch = self._build_validation_batch()

    def _normalize_sample_type(self, sample_type):
        normalized = str(sample_type or "uniform").strip().lower()
        aliases = {
            "grid": "uniform",
            "gaussian": "uniform",
            "combined": "mixed",
        }
        return aliases.get(normalized, normalized)

    def _resolve_mixed_ratios(self, *, uniform_ratio, near_surface_ratio, feature_ratio):
        uniform = _DEFAULT_MIXED_RATIOS["uniform"] if uniform_ratio is None else float(uniform_ratio)
        near_surface = _DEFAULT_MIXED_RATIOS["near_surface"] if near_surface_ratio is None else float(near_surface_ratio)
        feature = _DEFAULT_MIXED_RATIOS["feature"] if feature_ratio is None else float(feature_ratio)
        ratios = np.array([uniform, near_surface, feature], dtype=np.float64)
        ratios = np.clip(ratios, 0.0, None)
        if np.allclose(ratios.sum(), 0.0):
            ratios = np.array(
                [
                    _DEFAULT_MIXED_RATIOS["uniform"],
                    _DEFAULT_MIXED_RATIOS["near_surface"],
                    _DEFAULT_MIXED_RATIOS["feature"],
                ],
                dtype=np.float64,
            )
        ratios /= ratios.sum()
        return tuple(float(value) for value in ratios)

    def _build_face_sampling_state(self):
        adjacency = self.mesh.face_adjacency
        face_count = self.points.shape[0]
        adjacency_count = np.zeros(face_count, dtype=np.int32)
        normal_variation = np.zeros(face_count, dtype=np.float64)

        if adjacency.size > 0:
            normals_a = self.mnfld_n[adjacency[:, 0]]
            normals_b = self.mnfld_n[adjacency[:, 1]]
            cosine = np.clip(np.sum(normals_a * normals_b, axis=1), -1.0, 1.0)
            angles = np.arccos(cosine)
            np.add.at(adjacency_count, adjacency[:, 0], 1)
            np.add.at(adjacency_count, adjacency[:, 1], 1)
            np.add.at(normal_variation, adjacency[:, 0], angles)
            np.add.at(normal_variation, adjacency[:, 1], angles)

        normal_variation = normal_variation / np.maximum(adjacency_count, 1)
        boundary_mask = adjacency_count < 3
        boundary_weights = boundary_mask.astype(np.float64)
        variation_weights = normal_variation.astype(np.float64)
        max_variation = float(np.max(variation_weights)) if variation_weights.size else 0.0
        if max_variation > 0.0:
            variation_weights = variation_weights / max_variation
        feature_weights = ((1.0 - self.boundary_ratio) * variation_weights) + (self.boundary_ratio * boundary_weights)
        feature_weights = np.clip(feature_weights, 0.0, None)
        if np.allclose(feature_weights.sum(), 0.0):
            feature_weights = np.ones(face_count, dtype=np.float64)
        feature_weights /= feature_weights.sum()

        self.face_adjacency_count = adjacency_count
        self.boundary_face_mask = boundary_mask
        self.feature_face_weights = feature_weights

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _rng_for_index(self, index, *, validation=False):
        base_seed = (self.seed * 1000003) + (self.epoch * 9176)
        offset = 7919 if validation else 0
        return np.random.default_rng(base_seed + int(index) + offset)

    def get_static_batch(self):
        return {
            "points": self.points,
            "mnfld_n": self.mnfld_n,
            "local_coordinates_u": self.vector_u,
            "local_coordinates_v": self.vector_v,
        }

    def get_validation_batch(self):
        return {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in self.validation_batch.items()
        }

    def get_face_center_points(self):
        points = np.asarray(self.mesh.triangles_center, dtype=np.float32)
        normals = np.asarray(self.mesh.face_normals, dtype=np.float32)
        if normals.shape[0] == 0:
            normals = np.zeros_like(points)
        self.cp = points.mean(axis=0)
        points = points - self.cp[None, :]
        self.scale = np.abs(points).max()
        points = points / self.scale
        return points, normals

    def vector_u_v(self, normals):
        flags = np.ones_like(normals, dtype=bool)
        min_value_index = np.argmin(np.absolute(normals), axis=1)
        flags[np.arange(normals.shape[0]), min_value_index] = False
        true_ind = np.argwhere(flags)
        flag_row = true_ind[:, 0].reshape(-1, 2)
        flag_col = true_ind[:, 1].reshape(-1, 2)

        vector_u = np.zeros_like(normals, dtype=np.float32)
        vector_u[flag_row[:, 0], flag_col[:, 0]] = normals[flag_row[:, 1], flag_col[:, 1]]
        vector_u[flag_row[:, 1], flag_col[:, 1]] = -normals[flag_row[:, 0], flag_col[:, 0]]

        vector_v = np.cross(normals, vector_u)
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
        vector_u = vector_u / (np.linalg.norm(vector_u, axis=1, keepdims=True) + 1e-12)
        vector_v = vector_v / (np.linalg.norm(vector_v, axis=1, keepdims=True) + 1e-12)
        return normals, vector_u, vector_v

    def sample_gaussian_noise_around_shape(self):
        kd_tree = spatial.KDTree(self.points)
        dist, _ = kd_tree.query(self.points, k=min(51, len(self.points)), workers=-1)
        if dist.ndim == 1:
            dist = dist[:, None]
        sigmas = dist[:, -1:]
        if self.near_surface_sigma_scale is not None:
            sigmas = np.full_like(sigmas, self.near_surface_sigma_scale, dtype=np.float64)
        self.sigmas = sigmas.astype(np.float32)

    def _sample_uniform_points(self, rng, count):
        if count <= 0:
            return np.zeros((0, 3), dtype=np.float32)
        return rng.uniform(-self.uniform_extent, self.uniform_extent, size=(count, 3)).astype(np.float32)

    def _sample_near_surface_points(self, rng, count):
        if count <= 0:
            return np.zeros((0, 3), dtype=np.float32)
        indices = rng.integers(0, self.points.shape[0], size=count)
        offsets = rng.normal(size=(count, 3)).astype(np.float32) * self.sigmas[indices]
        return (self.points[indices] + offsets).astype(np.float32)

    def _sample_feature_points(self, rng, count):
        if count <= 0:
            return np.zeros((0, 3), dtype=np.float32)
        indices = rng.choice(self.points.shape[0], size=count, replace=True, p=self.feature_face_weights)
        offsets = rng.normal(size=(count, 3)).astype(np.float32) * self.sigmas[indices]
        return (self.points[indices] + offsets).astype(np.float32)

    def _allocation_from_ratios(self, count):
        raw = np.array(
            [
                self.uniform_ratio,
                self.near_surface_ratio,
                self.feature_ratio,
            ],
            dtype=np.float64,
        ) * count
        allocation = np.floor(raw).astype(np.int32)
        remainder = int(count - allocation.sum())
        if remainder > 0:
            order = np.argsort(-(raw - allocation))
            for idx in order[:remainder]:
                allocation[idx] += 1
        return allocation.tolist()

    def _sample_nonmanifold_points(self, rng, count):
        sample_type = self.nonmnfld_sample_type
        if sample_type == "uniform":
            points = self._sample_uniform_points(rng, count)
            labels = np.full(count, _SAMPLE_LABEL_UNIFORM, dtype=np.int64)
            return points, labels
        if sample_type == "near_surface":
            points = self._sample_near_surface_points(rng, count)
            labels = np.full(count, _SAMPLE_LABEL_NEAR_SURFACE, dtype=np.int64)
            return points, labels
        if sample_type == "feature_biased":
            points = self._sample_feature_points(rng, count)
            labels = np.full(count, _SAMPLE_LABEL_FEATURE, dtype=np.int64)
            return points, labels
        if sample_type != "mixed":
            raise ValueError(f"Unsupported nonmnfld_sample_type: {sample_type}")

        uniform_count, near_surface_count, feature_count = self._allocation_from_ratios(count)
        point_chunks = []
        label_chunks = []
        if uniform_count:
            point_chunks.append(self._sample_uniform_points(rng, uniform_count))
            label_chunks.append(np.full(uniform_count, _SAMPLE_LABEL_UNIFORM, dtype=np.int64))
        if near_surface_count:
            point_chunks.append(self._sample_near_surface_points(rng, near_surface_count))
            label_chunks.append(np.full(near_surface_count, _SAMPLE_LABEL_NEAR_SURFACE, dtype=np.int64))
        if feature_count:
            point_chunks.append(self._sample_feature_points(rng, feature_count))
            label_chunks.append(np.full(feature_count, _SAMPLE_LABEL_FEATURE, dtype=np.int64))
        points = np.concatenate(point_chunks, axis=0) if point_chunks else np.zeros((0, 3), dtype=np.float32)
        labels = np.concatenate(label_chunks, axis=0) if label_chunks else np.zeros((0,), dtype=np.int64)
        if len(labels) > 1:
            order = rng.permutation(len(labels))
            points = points[order]
            labels = labels[order]
        return points.astype(np.float32), labels

    def _build_validation_batch(self):
        rng = self._rng_for_index(0, validation=True)
        nonmnfld_points, labels = self._sample_nonmanifold_points(rng, self.n_points)
        near_points = self._sample_near_surface_points(rng, self.points.shape[0])
        feature_indices = rng.choice(
            self.points.shape[0],
            size=min(self.points.shape[0], max(1, min(128, self.points.shape[0]))),
            replace=False,
            p=self.feature_face_weights,
        )
        boundary_indices = np.flatnonzero(self.boundary_face_mask)
        if boundary_indices.size == 0:
            boundary_indices = feature_indices[:0]
        return {
            "nonmnfld_points": nonmnfld_points,
            "near_points": near_points,
            "nonmnfld_sample_labels": labels,
            "validation_face_indices": feature_indices.astype(np.int64),
            "validation_feature_indices": feature_indices.astype(np.int64),
            "validation_boundary_indices": boundary_indices.astype(np.int64),
        }

    def __getitem__(self, index):
        rng = self._rng_for_index(index)
        nonmnfld_points, labels = self._sample_nonmanifold_points(rng, self.n_points)
        near_points = self._sample_near_surface_points(rng, self.points.shape[0])
        return {
            "nonmnfld_points": nonmnfld_points,
            "near_points": near_points,
            "nonmnfld_sample_labels": labels,
        }

    def __len__(self):
        return self.n_samples
