# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import warp as wp

import newton.ik as ik
from newton._src.sim.ik.ik_common import IKJacobianType


@wp.func
def _wp_smooth_joint_filter_func(
    x            : wp.float32,
    lower_limit  : wp.float32,
    upper_limit  : wp.float32,
    padding_limit: wp.float32,
    m            : wp.float32,
    p            : wp.float32
):
    c = (lower_limit + upper_limit) * 0.5
    lower_limit += (padding_limit - c)
    upper_limit -= (padding_limit + c)
    if lower_limit < x and x <= upper_limit:
        return 0.0

    diff = wp.where(x <= lower_limit, lower_limit-x, x-upper_limit) * m
    return 1.0 - wp.exp(-wp.pow(diff, p))


@wp.kernel
def _smooth_joint_filter_residuals(
    joint_q: wp.array2d(dtype=wp.float32),           # (n_batch, n_coords)
    dof_to_coord: wp.array1d(dtype=wp.int32),        # (n_dofs)
    joint_limit_lower: wp.array1d(dtype=wp.float32), # (n_dofs)
    joint_limit_upper: wp.array1d(dtype=wp.float32), # (n_dofs)
    coord_masks: wp.array1d(dtype=wp.float32),       # (n_coords)
    weight: wp.array1d(dtype=wp.float32),            # (1)
    start_idx: int,
    # outputs
    residuals: wp.array2d(dtype=wp.float32),     # (n_batch, n_residuals)
):
    problem, dof_idx = wp.tid()
    coord_idx = dof_to_coord[dof_idx]
    mask = coord_masks[coord_idx]

    if coord_idx < 0:
        return

    if mask > 0.0:
        lower = joint_limit_lower[dof_idx]
        upper = joint_limit_upper[dof_idx]
        c = (lower + upper) * 0.5

        q = joint_q[problem, coord_idx]
        error = (q - c)

        smoother = _wp_smooth_joint_filter_func(error, lower, upper, 1.02, 1.0, 6.5)
        residuals[problem, start_idx + dof_idx] = error * smoother * weight[0] * mask
    else:
        residuals[problem, start_idx + dof_idx] = 0.0


@wp.kernel
def _update_weight(
    in_value: wp.float32,
    out_weight: wp.array1d(dtype=wp.float32),  # (1)
):
    out_weight[0] = in_value


@wp.kernel
def _smooth_joint_filter_jac_analytic(
    dof_to_coord: wp.array1d(dtype=wp.int32),    # (n_dofs)
    coord_masks: wp.array1d(dtype=wp.float32),   # (n_coords)
    n_dofs: int,
    start_idx: int,
    weight: wp.array1d(dtype=wp.float32), # (1)
    # outputs
    jacobian: wp.array3d(dtype=wp.float32),      # (n_batch, n_residuals, n_dofs)
):
    problem, dof_idx = wp.tid()
    coord_idx = dof_to_coord[dof_idx]
    mask = coord_masks[coord_idx]

    if coord_idx < 0:
        return

    # Jacobian is diagonal: dr[dof]/dq[dof] = weight
    jacobian[problem, start_idx + dof_idx, dof_idx] = weight[0] * mask


class IKSmoothJointFilter(ik.IKObjective):
    """
    An IK objective that applies a smooth penalty to joint coordinates that approach or exceed specified limits
    using an inverse gaussian filter.

    Args:
        joint_limit_lower (wp.array1d): An array of shape (n_dofs,) containing the lower limits for each joint degree of freedom.
        joint_limit_upper (wp.array1d): An array of shape (n_dofs,) containing the upper limits for each joint degree of freedom.
        weight (float, optional): A scalar weight that controls the strength of the joint limit penalty. Defaults to 0.01.
        coord_masks (wp.array1d, optional): An array of shape (n_coords,) containing mask values for each joint coordinate.
            Mask values should be in the range [0, 1], where 0 means the coordinate is ignored by this objective and 1 means it is fully considered.
            All coords are used by default if no masks are specified.
    """
    def __init__(self, joint_limit_lower, joint_limit_upper, weight=0.01, coord_masks=None):
        super().__init__()
        self.joint_limit_lower = joint_limit_lower
        self.joint_limit_upper = joint_limit_upper
        self.n_dofs = len(joint_limit_lower)
        self.dof_to_coord = None
        self.e_array = None
        self._weight = wp.array([weight], dtype=wp.float32)

        self.coord_masks = None
        self.coord_masks_np = None
        if coord_masks is not None:
            if isinstance(coord_masks, np.ndarray):
                self.coord_masks_np = coord_masks.astype(np.float32)
                self.coord_masks = None
            elif isinstance(coord_masks, wp.array):
                self.coord_masks = coord_masks
                self.coord_masks_np = None

    def bind_device(self, device):
        super().bind_device(device)

    def init_buffers(self, model, jacobian_mode):
        self._require_batch_layout()

        if self.coord_masks_np is not None and len(self.coord_masks_np) == model.joint_coord_count:
            self.coord_masks = wp.array(self.coord_masks_np, dtype=wp.float32, device=self.device)

        # All coords are considered if no coord masks have been declared
        if self.coord_masks is None:
            self.coord_masks = wp.ones(shape=model.joint_coord_count, dtype=wp.float32, device=self.device)

        # Build DOF to coordinate mapping
        dof_to_coord_np = np.full(self.n_dofs, -1, dtype=np.int32)
        q_start_np = model.joint_q_start.numpy()
        qd_start_np = model.joint_qd_start.numpy()
        joint_dof_dim_np = model.joint_dof_dim.numpy()

        for j in range(model.joint_count):
            dof0 = qd_start_np[j]
            coord0 = q_start_np[j]
            lin, ang = joint_dof_dim_np[j]
            for k in range(lin + ang):
                if dof0 + k < self.n_dofs:
                    dof_to_coord_np[dof0 + k] = coord0 + k

        self.dof_to_coord = wp.array(dof_to_coord_np, dtype=wp.int32, device=self.device)

        # For autodiff mode
        if jacobian_mode == IKJacobianType.AUTODIFF:
            e = np.zeros((self.n_batch, self.total_residuals), dtype=np.float32)
            for prob_idx in range(self.n_batch):
                for dof_idx in range(self.n_dofs):
                    e[prob_idx, self.residual_offset + dof_idx] = 1.0
            self.e_array = wp.array(e.flatten(), dtype=wp.float32, device=self.device)

    def supports_analytic(self):
        return True

    def residual_dim(self):
        return self.n_dofs

    def set_weight(self, value):
        if self.coord_masks is None:
            return

        wp.launch(
            _update_weight,
            dim=1,
            inputs=[value],
            outputs=[self._weight],
            device=self.device)

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        count = joint_q.shape[0]
        wp.launch(
            _smooth_joint_filter_residuals,
            dim=[count, self.n_dofs],
            inputs=[
                joint_q,
                self.dof_to_coord,
                self.joint_limit_lower,
                self.joint_limit_upper,
                self.coord_masks,
                self._weight,
                start_idx,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof):
        self._require_batch_layout()
        tape.backward(grads={tape.outputs[0]: self.e_array})

        q_grad = tape.gradients[dq_dof]

        # Use the analytic Jacobian fill since it's simple
        wp.launch(
            _smooth_joint_filter_jac_analytic,
            dim=[self.n_batch, self.n_dofs],
            inputs=[
                self.dof_to_coord,
                self.coord_masks,
                self.n_dofs,
                start_idx,
                self._weight,
            ],
            outputs=[jacobian],
            device=self.device,
        )

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx):
        count = joint_q.shape[0]
        wp.launch(
            _smooth_joint_filter_jac_analytic,
            dim=[count, self.n_dofs],
            inputs=[
                self.dof_to_coord,
                self.coord_masks,
                self.n_dofs,
                start_idx,
                self._weight,
            ],
            outputs=[jacobian],
            device=self.device,
        )


@wp.kernel
def _temporal_reference_residuals(
    joint_q: wp.array2d(dtype=wp.float32),
    target_q: wp.array2d(dtype=wp.float32),
    coord_masks: wp.array1d(dtype=wp.float32),
    weight: wp.array1d(dtype=wp.float32),
    start_idx: int,
    residuals: wp.array2d(dtype=wp.float32),
):
    problem, coord_idx = wp.tid()
    residuals[problem, start_idx + coord_idx] = (
        joint_q[problem, coord_idx] - target_q[problem, coord_idx]
    ) * weight[0] * coord_masks[coord_idx]


@wp.kernel
def _temporal_reference_jac_analytic(
    coord_masks: wp.array1d(dtype=wp.float32),
    n_coords: int,
    start_idx: int,
    weight: wp.array1d(dtype=wp.float32),
    jacobian: wp.array3d(dtype=wp.float32),
):
    problem, coord_idx = wp.tid()
    if coord_idx < n_coords:
        jacobian[problem, start_idx + coord_idx, coord_idx] = weight[0] * coord_masks[coord_idx]


class IKTemporalJointReference(ik.IKObjective):
    """Penalize selected joint coordinates against an explicit q reference."""

    def __init__(self, joint_coord_count: int, weight=0.0, coord_masks=None):
        super().__init__()
        self.n_coords = int(joint_coord_count)
        self._weight = wp.array([weight], dtype=wp.float32)
        self.target_q = None
        self.coord_masks = None
        self.coord_masks_np = None
        if coord_masks is not None:
            if isinstance(coord_masks, np.ndarray):
                self.coord_masks_np = coord_masks.astype(np.float32)
            elif isinstance(coord_masks, wp.array):
                self.coord_masks = coord_masks

    def init_buffers(self, model, jacobian_mode):
        self._require_batch_layout()
        if self.coord_masks_np is not None and len(self.coord_masks_np) == model.joint_coord_count:
            self.coord_masks = wp.array(self.coord_masks_np, dtype=wp.float32, device=self.device)
        if self.coord_masks is None:
            self.coord_masks = wp.ones(shape=model.joint_coord_count, dtype=wp.float32, device=self.device)
        self.target_q = wp.zeros(
            shape=(self.n_batch, model.joint_coord_count),
            dtype=wp.float32,
            device=self.device,
        )

    def supports_analytic(self):
        return True

    def residual_dim(self):
        return self.n_coords

    def set_weight(self, value):
        wp.launch(
            _update_weight,
            dim=1,
            inputs=[value],
            outputs=[self._weight],
            device=self.device)

    def set_target(self, target_q_np):
        if self.target_q is None:
            return
        target = np.asarray(target_q_np, dtype=np.float32)
        if target.ndim == 1:
            target = target[None, :]
        wp.copy(self.target_q, wp.array(target, dtype=wp.float32, device=self.device))

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        count = joint_q.shape[0]
        wp.launch(
            _temporal_reference_residuals,
            dim=[count, self.n_coords],
            inputs=[
                joint_q,
                self.target_q,
                self.coord_masks,
                self._weight,
                start_idx,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof):
        self.compute_jacobian_analytic(None, None, model, jacobian, None, start_idx)

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx):
        wp.launch(
            _temporal_reference_jac_analytic,
            dim=[self.n_batch, self.n_coords],
            inputs=[
                self.coord_masks,
                self.n_coords,
                start_idx,
                self._weight,
            ],
            outputs=[jacobian],
            device=self.device,
        )


@wp.kernel
def _arm_bend_angle_residuals(
    body_q: wp.array2d(dtype=wp.transform),
    root_indices: wp.array1d(dtype=wp.int32),
    mid_indices: wp.array1d(dtype=wp.int32),
    end_indices: wp.array1d(dtype=wp.int32),
    target_angles: wp.array2d(dtype=wp.float32),
    active_mask: wp.array2d(dtype=wp.float32),
    weight: wp.array1d(dtype=wp.float32),
    start_idx: int,
    problem_idx_map: wp.array1d(dtype=wp.int32),
    residuals: wp.array2d(dtype=wp.float32),
):
    row, chain_idx = wp.tid()
    base = problem_idx_map[row]

    root_tf = body_q[row, root_indices[chain_idx]]
    mid_tf = body_q[row, mid_indices[chain_idx]]
    end_tf = body_q[row, end_indices[chain_idx]]

    root = wp.vec3(root_tf[0], root_tf[1], root_tf[2])
    mid = wp.vec3(mid_tf[0], mid_tf[1], mid_tf[2])
    end = wp.vec3(end_tf[0], end_tf[1], end_tf[2])

    v0 = root - mid
    v1 = end - mid
    denom = wp.max(wp.length(v0) * wp.length(v1), 1.0e-8)
    c = wp.clamp(wp.dot(v0, v1) / denom, -1.0, 1.0)
    theta = wp.acos(c)

    residuals[row, start_idx + chain_idx] = (
        (theta - target_angles[base, chain_idx])
        * active_mask[base, chain_idx]
        * weight[0]
    )


@wp.kernel
def _masked_position_residuals(
    body_q: wp.array2d(dtype=wp.transform),
    link_indices: wp.array1d(dtype=wp.int32),
    target_positions: wp.array3d(dtype=wp.float32),
    active_mask: wp.array2d(dtype=wp.float32),
    weight: wp.array1d(dtype=wp.float32),
    start_idx: int,
    problem_idx_map: wp.array1d(dtype=wp.int32),
    residuals: wp.array2d(dtype=wp.float32),
):
    row, chain_idx = wp.tid()
    base = problem_idx_map[row]
    body_tf = body_q[row, link_indices[chain_idx]]
    body_pos = wp.vec3(body_tf[0], body_tf[1], body_tf[2])
    target = wp.vec3(
        target_positions[base, chain_idx, 0],
        target_positions[base, chain_idx, 1],
        target_positions[base, chain_idx, 2],
    )
    scale = active_mask[base, chain_idx] * weight[0]
    offset = start_idx + chain_idx * 3
    diff = body_pos - target
    residuals[row, offset + 0] = diff[0] * scale
    residuals[row, offset + 1] = diff[1] * scale
    residuals[row, offset + 2] = diff[2] * scale


@wp.kernel
def _autodiff_jac_fill(
    q_grad: wp.array2d(dtype=wp.float32),
    n_dofs: int,
    start_idx: int,
    component: int,
    jacobian: wp.array3d(dtype=wp.float32),
):
    row = wp.tid()
    for dof_idx in range(n_dofs):
        jacobian[row, start_idx + component, dof_idx] = q_grad[row, dof_idx]


class IKArmBendAngleObjective(ik.IKObjective):
    """Weakly match bilateral shoulder-elbow-wrist bend angles.

    This objective intentionally has no analytic Jacobian. Use it with Newton
    IK `MIXED` or `AUTODIFF` Jacobian mode.
    """

    def __init__(self, root_indices, mid_indices, end_indices, n_problems: int, weight=0.0):
        super().__init__()
        self.root_indices_np = np.asarray(root_indices, dtype=np.int32)
        self.mid_indices_np = np.asarray(mid_indices, dtype=np.int32)
        self.end_indices_np = np.asarray(end_indices, dtype=np.int32)
        self.n_chains = int(len(self.root_indices_np))
        self.n_problems = int(n_problems)
        self._weight = wp.array([weight], dtype=wp.float32)
        self.root_indices = None
        self.mid_indices = None
        self.end_indices = None
        self.target_angles = None
        self.active_mask = None
        self.e_arrays = None

    def init_buffers(self, model, jacobian_mode):
        self._require_batch_layout()
        self.root_indices = wp.array(self.root_indices_np, dtype=wp.int32, device=self.device)
        self.mid_indices = wp.array(self.mid_indices_np, dtype=wp.int32, device=self.device)
        self.end_indices = wp.array(self.end_indices_np, dtype=wp.int32, device=self.device)
        self.target_angles = wp.zeros(
            shape=(self.n_problems, self.n_chains),
            dtype=wp.float32,
            device=self.device,
        )
        self.active_mask = wp.zeros(
            shape=(self.n_problems, self.n_chains),
            dtype=wp.float32,
            device=self.device,
        )
        if jacobian_mode in (IKJacobianType.AUTODIFF, IKJacobianType.MIXED):
            self.e_arrays = []
            for component in range(self.residual_dim()):
                e = np.zeros((self.n_batch, self.total_residuals), dtype=np.float32)
                for prob_idx in range(self.n_batch):
                    e[prob_idx, self.residual_offset + component] = 1.0
                self.e_arrays.append(wp.array(e.flatten(), dtype=wp.float32, device=self.device))

    def supports_analytic(self):
        return False

    def residual_dim(self):
        return self.n_chains

    def set_weight(self, value):
        wp.launch(_update_weight, dim=1, inputs=[value], outputs=[self._weight], device=self.device)

    def set_targets(self, target_angles_np, active_mask_np):
        if self.target_angles is None or self.active_mask is None:
            return
        angles = np.asarray(target_angles_np, dtype=np.float32)
        mask = np.asarray(active_mask_np, dtype=np.float32)
        wp.copy(self.target_angles, wp.array(angles, dtype=wp.float32, device=self.device))
        wp.copy(self.active_mask, wp.array(mask, dtype=wp.float32, device=self.device))

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        wp.launch(
            _arm_bend_angle_residuals,
            dim=[body_q.shape[0], self.n_chains],
            inputs=[
                body_q,
                self.root_indices,
                self.mid_indices,
                self.end_indices,
                self.target_angles,
                self.active_mask,
                self._weight,
                start_idx,
                problem_idx,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof):
        self._require_batch_layout()
        if self.e_arrays is None:
            return
        for component in range(self.residual_dim()):
            tape.backward(grads={tape.outputs[0]: self.e_arrays[component].flatten()})
            q_grad = tape.gradients[dq_dof]
            wp.launch(
                _autodiff_jac_fill,
                dim=self.n_batch,
                inputs=[q_grad, model.joint_dof_count, start_idx, component],
                outputs=[jacobian],
                device=self.device,
            )
            tape.zero()


@wp.kernel
def _limb_plane_normal_residuals(
    body_q: wp.array2d(dtype=wp.transform),
    root_indices: wp.array1d(dtype=wp.int32),
    mid_indices: wp.array1d(dtype=wp.int32),
    end_indices: wp.array1d(dtype=wp.int32),
    target_normals: wp.array3d(dtype=wp.float32),
    active_mask: wp.array2d(dtype=wp.float32),
    weight: wp.array1d(dtype=wp.float32),
    start_idx: int,
    problem_idx_map: wp.array1d(dtype=wp.int32),
    residuals: wp.array2d(dtype=wp.float32),
):
    row, chain_idx = wp.tid()
    base = problem_idx_map[row]

    root_tf = body_q[row, root_indices[chain_idx]]
    mid_tf = body_q[row, mid_indices[chain_idx]]
    end_tf = body_q[row, end_indices[chain_idx]]

    root = wp.vec3(root_tf[0], root_tf[1], root_tf[2])
    mid = wp.vec3(mid_tf[0], mid_tf[1], mid_tf[2])
    end = wp.vec3(end_tf[0], end_tf[1], end_tf[2])

    upper = mid - root
    lower = end - mid
    normal_raw = wp.cross(upper, lower)
    normal_len = wp.length(normal_raw)
    normal = wp.vec3(0.0, 0.0, 0.0)
    if normal_len > 1.0e-8:
        normal = normal_raw / normal_len

    target = wp.vec3(
        target_normals[base, chain_idx, 0],
        target_normals[base, chain_idx, 1],
        target_normals[base, chain_idx, 2],
    )
    scale = active_mask[base, chain_idx] * weight[0]
    offset = start_idx + chain_idx * 3
    diff = normal - target
    residuals[row, offset + 0] = diff[0] * scale
    residuals[row, offset + 1] = diff[1] * scale
    residuals[row, offset + 2] = diff[2] * scale


class IKLimbPlaneNormalObjective(ik.IKObjective):
    """Weakly match limb bend-plane normals for elbow/knee chain shape."""

    def __init__(self, root_indices, mid_indices, end_indices, n_problems: int, weight=0.0):
        super().__init__()
        self.root_indices_np = np.asarray(root_indices, dtype=np.int32)
        self.mid_indices_np = np.asarray(mid_indices, dtype=np.int32)
        self.end_indices_np = np.asarray(end_indices, dtype=np.int32)
        self.n_chains = int(len(self.root_indices_np))
        self.n_problems = int(n_problems)
        self._weight = wp.array([weight], dtype=wp.float32)
        self.root_indices = None
        self.mid_indices = None
        self.end_indices = None
        self.target_normals = None
        self.active_mask = None
        self.e_arrays = None

    def init_buffers(self, model, jacobian_mode):
        self._require_batch_layout()
        self.root_indices = wp.array(self.root_indices_np, dtype=wp.int32, device=self.device)
        self.mid_indices = wp.array(self.mid_indices_np, dtype=wp.int32, device=self.device)
        self.end_indices = wp.array(self.end_indices_np, dtype=wp.int32, device=self.device)
        self.target_normals = wp.zeros(
            shape=(self.n_problems, self.n_chains, 3),
            dtype=wp.float32,
            device=self.device,
        )
        self.active_mask = wp.zeros(
            shape=(self.n_problems, self.n_chains),
            dtype=wp.float32,
            device=self.device,
        )
        if jacobian_mode in (IKJacobianType.AUTODIFF, IKJacobianType.MIXED):
            self.e_arrays = []
            for component in range(self.residual_dim()):
                e = np.zeros((self.n_batch, self.total_residuals), dtype=np.float32)
                for prob_idx in range(self.n_batch):
                    e[prob_idx, self.residual_offset + component] = 1.0
                self.e_arrays.append(wp.array(e.flatten(), dtype=wp.float32, device=self.device))

    def supports_analytic(self):
        return False

    def residual_dim(self):
        return self.n_chains * 3

    def set_weight(self, value):
        wp.launch(_update_weight, dim=1, inputs=[value], outputs=[self._weight], device=self.device)

    def set_targets(self, target_normals_np, active_mask_np):
        if self.target_normals is None or self.active_mask is None:
            return
        normals = np.asarray(target_normals_np, dtype=np.float32)
        mask = np.asarray(active_mask_np, dtype=np.float32)
        wp.copy(self.target_normals, wp.array(normals, dtype=wp.float32, device=self.device))
        wp.copy(self.active_mask, wp.array(mask, dtype=wp.float32, device=self.device))

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        wp.launch(
            _limb_plane_normal_residuals,
            dim=[body_q.shape[0], self.n_chains],
            inputs=[
                body_q,
                self.root_indices,
                self.mid_indices,
                self.end_indices,
                self.target_normals,
                self.active_mask,
                self._weight,
                start_idx,
                problem_idx,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof):
        self._require_batch_layout()
        if self.e_arrays is None:
            return
        for component in range(self.residual_dim()):
            tape.backward(grads={tape.outputs[0]: self.e_arrays[component].flatten()})
            q_grad = tape.gradients[dq_dof]
            wp.launch(
                _autodiff_jac_fill,
                dim=self.n_batch,
                inputs=[q_grad, model.joint_dof_count, start_idx, component],
                outputs=[jacobian],
                device=self.device,
            )
            tape.zero()


class IKMaskedBodyPositionObjective(ik.IKObjective):
    """Per-problem masked body position objective for weak auxiliary targets."""

    def __init__(self, link_indices, n_problems: int, weight=0.0):
        super().__init__()
        self.link_indices_np = np.asarray(link_indices, dtype=np.int32)
        self.n_chains = int(len(self.link_indices_np))
        self.n_problems = int(n_problems)
        self._weight = wp.array([weight], dtype=wp.float32)
        self.link_indices = None
        self.target_positions = None
        self.active_mask = None
        self.e_arrays = None

    def init_buffers(self, model, jacobian_mode):
        self._require_batch_layout()
        self.link_indices = wp.array(self.link_indices_np, dtype=wp.int32, device=self.device)
        self.target_positions = wp.zeros(
            shape=(self.n_problems, self.n_chains, 3),
            dtype=wp.float32,
            device=self.device,
        )
        self.active_mask = wp.zeros(
            shape=(self.n_problems, self.n_chains),
            dtype=wp.float32,
            device=self.device,
        )
        if jacobian_mode in (IKJacobianType.AUTODIFF, IKJacobianType.MIXED):
            self.e_arrays = []
            for component in range(self.residual_dim()):
                e = np.zeros((self.n_batch, self.total_residuals), dtype=np.float32)
                for prob_idx in range(self.n_batch):
                    e[prob_idx, self.residual_offset + component] = 1.0
                self.e_arrays.append(wp.array(e.flatten(), dtype=wp.float32, device=self.device))

    def supports_analytic(self):
        return False

    def residual_dim(self):
        return self.n_chains * 3

    def set_weight(self, value):
        wp.launch(_update_weight, dim=1, inputs=[value], outputs=[self._weight], device=self.device)

    def set_targets(self, target_positions_np, active_mask_np):
        if self.target_positions is None or self.active_mask is None:
            return
        targets = np.asarray(target_positions_np, dtype=np.float32)
        mask = np.asarray(active_mask_np, dtype=np.float32)
        wp.copy(self.target_positions, wp.array(targets, dtype=wp.float32, device=self.device))
        wp.copy(self.active_mask, wp.array(mask, dtype=wp.float32, device=self.device))

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        wp.launch(
            _masked_position_residuals,
            dim=[body_q.shape[0], self.n_chains],
            inputs=[
                body_q,
                self.link_indices,
                self.target_positions,
                self.active_mask,
                self._weight,
                start_idx,
                problem_idx,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof):
        self._require_batch_layout()
        if self.e_arrays is None:
            return
        for component in range(self.residual_dim()):
            tape.backward(grads={tape.outputs[0]: self.e_arrays[component].flatten()})
            q_grad = tape.gradients[dq_dof]
            wp.launch(
                _autodiff_jac_fill,
                dim=self.n_batch,
                inputs=[q_grad, model.joint_dof_count, start_idx, component],
                outputs=[jacobian],
                device=self.device,
            )
            tape.zero()


@wp.kernel
def _point_segment_barrier_residuals(
    body_q: wp.array2d(dtype=wp.transform),
    point_indices: wp.array1d(dtype=wp.int32),
    segment_a_indices: wp.array1d(dtype=wp.int32),
    segment_b_indices: wp.array1d(dtype=wp.int32),
    active_mask: wp.array2d(dtype=wp.float32),
    clearance: wp.array1d(dtype=wp.float32),
    weight: wp.array1d(dtype=wp.float32),
    start_idx: int,
    problem_idx_map: wp.array1d(dtype=wp.int32),
    residuals: wp.array2d(dtype=wp.float32),
):
    row, pair_idx = wp.tid()
    base = problem_idx_map[row]

    p_tf = body_q[row, point_indices[pair_idx]]
    a_tf = body_q[row, segment_a_indices[pair_idx]]
    b_tf = body_q[row, segment_b_indices[pair_idx]]

    p = wp.vec3(p_tf[0], p_tf[1], p_tf[2])
    a = wp.vec3(a_tf[0], a_tf[1], a_tf[2])
    b = wp.vec3(b_tf[0], b_tf[1], b_tf[2])
    ab = b - a
    denom = wp.max(wp.dot(ab, ab), 1.0e-8)
    t = wp.clamp(wp.dot(p - a, ab) / denom, 0.0, 1.0)
    closest = a + ab * t
    distance = wp.length(p - closest)
    violation = wp.max(clearance[pair_idx] - distance, 0.0)
    residuals[row, start_idx + pair_idx] = (
        violation * active_mask[base, pair_idx] * weight[0]
    )


class IKPointSegmentDistanceBarrier(ik.IKObjective):
    """Masked point-to-segment clearance barrier.

    This is intentionally autodiff-only. It is used for low-weight contact
    proxies after an offline proxy/contact correlation gate has passed.
    """

    def __init__(
        self,
        point_indices,
        segment_a_indices,
        segment_b_indices,
        n_problems: int,
        clearance_m=0.035,
        weight=0.0,
    ):
        super().__init__()
        self.point_indices_np = np.asarray(point_indices, dtype=np.int32)
        self.segment_a_indices_np = np.asarray(segment_a_indices, dtype=np.int32)
        self.segment_b_indices_np = np.asarray(segment_b_indices, dtype=np.int32)
        self.n_pairs = int(len(self.point_indices_np))
        self.n_problems = int(n_problems)
        clearance = np.full((self.n_pairs,), float(clearance_m), dtype=np.float32)
        self.clearance_np = clearance
        self._weight = wp.array([weight], dtype=wp.float32)
        self.point_indices = None
        self.segment_a_indices = None
        self.segment_b_indices = None
        self.clearance = None
        self.active_mask = None
        self.e_arrays = None

    def init_buffers(self, model, jacobian_mode):
        self._require_batch_layout()
        self.point_indices = wp.array(self.point_indices_np, dtype=wp.int32, device=self.device)
        self.segment_a_indices = wp.array(self.segment_a_indices_np, dtype=wp.int32, device=self.device)
        self.segment_b_indices = wp.array(self.segment_b_indices_np, dtype=wp.int32, device=self.device)
        self.clearance = wp.array(self.clearance_np, dtype=wp.float32, device=self.device)
        self.active_mask = wp.zeros(
            shape=(self.n_problems, self.n_pairs),
            dtype=wp.float32,
            device=self.device,
        )
        if jacobian_mode in (IKJacobianType.AUTODIFF, IKJacobianType.MIXED):
            self.e_arrays = []
            for component in range(self.residual_dim()):
                e = np.zeros((self.n_batch, self.total_residuals), dtype=np.float32)
                for prob_idx in range(self.n_batch):
                    e[prob_idx, self.residual_offset + component] = 1.0
                self.e_arrays.append(wp.array(e.flatten(), dtype=wp.float32, device=self.device))

    def supports_analytic(self):
        return False

    def residual_dim(self):
        return self.n_pairs

    def set_weight(self, value):
        wp.launch(_update_weight, dim=1, inputs=[value], outputs=[self._weight], device=self.device)

    def set_active_mask(self, active_mask_np):
        if self.active_mask is None:
            return
        mask = np.asarray(active_mask_np, dtype=np.float32)
        wp.copy(self.active_mask, wp.array(mask, dtype=wp.float32, device=self.device))

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        wp.launch(
            _point_segment_barrier_residuals,
            dim=[body_q.shape[0], self.n_pairs],
            inputs=[
                body_q,
                self.point_indices,
                self.segment_a_indices,
                self.segment_b_indices,
                self.active_mask,
                self.clearance,
                self._weight,
                start_idx,
                problem_idx,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof):
        self._require_batch_layout()
        if self.e_arrays is None:
            return
        for component in range(self.residual_dim()):
            tape.backward(grads={tape.outputs[0]: self.e_arrays[component].flatten()})
            q_grad = tape.gradients[dq_dof]
            wp.launch(
                _autodiff_jac_fill,
                dim=self.n_batch,
                inputs=[q_grad, model.joint_dof_count, start_idx, component],
                outputs=[jacobian],
                device=self.device,
            )
            tape.zero()
