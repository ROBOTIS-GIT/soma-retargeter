# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AI Sapiens robot asset constants and path helpers."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

import soma_retargeter.assets.ai_sapiens_mjcf as ai_sapiens_mjcf
import soma_retargeter.utils.io_utils as io_utils


AI_SAPIENS_JOINT_NAMES: list[str] = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
]

AI_SAPIENS_HAND_TCP_LOCAL: dict[str, tuple[float, float, float]] = {
    "left": (0.123893, -0.006614, 0.010667),
    "right": (0.123893, 0.006411, 0.010664),
}

def resolve_ai_sapiens_mjcf_path(config_value: str | Path | None = None) -> Path:
    """Resolve the MJCF path for AI Sapiens.

    Resolution order:
      1. Explicit config value.
      2. ``SOMA_RETARGETER_AI_SAPIENS_MJCF_PATH``.
      3. ``KIMODO_AI_SAPIENS_MJCF_PATH`` for compatibility with existing local scripts.
      4. Generated ``configs/ai_sapiens/ai_sapiens_retarget.xml``.
      5. Packaged ``configs/ai_sapiens/ai_sapiens.xml`` for compatibility.
    """
    default_retarget_mjcf = ai_sapiens_mjcf.DEFAULT_OUTPUT_PATH
    default_retarget_rel = Path("ai_sapiens") / default_retarget_mjcf.name

    def _config_path(candidate: str | Path) -> Path:
        path = Path(candidate)
        if not path.is_absolute():
            path = io_utils.get_config_file(str(path))
        return path

    def _can_generate(candidate: str | Path, path: Path) -> bool:
        candidate_path = Path(candidate)
        if candidate_path.is_absolute():
            return False
        return candidate_path == default_retarget_rel and path == default_retarget_mjcf

    candidates: list[tuple[str | Path, bool]] = []
    if config_value:
        path = _config_path(config_value)
        candidates.append((config_value, _can_generate(config_value, path)))
    candidates.extend(
        (env_value, False)
        for env_value in (
            os.environ.get("SOMA_RETARGETER_AI_SAPIENS_MJCF_PATH"),
            os.environ.get("KIMODO_AI_SAPIENS_MJCF_PATH"),
        )
        if env_value
    )
    candidates.append((default_retarget_rel, True))
    candidates.append((io_utils.get_config_file("ai_sapiens", "ai_sapiens.xml"), False))

    for candidate, can_generate in candidates:
        path = _config_path(candidate)
        if path.exists():
            if can_generate:
                try:
                    ai_sapiens_mjcf.validate_retarget_mjcf(path, validate_newton=False)
                except ai_sapiens_mjcf.AiSapiensMJCFError as exc:
                    raise FileNotFoundError(f"[ERROR]: {exc}") from exc
            return path
        if can_generate:
            try:
                ai_sapiens_mjcf.ensure_default_retarget_mjcf()
            except ai_sapiens_mjcf.AiSapiensMJCFError as exc:
                raise FileNotFoundError(f"[ERROR]: {exc}") from exc
            if path.exists():
                return path

    formatted = ", ".join(str(candidate) for candidate, _ in candidates)
    raise FileNotFoundError(f"[ERROR]: AI Sapiens MJCF not found. Tried: {formatted}")


def apply_root_convention_to_rows(
    raw_data,
    *,
    root_translation_yaw_deg: float = 0.0,
    root_orientation_yaw_deg: float = 0.0,
    root_translation_xy_scale: float = 1.0,
) -> np.ndarray:
    """Return AI Sapiens anim rows with the configured world-root convention applied.

    Rows use the internal animation layout:
    ``root_pos(3), root_quat_xyzw(4), AI Sapiens joints``.
    """
    data = np.stack([np.asarray(row, dtype=np.float64) for row in raw_data], axis=0)
    expected_width = 7 + len(AI_SAPIENS_JOINT_NAMES)
    if data.ndim != 2 or data.shape[1] != expected_width:
        raise ValueError(
            f"AI Sapiens rows must have shape (T, {expected_width}), got {data.shape}"
        )

    out = data.copy()

    root_origin = out[0, 0:3].copy()
    root_translation = out[:, 0:3] - root_origin
    root_translation[:, 0:2] *= float(root_translation_xy_scale)
    if abs(float(root_translation_yaw_deg)) > 1e-12:
        root_translation = R.from_euler(
            "z", float(root_translation_yaw_deg), degrees=True
        ).apply(root_translation)
    out[:, 0:3] = root_origin + root_translation

    if abs(float(root_orientation_yaw_deg)) > 1e-12:
        yaw = R.from_euler("z", float(root_orientation_yaw_deg), degrees=True)
        out[:, 3:7] = (yaw * R.from_quat(out[:, 3:7])).as_quat()

    return out.astype(np.float32)


def apply_root_convention_to_buffer(
    buffer,
    *,
    root_translation_yaw_deg: float = 0.0,
    root_orientation_yaw_deg: float = 0.0,
    root_translation_xy_scale: float = 1.0,
):
    """Mutate a AI Sapiens CSVAnimationBuffer so saved CSV and diagnostics agree."""
    buffer.data = apply_root_convention_to_rows(
        buffer.data,
        root_translation_yaw_deg=root_translation_yaw_deg,
        root_orientation_yaw_deg=root_orientation_yaw_deg,
        root_translation_xy_scale=root_translation_xy_scale,
    )
    buffer.num_frames = int(buffer.data.shape[0])
    return buffer


def _rows_to_mujoco_qpos(rows: np.ndarray) -> np.ndarray:
    data = np.asarray(rows, dtype=np.float64)
    expected_width = 7 + len(AI_SAPIENS_JOINT_NAMES)
    if data.ndim != 2 or data.shape[1] != expected_width:
        raise ValueError(
            f"AI Sapiens rows must have shape (T, {expected_width}), got {data.shape}"
        )
    qpos = np.zeros_like(data, dtype=np.float64)
    qpos[:, 0:3] = data[:, 0:3]
    # CSVAnimationBuffer stores quaternions as xyzw; MuJoCo freejoint qpos is wxyz.
    qpos[:, 3:7] = data[:, [6, 3, 4, 5]]
    qpos[:, 7:] = data[:, 7:]
    return qpos


def compute_ground_alignment_summary(
    rows,
    *,
    mjcf_path: str | Path,
    ground_z: float = 0.0,
) -> dict:
    """Compute a root-z offset that keeps all non-plane geoms above ground."""
    import mujoco

    qpos = _rows_to_mujoco_qpos(
        np.stack([np.asarray(row, dtype=np.float64) for row in rows], axis=0)
    )
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    if qpos.shape[1] != model.nq:
        raise ValueError(f"AI Sapiens qpos width {qpos.shape[1]} != model.nq {model.nq}")

    data = mujoco.MjData(model)
    min_lower = float("inf")
    min_frame = -1
    min_geom = ""
    frame_min_lowers: list[float] = []
    for frame_idx, frame_qpos in enumerate(qpos):
        data.qpos[:] = frame_qpos
        mujoco.mj_forward(model, data)
        frame_min_lower = float("inf")
        for geom_idx in range(model.ngeom):
            if model.geom_type[geom_idx] == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            lower = float(data.geom_xpos[geom_idx, 2] - model.geom_rbound[geom_idx])
            frame_min_lower = min(frame_min_lower, lower)
            if lower < min_lower:
                min_lower = lower
                min_frame = frame_idx
                min_geom = (
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_idx)
                    or f"geom_{geom_idx}"
                )
        frame_min_lowers.append(frame_min_lower)

    offset = max(0.0, float(ground_z) - min_lower)
    before = np.asarray(frame_min_lowers, dtype=np.float64)
    after = before + float(offset)
    return {
        "enabled": True,
        "ground_z": float(ground_z),
        "min_geom_lower_before_m": float(min_lower),
        "root_z_offset_m": float(offset),
        "mesh_min_z_before_min_m": float(np.min(before)) if before.size else float("nan"),
        "mesh_min_z_before_mean_m": float(np.mean(before)) if before.size else float("nan"),
        "mesh_min_z_before_max_m": float(np.max(before)) if before.size else float("nan"),
        "mesh_min_z_after_min_m": float(np.min(after)) if after.size else float("nan"),
        "mesh_min_z_after_mean_m": float(np.mean(after)) if after.size else float("nan"),
        "mesh_min_z_after_max_m": float(np.max(after)) if after.size else float("nan"),
        "min_frame": int(min_frame),
        "min_geom": min_geom,
        "method": "all-frame non-plane geom_xpos.z - geom_rbound lower bound",
    }


def apply_ground_alignment_to_buffer(
    buffer,
    *,
    mjcf_path: str | Path,
    ground_z: float = 0.0,
) -> dict:
    """Mutate root z so the lowest non-plane geom over the sequence touches ground."""
    summary = compute_ground_alignment_summary(
        buffer.data,
        mjcf_path=mjcf_path,
        ground_z=ground_z,
    )
    offset = float(summary["root_z_offset_m"])
    if offset:
        data = np.stack([np.asarray(row, dtype=np.float64) for row in buffer.data], axis=0)
        data[:, 2] += offset
        buffer.data = data.astype(np.float32)
        buffer.num_frames = int(buffer.data.shape[0])
    return summary
