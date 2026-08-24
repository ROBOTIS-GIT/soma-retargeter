# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert SMPL-family motion files to Soma Retargeter SOMA77 NPZ files.

The conversion keeps NVIDIA SOMA-X's PoseInversion algorithm, but performs
source posing, pose inversion, target forward kinematics, coordinate
conversion, and export in one process. This avoids intermediate SOMA files
while retaining the SO(3) projection required for valid output rotations.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import pickle
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.spatial.transform import Rotation

from soma_retargeter.assets.kimodo_npz import joint_names as soma77_joint_names


SOMA_X_REQUIRED_VERSION = "0.2.1"
SOMA_X_HUMAN_MODEL_ENV = "SOMA_RETARGETER_SOMA_X_HUMAN_MODEL"
SOMA_X_SMPLX_MODEL_ENV = "SOMA_RETARGETER_SOMA_X_SMPLX_MODEL"
SOMA_X_MODEL_ENV = SOMA_X_SMPLX_MODEL_ENV
SUPPORTED_SMPL_MODEL_TYPES = ("smpl", "smplh", "smplx")
SMPL_MODEL_LABELS = {
    "smpl": "SMPL",
    "smplh": "SMPL-H",
    "smplx": "SMPL-X",
}


C_AMASS_TO_KIMODO = np.array(
    [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
    dtype=np.float32,
)

_SOMA77_HEADING_LANDMARKS = {
    "left_shoulder": 11,
    "right_shoulder": 39,
    "left_hip": 67,
    "right_hip": 72,
}

_WORLD_POSITION_KEYS = (
    "posed_joints",
    "root_positions",
    "smooth_root_pos",
    "world_root_position",
    "root_translation",
    "transl",
    "trans",
)


@dataclass(frozen=True)
class HumanModelInfo:
    """Structurally detected licensed SMPL-family model metadata."""

    path: Path
    model_type: str
    display_name: str
    vertex_count: int
    joint_count: int
    shape_coefficient_count: int


@dataclass(frozen=True)
class SMPLMotion:
    """Normalized SMPL-family animation fields."""

    model_type: str
    frame_count: int
    global_orient: np.ndarray
    body_pose: np.ndarray
    left_hand_pose: np.ndarray
    right_hand_pose: np.ndarray
    jaw_pose: np.ndarray
    leye_pose: np.ndarray
    reye_pose: np.ndarray
    expression: np.ndarray
    transl: np.ndarray
    betas: np.ndarray
    gender: str
    fps: float
    has_expression: bool
    source_coordinate: str
    transl_is_root_position: bool


# Public compatibility name retained for existing imports.
SMPLXMotion = SMPLMotion


@dataclass(frozen=True)
class ConversionMetrics:
    """Timing and fitting metrics from a conversion run."""

    frame_count: int
    fps: float
    initialization_seconds: float
    conversion_seconds: float
    coordinate_seconds: float
    total_seconds: float
    conversion_frames_per_second: float
    mean_vertex_error_m: float
    median_vertex_error_m: float
    max_vertex_error_m: float


@dataclass(frozen=True)
class SomaXDependencyStatus:
    """Availability of the optional SOMA-X runtime."""

    available: bool
    version: str | None
    reason: str


class SomaXDependencyError(RuntimeError):
    """Raised when a requested SOMA-X conversion cannot start."""


def probe_soma_x_dependencies() -> SomaXDependencyStatus:
    """Check the optional runtime without importing Torch or SOMA-X."""

    try:
        version = importlib.metadata.version("py-soma-x")
    except importlib.metadata.PackageNotFoundError:
        return SomaXDependencyStatus(False, None, "py-soma-x is not installed")

    if version != SOMA_X_REQUIRED_VERSION:
        return SomaXDependencyStatus(
            False,
            version,
            f"py-soma-x {version} is installed; required version is {SOMA_X_REQUIRED_VERSION}",
        )

    missing = []
    for module_name in ("soma", "smplx", "torch"):
        try:
            found = importlib.util.find_spec(module_name)
        except (ImportError, ModuleNotFoundError, ValueError):
            found = None
        if found is None:
            missing.append(module_name)
    if missing:
        return SomaXDependencyStatus(
            False,
            version,
            "missing Python modules: " + ", ".join(missing),
        )
    return SomaXDependencyStatus(True, version, "available")


def soma_x_install_command() -> str:
    return f"{sys.executable} -m pip install -e '.[soma-x]'"


def require_soma_x_dependencies() -> SomaXDependencyStatus:
    status = probe_soma_x_dependencies()
    if not status.available:
        raise SomaXDependencyError(
            f"SOMA-X conversion is unavailable: {status.reason}. "
            f"Install it with `{soma_x_install_command()}`."
        )
    return status


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def normalize_model_type(value: str | None) -> str:
    model_type = str(value or "auto").strip().lower().replace("-", "")
    if model_type == "auto":
        return model_type
    if model_type not in SUPPORTED_SMPL_MODEL_TYPES:
        supported = ", ".join(("auto", *SUPPORTED_SMPL_MODEL_TYPES))
        raise ValueError(f"Unsupported human model type {value!r}; expected {supported}")
    return model_type


def resolve_human_model_path(
    explicit: str | Path | None = None,
    config: dict[str, Any] | None = None,
    *,
    model_type: str = "auto",
) -> Path | None:
    """Resolve CLI/session, environment, then config model path precedence."""

    value = explicit
    if value is None or str(value).strip() == "":
        value = os.environ.get(SOMA_X_HUMAN_MODEL_ENV)
    if (value is None or str(value).strip() == "") and config is not None:
        value = config.get("soma_x_human_model")
    if value is None or str(value).strip() == "":
        requested_type = normalize_model_type(model_type)
        if requested_type in {"auto", "smplx"}:
            value = os.environ.get(SOMA_X_SMPLX_MODEL_ENV)
            if (value is None or str(value).strip() == "") and config is not None:
                value = config.get("soma_x_smplx_model")
    if value is None or str(value).strip() == "":
        return None

    candidate = Path(value).expanduser()
    if candidate.is_absolute() or candidate.exists():
        return candidate.resolve()
    return (_repo_root() / candidate).resolve()


def resolve_smplx_model_path(
    explicit: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> Path | None:
    """Compatibility resolver for existing SMPL-X-only callers."""

    return resolve_human_model_path(explicit, config, model_type="smplx")


def _model_data_mapping(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            return {key: np.asarray(data[key]) for key in data.files}
    if suffix == ".pkl":
        with path.open("rb") as stream:
            try:
                data = pickle.load(stream, encoding="latin1")
            except TypeError:
                stream.seek(0)
                data = pickle.load(stream)
        if isinstance(data, dict):
            return data
        if hasattr(data, "__dict__"):
            return vars(data)
        raise ValueError(f"Unsupported model pickle payload: {type(data).__name__}")
    raise ValueError(f"Human model must be .npz or .pkl: {path}")


def _array_shape(value: Any) -> tuple[int, ...]:
    if hasattr(value, "r"):
        value = value.r
    return tuple(np.asarray(value).shape)


def inspect_human_model(
    path: str | Path,
    requested_type: str = "auto",
) -> HumanModelInfo:
    """Detect an SMPL-family model from topology and parameter fields."""

    model_path = Path(path).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    data = _model_data_mapping(model_path)
    if "v_template" not in data or "shapedirs" not in data:
        raise ValueError(
            f"{model_path} is not a usable SMPL-family model: "
            "v_template and shapedirs are required"
        )

    vertex_shape = _array_shape(data["v_template"])
    if len(vertex_shape) != 2 or vertex_shape[1] != 3:
        raise ValueError(f"Invalid v_template shape in {model_path}: {vertex_shape}")
    vertex_count = int(vertex_shape[0])

    kintree = data.get("kintree_table")
    if kintree is None:
        raise ValueError(f"{model_path} has no kintree_table")
    kintree_shape = _array_shape(kintree)
    if len(kintree_shape) != 2:
        raise ValueError(f"Invalid kintree_table shape in {model_path}: {kintree_shape}")
    joint_count = int(kintree_shape[1] if kintree_shape[0] == 2 else kintree_shape[0])

    shapedirs_shape = _array_shape(data["shapedirs"])
    if len(shapedirs_shape) == 3:
        shape_coefficient_count = int(shapedirs_shape[-1])
    elif len(shapedirs_shape) == 2:
        shape_coefficient_count = int(shapedirs_shape[-1])
    else:
        raise ValueError(f"Invalid shapedirs shape in {model_path}: {shapedirs_shape}")

    keys = set(data)
    has_hands = bool(
        keys
        & {
            "hands_componentsl",
            "hands_componentsr",
            "hands_meanl",
            "hands_meanr",
        }
    )
    has_face = bool(
        keys
        & {
            "lmk_faces_idx",
            "lmk_bary_coords",
            "dynamic_lmk_faces_idx",
            "dynamic_lmk_bary_coords",
            "expr_dirs",
        }
    )

    if vertex_count == 10475 or has_face:
        detected = "smplx"
    elif vertex_count == 6890 and (has_hands or joint_count > 24):
        detected = "smplh"
    elif vertex_count == 6890 and joint_count == 24:
        detected = "smpl"
    else:
        raise ValueError(
            f"Could not identify SMPL family for {model_path}: "
            f"vertices={vertex_count}, joints={joint_count}, "
            f"hand_fields={has_hands}, face_fields={has_face}"
        )

    requested = normalize_model_type(requested_type)
    if requested != "auto" and requested != detected:
        raise ValueError(
            f"Human model type mismatch: requested {SMPL_MODEL_LABELS[requested]}, "
            f"but {model_path} is {SMPL_MODEL_LABELS[detected]}"
        )
    return HumanModelInfo(
        path=model_path,
        model_type=detected,
        display_name=SMPL_MODEL_LABELS[detected],
        vertex_count=vertex_count,
        joint_count=joint_count,
        shape_coefficient_count=shape_coefficient_count,
    )


def validate_human_model(
    path: str | Path,
    requested_type: str = "auto",
) -> HumanModelInfo:
    """Structurally inspect and instantiate the matching SMPL class on CPU."""

    info = inspect_human_model(path, requested_type)
    require_soma_x_dependencies()
    try:
        import smplx
    except (ImportError, ModuleNotFoundError) as exc:
        raise SomaXDependencyError(
            f"Could not import smplx while validating {info.path}: {exc}"
        ) from exc
    source_model = _create_source_model(
        smplx,
        info,
        gender="neutral",
        num_betas=min(10, info.shape_coefficient_count),
        num_expression_coeffs=10,
        flat_hand_mean=True,
    )
    del source_model
    return info


def resolve_soma_x_device(requested: str | None, torch_module: Any) -> str:
    device_name = str(requested or "auto").strip().lower()
    if device_name == "auto":
        return "cuda:0" if torch_module.cuda.is_available() else "cpu"
    if device_name == "cuda":
        device_name = "cuda:0"
    if device_name.startswith("cuda") and not torch_module.cuda.is_available():
        raise RuntimeError(
            f"SOMA-X device {device_name!r} was requested but Torch cannot access CUDA"
        )
    return device_name


def build_conversion_signature(
    input_path: str | Path,
    model_path: str | Path,
    options: dict[str, Any],
) -> str:
    """Hash source/model identity and conversion options for safe resume/cache."""

    source = Path(input_path).expanduser().resolve()
    model = Path(model_path).expanduser().resolve()
    source_stat = source.stat()
    model_stat = model.stat()
    payload = {
        "schema": 1,
        "soma_x_version": SOMA_X_REQUIRED_VERSION,
        "source": str(source),
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "model": str(model),
        "model_size": model_stat.st_size,
        "model_mtime_ns": model_stat.st_mtime_ns,
        "options": options,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def temp_retarget_npz_path_for_smpl(
    input_path: str | Path,
    model_path: str | Path,
    options: dict[str, Any],
) -> Path:
    source = Path(input_path).expanduser().resolve()
    digest = build_conversion_signature(source, model_path, options)[:16]
    cache_root = Path(tempfile.gettempdir()) / "soma_retargeter_soma_x"
    return cache_root / f"{source.stem}_{digest}.npz"


def temp_retarget_npz_path_for_smplx(
    input_path: str | Path,
    model_path: str | Path,
    options: dict[str, Any],
) -> Path:
    """Compatibility cache path helper for existing SMPL-X callers."""

    return temp_retarget_npz_path_for_smpl(input_path, model_path, options)


SMPLRuntimeCache = dict[
    tuple[str, str, str, int, bool, str],
    tuple[Any, Any, Any],
]
SMPLXRuntimeCache = SMPLRuntimeCache


def _first_array(data: np.lib.npyio.NpzFile, *keys: str) -> np.ndarray | None:
    for key in keys:
        if key in data.files:
            return np.asarray(data[key])
    return None


def _scalar_string(value: Any, default: str) -> str:
    if value is None:
        return default
    array = np.asarray(value)
    if array.shape == ():
        value = array.item()
    elif array.size == 1:
        value = array.reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)


def _normalize_pose_field(
    value: np.ndarray | None,
    frame_count: int,
    width: int,
    name: str,
) -> np.ndarray:
    if value is None:
        return np.zeros((frame_count, width), dtype=np.float32)

    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim == 3 and array.shape[-1] == 3:
        array = array.reshape(array.shape[0], -1)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(
            f"{name} must have shape (T, {width}) or (T, J, 3), got {array.shape}. "
            "Full axis-angle parameters are required; hand PCA is unsupported."
        )
    if array.shape[0] == 1 and frame_count > 1:
        array = np.repeat(array, frame_count, axis=0)
    if array.shape[0] != frame_count:
        raise ValueError(f"{name} has {array.shape[0]} frames; expected {frame_count}")
    return np.ascontiguousarray(array, dtype=np.float32)


def _infer_frame_count(data: np.lib.npyio.NpzFile) -> int:
    for key in (
        "poses",
        "global_orient",
        "root_orient",
        "body_pose",
        "body_pose_axis",
        "pose_body",
        "transl",
        "trans",
        "left_hand_pose",
        "pose_hand",
    ):
        value = _first_array(data, key)
        if value is not None:
            return 1 if value.ndim == 1 else int(value.shape[0])
    raise ValueError("Could not infer frame count from the SMPL-family NPZ")


def _empty_pose_fields(frame_count: int) -> dict[str, np.ndarray]:
    return {
        "left_hand_pose": np.zeros((frame_count, 45), dtype=np.float32),
        "right_hand_pose": np.zeros((frame_count, 45), dtype=np.float32),
        "jaw_pose": np.zeros((frame_count, 3), dtype=np.float32),
        "leye_pose": np.zeros((frame_count, 3), dtype=np.float32),
        "reye_pose": np.zeros((frame_count, 3), dtype=np.float32),
    }


def _split_combined_poses(
    poses: np.ndarray,
    model_type: str = "smplx",
) -> dict[str, np.ndarray]:
    frame_count, width = poses.shape
    model_type = normalize_model_type(model_type)
    if model_type == "auto":
        raise ValueError("A concrete model type is required to parse combined poses")
    empty = _empty_pose_fields(frame_count)
    if model_type == "smpl":
        if width != 72:
            raise ValueError(
                f"SMPL poses must have shape (T, 72), got {poses.shape}"
            )
        return {
            "global_orient": poses[:, 0:3],
            "body_pose": poses[:, 3:72],
            **empty,
        }
    if model_type == "smplh":
        if width == 156:
            return {
                "global_orient": poses[:, 0:3],
                "body_pose": poses[:, 3:66],
                **empty,
                "left_hand_pose": poses[:, 66:111],
                "right_hand_pose": poses[:, 111:156],
            }
        if width == 66:
            return {
                "global_orient": poses[:, 0:3],
                "body_pose": poses[:, 3:66],
                **empty,
            }
        raise ValueError(
            f"SMPL-H poses must have shape (T, 156) or (T, 66), got {poses.shape}"
        )

    if width == 165:
        return {
            "global_orient": poses[:, 0:3],
            "body_pose": poses[:, 3:66],
            "jaw_pose": poses[:, 66:69],
            "leye_pose": poses[:, 69:72],
            "reye_pose": poses[:, 72:75],
            "left_hand_pose": poses[:, 75:120],
            "right_hand_pose": poses[:, 120:165],
        }
    if width == 156:
        return {
            "global_orient": poses[:, 0:3],
            "body_pose": poses[:, 3:66],
            **empty,
            "left_hand_pose": poses[:, 66:111],
            "right_hand_pose": poses[:, 111:156],
        }
    if width == 66:
        return {
            "global_orient": poses[:, 0:3],
            "body_pose": poses[:, 3:66],
            **empty,
        }
    raise ValueError(
        f"Unsupported poses shape {poses.shape}; expected (T, 165), "
        "(T, 156), or (T, 66)"
    )


def _load_local_rotation_matrices(data: np.lib.npyio.NpzFile) -> np.ndarray:
    local_rot_mats = np.asarray(data["local_rot_mats"], dtype=np.float64)
    if local_rot_mats.ndim == 5:
        if local_rot_mats.shape[0] != 1:
            raise ValueError(
                "local_rot_mats batch dimension must be 1, got "
                f"{local_rot_mats.shape[0]}"
            )
        local_rot_mats = local_rot_mats[0]
    if local_rot_mats.ndim != 4 or local_rot_mats.shape[-2:] != (3, 3):
        raise ValueError(
            "local_rot_mats must have shape (T, J, 3, 3), got "
            f"{local_rot_mats.shape}"
        )
    if local_rot_mats.shape[0] == 0:
        raise ValueError("local_rot_mats must contain at least one frame")
    if not np.all(np.isfinite(local_rot_mats)):
        raise ValueError("local_rot_mats contains non-finite values")
    return local_rot_mats


def _split_kimodo_smplx_body_rotations(
    local_rot_mats: np.ndarray,
) -> dict[str, np.ndarray]:
    """Convert Kimodo's root+21 SMPL-X body matrices to axis-angle fields."""

    frame_count, joint_count = local_rot_mats.shape[:2]
    if joint_count != 22:
        raise ValueError(
            "Kimodo SMPL-X matrix input requires root plus 21 body joints; "
            f"got {joint_count} joints"
        )
    determinants = np.linalg.det(local_rot_mats)
    if np.any(determinants <= 0.0):
        raise ValueError("local_rot_mats contains reflection or singular matrices")
    rotvec = Rotation.from_matrix(local_rot_mats.reshape(-1, 3, 3)).as_rotvec()
    rotvec = rotvec.reshape(frame_count, joint_count, 3).astype(np.float32)
    zeros3 = np.zeros((frame_count, 3), dtype=np.float32)
    zeros45 = np.zeros((frame_count, 45), dtype=np.float32)
    return {
        "global_orient": np.ascontiguousarray(rotvec[:, 0]),
        "body_pose": np.ascontiguousarray(rotvec[:, 1:].reshape(frame_count, 63)),
        "left_hand_pose": zeros45.copy(),
        "right_hand_pose": zeros45.copy(),
        "jaw_pose": zeros3.copy(),
        "leye_pose": zeros3.copy(),
        "reye_pose": zeros3.copy(),
    }


def _kimodo_smplx_root_positions(
    data: np.lib.npyio.NpzFile,
    frame_count: int,
) -> np.ndarray:
    root_positions = _first_array(data, "root_positions")
    if root_positions is None:
        posed_joints = _first_array(data, "posed_joints")
        if posed_joints is None:
            raise ValueError(
                "Kimodo SMPL-X matrix input requires root_positions or "
                "posed_joints"
            )
        posed_joints = np.asarray(posed_joints, dtype=np.float32)
        if posed_joints.ndim == 4 and posed_joints.shape[0] == 1:
            posed_joints = posed_joints[0]
        if posed_joints.ndim != 3 or posed_joints.shape[0] != frame_count:
            raise ValueError(
                "posed_joints must have shape (T, J, 3), got "
                f"{posed_joints.shape}"
            )
        root_positions = posed_joints[:, 0]
    else:
        root_positions = np.asarray(root_positions)
        if root_positions.ndim == 3 and root_positions.shape[0] == 1:
            root_positions = root_positions[0]
    return _normalize_pose_field(
        root_positions, frame_count, 3, "root_positions"
    )


def _normalize_expression_field(
    value: np.ndarray | None,
    frame_count: int,
) -> np.ndarray:
    if value is None:
        return np.zeros((frame_count, 10), dtype=np.float32)
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError(f"expression must have shape (T, E), got {array.shape}")
    if array.shape[0] == 1 and frame_count > 1:
        array = np.repeat(array, frame_count, axis=0)
    if array.shape[0] != frame_count:
        raise ValueError(
            f"expression has {array.shape[0]} frames; expected {frame_count}"
        )
    return np.ascontiguousarray(array, dtype=np.float32)


def load_smpl_motion(
    path: str | Path,
    *,
    model_type: str,
    fps_override: float | None = None,
) -> SMPLMotion:
    """Load parameters matching one concrete SMPL-family model type."""

    input_path = Path(path).expanduser()
    model_type = normalize_model_type(model_type)
    if model_type == "auto":
        raise ValueError(
            "model_type='auto' requires a selected human model before loading motion"
        )
    with np.load(input_path, allow_pickle=True) as data:
        source_coordinate = "amass"
        transl_is_root_position = False
        transl_value = None
        if "local_rot_mats" in data.files:
            if model_type != "smplx":
                raise ValueError(
                    "Kimodo 22-joint matrix input is only compatible with SMPL-X"
                )
            local_rot_mats = _load_local_rotation_matrices(data)
            frame_count, joint_count = local_rot_mats.shape[:2]
            if joint_count == 77:
                raise ValueError(
                    f"{input_path} is already a Soma Retargeter SOMA77 NPZ. "
                    "Load it directly with the GUI NPZ Motion button."
                )
            if joint_count != 22:
                raise ValueError(
                    f"Unsupported local_rot_mats joint count {joint_count}; "
                    "expected 22 for Kimodo SMPL-X input or 77 for an "
                    "already converted Soma Retargeter motion"
                )
            fields = _split_kimodo_smplx_body_rotations(local_rot_mats)
            direct_transl = _first_array(data, "transl", "trans")
            if direct_transl is None:
                transl_value = _kimodo_smplx_root_positions(data, frame_count)
                transl_is_root_position = True
            else:
                direct_transl = np.asarray(direct_transl)
                if direct_transl.ndim == 3 and direct_transl.shape[0] == 1:
                    direct_transl = direct_transl[0]
                transl_value = direct_transl
            source_coordinate = "kimodo"
        else:
            frame_count = _infer_frame_count(data)
            combined = _first_array(data, "poses")
            if combined is not None:
                poses = np.asarray(combined, dtype=np.float32)
                if poses.ndim == 3 and poses.shape[1:] == (77, 3):
                    raise ValueError(
                        f"{input_path} looks like a SOMA-X intermediate NPZ "
                        "(poses shape is (T, 77, 3)). Convert the original "
                        "SMPL-X/Stage-II motion instead, or use the existing "
                        "assets/motions SOMA-X-to-Kimodo pipeline explicitly."
                    )
                if poses.ndim == 1:
                    poses = poses.reshape(1, -1)
                elif poses.ndim == 3 and poses.shape[-1] == 3:
                    poses = poses.reshape(poses.shape[0], -1)
                if poses.ndim != 2 or poses.shape[0] != frame_count:
                    raise ValueError(f"Invalid poses shape: {poses.shape}")
                fields = _split_combined_poses(
                    np.ascontiguousarray(poses), model_type
                )
            else:
                pose_hand = _first_array(data, "pose_hand")
                left_hand = _first_array(data, "left_hand_pose")
                right_hand = _first_array(data, "right_hand_pose")
                if pose_hand is not None and left_hand is None and right_hand is None:
                    pose_hand = _normalize_pose_field(
                        pose_hand, frame_count, 90, "pose_hand"
                    )
                    left_hand = pose_hand[:, :45]
                    right_hand = pose_hand[:, 45:]

                pose_eye = _first_array(data, "pose_eye")
                left_eye = _first_array(data, "leye_pose", "left_eye_pose")
                right_eye = _first_array(data, "reye_pose", "right_eye_pose")
                if pose_eye is not None and left_eye is None and right_eye is None:
                    pose_eye = _normalize_pose_field(
                        pose_eye, frame_count, 6, "pose_eye"
                    )
                    left_eye = pose_eye[:, :3]
                    right_eye = pose_eye[:, 3:]

                body_width = 69 if model_type == "smpl" else 63
                has_hand_fields = any(
                    value is not None
                    for value in (pose_hand, left_hand, right_hand)
                )
                has_face_fields = any(
                    value is not None
                    for value in (
                        pose_eye,
                        left_eye,
                        right_eye,
                        _first_array(data, "jaw_pose", "pose_jaw"),
                    )
                )
                if model_type == "smpl" and has_hand_fields:
                    raise ValueError(
                        "SMPL motion cannot contain SMPL-H/SMPL-X hand pose fields"
                    )
                if model_type != "smplx" and has_face_fields:
                    raise ValueError(
                        f"{SMPL_MODEL_LABELS[model_type]} motion cannot contain "
                        "SMPL-X face pose fields"
                    )
                fields = {
                    "global_orient": _normalize_pose_field(
                        _first_array(data, "global_orient", "root_orient"),
                        frame_count,
                        3,
                        "global_orient",
                    ),
                    "body_pose": _normalize_pose_field(
                        _first_array(data, "body_pose", "body_pose_axis", "pose_body"),
                        frame_count,
                        body_width,
                        "body_pose",
                    ),
                    "left_hand_pose": _normalize_pose_field(
                        left_hand, frame_count, 45, "left_hand_pose"
                    ),
                    "right_hand_pose": _normalize_pose_field(
                        right_hand, frame_count, 45, "right_hand_pose"
                    ),
                    "jaw_pose": _normalize_pose_field(
                        _first_array(data, "jaw_pose", "pose_jaw"),
                        frame_count,
                        3,
                        "jaw_pose",
                    ),
                    "leye_pose": _normalize_pose_field(
                        left_eye, frame_count, 3, "leye_pose"
                    ),
                    "reye_pose": _normalize_pose_field(
                        right_eye, frame_count, 3, "reye_pose"
                    ),
                }

        transl = _normalize_pose_field(
            transl_value
            if transl_value is not None
            else _first_array(data, "transl", "trans"),
            frame_count,
            3,
            "transl",
        )
        betas_value = _first_array(data, "betas", "beta")
        if betas_value is None:
            betas = np.zeros((1, 10), dtype=np.float32)
        else:
            betas = np.asarray(betas_value, dtype=np.float32)
            if betas.ndim == 1:
                betas = betas.reshape(1, -1)
            if betas.ndim != 2:
                raise ValueError(f"betas must be 1D or 2D, got {betas.shape}")
            if betas.shape[0] == frame_count:
                variation = float(np.max(np.abs(betas - betas[:1])))
                if variation > 1e-5:
                    raise ValueError(
                        "Per-frame betas vary; one motion requires one identity"
                    )
                betas = betas[:1]
            elif betas.shape[0] != 1:
                raise ValueError(
                    f"betas has {betas.shape[0]} rows; expected 1 or {frame_count}"
                )
            betas = np.ascontiguousarray(betas, dtype=np.float32)

        gender = _scalar_string(_first_array(data, "gender"), "neutral").lower()
        if gender not in {"neutral", "male", "female"}:
            gender = "neutral"
        fps_value = _first_array(
            data,
            "mocap_frame_rate",
            "mocap_framerate",
            "fps",
            "sample_rate",
            "frame_rate",
            "framerate",
            "source_fps",
        )
        if fps_override is not None:
            fps = float(fps_override)
        elif fps_value is not None:
            fps = float(np.asarray(fps_value).reshape(-1)[0])
        else:
            fps = 30.0
        if not math.isfinite(fps) or fps <= 0.0:
            raise ValueError(f"FPS must be finite and positive, got {fps}")
        expression_value = _first_array(data, "expression", "expr")
        if model_type != "smplx" and expression_value is not None:
            raise ValueError(
                f"{SMPL_MODEL_LABELS[model_type]} motion cannot contain expression fields"
            )
        expression = _normalize_expression_field(
            expression_value if model_type == "smplx" else None,
            frame_count,
        )
        has_expression = expression_value is not None

    return SMPLMotion(
        model_type=model_type,
        frame_count=frame_count,
        global_orient=fields["global_orient"],
        body_pose=fields["body_pose"],
        left_hand_pose=fields["left_hand_pose"],
        right_hand_pose=fields["right_hand_pose"],
        jaw_pose=fields["jaw_pose"],
        leye_pose=fields["leye_pose"],
        reye_pose=fields["reye_pose"],
        expression=expression,
        transl=transl,
        betas=betas,
        gender=gender,
        fps=fps,
        has_expression=has_expression,
        source_coordinate=source_coordinate,
        transl_is_root_position=transl_is_root_position,
    )


def load_smplx_motion(
    path: str | Path,
    *,
    fps_override: float | None = None,
) -> SMPLMotion:
    """Compatibility loader for existing SMPL-X-only callers."""

    return load_smpl_motion(
        path,
        model_type="smplx",
        fps_override=fps_override,
    )


def resolve_source_coordinate(requested: str, motion: SMPLMotion) -> str:
    """Resolve an explicit coordinate override or use the input schema."""

    if requested == "auto":
        return motion.source_coordinate
    if requested not in {"amass", "kimodo"}:
        raise ValueError(f"Unsupported source coordinate: {requested}")
    return requested


def yaw_rotation_y(degrees: float) -> np.ndarray:
    radians = math.radians(float(degrees))
    cosine = math.cos(radians)
    sine = math.sin(radians)
    return np.array(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float32,
    )


def compute_anatomical_heading(posed_joints: np.ndarray) -> np.ndarray:
    """Return frame-zero body forward in the Kimodo horizontal XZ plane.

    The SOMA root rotation is not a reliable anatomical heading after pose
    inversion. Shoulder and hip landmarks retain the body's left/right axis,
    so crossing that axis with Kimodo +Y gives a pose-independent forward
    direction. Both landmark pairs are used when available to reduce noise.
    """

    joints = np.asarray(posed_joints, dtype=np.float32)
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(
            f"posed_joints must have shape (T, J, 3), got {joints.shape}"
        )
    if joints.shape[0] == 0:
        raise ValueError("posed_joints must contain at least one frame")
    required_joint_count = max(_SOMA77_HEADING_LANDMARKS.values()) + 1
    if joints.shape[1] < required_joint_count:
        raise ValueError(
            "posed_joints does not contain the SOMA77 shoulder/hip landmarks: "
            f"got {joints.shape[1]} joints, need {required_joint_count}"
        )

    frame = joints[0]
    lateral_vectors = []
    for left_name, right_name in (
        ("left_shoulder", "right_shoulder"),
        ("left_hip", "right_hip"),
    ):
        lateral = (
            frame[_SOMA77_HEADING_LANDMARKS[left_name]]
            - frame[_SOMA77_HEADING_LANDMARKS[right_name]]
        )
        lateral[1] = 0.0
        norm = float(np.linalg.norm(lateral))
        if norm > 1e-8:
            lateral_vectors.append(lateral / norm)

    if not lateral_vectors:
        raise ValueError(
            "Could not determine frame-zero heading because shoulder and hip "
            "landmarks have no horizontal separation"
        )

    lateral = np.sum(lateral_vectors, axis=0)
    lateral_norm = float(np.linalg.norm(lateral))
    if lateral_norm <= 1e-8:
        raise ValueError(
            "Could not determine frame-zero heading because shoulder and hip "
            "lateral axes cancel each other"
        )
    lateral /= lateral_norm

    forward = np.cross(
        lateral,
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )
    forward[1] = 0.0
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm <= 1e-8:
        raise ValueError("Could not determine a horizontal anatomical heading")
    return (forward / forward_norm).astype(np.float32)


def canonical_heading_yaw_degrees(posed_joints: np.ndarray) -> float:
    """Return the Kimodo +Y yaw that maps frame-zero body forward to +Z."""

    forward = compute_anatomical_heading(posed_joints)
    return math.degrees(-math.atan2(float(forward[0]), float(forward[2])))


def compute_root_heading(global_rot_mats: np.ndarray) -> np.ndarray:
    root_rotation = np.asarray(global_rot_mats, dtype=np.float32)[:, 0]
    forward = np.einsum(
        "tij,j->ti",
        root_rotation,
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )
    heading = forward[:, [0, 2]]
    norm = np.linalg.norm(heading, axis=-1, keepdims=True)
    valid = norm[:, 0] > 1e-8
    heading[valid] /= norm[valid]
    heading[~valid] = np.array([0.0, 1.0], dtype=np.float32)
    return heading.astype(np.float32)


def transform_to_kimodo_frame(
    local_rot_mats: np.ndarray,
    global_rot_mats: np.ndarray,
    position_arrays: dict[str, np.ndarray],
    *,
    source_coordinate: str,
    heading_yaw_degrees: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Apply one vectorized world-frame conversion to all motion fields."""

    local = np.asarray(local_rot_mats, dtype=np.float32).copy()
    global_mats = np.asarray(global_rot_mats, dtype=np.float32).copy()
    positions = {
        key: np.asarray(value, dtype=np.float32).copy()
        for key, value in position_arrays.items()
    }
    if source_coordinate == "amass":
        local[:, 0] = np.einsum(
            "ab,tbc->tac", C_AMASS_TO_KIMODO, local[:, 0]
        )
        global_mats = np.einsum(
            "ab,...bc->...ac", C_AMASS_TO_KIMODO, global_mats
        )
        for key, value in positions.items():
            positions[key] = np.matmul(
                value, C_AMASS_TO_KIMODO.T
            ).astype(np.float32)
    elif source_coordinate != "kimodo":
        raise ValueError(f"Unsupported source coordinate: {source_coordinate}")

    if not math.isclose(float(heading_yaw_degrees), 0.0, abs_tol=1e-12):
        yaw = yaw_rotation_y(heading_yaw_degrees)
        pivot = positions["root_positions"][0].copy()
        local[:, 0] = np.einsum("ab,tbc->tac", yaw, local[:, 0])
        global_mats = np.einsum("ab,...bc->...ac", yaw, global_mats)
        for key, value in positions.items():
            pivot_shape = (1,) * (value.ndim - 1) + (3,)
            pivot_view = pivot.reshape(pivot_shape)
            positions[key] = (
                (value - pivot_view) @ yaw.T + pivot_view
            ).astype(np.float32)
    return local, global_mats.astype(np.float32), positions


def rebase_root_horizontal_positions(
    position_arrays: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Move frame-zero root X/Z to the Kimodo horizontal origin.

    Kimodo uses Y as up, so the source height and every relative trajectory
    delta are retained. The same translation is applied to every world-space
    position field to keep the exported NPZ internally consistent.
    """

    if "root_positions" not in position_arrays:
        raise KeyError("root_positions is required for horizontal rebasing")

    root_positions = np.asarray(
        position_arrays["root_positions"], dtype=np.float32
    )
    if (
        root_positions.ndim != 2
        or root_positions.shape[0] == 0
        or root_positions.shape[1] != 3
    ):
        raise ValueError(
            "root_positions must have non-empty shape (frames, 3), got "
            f"{root_positions.shape}"
        )
    if not np.all(np.isfinite(root_positions[0])):
        raise ValueError("frame-zero root position contains non-finite values")

    offset = np.array(
        [root_positions[0, 0], 0.0, root_positions[0, 2]],
        dtype=np.float32,
    )
    rebased: dict[str, np.ndarray] = {}
    for key, value in position_arrays.items():
        array = np.asarray(value, dtype=np.float32)
        if array.shape[0] != root_positions.shape[0] or array.shape[-1] != 3:
            raise ValueError(
                f"{key} must have one 3D position per frame, got {array.shape}"
            )
        offset_shape = (1,) * (array.ndim - 1) + (3,)
        rebased[key] = (
            array - offset.reshape(offset_shape)
        ).astype(np.float32)
    return rebased, offset


def _runtime_imports() -> tuple[Any, Any, Any, Any, Any, Any]:
    require_soma_x_dependencies()
    try:
        import smplx
        import torch
        from soma import SOMALayer
        from soma.geometry.rig_utils import remove_joint_orient_local
        from soma.geometry.transforms import matrix_to_rotvec
        from soma.pose_inversion import PoseInversion
    except (ImportError, ModuleNotFoundError) as exc:
        raise SomaXDependencyError(
            "SOMA-X conversion dependencies could not be imported after "
            f"availability checks: {exc}. Install with "
            f"`{soma_x_install_command()}`."
        ) from exc
    return (
        smplx,
        torch,
        SOMALayer,
        remove_joint_orient_local,
        matrix_to_rotvec,
        PoseInversion,
    )


def _synchronize(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _create_source_model(
    smplx_module: Any,
    model_info: HumanModelInfo,
    *,
    gender: str,
    num_betas: int,
    num_expression_coeffs: int,
    flat_hand_mean: bool,
) -> Any:
    common = {
        "model_path": str(model_info.path),
        "gender": gender,
        "num_betas": num_betas,
        "batch_size": 1,
    }
    if model_info.model_type == "smpl":
        if model_info.path.suffix.lower() == ".npz":
            common["data_struct"] = smplx_module.utils.Struct(
                **_model_data_mapping(model_info.path)
            )
        return smplx_module.SMPL(**common)
    hand_options = {
        "ext": model_info.path.suffix.lstrip("."),
        "use_pca": False,
        "flat_hand_mean": flat_hand_mean,
    }
    if model_info.model_type == "smplh":
        return smplx_module.SMPLH(**common, **hand_options)
    return smplx_module.SMPLX(
        **common,
        **hand_options,
        num_expression_coeffs=num_expression_coeffs,
    )


def _source_forward_kwargs(
    motion: SMPLMotion,
    fields: dict[str, Any],
    betas: Any,
    start: int,
    end: int,
    *,
    return_verts: bool,
) -> dict[str, Any]:
    count = end - start
    kwargs = {
        "betas": betas.expand(count, -1),
        "global_orient": fields["global_orient"][start:end],
        "body_pose": fields["body_pose"][start:end],
        "transl": fields["transl"][start:end],
        "return_verts": return_verts,
    }
    if motion.model_type in {"smplh", "smplx"}:
        kwargs.update(
            left_hand_pose=fields["left_hand_pose"][start:end],
            right_hand_pose=fields["right_hand_pose"][start:end],
        )
    if motion.model_type == "smplx":
        kwargs.update(
            jaw_pose=fields["jaw_pose"][start:end],
            leye_pose=fields["leye_pose"][start:end],
            reye_pose=fields["reye_pose"][start:end],
            expression=fields["expression"][start:end],
        )
    return kwargs


def convert_smpl_to_retarget_arrays(
    input_path: str | Path,
    model_path: str | Path,
    *,
    model_type: str = "auto",
    device_name: str = "auto",
    batch_size: int = 32,
    body_iters: int = 2,
    finger_iters: int = 1,
    full_iters: int = 1,
    lie_iters: int = 3,
    lie_lambda: float = 1e-1,
    autograd_iters: int = 0,
    autograd_lr: float = 5e-3,
    flat_hand_mean: bool = True,
    fps_override: float | None = None,
    source_coordinate: str = "auto",
    canonicalize_heading: bool = True,
    heading_yaw_degrees: float = 0.0,
    rebase_root_horizontal: bool = True,
    progress: Callable[[int, int], None] | None = None,
    runtime_cache: SMPLRuntimeCache | None = None,
) -> tuple[dict[str, np.ndarray], ConversionMetrics]:
    """Convert one SMPL-family motion to retargeter SOMA77 arrays.

    Pass the same ``runtime_cache`` to consecutive calls to reuse compatible
    source/target models. The identity is prepared again for every motion.
    """

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    source_path = Path(input_path).expanduser().resolve()
    model = Path(model_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if not model.is_file():
        raise FileNotFoundError(model)
    model_info = inspect_human_model(model, model_type)

    (
        smplx,
        torch,
        SOMALayer,
        remove_joint_orient_local,
        matrix_to_rotvec,
        PoseInversion,
    ) = _runtime_imports()
    device_name = resolve_soma_x_device(device_name, torch)

    total_start = time.perf_counter()
    motion = load_smpl_motion(
        source_path,
        model_type=model_info.model_type,
        fps_override=fps_override,
    )
    effective_source_coordinate = resolve_source_coordinate(
        source_coordinate, motion
    )
    device = torch.device(device_name)
    num_betas = min(
        int(motion.betas.shape[1]),
        int(model_info.shape_coefficient_count),
    )
    if num_betas <= 0:
        raise ValueError(f"Human model has no usable shape coefficients: {model}")
    betas = torch.from_numpy(motion.betas[:, :num_betas]).float().to(device)
    num_betas = int(betas.shape[1])
    cache = {} if runtime_cache is None else runtime_cache
    runtime_key = (
        str(model),
        model_info.model_type,
        motion.gender,
        num_betas,
        flat_hand_mean,
        str(device),
    )
    runtime = cache.get(runtime_key)
    if runtime is None:
        source_model = _create_source_model(
            smplx,
            model_info,
            gender=motion.gender,
            num_betas=num_betas,
            num_expression_coeffs=int(motion.expression.shape[1]),
            flat_hand_mean=flat_hand_mean,
        ).to(device)
        source_model.eval()

        identity_model_type = (
            "smplx" if model_info.model_type == "smplx" else "smpl"
        )
        target_layer = SOMALayer(
            identity_model_type=identity_model_type,
            identity_model_kwargs={
                "model_path": str(model),
                "gender": motion.gender,
                "num_betas": num_betas,
            },
            device=device,
            mode="warp",
            enable_procedural_transforms=False,
            correctives_model_path=None,
        )
        inverter = PoseInversion(target_layer, low_lod=True)
        cache[runtime_key] = (source_model, target_layer, inverter)
    else:
        source_model, target_layer, inverter = runtime

    inverter.prepare_identity(betas)
    # PoseInversion.fit() consumes the layer's cached pose identity, so target
    # forward kinematics requires a separate prepared identity.
    target_layer.prepare_identity(betas)

    fields = {
        name: torch.from_numpy(getattr(motion, name)).float().to(device)
        for name in (
            "global_orient",
            "body_pose",
            "left_hand_pose",
            "right_hand_pose",
            "jaw_pose",
            "leye_pose",
            "reye_pose",
            "expression",
            "transl",
        )
    }
    if motion.transl_is_root_position:
        neutral_fields = {
            name: torch.zeros_like(value[:1])
            for name, value in fields.items()
        }
        with torch.no_grad():
            neutral_output = source_model(
                **_source_forward_kwargs(
                    motion,
                    neutral_fields,
                    betas,
                    0,
                    1,
                    return_verts=False,
                )
            )
        model_root_offset = neutral_output.joints[:, 0]
        fields["transl"] = fields["transl"] - model_root_offset

    frame_count = motion.frame_count
    local_rot_mats = np.empty((frame_count, 77, 3, 3), dtype=np.float32)
    global_rot_mats = np.empty((frame_count, 77, 3, 3), dtype=np.float32)
    posed_joints = np.empty((frame_count, 77, 3), dtype=np.float32)
    root_translation = np.empty((frame_count, 3), dtype=np.float32)
    fit_errors: list[np.ndarray] = []
    orient = inverter.soma._t_pose_orient
    orient_parent_t = inverter.soma._t_pose_orient_parent_T
    initialization_seconds = time.perf_counter() - total_start

    _synchronize(torch, device)
    conversion_start = time.perf_counter()
    for start in range(0, frame_count, batch_size):
        end = min(start + batch_size, frame_count)
        count = end - start
        with torch.no_grad():
            source_output = source_model(
                **_source_forward_kwargs(
                    motion,
                    fields,
                    betas,
                    start,
                    end,
                    return_verts=True,
                )
            )

        result = inverter.fit(
            source_output.vertices,
            body_iters=body_iters,
            finger_iters=finger_iters,
            full_iters=full_iters,
            lie_iters=lie_iters,
            lie_lambda=lie_lambda,
            autograd_iters=autograd_iters,
            autograd_lr=autograd_lr,
        )
        relative = remove_joint_orient_local(
            result.rotations, orient, orient_parent_t
        )
        if relative.shape[1] == 78:
            relative = relative[:, 1:]
        if relative.shape[1] != 77:
            raise RuntimeError(
                f"PoseInversion returned {relative.shape[1]} joints; expected 77"
            )
        relative_rotvec = matrix_to_rotvec(
            relative.reshape(-1, 3, 3)
        ).reshape(count, 77, 3)
        relative_np = Rotation.from_rotvec(
            relative_rotvec.detach().cpu().numpy().reshape(-1, 3)
        ).as_matrix().reshape(count, 77, 3, 3).astype(np.float32)
        relative = torch.from_numpy(relative_np).to(device)

        with torch.no_grad():
            target_output = target_layer.pose(
                relative,
                transl=result.root_translation,
                pose2rot=False,
                absolute_pose=False,
                apply_correctives=False,
                fk_only=True,
            )
        transforms = target_output["transforms"]
        if transforms.shape[1] == 78:
            transforms = transforms[:, 1:]
        joints = target_output["joints"]
        if joints.shape[1] == 78:
            joints = joints[:, 1:]
        if transforms.shape[1] != 77 or joints.shape[1] != 77:
            raise RuntimeError(
                f"SOMA FK returned transforms={transforms.shape}, "
                f"joints={joints.shape}; expected 77"
            )

        local_rot_mats[start:end] = relative_np
        global_rot_mats[start:end] = (
            transforms[:, :, :3, :3].detach().cpu().numpy()
        )
        posed_joints[start:end] = joints.detach().cpu().numpy()
        root_translation[start:end] = (
            result.root_translation.detach().cpu().numpy()
        )
        fit_errors.append(result.per_vertex_error.detach().cpu().numpy())
        if progress is not None:
            progress(end, frame_count)

    _synchronize(torch, device)
    conversion_seconds = time.perf_counter() - conversion_start

    coordinate_start = time.perf_counter()
    root_positions = posed_joints[:, 0].copy()
    if effective_source_coordinate == "amass":
        heading_joints = np.matmul(
            posed_joints, C_AMASS_TO_KIMODO.T
        ).astype(np.float32)
    elif effective_source_coordinate == "kimodo":
        heading_joints = posed_joints
    else:
        raise ValueError(
            f"Unsupported source coordinate: {effective_source_coordinate}"
        )

    source_anatomical_heading = compute_anatomical_heading(heading_joints)
    canonical_yaw_degrees = (
        canonical_heading_yaw_degrees(heading_joints)
        if canonicalize_heading
        else 0.0
    )
    applied_heading_yaw_degrees = (
        canonical_yaw_degrees + float(heading_yaw_degrees)
    )
    local_rot_mats, global_rot_mats, positions = transform_to_kimodo_frame(
        local_rot_mats,
        global_rot_mats,
        {
            "posed_joints": posed_joints,
            "root_positions": root_positions,
            "root_translation": root_translation,
        },
        source_coordinate=effective_source_coordinate,
        heading_yaw_degrees=applied_heading_yaw_degrees,
    )
    root_horizontal_rebase_offset = np.zeros(3, dtype=np.float32)
    if rebase_root_horizontal:
        positions, root_horizontal_rebase_offset = (
            rebase_root_horizontal_positions(positions)
        )
    posed_joints = positions["posed_joints"]
    root_positions = positions["root_positions"]
    root_translation = positions["root_translation"]
    coordinate_seconds = time.perf_counter() - coordinate_start

    expected_names = soma77_joint_names()
    public_names = list(
        getattr(target_layer, "public_joint_names", expected_names)
    )
    if public_names[:1] == ["Root"]:
        public_names = public_names[1:]
    if public_names != expected_names:
        raise RuntimeError(
            "SOMA-X public joint order does not match Soma Retargeter SOMA77"
        )

    fps_value = np.float32(motion.fps)
    output = {
        "local_rot_mats": local_rot_mats,
        "global_rot_mats": global_rot_mats,
        "posed_joints": posed_joints,
        "root_positions": root_positions,
        "smooth_root_pos": root_positions.copy(),
        "global_root_heading": compute_root_heading(global_rot_mats),
        "foot_contacts": np.zeros((frame_count, 4), dtype=np.bool_),
        "world_root_position": root_positions.copy(),
        "root_translation": root_translation,
        "transl": root_translation.copy(),
        "trans": root_translation.copy(),
        "fps": fps_value,
        "sample_rate": fps_value,
        "mocap_frame_rate": fps_value,
        "mocap_framerate": fps_value,
        "frame_rate": fps_value,
        "framerate": fps_value,
        "source_fps": fps_value,
        "mocap_time_length": np.float32(frame_count / motion.fps),
        "joint_names": np.asarray(expected_names),
        "skeleton": np.asarray("somaskel77"),
        "identity_model_type": np.asarray(
            "smplx" if model_info.model_type == "smplx" else "smpl"
        ),
        "identity_coeffs": motion.betas[:, :num_betas],
        "absolute_pose": np.bool_(False),
        "unit": np.asarray("meters"),
        "source_file": np.asarray(str(source_path)),
        "source_size_bytes": np.int64(source_path.stat().st_size),
        "source_mtime_ns": np.int64(source_path.stat().st_mtime_ns),
        "source_model_type": np.asarray(model_info.model_type),
        "human_model_file": np.asarray(str(model)),
        "source_num_betas": np.int32(num_betas),
        "soma_x_version": np.asarray(SOMA_X_REQUIRED_VERSION),
        "conversion_device": np.asarray(str(device)),
        "source_gender": np.asarray(motion.gender),
        "source_coordinate": np.asarray(effective_source_coordinate),
        "source_translation_is_root_position": np.bool_(
            motion.transl_is_root_position
        ),
        "coordinate_system": np.asarray("kimodo_y_up_z_forward"),
        "heading_canonicalized": np.bool_(canonicalize_heading),
        "heading_source_forward_xz": source_anatomical_heading[[0, 2]],
        "heading_canonical_yaw_degrees": np.float32(canonical_yaw_degrees),
        "heading_additional_yaw_degrees": np.float32(heading_yaw_degrees),
        "heading_correction_degrees": np.float32(
            applied_heading_yaw_degrees
        ),
        "root_horizontal_rebased": np.bool_(rebase_root_horizontal),
        "root_horizontal_rebase_offset_m": (
            root_horizontal_rebase_offset.astype(np.float32)
        ),
        "converter": np.asarray("soma_retargeter.smpl_motion"),
    }
    if model_info.model_type == "smplx":
        output["smplx_model_file"] = np.asarray(str(model))

    errors = np.concatenate(fit_errors, axis=0)
    total_seconds = time.perf_counter() - total_start
    metrics = ConversionMetrics(
        frame_count=frame_count,
        fps=motion.fps,
        initialization_seconds=initialization_seconds,
        conversion_seconds=conversion_seconds,
        coordinate_seconds=coordinate_seconds,
        total_seconds=total_seconds,
        conversion_frames_per_second=(
            frame_count / max(conversion_seconds, 1e-12)
        ),
        mean_vertex_error_m=float(errors.mean()),
        median_vertex_error_m=float(np.median(errors)),
        max_vertex_error_m=float(errors.max()),
    )
    return output, metrics


def convert_smplx_to_retarget_arrays(
    input_path: str | Path,
    model_path: str | Path,
    **kwargs: Any,
) -> tuple[dict[str, np.ndarray], ConversionMetrics]:
    """Compatibility converter for existing SMPL-X-only callers."""

    requested = kwargs.pop("model_type", "smplx")
    if normalize_model_type(requested) != "smplx":
        raise ValueError("convert_smplx_to_retarget_arrays only accepts SMPL-X")
    return convert_smpl_to_retarget_arrays(
        input_path,
        model_path,
        model_type="smplx",
        **kwargs,
    )


def normalize_retarget_heading_arrays(
    arrays: dict[str, np.ndarray],
    *,
    additional_yaw_degrees: float = 0.0,
    rebase_root_horizontal: bool = True,
) -> tuple[dict[str, np.ndarray], float]:
    """Canonicalize an existing SOMA77 NPZ without rerunning pose inversion."""

    output = {key: np.array(value, copy=True) for key, value in arrays.items()}
    required = ("local_rot_mats", "global_rot_mats", "posed_joints")
    missing = [key for key in required if key not in output]
    if missing:
        raise KeyError(f"Converted NPZ is missing required fields: {missing}")

    local_rot_mats = np.asarray(output["local_rot_mats"], dtype=np.float32)
    global_rot_mats = np.asarray(output["global_rot_mats"], dtype=np.float32)
    posed_joints = np.asarray(output["posed_joints"], dtype=np.float32)
    if "root_positions" not in output:
        output["root_positions"] = posed_joints[:, 0].copy()

    source_heading = compute_anatomical_heading(posed_joints)
    canonical_yaw = canonical_heading_yaw_degrees(posed_joints)
    applied_yaw = canonical_yaw + float(additional_yaw_degrees)
    previous_correction = float(
        np.asarray(output.get("heading_correction_degrees", 0.0)).reshape(-1)[0]
    )
    cumulative_correction = (
        previous_correction + applied_yaw + 180.0
    ) % 360.0 - 180.0
    positions = {
        key: np.asarray(output[key], dtype=np.float32)
        for key in _WORLD_POSITION_KEYS
        if key in output
    }
    local_rot_mats, global_rot_mats, positions = transform_to_kimodo_frame(
        local_rot_mats,
        global_rot_mats,
        positions,
        source_coordinate="kimodo",
        heading_yaw_degrees=applied_yaw,
    )
    root_horizontal_rebase_offset = np.zeros(3, dtype=np.float32)
    if rebase_root_horizontal:
        positions, root_horizontal_rebase_offset = (
            rebase_root_horizontal_positions(positions)
        )
    previous_root_horizontal_rebase_offset = np.asarray(
        output.get(
            "root_horizontal_rebase_offset_m",
            np.zeros(3, dtype=np.float32),
        ),
        dtype=np.float32,
    ).reshape(3)
    output["local_rot_mats"] = local_rot_mats
    output["global_rot_mats"] = global_rot_mats
    output.update(positions)
    output["global_root_heading"] = compute_root_heading(global_rot_mats)
    output["coordinate_system"] = np.asarray("kimodo_y_up_z_forward")
    output["heading_canonicalized"] = np.bool_(True)
    output["heading_source_forward_xz"] = source_heading[[0, 2]]
    output["heading_canonical_yaw_degrees"] = np.float32(canonical_yaw)
    output["heading_additional_yaw_degrees"] = np.float32(
        additional_yaw_degrees
    )
    output["heading_previous_correction_degrees"] = np.float32(
        previous_correction
    )
    output["heading_normalization_yaw_degrees"] = np.float32(applied_yaw)
    output["heading_correction_degrees"] = np.float32(cumulative_correction)
    output["heading_normalizer"] = np.asarray(
        "soma_retargeter.smplx_motion.anatomical_frame0"
    )
    if rebase_root_horizontal:
        output["root_horizontal_rebased"] = np.bool_(True)
        output["root_horizontal_rebase_normalization_m"] = (
            root_horizontal_rebase_offset.astype(np.float32)
        )
        output["root_horizontal_rebase_offset_m"] = (
            previous_root_horizontal_rebase_offset
            + root_horizontal_rebase_offset
        ).astype(np.float32)
        output["position_normalizer"] = np.asarray(
            "soma_retargeter.smplx_motion.frame0_horizontal_origin"
        )
    return output, applied_yaw


def save_retarget_npz(
    output_path: str | Path,
    arrays: dict[str, np.ndarray],
    *,
    compressed: bool = True,
) -> None:
    """Atomically save a converted motion."""

    path = Path(output_path).expanduser()
    if path.suffix.lower() != ".npz":
        raise ValueError(f"Output must use the .npz extension: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        saver = np.savez_compressed if compressed else np.savez
        saver(temporary, **arrays)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
