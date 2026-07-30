# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert Kimodo SOMA77 NPZ motions to BVH without importing Kimodo."""

from __future__ import annotations

import hashlib
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


SOMA77_JOINTS_WITH_PARENTS = [
    ("Hips", None),
    ("Spine1", "Hips"),
    ("Spine2", "Spine1"),
    ("Chest", "Spine2"),
    ("Neck1", "Chest"),
    ("Neck2", "Neck1"),
    ("Head", "Neck2"),
    ("HeadEnd", "Head"),
    ("Jaw", "Head"),
    ("LeftEye", "Head"),
    ("RightEye", "Head"),
    ("LeftShoulder", "Chest"),
    ("LeftArm", "LeftShoulder"),
    ("LeftForeArm", "LeftArm"),
    ("LeftHand", "LeftForeArm"),
    ("LeftHandThumb1", "LeftHand"),
    ("LeftHandThumb2", "LeftHandThumb1"),
    ("LeftHandThumb3", "LeftHandThumb2"),
    ("LeftHandThumbEnd", "LeftHandThumb3"),
    ("LeftHandIndex1", "LeftHand"),
    ("LeftHandIndex2", "LeftHandIndex1"),
    ("LeftHandIndex3", "LeftHandIndex2"),
    ("LeftHandIndex4", "LeftHandIndex3"),
    ("LeftHandIndexEnd", "LeftHandIndex4"),
    ("LeftHandMiddle1", "LeftHand"),
    ("LeftHandMiddle2", "LeftHandMiddle1"),
    ("LeftHandMiddle3", "LeftHandMiddle2"),
    ("LeftHandMiddle4", "LeftHandMiddle3"),
    ("LeftHandMiddleEnd", "LeftHandMiddle4"),
    ("LeftHandRing1", "LeftHand"),
    ("LeftHandRing2", "LeftHandRing1"),
    ("LeftHandRing3", "LeftHandRing2"),
    ("LeftHandRing4", "LeftHandRing3"),
    ("LeftHandRingEnd", "LeftHandRing4"),
    ("LeftHandPinky1", "LeftHand"),
    ("LeftHandPinky2", "LeftHandPinky1"),
    ("LeftHandPinky3", "LeftHandPinky2"),
    ("LeftHandPinky4", "LeftHandPinky3"),
    ("LeftHandPinkyEnd", "LeftHandPinky4"),
    ("RightShoulder", "Chest"),
    ("RightArm", "RightShoulder"),
    ("RightForeArm", "RightArm"),
    ("RightHand", "RightForeArm"),
    ("RightHandThumb1", "RightHand"),
    ("RightHandThumb2", "RightHandThumb1"),
    ("RightHandThumb3", "RightHandThumb2"),
    ("RightHandThumbEnd", "RightHandThumb3"),
    ("RightHandIndex1", "RightHand"),
    ("RightHandIndex2", "RightHandIndex1"),
    ("RightHandIndex3", "RightHandIndex2"),
    ("RightHandIndex4", "RightHandIndex3"),
    ("RightHandIndexEnd", "RightHandIndex4"),
    ("RightHandMiddle1", "RightHand"),
    ("RightHandMiddle2", "RightHandMiddle1"),
    ("RightHandMiddle3", "RightHandMiddle2"),
    ("RightHandMiddle4", "RightHandMiddle3"),
    ("RightHandMiddleEnd", "RightHandMiddle4"),
    ("RightHandRing1", "RightHand"),
    ("RightHandRing2", "RightHandRing1"),
    ("RightHandRing3", "RightHandRing2"),
    ("RightHandRing4", "RightHandRing3"),
    ("RightHandRingEnd", "RightHandRing4"),
    ("RightHandPinky1", "RightHand"),
    ("RightHandPinky2", "RightHandPinky1"),
    ("RightHandPinky3", "RightHandPinky2"),
    ("RightHandPinky4", "RightHandPinky3"),
    ("RightHandPinkyEnd", "RightHandPinky4"),
    ("LeftLeg", "Hips"),
    ("LeftShin", "LeftLeg"),
    ("LeftFoot", "LeftShin"),
    ("LeftToeBase", "LeftFoot"),
    ("LeftToeEnd", "LeftToeBase"),
    ("RightLeg", "Hips"),
    ("RightShin", "RightLeg"),
    ("RightFoot", "RightShin"),
    ("RightToeBase", "RightFoot"),
    ("RightToeEnd", "RightToeBase"),
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def joint_names() -> list[str]:
    return [name for name, _ in SOMA77_JOINTS_WITH_PARENTS]


def joint_parents() -> np.ndarray:
    name_to_index = {name: i for i, (name, _) in enumerate(SOMA77_JOINTS_WITH_PARENTS)}
    return np.array(
        [-1 if parent is None else name_to_index[parent] for _, parent in SOMA77_JOINTS_WITH_PARENTS],
        dtype=np.int64,
    )


def resolve_path(path: str | Path | None) -> Path | None:
    if path is None or str(path).strip() == "":
        return None
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    return (repo_root() / candidate).resolve()


def default_template_bvh_path() -> Path:
    return repo_root() / "assets" / "motions" / "bvh" / "Neutral_walk_forward_002__A057.bvh"


def find_default_offsets_path() -> Path | None:
    rel = Path("assets") / "skeletons" / "somaskel77" / "standard_t_pose_global_offsets_rots.p"
    candidates = [
        repo_root() / "assets" / "skeletons" / "somaskel77" / "standard_t_pose_global_offsets_rots.p",
        repo_root() / "soma_retargeter" / "configs" / "soma" / "standard_t_pose_global_offsets_rots.p",
    ]
    for parent in repo_root().parents:
        candidates.append(parent / "kimodo" / rel)
        candidates.append(parent / "kimodo" / "kimodo" / rel)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def temp_bvh_path_for_npz(npz_path: str | Path) -> Path:
    source = Path(npz_path).expanduser()
    digest = hashlib.sha1(str(source.resolve()).encode("utf-8")).hexdigest()[:12]
    tmp_dir = Path(tempfile.gettempdir()) / "soma_retargeter_npz_to_bvh"
    return tmp_dir / f"{source.stem}_{digest}_fixed.bvh"


def load_npz_motion(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path).expanduser()
    data = np.load(path)
    if "local_rot_mats" not in data:
        raise ValueError(f"{path} does not contain local_rot_mats")

    local_rot_mats = np.asarray(data["local_rot_mats"], dtype=np.float64)
    if local_rot_mats.ndim == 5:
        if local_rot_mats.shape[0] != 1:
            raise ValueError(f"local_rot_mats batch dimension must be 1, got {local_rot_mats.shape[0]}")
        local_rot_mats = local_rot_mats[0]
    if local_rot_mats.shape[1:] != (77, 3, 3):
        raise ValueError(f"local_rot_mats must have shape (T, 77, 3, 3), got {local_rot_mats.shape}")

    if "root_positions" in data:
        root_positions = np.asarray(data["root_positions"], dtype=np.float64)
    elif "posed_joints" in data:
        root_positions = np.asarray(data["posed_joints"], dtype=np.float64)[:, 0]
    else:
        raise ValueError(f"{path} does not contain root_positions or posed_joints")

    if root_positions.ndim == 3:
        if root_positions.shape[0] != 1:
            raise ValueError(f"root_positions batch dimension must be 1, got {root_positions.shape[0]}")
        root_positions = root_positions[0]
    if root_positions.shape != (local_rot_mats.shape[0], 3):
        raise ValueError(
            f"root_positions must have shape ({local_rot_mats.shape[0]}, 3), got {root_positions.shape}"
        )
    return local_rot_mats, root_positions


def _load_torch_float_storage_zip(path: Path) -> np.ndarray:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        data_entries = [name for name in names if name.endswith("/data/0")]
        if len(data_entries) != 1:
            raise ValueError(f"{path} does not look like the expected torch tensor zip")
        byteorder_entries = [name for name in names if name.endswith("/byteorder")]
        byteorder = "little"
        if byteorder_entries:
            byteorder = archive.read(byteorder_entries[0]).decode("utf-8").strip()
        dtype = "<f4" if byteorder == "little" else ">f4"
        tensor = np.frombuffer(archive.read(data_entries[0]), dtype=dtype).astype(np.float64)
    if tensor.size != 77 * 3 * 3:
        raise ValueError(f"{path} tensor storage must contain {77 * 3 * 3} float values, got {tensor.size}")
    return tensor.reshape(77, 3, 3)


def load_global_offsets(path: str | Path) -> np.ndarray:
    path = Path(path).expanduser()
    if path.suffix == ".npy":
        offsets = np.load(path)
    elif zipfile.is_zipfile(path):
        offsets = _load_torch_float_storage_zip(path)
    else:
        raise ValueError(f"Unsupported offsets file format: {path}")

    offsets = np.asarray(offsets, dtype=np.float64)
    if offsets.shape != (77, 3, 3):
        raise ValueError(f"offset rotations must have shape (77, 3, 3), got {offsets.shape}")
    return offsets


def standard_tpose_local_to_bvh_local(local_std: np.ndarray, global_offsets: np.ndarray) -> np.ndarray:
    parents = joint_parents()
    global_std = np.empty_like(local_std)
    for joint_idx, parent in enumerate(parents):
        if parent < 0:
            global_std[:, joint_idx] = local_std[:, joint_idx]
        else:
            global_std[:, joint_idx] = global_std[:, parent] @ local_std[:, joint_idx]

    global_bvh = global_std @ global_offsets[None]
    local_bvh = np.empty_like(global_bvh)
    for joint_idx, parent in enumerate(parents):
        if parent < 0:
            local_bvh[:, joint_idx] = global_bvh[:, joint_idx]
        else:
            local_bvh[:, joint_idx] = np.swapaxes(global_bvh[:, parent], -1, -2) @ global_bvh[:, joint_idx]
    return local_bvh


def rotmats_to_bvh_eulers(local_rot_mats: np.ndarray) -> np.ndarray:
    frames, joints = local_rot_mats.shape[:2]
    euler_rad = Rotation.from_matrix(local_rot_mats.reshape(-1, 3, 3)).as_euler("ZYX")
    euler_rad = euler_rad.reshape(frames, joints, 3)
    euler_rad = np.unwrap(euler_rad, axis=0)
    return np.rad2deg(euler_rad)


def read_template_header(path: str | Path) -> tuple[list[str], float]:
    path = Path(path).expanduser()
    lines = path.read_text(encoding="utf-8").splitlines()
    frame_line = next((i for i, line in enumerate(lines) if line.strip().startswith("Frames:")), None)
    frame_time_line = next((i for i, line in enumerate(lines) if line.strip().startswith("Frame Time:")), None)
    if frame_line is None or frame_time_line is None:
        raise ValueError(f"{path} does not look like a BVH file")

    found_joints = []
    channels = []
    for line in lines[:frame_line]:
        stripped = line.strip()
        if stripped.startswith("ROOT ") or stripped.startswith("JOINT "):
            found_joints.append(stripped.split()[1])
        elif stripped.startswith("CHANNELS "):
            channels.append(int(stripped.split()[1]))

    expected = ["Root", *joint_names()]
    if found_joints != expected:
        raise ValueError(
            "template BVH joint order does not match Kimodo SOMA77 export\n"
            f"expected first joints: {expected[:8]}\n"
            f"found first joints: {found_joints[:8]}\n"
            f"joint count: expected {len(expected)}, found {len(found_joints)}"
        )
    if channels != [6, 6, *([3] * 76)]:
        raise ValueError(f"template BVH channels do not match Root+SOMA77 layout: {channels[:10]}...")

    frame_time = float(lines[frame_time_line].split()[-1])
    header = lines[: frame_time_line + 1]
    return header, frame_time


def detect_sidecar_bvh_fps(input_npz: str | Path) -> float | None:
    sidecar_bvh = Path(input_npz).expanduser().with_suffix(".bvh")
    if not sidecar_bvh.exists():
        return None
    _, frame_time = read_template_header(sidecar_bvh)
    return 1.0 / frame_time


def write_bvh(
    path: str | Path,
    header: list[str],
    root_positions: np.ndarray,
    eulers_zyx_deg: np.ndarray,
    *,
    frame_time: float,
    position_scale: float,
) -> None:
    path = Path(path).expanduser()
    frames = eulers_zyx_deg.shape[0]
    out = []
    for line in header:
        stripped = line.strip()
        if stripped.startswith("Frames:"):
            out.append(f"Frames: {frames}")
        elif stripped.startswith("Frame Time:"):
            out.append(f"Frame Time: {frame_time:.8f}")
        else:
            out.append(line)

    root_cm = root_positions * position_scale
    zero_root = [0.0] * 6
    for frame in range(frames):
        values = [
            *zero_root,
            float(root_cm[frame, 0]),
            float(root_cm[frame, 1]),
            float(root_cm[frame, 2]),
            *[float(x) for x in eulers_zyx_deg[frame, 0]],
        ]
        for joint_idx in range(1, 77):
            values.extend(float(x) for x in eulers_zyx_deg[frame, joint_idx])
        out.append(" ".join(f"{value:.6f}" for value in values))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def compare_euler_roundtrip(target_rot_mats: np.ndarray, eulers_zyx_deg: np.ndarray) -> tuple[float, float, float]:
    reconstructed = Rotation.from_euler("ZYX", eulers_zyx_deg.reshape(-1, 3), degrees=True).as_matrix()
    reconstructed = reconstructed.reshape(target_rot_mats.shape)
    rel = np.swapaxes(reconstructed, -1, -2) @ target_rot_mats
    angles = Rotation.from_matrix(rel.reshape(-1, 3, 3)).magnitude()
    angles_deg = np.rad2deg(angles)
    return float(angles_deg.max()), float(np.percentile(angles_deg, 99.0)), float(angles_deg.mean())


def convert_npz_to_bvh(
    input_npz: str | Path,
    output_bvh: str | Path,
    *,
    template_bvh: str | Path | None = None,
    offsets: str | Path | None = None,
    fps: float | None = None,
    position_scale: float = 100.0,
    compare: bool = False,
) -> dict[str, float | int | str | None]:
    template_path = resolve_path(template_bvh) or default_template_bvh_path()
    offsets_path = resolve_path(offsets) or find_default_offsets_path()
    if offsets_path is None:
        raise FileNotFoundError(
            "Kimodo SOMA77 rest-pose offsets file not found. Set kimodo_npz_offsets in the config "
            "or SOMA_RETARGETER_KIMODO_NPZ_OFFSETS to standard_t_pose_global_offsets_rots.p."
        )
    if not template_path.exists():
        raise FileNotFoundError(f"Kimodo NPZ template BVH not found: {template_path}")
    if not offsets_path.exists():
        raise FileNotFoundError(f"Kimodo NPZ offsets file not found: {offsets_path}")

    local_std, root_positions = load_npz_motion(input_npz)
    global_offsets = load_global_offsets(offsets_path)
    local_bvh = standard_tpose_local_to_bvh_local(local_std, global_offsets)
    eulers = rotmats_to_bvh_eulers(local_bvh)

    header, template_frame_time = read_template_header(template_path)
    frame_time = template_frame_time if fps is None else 1.0 / float(fps)
    write_bvh(
        output_bvh,
        header,
        root_positions,
        eulers,
        frame_time=frame_time,
        position_scale=float(position_scale),
    )

    result: dict[str, float | int | str | None] = {
        "frames": int(local_std.shape[0]),
        "frame_time": float(frame_time),
        "fps": float(1.0 / frame_time),
        "joints": 77,
        "template_bvh": str(template_path),
        "offsets": str(offsets_path),
        "output_bvh": str(Path(output_bvh).expanduser()),
    }
    if compare:
        max_err, p99_err, mean_err = compare_euler_roundtrip(local_bvh, eulers)
        result.update(
            {
                "euler_roundtrip_error_max_deg": max_err,
                "euler_roundtrip_error_p99_deg": p99_err,
                "euler_roundtrip_error_mean_deg": mean_err,
            }
        )
    return result
