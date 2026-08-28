# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare BVH and SMPL-family motion inputs for headless retargeting."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from soma_retargeter.assets import kimodo_npz
from soma_retargeter.assets import smplx_motion


REQUIRED_SOMA_OUTPUT_FIELDS = (
    "local_rot_mats",
    "global_rot_mats",
    "posed_joints",
    "root_positions",
    "fps",
    "conversion_signature",
)

SOMA_NPZ_OUTPUT_DIRECTORY = "soma_npz"
SOMA_BVH_OUTPUT_DIRECTORY = "soma_bvh"
RETARGETED_CSV_OUTPUT_DIRECTORY = "retargeted_csv"


class MotionInputKind(str, Enum):
    """Supported source formats for the unified headless entrypoint."""

    BVH = "bvh"
    SOMA77_NPZ = "soma77_npz"
    RAW_SMPL_NPZ = "raw_smpl_npz"


@dataclass(frozen=True)
class MotionInputJob:
    """One source motion and its deterministic output paths."""

    source_path: Path
    relative_path: Path
    kind: MotionInputKind
    csv_path: Path
    soma_npz_path: Path | None
    bvh_path: Path


def _normalized_local_rot_shape(data: np.lib.npyio.NpzFile) -> tuple[int, ...]:
    local = np.asarray(data["local_rot_mats"])
    if local.ndim == 5 and local.shape[0] == 1:
        local = local[0]
    return tuple(local.shape)


def classify_motion_input(path: str | Path) -> MotionInputKind:
    """Classify a BVH, SOMA77 NPZ, or raw SMPL-family NPZ by its schema."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    suffix = source.suffix.lower()
    if suffix == ".bvh":
        return MotionInputKind.BVH
    if suffix != ".npz":
        raise ValueError(f"Unsupported motion file extension: {source}")

    with np.load(source, allow_pickle=False) as data:
        fields = set(data.files)
        if {"v_template", "shapedirs"}.issubset(fields):
            raise ValueError(
                f"Human model file is not a motion input: {source}. "
                "Pass it with --human-model instead."
            )

        if "local_rot_mats" in fields:
            shape = _normalized_local_rot_shape(data)
            if len(shape) != 4 or shape[-2:] != (3, 3):
                raise ValueError(
                    "local_rot_mats must have shape (T, J, 3, 3), got "
                    f"{shape} in {source}"
                )
            if shape[1] == 77:
                return MotionInputKind.SOMA77_NPZ
            if shape[1] == 22:
                return MotionInputKind.RAW_SMPL_NPZ
            raise ValueError(
                f"Unsupported local_rot_mats joint count {shape[1]} in {source}; "
                "expected 22 for raw Kimodo SMPL-X or 77 for SOMA77"
            )

        if "poses" in fields:
            poses = np.asarray(data["poses"])
            if poses.ndim == 3 and poses.shape[1:] == (77, 3):
                raise ValueError(
                    f"{source} is a SOMA-X intermediate NPZ with poses shape "
                    "(T, 77, 3). Use the original SMPL-family motion instead."
                )
            return MotionInputKind.RAW_SMPL_NPZ

        raw_fields = {
            "global_orient",
            "root_orient",
            "body_pose",
            "body_pose_axis",
            "pose_body",
        }
        if fields & raw_fields:
            return MotionInputKind.RAW_SMPL_NPZ

    raise ValueError(f"Could not identify a supported motion schema in {source}")


def _iter_motion_files(input_root: Path) -> Iterable[Path]:
    for path in input_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".bvh", ".npz"}:
            yield path


def plan_motion_jobs(
    input_path: str | Path,
    output_root: str | Path,
    *,
    human_model: str | Path | None = None,
) -> list[MotionInputJob]:
    """Discover inputs, classify them, and reject output-path collisions."""

    source = Path(input_path).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    model = Path(human_model).expanduser().resolve() if human_model else None

    if source.is_file():
        candidates = [source]
        relative_root = source.parent
    elif source.is_dir():
        if destination == source or destination.is_relative_to(source):
            raise ValueError("--output-dir must be outside the input directory")
        candidates = sorted(_iter_motion_files(source))
        relative_root = source
    else:
        raise FileNotFoundError(source)

    if model is not None:
        candidates = [path for path in candidates if path.resolve() != model]
    if not candidates:
        raise FileNotFoundError(f"No BVH or NPZ motion files found under {source}")

    jobs = []
    destinations: dict[Path, Path] = {}
    for candidate in candidates:
        relative = candidate.relative_to(relative_root)
        csv_path = (
            destination
            / RETARGETED_CSV_OUTPUT_DIRECTORY
            / relative.with_suffix(".csv")
        )
        previous = destinations.get(csv_path)
        if previous is not None:
            raise ValueError(
                "Multiple inputs map to the same CSV output: "
                f"{previous} and {candidate} -> {csv_path}"
            )
        destinations[csv_path] = candidate

        kind = classify_motion_input(candidate)
        if kind == MotionInputKind.BVH:
            soma_npz_path = None
            bvh_path = candidate
        else:
            soma_npz_path = (
                destination
                / SOMA_NPZ_OUTPUT_DIRECTORY
                / relative.with_suffix(".npz")
                if kind == MotionInputKind.RAW_SMPL_NPZ
                else None
            )
            bvh_path = (
                destination
                / SOMA_BVH_OUTPUT_DIRECTORY
                / relative.with_suffix(".bvh")
            )
        jobs.append(
            MotionInputJob(
                source_path=candidate,
                relative_path=relative,
                kind=kind,
                csv_path=csv_path,
                soma_npz_path=soma_npz_path,
                bvh_path=bvh_path,
            )
        )
    return jobs


def validate_existing_soma_output(
    input_path: str | Path,
    output_path: str | Path,
    model_path: str | Path,
    conversion_options: dict[str, Any],
) -> tuple[bool, str]:
    """Check whether an existing SOMA77 output matches its conversion inputs."""

    try:
        expected = smplx_motion.build_conversion_signature(
            input_path, model_path, conversion_options
        )
        with np.load(Path(output_path).expanduser(), allow_pickle=False) as data:
            missing = [
                field for field in REQUIRED_SOMA_OUTPUT_FIELDS if field not in data.files
            ]
            if missing:
                return False, "missing fields: " + ", ".join(missing)
            actual = str(np.asarray(data["conversion_signature"]).reshape(-1)[0])
            local = np.asarray(data["local_rot_mats"])
            if local.ndim != 4 or local.shape[1:] != (77, 3, 3):
                return False, f"invalid local_rot_mats shape: {local.shape}"
        if actual != expected:
            return False, "conversion signature differs"
        return True, "matching conversion signature"
    except Exception as exc:
        return False, f"could not validate output: {exc}"


def convert_raw_smpl_to_soma_npz(
    input_path: str | Path,
    output_path: str | Path,
    model_path: str | Path,
    conversion_options: dict[str, Any],
    *,
    runtime_cache: dict,
    progress: Callable[[int, int], None] | None = None,
    compressed: bool = True,
) -> dict[str, Any]:
    """Convert one raw SMPL-family motion and persist a signed SOMA77 NPZ."""

    arrays, metrics = smplx_motion.convert_smpl_to_retarget_arrays(
        input_path,
        model_path,
        **conversion_options,
        runtime_cache=runtime_cache,
        progress=progress,
    )
    signature = smplx_motion.build_conversion_signature(
        input_path, model_path, conversion_options
    )
    arrays["conversion_signature"] = np.asarray(signature)
    smplx_motion.save_retarget_npz(output_path, arrays, compressed=compressed)
    return {
        "input": str(Path(input_path).expanduser().resolve()),
        "output": str(Path(output_path).expanduser().resolve()),
        "conversion_signature": signature,
        **asdict(metrics),
    }


def resolve_npz_fps(
    path: str | Path,
    configured_fps: float | None = None,
) -> tuple[float, str]:
    """Resolve NPZ playback rate using the same precedence as the GUI."""

    if configured_fps is not None:
        fps = float(configured_fps)
        source = "override"
    else:
        fps = kimodo_npz.detect_npz_fps(path)
        source = "metadata"
        if fps is None:
            fps = kimodo_npz.detect_sidecar_bvh_fps(path)
            source = "sidecar_bvh"
        if fps is None:
            fps = 30.0
            source = "fallback_30"
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"FPS must be finite and positive, got {fps}")
    return fps, source


def convert_soma_npz_to_bvh(
    input_path: str | Path,
    output_path: str | Path,
    *,
    template_bvh: str | Path | None,
    offsets: str | Path | None,
    configured_fps: float | None,
    position_scale: float,
    compare: bool,
) -> dict[str, Any]:
    """Persist a fixed-Euler BVH using the GUI's conversion implementation."""

    fps, fps_source = resolve_npz_fps(input_path, configured_fps)
    result = kimodo_npz.convert_npz_to_bvh(
        input_path,
        output_path,
        template_bvh=template_bvh,
        offsets=offsets,
        fps=fps,
        position_scale=position_scale,
        compare=compare,
    )
    result["fps_source"] = fps_source
    return result
