#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert raw SMPL-family motions to Soma Retargeter SOMA77 NPZ."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np

from soma_retargeter.assets import kimodo_npz
from soma_retargeter.assets.smplx_motion import (
    build_conversion_signature,
    convert_smpl_to_retarget_arrays,
    inspect_human_model,
    load_smpl_motion,
    normalize_model_type,
    require_soma_x_dependencies,
    resolve_human_model_path,
    save_retarget_npz,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "assets/default_ai_sapiens_bvh_to_csv_converter_config.json"
REQUIRED_OUTPUT_FIELDS = (
    "local_rot_mats",
    "global_rot_mats",
    "posed_joints",
    "root_positions",
    "fps",
    "conversion_signature",
)


def parse_args(
    argv: list[str] | None = None,
    *,
    default_model_type: str = "auto",
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert SMPL/SMPL-H/SMPL-X motion parameters to a SOMA77 NPZ that Soma "
            "Retargeter can load directly. SOMA-X is only required when a "
            "conversion is requested."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", type=Path)
    input_group.add_argument(
        "--input-dir",
        type=Path,
        help="Recursively convert matching NPZ files below this directory.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory that receives recursively converted SOMA77 NPZ files.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Soma Retargeter config (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help=(
            "Licensed SMPL-family .npz or .pkl model. Resolution order is this "
            "argument, SOMA_RETARGETER_SOMA_X_HUMAN_MODEL, then "
            "soma_x_human_model in --config. The legacy SMPL-X settings remain "
            "a fallback for SMPL-X."
        ),
    )
    parser.add_argument(
        "--model-type",
        choices=("auto", "smpl", "smplh", "smplx"),
        default=default_model_type,
        help="Human model family. Auto detects it from the selected model file.",
    )
    parser.add_argument(
        "--device",
        help="SOMA-X device: auto, cpu, cuda, or cuda:N (default: auto).",
    )
    parser.add_argument(
        "--input-fps",
        type=float,
        help=(
            "Override source FPS. Internal NPZ metadata is used when omitted; "
            "inputs without FPS metadata retain the 30 FPS compatibility default."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="PoseInversion frame batch size (config/default: 32).",
    )
    parser.add_argument("--body-iters", type=int, default=2)
    parser.add_argument("--finger-iters", type=int, default=1)
    parser.add_argument("--full-iters", type=int, default=1)
    parser.add_argument("--lie-iters", type=int, default=3)
    parser.add_argument("--lie-lambda", type=float, default=1e-1)
    parser.add_argument("--autograd-iters", type=int, default=0)
    parser.add_argument("--autograd-lr", type=float, default=5e-3)
    parser.add_argument(
        "--flat-hand-mean",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--source-coordinate",
        choices=("auto", "amass", "kimodo"),
        help=(
            "Source world frame. Auto selects AMASS for parameter NPZ files "
            "and Kimodo for 22-joint matrix exports."
        ),
    )
    parser.add_argument(
        "--canonicalize-heading",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Rotate frame-zero anatomical forward to Kimodo +Z.",
    )
    parser.add_argument(
        "--heading-yaw-degrees",
        type=float,
        default=0.0,
        help="Additional Kimodo +Y yaw after heading canonicalization.",
    )
    parser.add_argument(
        "--rebase-root-horizontal",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Move frame-zero root X/Z to the horizontal origin.",
    )
    parser.add_argument(
        "--emit-bvh",
        action="store_true",
        help="Also emit a fixed-Euler BVH accepted by the existing batch retargeter.",
    )
    parser.add_argument("--bvh-output", type=Path, help="Single-input BVH output path.")
    parser.add_argument(
        "--bvh-output-dir",
        type=Path,
        help="Recursive BVH output root; relative input paths are preserved.",
    )
    parser.add_argument("--bvh-template", type=Path)
    parser.add_argument("--bvh-offsets", type=Path)
    parser.add_argument("--bvh-position-scale", type=float)
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Use uncompressed NPZ output to reduce export CPU time.",
    )
    parser.add_argument("--force", action="store_true", help="Replace existing outputs.")
    parser.add_argument(
        "--pattern",
        default="*_stageii.npz",
        help="Recursive filename pattern (default: *_stageii.npz).",
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        type=Path,
        metavar="RELATIVE_PATH",
        help="Input subtree to exclude; repeat for multiple directories.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print normalized source metadata without importing SOMA-X.",
    )
    args = parser.parse_args(argv)

    if args.input is not None:
        if args.output_dir is not None:
            parser.error("--output-dir can only be used with --input-dir")
        if args.bvh_output_dir is not None:
            parser.error("--bvh-output-dir can only be used with --input-dir")
        if args.exclude_dir:
            parser.error("--exclude-dir can only be used with --input-dir")
        if not args.inspect and args.output is None:
            parser.error("--output is required with --input")
    else:
        if args.output is not None:
            parser.error("--output can only be used with --input")
        if args.bvh_output is not None:
            parser.error("--bvh-output can only be used with --input")
        if not args.inspect and args.output_dir is None:
            parser.error("--output-dir is required with --input-dir")
    if not args.emit_bvh and (args.bvh_output or args.bvh_output_dir):
        parser.error("--bvh-output and --bvh-output-dir require --emit-bvh")
    return args


def _bool_value(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _optional_path(value: object) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute() or path.exists():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def load_config(path: Path) -> dict[str, object]:
    config_path = path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Soma Retargeter config not found: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def configure_args(args: argparse.Namespace) -> argparse.Namespace:
    """Apply CLI, environment, then config/default precedence."""

    config = load_config(args.config)
    args.config = args.config.expanduser().resolve()
    args.model_type = normalize_model_type(args.model_type)
    args.model = resolve_human_model_path(
        args.model,
        config,
        model_type=args.model_type,
    )
    if args.model is not None:
        model_info = inspect_human_model(args.model, args.model_type)
        args.model_type = model_info.model_type
    args.device = (
        args.device
        or os.environ.get("SOMA_RETARGETER_SOMA_X_DEVICE")
        or config.get("soma_x_device")
        or "auto"
    )
    args.batch_size = int(
        args.batch_size
        or os.environ.get("SOMA_RETARGETER_SOMA_X_BATCH_SIZE")
        or config.get("soma_x_batch_size")
        or 32
    )
    args.source_coordinate = str(
        args.source_coordinate
        or os.environ.get("SOMA_RETARGETER_SOMA_X_SOURCE_COORDINATE")
        or config.get("soma_x_source_coordinate")
        or "auto"
    )
    if args.source_coordinate not in {"auto", "amass", "kimodo"}:
        raise ValueError(f"Unsupported source coordinate: {args.source_coordinate}")
    if args.canonicalize_heading is None:
        args.canonicalize_heading = _bool_value(
            os.environ.get(
                "SOMA_RETARGETER_SOMA_X_CANONICALIZE_HEADING",
                config.get("soma_x_canonicalize_heading"),
            ),
            True,
        )
    if args.rebase_root_horizontal is None:
        args.rebase_root_horizontal = _bool_value(
            os.environ.get(
                "SOMA_RETARGETER_SOMA_X_REBASE_ROOT_HORIZONTAL",
                config.get("soma_x_rebase_root_horizontal"),
            ),
            True,
        )
    args.bvh_template = args.bvh_template or _optional_path(
        config.get("kimodo_npz_template_bvh")
    )
    args.bvh_offsets = args.bvh_offsets or _optional_path(
        config.get("kimodo_npz_offsets")
    )
    args.bvh_position_scale = float(
        args.bvh_position_scale
        if args.bvh_position_scale is not None
        else config.get("kimodo_npz_position_scale", 100.0)
    )
    if not args.inspect:
        require_soma_x_dependencies()
        if args.model is None:
            raise FileNotFoundError(
                "No licensed SMPL-family model was configured. Pass --model, set "
                "SOMA_RETARGETER_SOMA_X_HUMAN_MODEL, or set "
                "soma_x_human_model in the config."
            )
        if not args.model.is_file():
            raise FileNotFoundError(f"Human model file not found: {args.model}")
    elif args.model_type == "auto":
        raise ValueError(
            "--inspect without --model requires an explicit --model-type because "
            "SMPL-H and SMPL-X body-only motion schemas can be ambiguous"
        )
    return args


def motion_metadata(path: Path, motion) -> dict[str, object]:
    return {
        "input": str(path.expanduser()),
        "frames": motion.frame_count,
        "fps": motion.fps,
        "duration_seconds": motion.frame_count / motion.fps,
        "gender": motion.gender,
        "model_type": motion.model_type,
        "betas": list(motion.betas.shape),
        "has_expression": motion.has_expression,
        "source_coordinate": motion.source_coordinate,
        "translation_is_root_position": motion.transl_is_root_position,
    }


def build_directory_jobs(
    input_root: Path,
    output_root: Path,
    candidates: Iterable[Path],
) -> list[tuple[Path, Path]]:
    jobs = []
    for source in sorted(candidates):
        jobs.append((source, output_root / source.relative_to(input_root)))
    return jobs


def normalize_excluded_directories(exclude_dirs: Iterable[Path]) -> tuple[Path, ...]:
    normalized = []
    for exclude_dir in exclude_dirs:
        path = Path(exclude_dir)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"--exclude-dir must be relative to --input-dir: {path}")
        path = Path(*[part for part in path.parts if part not in ("", ".")])
        if not path.parts:
            raise ValueError("--exclude-dir cannot refer to the input root")
        if path not in normalized:
            normalized.append(path)
    return tuple(normalized)


def collect_directory_candidates(
    input_dir: Path,
    pattern: str,
    exclude_dirs: Iterable[Path] = (),
) -> list[Path]:
    input_root = input_dir.expanduser().resolve()
    if not input_root.is_dir():
        raise NotADirectoryError(input_root)
    excluded = normalize_excluded_directories(exclude_dirs)
    candidates = []
    for source in input_root.rglob(pattern):
        if not source.is_file():
            continue
        relative = source.relative_to(input_root)
        if any(relative == item or item in relative.parents for item in excluded):
            continue
        candidates.append(source)
    return candidates


def collect_directory_jobs(
    input_dir: Path,
    output_dir: Path,
    pattern: str,
    exclude_dirs: Iterable[Path] = (),
) -> list[tuple[Path, Path]]:
    input_root = input_dir.expanduser().resolve()
    output_root = output_dir.expanduser().resolve()
    if not input_root.is_dir():
        raise NotADirectoryError(input_root)
    if input_root == output_root:
        raise ValueError("--input-dir and --output-dir must be different")
    candidates = [
        source
        for source in collect_directory_candidates(input_root, pattern, exclude_dirs)
        if not source.resolve().is_relative_to(output_root)
    ]
    return build_directory_jobs(input_root, output_root, candidates)


def conversion_options(args: argparse.Namespace) -> dict[str, object]:
    return {
        "model_type": args.model_type,
        "device_name": args.device,
        "batch_size": args.batch_size,
        "body_iters": args.body_iters,
        "finger_iters": args.finger_iters,
        "full_iters": args.full_iters,
        "lie_iters": args.lie_iters,
        "lie_lambda": args.lie_lambda,
        "autograd_iters": args.autograd_iters,
        "autograd_lr": args.autograd_lr,
        "flat_hand_mean": args.flat_hand_mean,
        "fps_override": args.input_fps,
        "source_coordinate": args.source_coordinate,
        "canonicalize_heading": args.canonicalize_heading,
        "heading_yaw_degrees": args.heading_yaw_degrees,
        "rebase_root_horizontal": args.rebase_root_horizontal,
    }


def expected_signature(args: argparse.Namespace, input_path: Path) -> str:
    return build_conversion_signature(input_path, args.model, conversion_options(args))


def validate_existing_output(
    args: argparse.Namespace,
    input_path: Path,
    output_path: Path,
) -> tuple[bool, str]:
    try:
        expected = expected_signature(args, input_path)
        with np.load(output_path, allow_pickle=False) as data:
            missing = [field for field in REQUIRED_OUTPUT_FIELDS if field not in data.files]
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


def bvh_path_for_job(
    args: argparse.Namespace,
    input_path: Path,
    output_path: Path,
) -> Path | None:
    if not args.emit_bvh:
        return None
    if args.input is not None:
        return (args.bvh_output or output_path.with_suffix(".bvh")).expanduser()
    if args.bvh_output_dir is None:
        return output_path.with_suffix(".bvh")
    relative = input_path.relative_to(args.input_dir.expanduser().resolve())
    return args.bvh_output_dir.expanduser().resolve() / relative.with_suffix(".bvh")


def emit_bvh(args: argparse.Namespace, npz_path: Path, bvh_path: Path) -> dict[str, object]:
    fps = kimodo_npz.detect_npz_fps(npz_path)
    if fps is None:
        raise ValueError(f"Converted NPZ has no valid FPS metadata: {npz_path}")
    return kimodo_npz.convert_npz_to_bvh(
        npz_path,
        bvh_path,
        template_bvh=args.bvh_template,
        offsets=args.bvh_offsets,
        fps=fps,
        position_scale=args.bvh_position_scale,
        compare=False,
    )


def convert_one(
    args: argparse.Namespace,
    input_path: Path,
    output_path: Path,
    runtime_cache: dict,
    *,
    file_index: int | None = None,
    file_count: int | None = None,
) -> dict[str, object]:
    prefix = f"[{file_index}/{file_count}] " if file_index and file_count else ""
    arrays, metrics = convert_smpl_to_retarget_arrays(
        input_path,
        args.model,
        **conversion_options(args),
        runtime_cache=runtime_cache,
        progress=lambda current, total: print(
            f"{prefix}Processed {current}/{total} frames", flush=True
        ),
    )
    signature = expected_signature(args, input_path)
    arrays["conversion_signature"] = np.asarray(signature)
    save_retarget_npz(output_path, arrays, compressed=not args.no_compress)
    result: dict[str, object] = {
        "input": str(input_path),
        "output": str(output_path),
        "conversion_signature": signature,
        **asdict(metrics),
    }
    bvh_path = bvh_path_for_job(args, input_path, output_path)
    if bvh_path is not None:
        result["bvh"] = emit_bvh(args, output_path, bvh_path)
    print(json.dumps(result, indent=2), flush=True)
    return result


def _handle_existing_output(
    args: argparse.Namespace,
    input_path: Path,
    output_path: Path,
) -> dict[str, object]:
    valid, reason = validate_existing_output(args, input_path, output_path)
    if not valid:
        raise FileExistsError(
            f"Existing output is stale or invalid: {output_path} ({reason}). "
            "Pass --force to replace it."
        )
    result: dict[str, object] = {
        "input": str(input_path),
        "output": str(output_path),
        "status": "reused",
        "reason": reason,
    }
    bvh_path = bvh_path_for_job(args, input_path, output_path)
    if bvh_path is not None and not bvh_path.exists():
        result["bvh"] = emit_bvh(args, output_path, bvh_path)
        result["status"] = "reused_npz_emitted_bvh"
    return result


def run_single(args: argparse.Namespace) -> None:
    input_path = args.input.expanduser().resolve()
    motion = load_smpl_motion(
        input_path,
        model_type=args.model_type,
        fps_override=args.input_fps,
    )
    print(json.dumps(motion_metadata(input_path, motion), indent=2))
    if args.inspect:
        return
    output_path = args.output.expanduser().resolve()
    if input_path == output_path:
        raise ValueError("Input and output paths must be different")
    if output_path.exists() and not args.force:
        print(json.dumps(_handle_existing_output(args, input_path, output_path), indent=2))
        return
    convert_one(args, input_path, output_path, {})


def run_directory(args: argparse.Namespace) -> None:
    input_root = args.input_dir.expanduser().resolve()
    if args.inspect:
        candidates = sorted(
            collect_directory_candidates(input_root, args.pattern, args.exclude_dir)
        )
        if not candidates:
            raise FileNotFoundError(f"No files matching {args.pattern!r} below {input_root}")
        for path in candidates:
            try:
                motion = load_smpl_motion(
                    path,
                    model_type=args.model_type,
                    fps_override=args.input_fps,
                )
                print(json.dumps(motion_metadata(path, motion)))
            except Exception as exc:
                print(json.dumps({"input": str(path), "error": str(exc)}))
                if args.fail_fast:
                    raise
        return

    jobs = collect_directory_jobs(input_root, args.output_dir, args.pattern, args.exclude_dir)
    if not jobs:
        raise FileNotFoundError(f"No files matching {args.pattern!r} below {input_root}")
    runtime_cache: dict = {}
    converted: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    failed: list[dict[str, str]] = []
    started = time.perf_counter()
    for index, (input_path, output_path) in enumerate(jobs, start=1):
        try:
            if output_path.exists() and not args.force:
                skipped.append(_handle_existing_output(args, input_path, output_path))
                print(f"[{index}/{len(jobs)}] Reused validated output: {output_path}")
                continue
            converted.append(
                convert_one(
                    args,
                    input_path,
                    output_path,
                    runtime_cache,
                    file_index=index,
                    file_count=len(jobs),
                )
            )
        except Exception as exc:
            failure = {"input": str(input_path), "output": str(output_path), "error": str(exc)}
            failed.append(failure)
            print(json.dumps(failure), file=sys.stderr, flush=True)
            if args.fail_fast:
                raise
    summary = {
        "input_root": str(input_root),
        "output_root": str(args.output_dir.expanduser().resolve()),
        "pattern": args.pattern,
        "excluded_directories": [str(path) for path in args.exclude_dir],
        "discovered": len(jobs),
        "converted": len(converted),
        "reused": len(skipped),
        "failed": len(failed),
        "runtime_cache_entries": len(runtime_cache),
        "elapsed_seconds": time.perf_counter() - started,
        "failures": failed,
    }
    print(json.dumps(summary, indent=2))
    if failed:
        raise SystemExit(1)


def main(
    argv: list[str] | None = None,
    *,
    default_model_type: str = "smplx",
) -> None:
    args = configure_args(
        parse_args(argv, default_model_type=default_model_type)
    )
    if args.input is not None:
        run_single(args)
    else:
        run_directory(args)


if __name__ == "__main__":
    main(default_model_type="smplx")
