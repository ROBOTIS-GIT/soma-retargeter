#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Canonicalize existing SOMA77 NPZ heading and horizontal root origin."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from soma_retargeter.assets.smplx_motion import (
    compute_anatomical_heading,
    normalize_retarget_heading_arrays,
    save_retarget_npz,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--input-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Atomically replace each input NPZ instead of writing a copy.",
    )
    parser.add_argument("--pattern", default="*_stageii.npz")
    parser.add_argument("--additional-yaw-degrees", type=float, default=0.0)
    parser.add_argument(
        "--rebase-root-horizontal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Move frame-zero root X/Z to the Kimodo horizontal origin while "
            "preserving height and relative motion (default: enabled)."
        ),
    )
    parser.add_argument("--no-compress", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    if args.in_place and (args.output is not None or args.output_dir is not None):
        parser.error("--in-place cannot be combined with --output or --output-dir")
    if args.input is not None:
        if args.output_dir is not None:
            parser.error("--output-dir requires --input-dir")
        if not args.in_place and args.output is None:
            parser.error("--output is required unless --in-place is used")
    else:
        if args.output is not None:
            parser.error("--output requires --input")
        if not args.in_place and args.output_dir is None:
            parser.error("--output-dir is required unless --in-place is used")
    return args


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as source:
        return {key: np.array(source[key], copy=True) for key in source.files}


def normalize_one(
    input_path: Path,
    output_path: Path,
    *,
    additional_yaw_degrees: float,
    rebase_root_horizontal: bool,
    compressed: bool,
) -> dict[str, object]:
    arrays = load_arrays(input_path)
    before = compute_anatomical_heading(arrays["posed_joints"])
    normalized, applied_yaw = normalize_retarget_heading_arrays(
        arrays,
        additional_yaw_degrees=additional_yaw_degrees,
        rebase_root_horizontal=rebase_root_horizontal,
    )
    after = compute_anatomical_heading(normalized["posed_joints"])
    save_retarget_npz(output_path, normalized, compressed=compressed)
    result = {
        "input": str(input_path),
        "output": str(output_path),
        "source_heading_xz": [float(before[0]), float(before[2])],
        "applied_yaw_degrees": applied_yaw,
        "output_heading_xz": [float(after[0]), float(after[2])],
        "root_horizontal_rebase_m": [
            float(value)
            for value in np.asarray(
                normalized.get(
                    "root_horizontal_rebase_normalization_m",
                    np.zeros(3, dtype=np.float32),
                )
            ).reshape(3)
        ],
    }
    print(json.dumps(result))
    return result


def main() -> None:
    args = parse_args()
    if args.input is not None:
        input_path = args.input.expanduser().resolve()
        output_path = input_path if args.in_place else args.output.expanduser().resolve()
        if output_path.exists() and output_path != input_path and not args.force:
            raise FileExistsError(f"Output exists: {output_path}")
        normalize_one(
            input_path,
            output_path,
            additional_yaw_degrees=args.additional_yaw_degrees,
            rebase_root_horizontal=args.rebase_root_horizontal,
            compressed=not args.no_compress,
        )
        return

    input_root = args.input_dir.expanduser().resolve()
    output_root = (
        input_root if args.in_place else args.output_dir.expanduser().resolve()
    )
    if not input_root.is_dir():
        raise NotADirectoryError(input_root)
    candidates = sorted(
        path
        for path in input_root.rglob(args.pattern)
        if path.is_file()
        and (
            args.in_place
            or not path.resolve().is_relative_to(output_root)
        )
    )
    if not candidates:
        raise FileNotFoundError(
            f"No files matching {args.pattern!r} below {input_root}"
        )

    converted = 0
    skipped = 0
    failures: list[dict[str, str]] = []
    for index, input_path in enumerate(candidates, start=1):
        output_path = (
            input_path
            if args.in_place
            else output_root / input_path.relative_to(input_root)
        )
        if output_path.exists() and output_path != input_path and not args.force:
            skipped += 1
            print(f"[{index}/{len(candidates)}] Skipped existing: {output_path}")
            continue
        try:
            normalize_one(
                input_path,
                output_path,
                additional_yaw_degrees=args.additional_yaw_degrees,
                rebase_root_horizontal=args.rebase_root_horizontal,
                compressed=not args.no_compress,
            )
            converted += 1
        except Exception as exc:
            failures.append({"input": str(input_path), "error": str(exc)})
            if args.fail_fast:
                raise

    print(
        json.dumps(
            {
                "discovered": len(candidates),
                "normalized": converted,
                "skipped": skipped,
                "failed": len(failures),
                "failures": failures,
            },
            indent=2,
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
