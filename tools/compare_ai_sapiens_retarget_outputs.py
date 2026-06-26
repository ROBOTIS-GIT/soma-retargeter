#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare AI Sapiens CSV output against an older retarget MJCF branch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RETARGET_MJCF_PATH = "soma_retargeter/configs/ai_sapiens/ai_sapiens_retarget.xml"
RETARGET_CONFIG_PATH = REPO_ROOT / "soma_retargeter/configs/ai_sapiens/soma_to_ai_sapiens_retargeter_config.json"
CONVERTER_CONFIG_PATH = REPO_ROOT / "assets/default_ai_sapiens_bvh_to_csv_converter_config.json"
CURRENT_MJCF_PATH = REPO_ROOT / RETARGET_MJCF_PATH
OLD_MESH_DIR = REPO_ROOT / "assets/robot_assets/ai_sapiens/meshes"
DEFAULT_SAMPLES = [
    "Neutral_throw_ball_001__A057.bvh",
    "high_jump_R_001__A277.bvh",
    "dance_hiphop_shuffle_square_R_fast_002__A318.bvh",
]


def _run(cmd: list[str], *, input_bytes: bytes | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        input=input_bytes,
        capture_output=True,
        check=False,
    )


def _extract_old_mjcf(ref: str, output_path: Path) -> None:
    result = _run(["git", "show", f"{ref}:{RETARGET_MJCF_PATH}"])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode(errors="replace").strip())

    data = result.stdout
    if data.startswith(b"version https://git-lfs.github.com/spec/v1"):
        smudge = _run(["git", "lfs", "smudge"], input_bytes=data)
        if smudge.returncode != 0:
            raise RuntimeError(smudge.stderr.decode(errors="replace").strip())
        data = smudge.stdout

    output_path.write_bytes(data)
    tree = ET.parse(output_path)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("meshdir", str(OLD_MESH_DIR))
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def _truncate_bvh_to_one_frame(source_path: Path, output_path: Path) -> None:
    lines = source_path.read_text(encoding="utf-8").splitlines()
    frames_idx = next(i for i, line in enumerate(lines) if line.strip().startswith("Frames:"))
    frame_time_idx = next(i for i, line in enumerate(lines) if line.strip().startswith("Frame Time:"))
    data_idx = next(i for i in range(frame_time_idx + 1, len(lines)) if lines[i].strip())
    lines[frames_idx] = "Frames: 1"
    output_path.write_text("\n".join(lines[: data_idx + 1]) + "\n", encoding="utf-8")


def _write_converter_config(base_dir: Path, label: str, mjcf_path: Path, input_dir: Path) -> Path:
    retarget_config = json.loads(RETARGET_CONFIG_PATH.read_text(encoding="utf-8"))
    retarget_config["robot_mjcf"] = str(mjcf_path)
    retarget_config_path = base_dir / f"{label}_retargeter_config.json"
    retarget_config_path.write_text(json.dumps(retarget_config, indent=2), encoding="utf-8")

    converter_config = json.loads(CONVERTER_CONFIG_PATH.read_text(encoding="utf-8"))
    converter_config["import_folder"] = str(input_dir)
    converter_config["export_folder"] = str(base_dir / label)
    converter_config["timestamp_result_folder"] = False
    converter_config["retargeter_config"] = str(retarget_config_path)
    converter_config["ai_sapiens_mjcf"] = str(mjcf_path)
    converter_config_path = base_dir / f"{label}_converter_config.json"
    converter_config_path.write_text(json.dumps(converter_config, indent=2), encoding="utf-8")
    return converter_config_path


def _run_converter(config_path: Path, label: str) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "app/bvh_to_csv_converter.py"),
        "--config",
        str(config_path),
        "--viewer",
        "null",
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"{label} conversion failed.\nSTDOUT:\n{result.stdout[-4000:]}\nSTDERR:\n{result.stderr[-4000:]}"
        )


def _read_csv(path: Path) -> tuple[list[str], list[list[float]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [[float(value) for value in row] for row in reader]
    return header, rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compare_csv(old_path: Path, current_path: Path) -> dict:
    old_header, old_rows = _read_csv(old_path)
    current_header, current_rows = _read_csv(current_path)
    header_equal = old_header == current_header
    shape_old = [len(old_rows), len(old_header)]
    shape_current = [len(current_rows), len(current_header)]

    max_abs_diff = 0.0
    mean_abs_diff = 0.0
    max_col = ""
    old_value = 0.0
    current_value = 0.0
    count = 0
    if header_equal and shape_old == shape_current:
        for row_idx, (old_row, current_row) in enumerate(zip(old_rows, current_rows)):
            for col_idx, (old_item, current_item) in enumerate(zip(old_row, current_row)):
                diff = abs(current_item - old_item)
                mean_abs_diff += diff
                count += 1
                if diff > max_abs_diff:
                    max_abs_diff = diff
                    max_col = old_header[col_idx]
                    old_value = old_item
                    current_value = current_item
        if count:
            mean_abs_diff /= count

    return {
        "old_sha256": _sha256(old_path),
        "current_sha256": _sha256(current_path),
        "header_equal": header_equal,
        "shape_old": shape_old,
        "shape_current": shape_current,
        "identical": old_path.read_bytes() == current_path.read_bytes(),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "max_diff_column": max_col,
        "old_value": old_value,
        "current_value": current_value,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-ref", default="add-ai-sapiens-retargeting")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--samples", nargs="*", default=DEFAULT_SAMPLES)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    base_dir = args.output_root or Path(tempfile.mkdtemp(prefix="ai-sapiens-branch-compare."))
    base_dir.mkdir(parents=True, exist_ok=True)
    input_dir = base_dir / "input"
    input_dir.mkdir(exist_ok=True)

    old_mjcf_path = base_dir / "old_branch_ai_sapiens_retarget.xml"
    _extract_old_mjcf(args.old_ref, old_mjcf_path)

    for sample in args.samples:
        _truncate_bvh_to_one_frame(
            REPO_ROOT / "assets/motions/bvh" / sample,
            input_dir / sample,
        )

    old_config = _write_converter_config(base_dir, "old", old_mjcf_path, input_dir)
    current_config = _write_converter_config(base_dir, "current", CURRENT_MJCF_PATH, input_dir)

    print(f"[INFO]: Output root: {base_dir}")
    _run_converter(old_config, "old")
    _run_converter(current_config, "current")

    any_difference = False
    for sample in args.samples:
        csv_name = sample.removesuffix(".bvh") + ".csv"
        old_csv = base_dir / "old" / csv_name
        current_csv = base_dir / "current" / csv_name
        result = _compare_csv(old_csv, current_csv)
        any_difference = any_difference or not result["identical"]
        print(f"[RESULT]: {csv_name}")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    if any_difference:
        print("[SUMMARY]: 동일하지 않음")
    else:
        print("[SUMMARY]: 동일")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
