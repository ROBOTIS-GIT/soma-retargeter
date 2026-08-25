# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app.bvh_to_csv_converter import Viewer
from soma_retargeter.assets.motion_input import (
    MotionInputKind,
    classify_motion_input,
    convert_raw_smpl_to_soma_npz,
    plan_motion_jobs,
    resolve_npz_fps,
    validate_existing_soma_output,
)
from soma_retargeter.assets.smplx_motion import ConversionMetrics
from tools.convert_smplx_to_retarget_npz import convert_one


def identity_rotations(frames: int, joints: int) -> np.ndarray:
    identity = np.eye(3, dtype=np.float32)
    return np.broadcast_to(identity, (frames, joints, 3, 3)).copy()


class MotionInputClassificationTests(unittest.TestCase):
    def test_classifies_supported_inputs_by_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bvh = root / "walk.bvh"
            soma = root / "walk_soma.npz"
            matrices = root / "walk_matrices.npz"
            raw = root / "walk_raw.npz"
            bvh.write_text("HIERARCHY\n", encoding="utf-8")
            np.savez(
                soma,
                local_rot_mats=identity_rotations(2, 77),
                root_positions=np.zeros((2, 3), dtype=np.float32),
            )
            np.savez(
                matrices,
                local_rot_mats=identity_rotations(2, 22),
                root_positions=np.zeros((2, 3), dtype=np.float32),
            )
            np.savez(raw, poses=np.zeros((2, 165), dtype=np.float32))

            self.assertEqual(classify_motion_input(bvh), MotionInputKind.BVH)
            self.assertEqual(
                classify_motion_input(soma), MotionInputKind.SOMA77_NPZ
            )
            self.assertEqual(
                classify_motion_input(matrices), MotionInputKind.RAW_SMPL_NPZ
            )
            self.assertEqual(
                classify_motion_input(raw), MotionInputKind.RAW_SMPL_NPZ
            )

    def test_rejects_soma_x_intermediate_and_human_model_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            intermediate = root / "intermediate.npz"
            model = root / "SMPLX_NEUTRAL.npz"
            np.savez(intermediate, poses=np.zeros((2, 77, 3), dtype=np.float32))
            np.savez(
                model,
                v_template=np.zeros((10475, 3), dtype=np.float32),
                shapedirs=np.zeros((10475, 3, 10), dtype=np.float32),
            )

            with self.assertRaisesRegex(ValueError, "SOMA-X intermediate"):
                classify_motion_input(intermediate)
            with self.assertRaisesRegex(ValueError, "Human model file"):
                classify_motion_input(model)


class MotionInputPlanningTests(unittest.TestCase):
    def test_preserves_relative_paths_and_excludes_selected_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "input"
            output = root / "output"
            nested = source / "nested"
            nested.mkdir(parents=True)
            bvh = source / "walk.bvh"
            soma = nested / "jump.npz"
            model = source / "SMPLX_NEUTRAL.npz"
            bvh.write_text("HIERARCHY\n", encoding="utf-8")
            np.savez(
                soma,
                local_rot_mats=identity_rotations(2, 77),
                root_positions=np.zeros((2, 3), dtype=np.float32),
            )
            np.savez(
                model,
                v_template=np.zeros((1, 3), dtype=np.float32),
                shapedirs=np.zeros((1, 3, 1), dtype=np.float32),
            )

            jobs = plan_motion_jobs(source, output, human_model=model)

        self.assertEqual(
            [job.relative_path for job in jobs],
            [Path("nested/jump.npz"), Path("walk.bvh")],
        )
        self.assertEqual(
            jobs[0].csv_path,
            output / "retargeted_csv/nested/jump.csv",
        )
        self.assertEqual(
            jobs[0].bvh_path,
            output / "soma_bvh/nested/jump.bvh",
        )
        self.assertIsNone(jobs[0].soma_npz_path)
        self.assertEqual(
            jobs[1].csv_path,
            output / "retargeted_csv/walk.csv",
        )
        self.assertEqual(jobs[1].bvh_path, bvh)

    def test_raw_input_uses_semantic_output_directory_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "input" / "nested" / "motion.npz"
            output = root / "output"
            source.parent.mkdir(parents=True)
            np.savez(source, poses=np.zeros((2, 165), dtype=np.float32))

            jobs = plan_motion_jobs(root / "input", output)

        self.assertEqual(len(jobs), 1)
        self.assertEqual(
            jobs[0].soma_npz_path,
            output / "soma_npz/nested/motion.npz",
        )
        self.assertEqual(
            jobs[0].bvh_path,
            output / "soma_bvh/nested/motion.bvh",
        )
        self.assertEqual(
            jobs[0].csv_path,
            output / "retargeted_csv/nested/motion.csv",
        )

    def test_rejects_csv_destination_collision(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "input"
            source.mkdir()
            (source / "walk.bvh").write_text("HIERARCHY\n", encoding="utf-8")
            np.savez(
                source / "walk.npz",
                local_rot_mats=identity_rotations(1, 77),
                root_positions=np.zeros((1, 3), dtype=np.float32),
            )

            with self.assertRaisesRegex(ValueError, "same CSV output"):
                plan_motion_jobs(source, root / "output")

    def test_rejects_output_below_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "input"
            source.mkdir()
            (source / "walk.bvh").write_text("HIERARCHY\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "outside the input"):
                plan_motion_jobs(source, source / "output")


class MotionPreparationTests(unittest.TestCase):
    def test_npz_fps_precedence_is_override_metadata_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metadata = root / "metadata.npz"
            fallback = root / "fallback.npz"
            np.savez(metadata, fps=np.float32(60.0))
            np.savez(fallback, values=np.zeros(1))

            self.assertEqual(resolve_npz_fps(metadata), (60.0, "metadata"))
            self.assertEqual(resolve_npz_fps(metadata, 24.0), (24.0, "override"))
            self.assertEqual(resolve_npz_fps(fallback), (30.0, "fallback_30"))

    def test_shared_converter_saves_and_validates_signed_soma_output(self) -> None:
        metrics = ConversionMetrics(
            frame_count=2,
            fps=30.0,
            initialization_seconds=0.1,
            conversion_seconds=0.2,
            coordinate_seconds=0.01,
            total_seconds=0.31,
            conversion_frames_per_second=10.0,
            mean_vertex_error_m=0.001,
            median_vertex_error_m=0.001,
            max_vertex_error_m=0.002,
        )
        arrays = {
            "local_rot_mats": identity_rotations(2, 77),
            "global_rot_mats": identity_rotations(2, 77),
            "posed_joints": np.zeros((2, 77, 3), dtype=np.float32),
            "root_positions": np.zeros((2, 3), dtype=np.float32),
            "fps": np.float32(30.0),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "raw.npz"
            model = root / "model.npz"
            output = root / "output.npz"
            source.touch()
            model.touch()
            options = {"model_type": "smplx"}
            with (
                patch(
                    "soma_retargeter.assets.motion_input.smplx_motion.convert_smpl_to_retarget_arrays",
                    return_value=(arrays, metrics),
                ),
                patch(
                    "soma_retargeter.assets.motion_input.smplx_motion.build_conversion_signature",
                    return_value="signature",
                ),
            ):
                result = convert_raw_smpl_to_soma_npz(
                    source,
                    output,
                    model,
                    options,
                    runtime_cache={},
                )
                valid, reason = validate_existing_soma_output(
                    source, output, model, options
                )

        self.assertEqual(result["conversion_signature"], "signature")
        self.assertTrue(valid, reason)

    def test_standalone_converter_uses_shared_conversion_api(self) -> None:
        args = SimpleNamespace(
            model=Path("/model.npz"),
            no_compress=False,
            emit_bvh=False,
            input=Path("/input.npz"),
            model_type="smplx",
            device="cpu",
            batch_size=32,
            body_iters=2,
            finger_iters=1,
            full_iters=1,
            lie_iters=3,
            lie_lambda=0.1,
            autograd_iters=0,
            autograd_lr=0.005,
            flat_hand_mean=True,
            input_fps=None,
            source_coordinate="auto",
            canonicalize_heading=True,
            heading_yaw_degrees=0.0,
            rebase_root_horizontal=True,
        )
        with patch(
            "tools.convert_smplx_to_retarget_npz.motion_input.convert_raw_smpl_to_soma_npz",
            return_value={"input": "/input.npz", "output": "/output.npz"},
        ) as shared:
            convert_one(args, Path("/input.npz"), Path("/output.npz"), {})

        shared.assert_called_once()

    def test_headless_soma_x_device_uses_config_until_cli_overrides_it(self) -> None:
        viewer = Viewer.__new__(Viewer)
        viewer.config = {
            "soma_x_device": "auto",
            "soma_x_batch_size": 32,
            "soma_x_source_coordinate": "auto",
            "soma_x_canonicalize_heading": True,
            "soma_x_rebase_root_horizontal": True,
        }
        args = SimpleNamespace(
            device=None,
            input_fps=None,
            heading_yaw_degrees=0.0,
            soma_x_batch_size=None,
            source_coordinate=None,
            canonicalize_heading=None,
            rebase_root_horizontal=None,
        )

        default_options = viewer._headless_soma_x_options(args, "smplh")
        args.device = "cuda:0"
        override_options = viewer._headless_soma_x_options(args, "smplh")

        self.assertEqual(default_options["device_name"], "auto")
        self.assertEqual(override_options["device_name"], "cuda:0")


if __name__ == "__main__":
    unittest.main()
