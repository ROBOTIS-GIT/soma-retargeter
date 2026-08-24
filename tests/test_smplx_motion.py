# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import importlib.metadata
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from soma_retargeter.assets.kimodo_npz import detect_npz_fps
from soma_retargeter.assets.smplx_motion import (
    C_AMASS_TO_KIMODO,
    SOMA_X_REQUIRED_VERSION,
    build_conversion_signature,
    canonical_heading_yaw_degrees,
    compute_anatomical_heading,
    inspect_human_model,
    load_smpl_motion,
    load_smplx_motion,
    normalize_retarget_heading_arrays,
    probe_soma_x_dependencies,
    rebase_root_horizontal_positions,
    resolve_smplx_model_path,
    resolve_human_model_path,
    resolve_soma_x_device,
    temp_retarget_npz_path_for_smplx,
    transform_to_kimodo_frame,
    yaw_rotation_y,
)
from tools.convert_smplx_to_retarget_npz import (
    bvh_path_for_job,
    build_directory_jobs,
    collect_directory_jobs,
    conversion_options,
    main as converter_main,
    run_directory,
    validate_existing_output,
)


class DirectoryConversionTests(unittest.TestCase):
    def test_preserves_recursive_relative_paths(self) -> None:
        input_root = Path("/input")
        output_root = Path("/output")
        jobs = build_directory_jobs(
            input_root,
            output_root,
            [
                input_root / "walk.npz",
                input_root / "nested" / "boxing.npz",
            ],
        )

        self.assertEqual(
            jobs,
            [
                (
                    input_root / "nested" / "boxing.npz",
                    output_root / "nested" / "boxing.npz",
                ),
                (input_root / "walk.npz", output_root / "walk.npz"),
            ],
        )

    def test_excludes_relative_directory_subtree(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_root = root / "input"
            output_root = root / "output"
            canonical = input_root / "CMU" / "walk_stageii.npz"
            derived = input_root / "K0_5" / "derived_stageii.npz"
            alias = input_root / "K0_5" / "alias_stageii.npz"
            canonical.parent.mkdir(parents=True)
            derived.parent.mkdir(parents=True)
            canonical.touch()
            derived.touch()
            alias.symlink_to(canonical)

            jobs = collect_directory_jobs(
                input_root,
                output_root,
                "*_stageii.npz",
                [Path("K0_5")],
            )

        self.assertEqual(
            jobs,
            [(canonical, output_root / "CMU" / "walk_stageii.npz")],
        )

    def test_reuses_one_runtime_cache_across_jobs(self) -> None:
        jobs = [
            (Path("/input/a.npz"), Path("/output/a.npz")),
            (Path("/input/nested/b.npz"), Path("/output/nested/b.npz")),
        ]
        cache_ids: list[int] = []

        def fake_convert_one(args, input_path, output_path, runtime_cache, **kwargs):
            cache_ids.append(id(runtime_cache))
            runtime_cache.setdefault("runtime", object())
            return {"input": str(input_path), "output": str(output_path)}

        args = SimpleNamespace(
            input_dir=Path("/input"),
            output_dir=Path("/output"),
            pattern="*.npz",
            exclude_dir=[],
            inspect=False,
            force=False,
            fail_fast=False,
        )
        with (
            patch(
                "tools.convert_smplx_to_retarget_npz.collect_directory_jobs",
                return_value=jobs,
            ),
            patch(
                "tools.convert_smplx_to_retarget_npz.convert_one",
                side_effect=fake_convert_one,
            ),
            patch("sys.stdout", new=io.StringIO()),
        ):
            run_directory(args)

        self.assertEqual(len(cache_ids), 2)
        self.assertEqual(len(set(cache_ids)), 1)


class HumanModelInspectionTests(unittest.TestCase):
    def test_detects_all_supported_model_families_structurally(self) -> None:
        cases = (
            ("smpl", 6890, 24, {}),
            ("smplh", 6890, 52, {"hands_componentsl": np.zeros((45, 45))}),
            ("smplx", 10475, 55, {"lmk_faces_idx": np.zeros(1)}),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for expected, vertices, joints, extra in cases:
                with self.subTest(model_type=expected):
                    model_path = root / f"model_{expected}.npz"
                    np.savez(
                        model_path,
                        v_template=np.zeros((vertices, 3), dtype=np.float32),
                        shapedirs=np.zeros((vertices, 3, 16), dtype=np.float32),
                        kintree_table=np.zeros((2, joints), dtype=np.int64),
                        **extra,
                    )
                    info = inspect_human_model(model_path)

                    self.assertEqual(info.model_type, expected)
                    self.assertEqual(info.vertex_count, vertices)
                    self.assertEqual(info.joint_count, joints)

    def test_rejects_requested_model_type_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "neutral.npz"
            np.savez(
                model_path,
                v_template=np.zeros((6890, 3), dtype=np.float32),
                shapedirs=np.zeros((6890, 3, 10), dtype=np.float32),
                kintree_table=np.zeros((2, 24), dtype=np.int64),
            )
            with self.assertRaisesRegex(ValueError, "model type mismatch"):
                inspect_human_model(model_path, "smplx")

    def test_generic_model_setting_precedes_legacy_smplx_setting(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            generic = root / "SMPL_NEUTRAL.npz"
            legacy = root / "SMPLX_NEUTRAL.npz"
            generic.touch()
            legacy.touch()
            with patch.dict(os.environ, {}, clear=True):
                resolved = resolve_human_model_path(
                    None,
                    {
                        "soma_x_human_model": str(generic),
                        "soma_x_smplx_model": str(legacy),
                    },
                )

        self.assertEqual(resolved, generic.resolve())


class SMPLXMotionLoaderTests(unittest.TestCase):
    def test_loads_smpl_combined_pose(self) -> None:
        with in_memory_npz(
            poses=np.zeros((2, 72), dtype=np.float32),
            trans=np.zeros((2, 3), dtype=np.float32),
        ) as path:
            motion = load_smpl_motion(path, model_type="smpl")

        self.assertEqual(motion.model_type, "smpl")
        self.assertEqual(motion.body_pose.shape, (2, 69))
        self.assertEqual(motion.left_hand_pose.shape, (2, 45))

    def test_loads_smplh_full_and_body_only_pose(self) -> None:
        for width in (156, 66):
            with self.subTest(width=width):
                with in_memory_npz(
                    poses=np.zeros((2, width), dtype=np.float32),
                    trans=np.zeros((2, 3), dtype=np.float32),
                ) as path:
                    motion = load_smpl_motion(path, model_type="smplh")

                self.assertEqual(motion.model_type, "smplh")
                self.assertEqual(motion.body_pose.shape, (2, 63))
                self.assertEqual(motion.left_hand_pose.shape, (2, 45))

    def test_rejects_motion_schema_for_selected_model(self) -> None:
        with in_memory_npz(
            poses=np.zeros((2, 165), dtype=np.float32),
            trans=np.zeros((2, 3), dtype=np.float32),
        ) as path:
            with self.assertRaisesRegex(ValueError, "SMPL poses must"):
                load_smpl_motion(path, model_type="smpl")

    def test_loads_full_stageii_pose_and_fps(self) -> None:
        with in_memory_npz(
            poses=np.zeros((3, 165), dtype=np.float32),
            trans=np.ones((3, 3), dtype=np.float32),
            betas=np.zeros(16, dtype=np.float32),
            gender=np.asarray("neutral"),
            mocap_frame_rate=np.float32(120.0),
        ) as path:
            motion = load_smplx_motion(path)

        self.assertEqual(motion.frame_count, 3)
        self.assertEqual(motion.body_pose.shape, (3, 63))
        self.assertEqual(motion.left_hand_pose.shape, (3, 45))
        self.assertEqual(motion.right_hand_pose.shape, (3, 45))
        self.assertEqual(motion.betas.shape, (1, 16))
        self.assertEqual(motion.fps, 120.0)
        self.assertEqual(motion.source_coordinate, "amass")
        self.assertFalse(motion.transl_is_root_position)

    def test_loads_stageii_separate_aliases(self) -> None:
        with in_memory_npz(
            root_orient=np.zeros((2, 3), dtype=np.float32),
            pose_body=np.zeros((2, 63), dtype=np.float32),
            pose_hand=np.zeros((2, 90), dtype=np.float32),
            pose_jaw=np.zeros((2, 3), dtype=np.float32),
            pose_eye=np.zeros((2, 6), dtype=np.float32),
            trans=np.zeros((2, 3), dtype=np.float32),
        ) as path:
            motion = load_smplx_motion(path)

        self.assertEqual(motion.left_hand_pose.shape, (2, 45))
        self.assertEqual(motion.right_hand_pose.shape, (2, 45))
        self.assertEqual(motion.leye_pose.shape, (2, 3))
        self.assertEqual(motion.reye_pose.shape, (2, 3))

    def test_loads_source_fps_alias(self) -> None:
        with in_memory_npz(
            poses=np.zeros((2, 165), dtype=np.float32),
            trans=np.zeros((2, 3), dtype=np.float32),
            source_fps=np.float32(59.94),
        ) as path:
            motion = load_smplx_motion(path)

        self.assertAlmostEqual(motion.fps, 59.94, places=4)

    def test_loads_kimodo_smplx_22_joint_matrix_export(self) -> None:
        local_rot_mats = np.broadcast_to(
            np.eye(3, dtype=np.float32), (3, 22, 3, 3)
        ).copy()
        root_angles = np.deg2rad([0.0, 15.0, 30.0])
        shoulder_angles = np.deg2rad([5.0, 10.0, 15.0])
        local_rot_mats[:, 0] = Rotation.from_rotvec(
            np.column_stack(
                [np.zeros(3), root_angles, np.zeros(3)]
            )
        ).as_matrix()
        local_rot_mats[:, 16] = Rotation.from_rotvec(
            np.column_stack(
                [np.zeros(3), np.zeros(3), shoulder_angles]
            )
        ).as_matrix()
        root_positions = np.array(
            [[1.0, 0.9, 2.0], [1.1, 0.95, 2.2], [1.2, 1.0, 2.4]],
            dtype=np.float32,
        )

        with in_memory_npz(
            local_rot_mats=local_rot_mats,
            root_positions=root_positions,
            frame_rate=np.float32(60.0),
        ) as path:
            motion = load_smplx_motion(path)

        self.assertEqual(motion.frame_count, 3)
        self.assertEqual(motion.global_orient.shape, (3, 3))
        self.assertEqual(motion.body_pose.shape, (3, 63))
        self.assertEqual(motion.left_hand_pose.shape, (3, 45))
        self.assertEqual(motion.right_hand_pose.shape, (3, 45))
        self.assertEqual(motion.source_coordinate, "kimodo")
        self.assertTrue(motion.transl_is_root_position)
        self.assertEqual(motion.fps, 60.0)
        np.testing.assert_allclose(motion.transl, root_positions)
        reconstructed = Rotation.from_rotvec(
            np.concatenate(
                [
                    motion.global_orient[:, None],
                    motion.body_pose.reshape(3, 21, 3),
                ],
                axis=1,
            ).reshape(-1, 3)
        ).as_matrix().reshape(3, 22, 3, 3)
        np.testing.assert_allclose(reconstructed, local_rot_mats, atol=1e-6)

    def test_uses_direct_translation_from_kimodo_matrix_export(self) -> None:
        local_rot_mats = np.broadcast_to(
            np.eye(3, dtype=np.float32), (2, 22, 3, 3)
        ).copy()
        transl = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32
        )

        with in_memory_npz(
            local_rot_mats=local_rot_mats,
            transl=transl,
        ) as path:
            motion = load_smplx_motion(path)

        self.assertFalse(motion.transl_is_root_position)
        np.testing.assert_allclose(motion.transl, transl)

    def test_loads_single_motion_batch_dimension_and_fps_override(self) -> None:
        local_rot_mats = np.broadcast_to(
            np.eye(3, dtype=np.float32), (1, 2, 22, 3, 3)
        ).copy()
        root_positions = np.array(
            [[[0.0, 1.0, 0.0], [0.1, 1.1, 0.2]]], dtype=np.float32
        )

        with in_memory_npz(
            local_rot_mats=local_rot_mats,
            root_positions=root_positions,
        ) as path:
            motion = load_smplx_motion(path, fps_override=50.0)

        self.assertEqual(motion.frame_count, 2)
        self.assertEqual(motion.fps, 50.0)
        np.testing.assert_allclose(motion.transl, root_positions[0])

    def test_rejects_unsupported_matrix_joint_count(self) -> None:
        local_rot_mats = np.broadcast_to(
            np.eye(3, dtype=np.float32), (2, 24, 3, 3)
        ).copy()

        with in_memory_npz(
            local_rot_mats=local_rot_mats,
            root_positions=np.zeros((2, 3), dtype=np.float32),
        ) as path:
            with self.assertRaisesRegex(
                ValueError, "expected 22.*or 77"
            ):
                load_smplx_motion(path)

    def test_rejects_already_converted_retarget_npz(self) -> None:
        with in_memory_npz(
            local_rot_mats=np.zeros((2, 77, 3, 3), dtype=np.float32),
            root_positions=np.zeros((2, 3), dtype=np.float32),
        ) as path:
            with self.assertRaisesRegex(ValueError, "already a Soma Retargeter"):
                load_smplx_motion(path)

    def test_rejects_soma_x_intermediate_pose_npz(self) -> None:
        with in_memory_npz(
            poses=np.zeros((2, 77, 3), dtype=np.float32),
            rotation_repr=np.asarray("rotvec"),
            identity_coeffs=np.zeros((1, 16), dtype=np.float32),
            transl=np.zeros((2, 3), dtype=np.float32),
        ) as path:
            with self.assertRaisesRegex(ValueError, "SOMA-X intermediate"):
                load_smplx_motion(path)


class CoordinateTransformTests(unittest.TestCase):
    def test_matches_sequential_amass_and_yaw_transforms(self) -> None:
        local = np.broadcast_to(np.eye(3, dtype=np.float32), (2, 77, 3, 3)).copy()
        global_mats = local.copy()
        root_positions = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]], dtype=np.float32)
        posed_joints = np.broadcast_to(root_positions[:, None], (2, 77, 3)).copy()

        actual_local, actual_global, actual_positions = transform_to_kimodo_frame(
            local,
            global_mats,
            {"root_positions": root_positions, "posed_joints": posed_joints},
            source_coordinate="amass",
            heading_yaw_degrees=180.0,
        )

        yaw = yaw_rotation_y(180.0)
        expected_local_root = yaw @ C_AMASS_TO_KIMODO
        expected_global = yaw @ C_AMASS_TO_KIMODO
        converted_root = root_positions @ C_AMASS_TO_KIMODO.T
        pivot = converted_root[0]
        expected_root = (converted_root - pivot) @ yaw.T + pivot

        np.testing.assert_allclose(
            actual_local[:, 0],
            np.broadcast_to(expected_local_root, (2, 3, 3)),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            actual_global,
            np.broadcast_to(expected_global, (2, 77, 3, 3)),
            atol=1e-6,
        )
        np.testing.assert_allclose(actual_positions["root_positions"], expected_root, atol=1e-6)

    def test_computes_anatomical_heading_from_soma77_landmarks(self) -> None:
        joints = soma77_joints_facing_positive_x()

        heading = compute_anatomical_heading(joints)

        np.testing.assert_allclose(
            heading,
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            atol=1e-6,
        )
        self.assertAlmostEqual(canonical_heading_yaw_degrees(joints), -90.0)

    def test_rebases_only_frame_zero_horizontal_root_offset(self) -> None:
        root_positions = np.array(
            [[1.0, 2.0, 3.0], [2.5, 2.25, 5.0]], dtype=np.float32
        )
        posed_joints = np.broadcast_to(
            root_positions[:, None], (2, 77, 3)
        ).copy()
        original_root = root_positions.copy()
        original_joints = posed_joints.copy()

        rebased, offset = rebase_root_horizontal_positions(
            {
                "root_positions": root_positions,
                "posed_joints": posed_joints,
            }
        )

        np.testing.assert_allclose(offset, [1.0, 0.0, 3.0])
        np.testing.assert_allclose(
            rebased["root_positions"],
            [[0.0, 2.0, 0.0], [1.5, 2.25, 2.0]],
        )
        np.testing.assert_allclose(
            rebased["posed_joints"][:, 0],
            rebased["root_positions"],
        )
        np.testing.assert_allclose(root_positions, original_root)
        np.testing.assert_allclose(posed_joints, original_joints)

    def test_normalizes_existing_motion_to_positive_z(self) -> None:
        joints = soma77_joints_facing_positive_x()
        joints += np.array([1.0, 2.0, 3.0], dtype=np.float32)
        local = np.broadcast_to(
            np.eye(3, dtype=np.float32), (1, 77, 3, 3)
        ).copy()
        global_mats = local.copy()
        arrays = {
            "local_rot_mats": local,
            "global_rot_mats": global_mats,
            "posed_joints": joints,
            "root_positions": joints[:, 0].copy(),
            "smooth_root_pos": joints[:, 0].copy(),
            "heading_correction_degrees": np.float32(180.0),
        }

        normalized, applied_yaw = normalize_retarget_heading_arrays(arrays)

        self.assertAlmostEqual(applied_yaw, -90.0)
        np.testing.assert_allclose(
            compute_anatomical_heading(normalized["posed_joints"]),
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            normalized["root_positions"][0],
            np.array([0.0, 2.0, 0.0], dtype=np.float32),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            normalized["posed_joints"][0, 0],
            normalized["root_positions"][0],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            normalized["root_horizontal_rebase_offset_m"],
            np.array([1.0, 0.0, 3.0], dtype=np.float32),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            normalized["local_rot_mats"][:, 1:],
            local[:, 1:],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            normalized["local_rot_mats"][:, 0],
            yaw_rotation_y(-90.0)[None],
            atol=1e-6,
        )
        self.assertTrue(bool(normalized["heading_canonicalized"]))
        self.assertAlmostEqual(
            float(normalized["heading_normalization_yaw_degrees"]), -90.0
        )
        self.assertAlmostEqual(
            float(normalized["heading_correction_degrees"]), 90.0
        )

        normalized_again, second_yaw = normalize_retarget_heading_arrays(
            normalized
        )
        self.assertAlmostEqual(second_yaw, 0.0, places=5)
        np.testing.assert_allclose(
            normalized_again["posed_joints"],
            normalized["posed_joints"],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            normalized_again["root_horizontal_rebase_normalization_m"],
            np.zeros(3, dtype=np.float32),
            atol=1e-6,
        )

    def test_rejects_degenerate_heading_landmarks(self) -> None:
        joints = np.zeros((1, 77, 3), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "no horizontal separation"):
            compute_anatomical_heading(joints)


class NPZFrameRateTests(unittest.TestCase):
    def test_detects_internal_fps(self) -> None:
        with in_memory_npz(fps=np.float32(120.0)) as path:
            self.assertEqual(detect_npz_fps(path), 120.0)

    def test_rejects_invalid_internal_fps(self) -> None:
        with in_memory_npz(fps=np.float32(0.0)) as path:
            self.assertIsNone(detect_npz_fps(path))


class SomaXOptionalDependencyTests(unittest.TestCase):
    def test_inspect_does_not_require_optional_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "motion_stageii.npz"
            np.savez(
                path,
                poses=np.zeros((1, 165), dtype=np.float32),
                trans=np.zeros((1, 3), dtype=np.float32),
            )
            with (
                patch(
                    "tools.convert_smplx_to_retarget_npz.require_soma_x_dependencies",
                    side_effect=AssertionError("optional runtime must stay lazy"),
                ),
                patch("sys.stdout", new=io.StringIO()),
            ):
                converter_main(["--input", str(path), "--inspect"])

    def test_reports_missing_optional_package_without_importing_it(self) -> None:
        with patch(
            "soma_retargeter.assets.smplx_motion.importlib.metadata.version",
            side_effect=importlib.metadata.PackageNotFoundError,
        ):
            status = probe_soma_x_dependencies()

        self.assertFalse(status.available)
        self.assertIsNone(status.version)
        self.assertIn("not installed", status.reason)

    def test_rejects_unpinned_soma_x_version(self) -> None:
        with patch(
            "soma_retargeter.assets.smplx_motion.importlib.metadata.version",
            return_value="9.9.9",
        ):
            status = probe_soma_x_dependencies()

        self.assertFalse(status.available)
        self.assertEqual(status.version, "9.9.9")
        self.assertIn(SOMA_X_REQUIRED_VERSION, status.reason)

    def test_auto_device_prefers_cuda_and_explicit_cpu_is_preserved(self) -> None:
        torch_with_cuda = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True)
        )
        torch_without_cuda = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False)
        )

        self.assertEqual(resolve_soma_x_device("auto", torch_with_cuda), "cuda:0")
        self.assertEqual(resolve_soma_x_device("auto", torch_without_cuda), "cpu")
        self.assertEqual(resolve_soma_x_device("cpu", torch_with_cuda), "cpu")
        with self.assertRaisesRegex(RuntimeError, "cannot access CUDA"):
            resolve_soma_x_device("cuda:0", torch_without_cuda)


class SomaXModelAndCacheTests(unittest.TestCase):
    def test_model_resolution_prefers_explicit_then_environment_then_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            explicit = root / "explicit.npz"
            environment = root / "environment.npz"
            configured = root / "configured.npz"
            for path in (explicit, environment, configured):
                path.touch()
            with patch.dict(
                os.environ,
                {"SOMA_RETARGETER_SOMA_X_SMPLX_MODEL": str(environment)},
            ):
                self.assertEqual(
                    resolve_smplx_model_path(explicit, {"soma_x_smplx_model": str(configured)}),
                    explicit.resolve(),
                )
                self.assertEqual(
                    resolve_smplx_model_path(None, {"soma_x_smplx_model": str(configured)}),
                    environment.resolve(),
                )
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(
                    resolve_smplx_model_path(None, {"soma_x_smplx_model": str(configured)}),
                    configured.resolve(),
                )

    def test_conversion_signature_changes_with_options(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "motion.npz"
            model = root / "SMPLX_NEUTRAL.npz"
            source.write_bytes(b"motion")
            model.write_bytes(b"model")

            first = build_conversion_signature(source, model, {"device_name": "cpu"})
            second = build_conversion_signature(source, model, {"device_name": "cuda:0"})
            cache_path = temp_retarget_npz_path_for_smplx(
                source,
                model,
                {"device_name": "cpu"},
            )

        self.assertNotEqual(first, second)
        self.assertEqual(cache_path.parent.name, "soma_retargeter_soma_x")
        self.assertIn(first[:16], cache_path.name)

    def test_existing_output_requires_matching_signature_and_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "motion.npz"
            model = root / "SMPLX_NEUTRAL.npz"
            output = root / "converted.npz"
            source.write_bytes(b"motion")
            model.write_bytes(b"model")
            args = conversion_args(model)
            signature = build_conversion_signature(
                source,
                model,
                conversion_options(args),
            )
            np.savez(
                output,
                local_rot_mats=np.zeros((1, 77, 3, 3), dtype=np.float32),
                global_rot_mats=np.zeros((1, 77, 3, 3), dtype=np.float32),
                posed_joints=np.zeros((1, 77, 3), dtype=np.float32),
                root_positions=np.zeros((1, 3), dtype=np.float32),
                fps=np.float32(30.0),
                conversion_signature=np.asarray(signature),
            )

            valid, reason = validate_existing_output(args, source, output)
            args.batch_size = 64
            stale, stale_reason = validate_existing_output(args, source, output)

        self.assertTrue(valid, reason)
        self.assertFalse(stale)
        self.assertIn("signature differs", stale_reason)

    def test_recursive_bvh_output_preserves_relative_path(self) -> None:
        args = SimpleNamespace(
            emit_bvh=True,
            input=None,
            input_dir=Path("/input"),
            bvh_output=None,
            bvh_output_dir=Path("/bvh"),
        )
        path = bvh_path_for_job(
            args,
            Path("/input/nested/walk_stageii.npz"),
            Path("/output/nested/walk_stageii.npz"),
        )

        self.assertEqual(path, Path("/bvh/nested/walk_stageii.bvh"))


def conversion_args(model: Path) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        model_type="smplx",
        device="auto",
        batch_size=32,
        body_iters=2,
        finger_iters=1,
        full_iters=1,
        lie_iters=3,
        lie_lambda=1e-1,
        autograd_iters=0,
        autograd_lr=5e-3,
        flat_hand_mean=True,
        input_fps=None,
        source_coordinate="auto",
        canonicalize_heading=True,
        heading_yaw_degrees=0.0,
        rebase_root_horizontal=True,
    )


@contextmanager
def in_memory_npz(**arrays):
    """Expose an NPZ through /proc without creating a disk artifact."""

    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    descriptor = os.memfd_create("soma_retargeter_test_npz")
    try:
        os.write(descriptor, buffer.getvalue())
        os.lseek(descriptor, 0, os.SEEK_SET)
        yield f"/proc/self/fd/{descriptor}"
    finally:
        os.close(descriptor)


def soma77_joints_facing_positive_x() -> np.ndarray:
    joints = np.zeros((1, 77, 3), dtype=np.float32)
    joints[0, 11] = np.array([0.0, 1.4, -0.3], dtype=np.float32)
    joints[0, 39] = np.array([0.0, 1.4, 0.3], dtype=np.float32)
    joints[0, 67] = np.array([0.0, 0.9, -0.2], dtype=np.float32)
    joints[0, 72] = np.array([0.0, 0.9, 0.2], dtype=np.float32)
    return joints


if __name__ == "__main__":
    unittest.main()
