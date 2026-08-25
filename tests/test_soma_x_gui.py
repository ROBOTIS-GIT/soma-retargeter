# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import tempfile
import unittest
import inspect
from collections import deque
from pathlib import Path
from unittest.mock import Mock, patch

from app.bvh_to_csv_converter import Viewer
from soma_retargeter.assets.smplx_motion import (
    HumanModelMotionMismatchError,
    HumanModelInfo,
    SomaXDependencyStatus,
)


class SomaXGuiProcessTests(unittest.TestCase):
    def make_viewer(self, model: Path) -> Viewer:
        viewer = Viewer.__new__(Viewer)
        viewer.config = {
            "soma_x_device": "cpu",
            "soma_x_batch_size": 8,
            "soma_x_source_coordinate": "amass",
            "soma_x_canonicalize_heading": True,
            "soma_x_rebase_root_horizontal": True,
        }
        viewer.soma_x_model_path = model
        viewer.soma_x_model_info = HumanModelInfo(
            path=model,
            model_type="smplx",
            display_name="SMPL-X",
            vertex_count=10475,
            joint_count=55,
            shape_coefficient_count=300,
        )
        viewer.soma_x_dependency_status = SomaXDependencyStatus(
            True,
            "0.2.1",
            "available",
        )
        viewer.soma_x_motion_path = None
        viewer.soma_x_process = None
        viewer.soma_x_reader_thread = None
        viewer.soma_x_output_path = None
        viewer.soma_x_retarget_ready = False
        viewer.soma_x_conversion_percent = None
        viewer.soma_x_error_message = None
        viewer.soma_x_log_lines = deque(maxlen=30)
        viewer.loaded_motion_kind = None
        viewer.animation_buffers = []
        viewer.skeleton_instances = []
        return viewer

    def test_missing_optional_runtime_reports_install_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "motion_stageii.npz"
            model = root / "SMPLX_NEUTRAL.npz"
            source.touch()
            model.touch()
            viewer = self.make_viewer(model)
            viewer._show_gui_error = Mock()

            with (
                patch(
                    "app.bvh_to_csv_converter.smplx_motion_utils.probe_soma_x_dependencies",
                    return_value=SomaXDependencyStatus(
                        False,
                        None,
                        "py-soma-x is not installed",
                    ),
                ),
                patch(
                    "app.bvh_to_csv_converter.smplx_motion_utils.soma_x_install_command",
                    return_value="python -m pip install -e '.[soma-x]'",
                ),
                patch("app.bvh_to_csv_converter.subprocess.Popen") as popen,
            ):
                viewer.start_soma_x_conversion(source)

            popen.assert_not_called()
            viewer._show_gui_error.assert_called_once()
            self.assertIn(
                "python -m pip install -e '.[soma-x]'",
                viewer._show_gui_error.call_args.args[1],
            )
            self.assertIsNone(viewer.soma_x_process)
            self.assertFalse(viewer.soma_x_dependency_status.available)

    def test_starts_converter_in_a_separate_process(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "motion_stageii.npz"
            model = root / "SMPLX_NEUTRAL.npz"
            source.write_bytes(b"motion")
            model.write_bytes(b"model")
            viewer = self.make_viewer(model)
            process = Mock()

            with (
                patch(
                    "app.bvh_to_csv_converter.smplx_motion_utils.probe_soma_x_dependencies",
                    return_value=SomaXDependencyStatus(True, "0.2.1", "available"),
                ),
                patch(
                    "app.bvh_to_csv_converter.subprocess.Popen",
                    return_value=process,
                ) as popen,
                patch("app.bvh_to_csv_converter.threading.Thread") as thread,
            ):
                viewer.start_soma_x_conversion(source)

            command = popen.call_args.args[0]
            self.assertIn("tools/convert_smpl_to_retarget_npz.py", command[1])
            self.assertEqual(command[command.index("--input") + 1], str(source))
            self.assertEqual(command[command.index("--model") + 1], str(model))
            self.assertEqual(command[command.index("--model-type") + 1], "smplx")
            self.assertEqual(command[command.index("--device") + 1], "cpu")
            self.assertEqual(command[command.index("--batch-size") + 1], "8")
            self.assertIn("--canonicalize-heading", command)
            self.assertIn("--rebase-root-horizontal", command)
            self.assertIs(viewer.soma_x_process, process)
            self.assertEqual(viewer.soma_x_conversion_percent, 0)
            self.assertEqual(popen.call_args.kwargs["stderr"], subprocess.STDOUT)
            thread.return_value.start.assert_called_once_with()

    def test_successful_process_loads_converted_npz_as_soma_x(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / "SMPLX_NEUTRAL.npz"
            model.touch()
            viewer = self.make_viewer(model)
            viewer.soma_x_output_path = root / "converted.npz"
            viewer.soma_x_process = Mock(poll=Mock(return_value=0))
            viewer.soma_x_reader_thread = Mock()
            viewer.load_npz_file = Mock()

            viewer._poll_soma_x_process()

            viewer.load_npz_file.assert_called_once_with(
                viewer.soma_x_output_path,
                motion_kind="soma_x",
            )
            self.assertTrue(viewer.soma_x_retarget_ready)
            self.assertEqual(viewer.soma_x_conversion_percent, 100)
            self.assertIsNone(viewer.soma_x_error_message)
            self.assertIsNone(viewer.soma_x_process)

    def test_model_motion_mismatch_is_shown_inline_without_popup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "smplx_motion.npz"
            source.touch()
            viewer = self.make_viewer(root / "SMPLH_NEUTRAL.npz")
            viewer.soma_x_model_info = HumanModelInfo(
                path=viewer.soma_x_model_path,
                model_type="smplh",
                display_name="SMPL-H",
                vertex_count=6890,
                joint_count=52,
                shape_coefficient_count=16,
            )
            viewer._show_gui_error = Mock()

            with (
                patch(
                    "app.bvh_to_csv_converter.smplx_motion_utils.probe_soma_x_dependencies",
                    return_value=SomaXDependencyStatus(True, "0.2.1", "available"),
                ),
                patch(
                    "app.bvh_to_csv_converter.smplx_motion_utils.load_smpl_motion",
                    side_effect=HumanModelMotionMismatchError("mismatch"),
                ),
                patch("app.bvh_to_csv_converter.subprocess.Popen") as popen,
            ):
                viewer.start_soma_x_conversion(source)

        popen.assert_not_called()
        viewer._show_gui_error.assert_not_called()
        self.assertEqual(
            viewer.soma_x_error_message,
            "Human model and motion do not match.",
        )
        self.assertFalse(viewer.soma_x_retarget_ready)
        self.assertIsNone(viewer.soma_x_conversion_percent)

    def test_control_state_follows_model_motion_retarget_sequence(self) -> None:
        viewer = Viewer.__new__(Viewer)
        viewer.soma_x_model_info = None
        viewer.soma_x_dependency_status = SomaXDependencyStatus(
            True,
            "0.2.1",
            "available",
        )
        viewer.soma_x_process = None
        viewer.soma_x_retarget_ready = False

        self.assertEqual(
            viewer._soma_x_controls_enabled(),
            {"human_model": True, "motion": False, "retarget": False},
        )
        self.assertEqual(viewer._soma_x_model_label(), "")
        self.assertEqual(viewer._soma_x_model_summary(), "")
        self.assertEqual(viewer._soma_x_model_tooltip(), "")

        viewer.soma_x_model_info = HumanModelInfo(
            path=Path("/model.pkl"),
            model_type="smplh",
            display_name="SMPL-H",
            vertex_count=6890,
            joint_count=52,
            shape_coefficient_count=16,
        )
        self.assertEqual(
            viewer._soma_x_controls_enabled(),
            {"human_model": True, "motion": True, "retarget": False},
        )
        self.assertEqual(viewer._soma_x_model_label(), "SMPL-H")
        self.assertEqual(
            viewer._soma_x_model_summary(),
            "SMPL-H | model.pkl",
        )
        self.assertEqual(
            viewer._soma_x_model_tooltip(),
            "/model.pkl\nVertices: 6,890\nJoints: 52\nShape coefficients: 16",
        )

        viewer.soma_x_process = Mock()
        self.assertEqual(
            viewer._soma_x_controls_enabled(),
            {"human_model": False, "motion": False, "retarget": False},
        )

        viewer.soma_x_process = None
        viewer.soma_x_retarget_ready = True
        self.assertEqual(
            viewer._soma_x_controls_enabled(),
            {"human_model": True, "motion": True, "retarget": True},
        )

    def test_unavailable_runtime_disables_all_soma_x_controls(self) -> None:
        viewer = self.make_viewer(Path("/models/SMPLX_NEUTRAL.npz"))
        viewer.soma_x_retarget_ready = True
        viewer.soma_x_dependency_status = SomaXDependencyStatus(
            False,
            None,
            "py-soma-x is not installed",
        )

        self.assertEqual(
            viewer._soma_x_controls_enabled(),
            {"human_model": False, "motion": False, "retarget": False},
        )

    def test_draws_unavailable_runtime_and_install_command_in_red(self) -> None:
        viewer = self.make_viewer(Path("/models/SMPLX_NEUTRAL.npz"))
        viewer.soma_x_dependency_status = SomaXDependencyStatus(
            False,
            None,
            "py-soma-x is not installed",
        )
        ui = Mock()
        color = ui.ImVec4.return_value

        with patch(
            "app.bvh_to_csv_converter.smplx_motion_utils.soma_x_install_command",
            return_value="python -m pip install -e '.[soma-x]'",
        ):
            viewer._draw_soma_x_model_info(ui)

        ui.ImVec4.assert_called_once_with(1.0, 0.2, 0.2, 1.0)
        ui.text_colored.assert_called_once_with(
            color,
            "SOMA-X is unavailable: py-soma-x is not installed\n"
            "Install with: python -m pip install -e '.[soma-x]'",
        )
        ui.set_tooltip.assert_not_called()

    def test_draws_human_model_info_on_a_small_dedicated_line(self) -> None:
        viewer = self.make_viewer(Path("/models/SMPLX_NEUTRAL.npz"))
        ui = Mock()
        ui.get_font_size.return_value = 15.0
        ui.is_item_hovered.return_value = True
        color = ui.ImVec4.return_value

        viewer._draw_soma_x_model_info(ui)

        pushed_font, pushed_size = ui.push_font.call_args.args
        self.assertIsNone(pushed_font)
        self.assertAlmostEqual(pushed_size, 12.0)
        ui.ImVec4.assert_called_once_with(1.0, 1.0, 1.0, 1.0)
        ui.text_colored.assert_called_once_with(
            color,
            "SMPL-X | SMPLX_NEUTRAL.npz",
        )
        ui.text.assert_not_called()
        ui.text_disabled.assert_not_called()
        ui.pop_font.assert_called_once_with()
        ui.set_tooltip.assert_called_once_with(
            "/models/SMPLX_NEUTRAL.npz\n"
            "Vertices: 10,475\n"
            "Joints: 55\n"
            "Shape coefficients: 300"
        )

    def test_draws_conversion_percentage_next_to_model_info(self) -> None:
        viewer = self.make_viewer(Path("/models/SMPLX_NEUTRAL.npz"))
        viewer.soma_x_conversion_percent = 37
        ui = Mock()
        ui.get_font_size.return_value = 15.0
        ui.is_item_hovered.return_value = False
        color = ui.ImVec4.return_value

        viewer._draw_soma_x_model_info(ui)

        ui.text_colored.assert_called_once_with(
            color,
            "SMPL-X | SMPLX_NEUTRAL.npz | 37%",
        )
        ui.text.assert_not_called()
        ui.text_disabled.assert_not_called()

    def test_draws_model_motion_mismatch_in_red(self) -> None:
        viewer = self.make_viewer(Path("/models/SMPLH_NEUTRAL.npz"))
        viewer.soma_x_error_message = "Human model and motion do not match."
        ui = Mock()
        ui.is_item_hovered.return_value = True
        color = ui.ImVec4.return_value

        viewer._draw_soma_x_model_info(ui)

        ui.ImVec4.assert_called_once_with(1.0, 0.2, 0.2, 1.0)
        ui.text_colored.assert_called_once_with(
            color,
            "Human model and motion do not match.",
        )
        ui.set_tooltip.assert_not_called()

    def test_parses_frame_conversion_progress_as_percentage(self) -> None:
        viewer = self.make_viewer(Path("/models/SMPLX_NEUTRAL.npz"))

        viewer._update_soma_x_progress_from_line("Processed 16/64 frames")
        self.assertEqual(viewer.soma_x_conversion_percent, 25)

        viewer._update_soma_x_progress_from_line(
            "[2/8] Processed 64/64 frames"
        )
        self.assertEqual(viewer.soma_x_conversion_percent, 100)

    def test_draws_supported_model_info_before_selection(self) -> None:
        viewer = Viewer.__new__(Viewer)
        viewer.soma_x_model_info = None
        viewer.soma_x_dependency_status = SomaXDependencyStatus(
            True,
            "0.2.1",
            "available",
        )
        ui = Mock()
        ui.get_font_size.return_value = 15.0
        color = ui.ImVec4.return_value

        viewer._draw_soma_x_model_info(ui)

        ui.text_colored.assert_called_once_with(
            color,
            "Supported models: SMPL / SMPL-H / SMPL-X",
        )
        ui.text.assert_not_called()
        ui.text_disabled.assert_not_called()
        ui.set_tooltip.assert_not_called()

    def test_failed_conversion_preserves_model_and_disables_retarget(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            viewer = self.make_viewer(Path(temp_dir) / "SMPLX_NEUTRAL.npz")
            model_info = viewer.soma_x_model_info
            viewer.soma_x_retarget_ready = True
            viewer.soma_x_process = Mock(poll=Mock(return_value=2))
            viewer.soma_x_reader_thread = Mock()
            viewer._show_gui_error = Mock()

            viewer._poll_soma_x_process()

        self.assertIs(viewer.soma_x_model_info, model_info)
        self.assertFalse(viewer.soma_x_retarget_ready)
        self.assertIsNone(viewer.soma_x_conversion_percent)
        self.assertTrue(viewer._soma_x_controls_enabled()["motion"])
        viewer._show_gui_error.assert_called_once()

    def test_subprocess_model_motion_mismatch_does_not_open_popup(self) -> None:
        viewer = self.make_viewer(Path("/models/SMPLH_NEUTRAL.npz"))
        viewer.soma_x_process = Mock(poll=Mock(return_value=2))
        viewer.soma_x_reader_thread = Mock()
        viewer.soma_x_log_lines.append(
            "HumanModelMotionMismatchError: SMPL-H model does not match motion"
        )
        viewer._show_gui_error = Mock()

        viewer._poll_soma_x_process()

        viewer._show_gui_error.assert_not_called()
        self.assertEqual(
            viewer.soma_x_error_message,
            "Human model and motion do not match.",
        )

    def test_model_replacement_invalidates_old_conversion_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            viewer = self.make_viewer(root / "old.npz")
            viewer.soma_x_motion_path = root / "old_motion.npz"
            viewer.soma_x_output_path = root / "old_output.npz"
            viewer.soma_x_retarget_ready = True
            replacement = HumanModelInfo(
                path=root / "new.pkl",
                model_type="smpl",
                display_name="SMPL",
                vertex_count=6890,
                joint_count=24,
                shape_coefficient_count=10,
            )
            with patch(
                "app.bvh_to_csv_converter.smplx_motion_utils.validate_human_model",
                return_value=replacement,
            ):
                viewer.set_soma_x_model(replacement.path)

        self.assertIs(viewer.soma_x_model_info, replacement)
        self.assertIsNone(viewer.soma_x_motion_path)
        self.assertIsNone(viewer.soma_x_output_path)
        self.assertFalse(viewer.soma_x_retarget_ready)
        self.assertIsNone(viewer.soma_x_conversion_percent)

    def test_gui_source_has_no_soma_x_runtime_status_text(self) -> None:
        source = inspect.getsource(Viewer.ui_scene_options)
        for forbidden in ("SOMA-X: Idle", "Converting", "Ready to retarget"):
            self.assertNotIn(forbidden, source)
        self.assertIn("self._draw_soma_x_model_info(ui)", source)


if __name__ == "__main__":
    unittest.main()
