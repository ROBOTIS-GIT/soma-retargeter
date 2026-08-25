# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import newton

import json
import pathlib
import re
import subprocess
import sys
import threading
import time
from collections import deque
import warp as wp
import numpy as np
from scipy.spatial.transform import Rotation as R

import soma_retargeter.utils.math_utils as math_utils
import soma_retargeter.assets.bvh as bvh_utils
import soma_retargeter.assets.csv as csv_utils
import soma_retargeter.assets.ai_sapiens as ai_sapiens_assets
import soma_retargeter.assets.kimodo_npz as kimodo_npz_utils
import soma_retargeter.assets.motion_input as motion_input_utils
import soma_retargeter.assets.smplx_motion as smplx_motion_utils
import soma_retargeter.utils.io_utils as io_utils
import soma_retargeter.pipelines.utils as pipeline_utils

from soma_retargeter.renderers.skeleton_renderer import SkeletonRenderer
from soma_retargeter.renderers.mesh_renderer import SkeletalMeshRenderer
from soma_retargeter.renderers.coordinate_renderer import CoordinateRenderer
from soma_retargeter.animation.skeleton import SkeletonInstance
from soma_retargeter.utils.space_conversion_utils import SpaceConverter, get_facing_direction_type_from_str

from tqdm import trange

_UI_NEWTON_PANEL_WIDTH  = 320
_UI_NEWTON_PANEL_MARGIN = 10
_UI_NEWTON_PANEL_ALPHA  = 0.9
_UI_SOMA_X_MODEL_INFO_FONT_SIZE = 12.0
_UI_SOMA_X_MODEL_INFO_COLOR = (1.0, 1.0, 1.0, 1.0)
_UI_SOMA_X_ERROR_INFO_COLOR = (1.0, 0.2, 0.2, 1.0)
_SOMA_X_SUPPORTED_MODEL_INFO = "Supported models: SMPL / SMPL-H / SMPL-X"
_SOMA_X_MODEL_MOTION_MISMATCH_INFO = "Human model and motion do not match."
_SOMA_X_FRAME_PROGRESS_PATTERN = re.compile(
    r"Processed\s+(?P<current>\d+)/(?P<total>\d+)\s+frames"
)
_DEFAULT_COLOR = (235.0 / 255.0, 245.0 / 255.0, 112.0 / 255.0)
_AI_SAPIENS_VIEWER_DEFAULT_ORIENTATION_YAW_DEG = -90.0


def _is_ai_sapiens_target(target: str) -> bool:
    return target == "ai_sapiens"


def _float_config_or_env(config: dict, config_key: str, env_key: str, default: float) -> float:
    if os.environ.get(env_key) is not None:
        return float(os.environ[env_key])
    if config_key in config:
        return float(config[config_key])
    return float(default)


def _bool_config_or_env(config: dict, config_key: str, env_key: str, default: bool) -> bool:
    if os.environ.get(env_key) is not None:
        value = os.environ[env_key].strip().lower()
        return value in {"1", "true", "yes", "on"}
    if config_key in config:
        value = config[config_key]
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    return bool(default)


def _str_config_or_env(config: dict, config_key: str, env_key: str, default: str) -> str:
    if os.environ.get(env_key) is not None:
        return str(os.environ[env_key])
    if config_key in config:
        return str(config[config_key])
    return str(default)


def _ai_sapiens_viewer_default_robot_offset(config: dict) -> wp.transform:
    viewer_orientation_yaw_deg = _float_config_or_env(
        config,
        "ai_sapiens_viewer_default_orientation_yaw_deg",
        "SOMA_RETARGETER_AI_SAPIENS_VIEWER_DEFAULT_ORIENTATION_YAW_DEG",
        _AI_SAPIENS_VIEWER_DEFAULT_ORIENTATION_YAW_DEG,
    )
    if abs(viewer_orientation_yaw_deg) <= 1e-12:
        return wp.transform_identity()
    return wp.transform(
        wp.vec3(0.0, 0.0, 0.0),
        wp.quat_from_axis_angle(
            wp.vec3(0.0, 0.0, 1.0),
            wp.radians(viewer_orientation_yaw_deg),
        ),
    )


def _load_retargeter_config(config: dict):
    retargeter_config_path = config.get("retargeter_config")
    if not retargeter_config_path:
        return None

    retargeter_config_path = pathlib.Path(retargeter_config_path)
    if not retargeter_config_path.is_absolute() and not retargeter_config_path.exists():
        retargeter_config_path = pathlib.Path(io_utils.get_config_file(str(retargeter_config_path)))
    with open(retargeter_config_path, "r", encoding="utf-8") as config_file:
        retargeter_config = json.load(config_file)
    print(f"[INFO]: Loaded retargeter config [{retargeter_config_path}]")
    return retargeter_config


def _ai_sapiens_mjcf_config_value(config: dict, retargeter_config: dict | None = None):
    return (
        (retargeter_config or {}).get("robot_mjcf")
        or config.get("ai_sapiens_mjcf")
    )


def _apply_target_yaw_to_pipeline(
    retarget_pipeline,
    *,
    position_yaw_deg: float,
    orientation_yaw_deg: float,
    pivot_mode: str = "origin",
):
    """Rotate every IK target transform by a fixed world yaw."""
    position_yaw_deg = float(position_yaw_deg)
    orientation_yaw_deg = float(orientation_yaw_deg)
    if abs(position_yaw_deg) <= 1e-12 and abs(orientation_yaw_deg) <= 1e-12:
        return

    pivot_mode = str(pivot_mode)
    position_yaw = R.from_euler("z", position_yaw_deg, degrees=True)
    orientation_yaw = R.from_euler("z", orientation_yaw_deg, degrees=True)

    def _rotate_targets(targets):
        target_array = np.asarray(targets, dtype=np.float64).copy()
        if target_array.ndim != 3 or target_array.shape[2] < 7:
            raise ValueError(
                f"AI Sapiens target transform array must have shape (T, N, 7), got {target_array.shape}"
            )
        if pivot_mode == "origin":
            pivot = np.zeros((1, 1, 3), dtype=np.float64)
        elif pivot_mode in {"first_root", "first_hips"}:
            root_idx = retarget_pipeline.mapped_joints.index("Hips")
            pivot = target_array[0:1, root_idx:root_idx + 1, 0:3].copy()
        elif pivot_mode in {"per_frame_root", "per_frame_hips"}:
            root_idx = retarget_pipeline.mapped_joints.index("Hips")
            pivot = target_array[:, root_idx:root_idx + 1, 0:3].copy()
        else:
            raise ValueError(
                "ai_sapiens_target_yaw_pivot must be one of: origin, first_root, per_frame_root"
            )
        if abs(position_yaw_deg) > 1e-12:
            rel_pos = target_array[:, :, 0:3] - pivot
            target_array[:, :, 0:3] = (
                position_yaw.apply(rel_pos.reshape(-1, 3)).reshape(rel_pos.shape) + pivot
            )
        quats = target_array[:, :, 3:7].reshape(-1, 4)
        if abs(orientation_yaw_deg) > 1e-12:
            target_array[:, :, 3:7] = (
                orientation_yaw * R.from_quat(quats)
            ).as_quat().reshape(target_array.shape[0], target_array.shape[1], 4)
        return target_array.astype(np.float32)

    for env_idx, targets in enumerate(retarget_pipeline.input_targets):
        retarget_pipeline.input_targets[env_idx] = _rotate_targets(targets)

    if hasattr(retarget_pipeline, "input_target_stage_traces"):
        for env_idx, trace in enumerate(retarget_pipeline.input_target_stage_traces):
            if env_idx >= len(retarget_pipeline.input_targets):
                continue
            for key, value in list(trace.items()):
                if key.startswith("target_"):
                    trace[key] = _rotate_targets(value)


def _apply_ai_sapiens_output_convention(config: dict, buffer, mjcf_config_value=None):
    root_translation_yaw_deg = _float_config_or_env(
        config,
        "ai_sapiens_root_translation_yaw_deg",
        "SOMA_RETARGETER_AI_SAPIENS_ROOT_TRANSLATION_YAW_DEG",
        0.0,
    )
    root_orientation_yaw_deg = _float_config_or_env(
        config,
        "ai_sapiens_root_orientation_yaw_deg",
        "SOMA_RETARGETER_AI_SAPIENS_ROOT_ORIENTATION_YAW_DEG",
        0.0,
    )
    root_translation_xy_scale = _float_config_or_env(
        config,
        "ai_sapiens_root_translation_xy_scale",
        "SOMA_RETARGETER_AI_SAPIENS_ROOT_TRANSLATION_XY_SCALE",
        1.0,
    )
    ai_sapiens_assets.apply_root_convention_to_buffer(
        buffer,
        root_translation_yaw_deg=root_translation_yaw_deg,
        root_orientation_yaw_deg=root_orientation_yaw_deg,
        root_translation_xy_scale=root_translation_xy_scale,
    )

    ground_align = _bool_config_or_env(
        config,
        "ai_sapiens_ground_align",
        "SOMA_RETARGETER_AI_SAPIENS_GROUND_ALIGN",
        False,
    )
    if ground_align:
        mjcf_path = ai_sapiens_assets.resolve_ai_sapiens_mjcf_path(mjcf_config_value)
        ai_sapiens_assets.apply_ground_alignment_to_buffer(
            buffer,
            mjcf_path=mjcf_path,
            ground_z=_float_config_or_env(
                config,
                "ai_sapiens_ground_z",
                "SOMA_RETARGETER_AI_SAPIENS_GROUND_Z",
                0.0,
            ),
        )


class Viewer:
    def __init__(self, viewer, config):
        self.viewer = viewer
        self.viewer.vsync = True
        self.config = config
        self.converter = SpaceConverter(get_facing_direction_type_from_str(self.config['retarget_source_facing_direction']))

        if isinstance(self.viewer, newton.viewer.ViewerNull):
            # Headless mode for batch processing
            return
        
        self.fps      = 60
        self.frame_dt = 1.0 / self.fps
        self.time     = 0.0

        self.is_playing          = True
        self.playback_time       = 0.0
        self.playback_speed      = 1.0
        self.playback_loop       = True
        self.playback_total_time = 0.0

        self.retarget_source_options = ['soma']
        self.retarget_target_options = ['unitree_g1', 'ai_sapiens']
        self.retarget_solver_options = ['Newton']
        self.retarget_solver_idx     = 0
        self.retarget_target_idx     = 0
        self.retarget_source_idx     = 0
        if self.config.get('retarget_target') in self.retarget_target_options:
            self.retarget_target_idx = self.retarget_target_options.index(self.config['retarget_target'])
        if self.config.get('retarget_source') in self.retarget_source_options:
            self.retarget_source_idx = self.retarget_source_options.index(self.config['retarget_source'])

        self.show_skeleton_mesh = True
        self.show_skeleton = False
        self.show_skeleton_joint_axes = False
        self.show_gizmos = True

        self.viewer.renderer.set_title("BVH to CSV Converter")
        self.viewer.register_ui_callback(lambda ui: self.gui(ui), position="free")

        target_builder = newton.ModelBuilder()
        if self.retarget_target_options[self.retarget_target_idx] == "unitree_g1":
            target_builder.add_mjcf(
                newton.utils.download_asset("unitree_g1") / "mjcf/g1_29dof_rev_1_0.xml")
        elif _is_ai_sapiens_target(self.retarget_target_options[self.retarget_target_idx]):
            target_builder.add_mjcf(
                ai_sapiens_assets.resolve_ai_sapiens_mjcf_path(self.config.get("ai_sapiens_mjcf")))
        else:
            raise ValueError(f"[ERROR]: Unknown retarget target [{self.retarget_target_options[self.retarget_target_idx]}].")
        
        self.num_robots = 1
        self.robot_offsets = [wp.transform(wp.vec3(0.0, i - (self.num_robots - 1) / 2.0, 0.0), wp.quat_identity()) for i in range(self.num_robots)]
        self.robot_default_display_offsets = [wp.transform_identity() for _ in range(self.num_robots)]
        if _is_ai_sapiens_target(self.retarget_target_options[self.retarget_target_idx]):
            ai_sapiens_offset = _ai_sapiens_viewer_default_robot_offset(self.config)
            self.robot_default_display_offsets = [ai_sapiens_offset for _ in range(self.num_robots)]
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        for _ in range(self.num_robots):
            builder.add_builder(target_builder, wp.transform_identity())
        self.model = builder.finalize()

        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets([0, 0, 0])
        self.state = self.model.state()

        self.robot_num_joint_q = self.model.joint_coord_count // self.model.articulation_count
        self.robot_joint_q_offsets = [int(i * self.robot_num_joint_q) for i in range(self.model.articulation_count)]
        self.robot_default_joint_q_values = self.model.joint_q.numpy()

        self.coordinate_renderer = CoordinateRenderer()
        self.skeleton = None
        self.skeleton_renderer = None
        self.skeletal_mesh_renderer = None
        self.loaded_npz_path = None
        self.loaded_npz_converted_bvh_path = None
        self.loaded_motion_kind = None
        self.soma_x_model_path = None
        self.soma_x_model_info = None
        self.soma_x_dependency_status = (
            smplx_motion_utils.probe_soma_x_dependencies()
        )
        self.soma_x_motion_path = None
        self.soma_x_process = None
        self.soma_x_reader_thread = None
        self.soma_x_output_path = None
        self.soma_x_retarget_ready = False
        self.soma_x_conversion_percent = None
        self.soma_x_error_message = None
        self.soma_x_log_lines = deque(maxlen=30)

        self.animation_offsets = []
        self.animation_buffers = []
        self.skeleton_instances = []
        self.robot_csv_animation_buffers = [None for _ in range(self.num_robots)]

    def gui(self, ui):
        self._poll_soma_x_process()
        self.ui_playback_controls(ui)
        self.ui_scene_options(ui)

    def load_csv_file(self, path):
        self.robot_csv_animation_buffers[0] = csv_utils.load_csv(
            path,
            csv_config=csv_utils.get_csv_config_for_target(
                self.retarget_target_options[self.retarget_target_idx]))
        self.compute_playback_total_time()

    def load_bvh_file(self, path, *, motion_kind="bvh"):
        self.animation_buffers = []
        self.skeleton_instances = []
        if self.skeleton_renderer is not None:
            self.skeleton_renderer.clear(self.viewer)
        if self.skeletal_mesh_renderer is not None:
            self.skeletal_mesh_renderer.clear(self.viewer)
        if self.coordinate_renderer is not None:
            self.coordinate_renderer.clear(self.viewer)

        self.skeleton, animation = bvh_utils.load_bvh(path)
        self.skeleton_renderer = SkeletonRenderer(self.skeleton, [0])
        self.skeleton_instances = [SkeletonInstance(self.skeleton, _DEFAULT_COLOR, self.converter.transform(wp.transform_identity()))]
        self.animation_offsets = [wp.transform_identity()] * len(self.skeleton_instances)
        self.animation_buffers = [animation]

        self.skeletal_mesh = pipeline_utils.get_source_model_mesh(pipeline_utils.SourceType.SOMA, self.skeleton)
        self.skeletal_mesh_renderer = SkeletalMeshRenderer(self.skeletal_mesh)
        self.loaded_motion_kind = motion_kind
        self.compute_playback_total_time()

    def load_npz_file(self, path, *, motion_kind="npz"):
        npz_path = pathlib.Path(path).expanduser()
        output_bvh = kimodo_npz_utils.temp_bvh_path_for_npz(npz_path)
        offsets = (
            os.environ.get("SOMA_RETARGETER_KIMODO_NPZ_OFFSETS")
            or self.config.get("kimodo_npz_offsets")
        )
        template_bvh = (
            os.environ.get("SOMA_RETARGETER_KIMODO_NPZ_TEMPLATE_BVH")
            or self.config.get("kimodo_npz_template_bvh")
        )
        fps = (
            float(os.environ["SOMA_RETARGETER_KIMODO_NPZ_FPS"])
            if os.environ.get("SOMA_RETARGETER_KIMODO_NPZ_FPS") is not None
            else self.config.get("kimodo_npz_fps")
        )
        if fps is None:
            fps = kimodo_npz_utils.detect_npz_fps(npz_path)
        if fps is None:
            fps = kimodo_npz_utils.detect_sidecar_bvh_fps(npz_path)
        if fps is None:
            fps = 30.0
        position_scale = _float_config_or_env(
            self.config,
            "kimodo_npz_position_scale",
            "SOMA_RETARGETER_KIMODO_NPZ_POSITION_SCALE",
            100.0,
        )
        compare = _bool_config_or_env(
            self.config,
            "kimodo_npz_compare",
            "SOMA_RETARGETER_KIMODO_NPZ_COMPARE",
            False,
        )
        result = kimodo_npz_utils.convert_npz_to_bvh(
            npz_path,
            output_bvh,
            template_bvh=template_bvh,
            offsets=offsets,
            fps=fps,
            position_scale=position_scale,
            compare=compare,
        )
        self.loaded_npz_path = npz_path
        self.loaded_npz_converted_bvh_path = pathlib.Path(result["output_bvh"])
        print(
            f"[INFO]: Converted NPZ [{npz_path}] to fixed BVH "
            f"[{self.loaded_npz_converted_bvh_path}] "
            f"frames={result['frames']} fps={result['fps']:.6g}")
        if compare:
            print(
                "[INFO]: NPZ Euler round-trip error deg "
                f"max={result['euler_roundtrip_error_max_deg']:.6g} "
                f"p99={result['euler_roundtrip_error_p99_deg']:.6g} "
                f"mean={result['euler_roundtrip_error_mean_deg']:.6g}")
        self.load_bvh_file(
            str(self.loaded_npz_converted_bvh_path),
            motion_kind=motion_kind,
        )

    def _show_gui_error(self, title, message):
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        try:
            messagebox.showerror(title, message, parent=root)
        finally:
            root.destroy()

    def _soma_x_conversion_options(self):
        return {
            "device_name": _str_config_or_env(
                self.config,
                "soma_x_device",
                "SOMA_RETARGETER_SOMA_X_DEVICE",
                "auto",
            ),
            "batch_size": int(
                _float_config_or_env(
                    self.config,
                    "soma_x_batch_size",
                    "SOMA_RETARGETER_SOMA_X_BATCH_SIZE",
                    32,
                )
            ),
            "body_iters": 2,
            "finger_iters": 1,
            "full_iters": 1,
            "lie_iters": 3,
            "lie_lambda": 1e-1,
            "autograd_iters": 0,
            "autograd_lr": 5e-3,
            "flat_hand_mean": True,
            "fps_override": None,
            "source_coordinate": _str_config_or_env(
                self.config,
                "soma_x_source_coordinate",
                "SOMA_RETARGETER_SOMA_X_SOURCE_COORDINATE",
                "auto",
            ),
            "canonicalize_heading": _bool_config_or_env(
                self.config,
                "soma_x_canonicalize_heading",
                "SOMA_RETARGETER_SOMA_X_CANONICALIZE_HEADING",
                True,
            ),
            "heading_yaw_degrees": 0.0,
            "rebase_root_horizontal": _bool_config_or_env(
                self.config,
                "soma_x_rebase_root_horizontal",
                "SOMA_RETARGETER_SOMA_X_REBASE_ROOT_HORIZONTAL",
                True,
            ),
        }

    def _invalidate_soma_x_motion(self):
        self.soma_x_motion_path = None
        self.soma_x_output_path = None
        self.soma_x_retarget_ready = False
        self.soma_x_conversion_percent = None
        self.soma_x_error_message = None
        if getattr(self, "loaded_motion_kind", None) != "soma_x":
            return
        for renderer_name in (
            "skeleton_renderer",
            "skeletal_mesh_renderer",
            "coordinate_renderer",
        ):
            renderer = getattr(self, renderer_name, None)
            if renderer is not None and hasattr(renderer, "clear"):
                renderer.clear(self.viewer)
        self.animation_buffers = []
        self.skeleton_instances = []
        self.loaded_motion_kind = None
        self.skeleton = None

    def set_soma_x_model(self, path):
        model = pathlib.Path(path).expanduser().resolve()
        info = smplx_motion_utils.validate_human_model(model)
        self._invalidate_soma_x_motion()
        self.soma_x_model_path = model
        self.soma_x_model_info = info
        return info

    def _select_soma_x_model(self):
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        try:
            selected = filedialog.askopenfilename(
                title="Select licensed SMPL-family human model",
                defaultextension=".npz",
                filetypes=[
                    ("SMPL-family model", "*.npz *.pkl"),
                    ("NPZ model", "*.npz"),
                    ("Pickle model", "*.pkl"),
                ],
                parent=root,
            )
        finally:
            root.destroy()
        if not selected:
            return None
        return pathlib.Path(selected).expanduser().resolve()

    def choose_soma_x_model(self):
        try:
            selected = self._select_soma_x_model()
            if selected is not None:
                self.set_soma_x_model(selected)
        except Exception as exc:
            self._show_gui_error("SOMA-X human model error", str(exc))

    def choose_soma_x_motion(self):
        import tkinter as tk
        from tkinter import filedialog

        if self.soma_x_model_info is None or self.soma_x_process is not None:
            return
        root = tk.Tk()
        root.withdraw()
        try:
            selected = filedialog.askopenfilename(
                title="Load raw SMPL-family motion",
                defaultextension=".npz",
                filetypes=[("NPZ motion", "*.npz")],
                parent=root,
            )
        finally:
            root.destroy()
        if selected:
            self.start_soma_x_conversion(selected)

    def _soma_x_controls_enabled(self):
        dependency_status = getattr(
            self,
            "soma_x_dependency_status",
            None,
        )
        soma_x_available = bool(
            dependency_status is not None and dependency_status.available
        )
        converting = self.soma_x_process is not None
        return {
            "human_model": soma_x_available and not converting,
            "motion": (
                soma_x_available
                and not converting
                and self.soma_x_model_info is not None
            ),
            "retarget": (
                soma_x_available
                and not converting
                and self.soma_x_retarget_ready
            ),
        }

    def _soma_x_dependency_message(self):
        status = getattr(self, "soma_x_dependency_status", None)
        if status is None or status.available:
            return ""
        return (
            f"SOMA-X is unavailable: {status.reason}\n"
            f"Install with: {smplx_motion_utils.soma_x_install_command()}"
        )

    def _soma_x_model_label(self):
        if self.soma_x_model_info is None:
            return ""
        return self.soma_x_model_info.display_name

    def _soma_x_model_summary(self):
        if self.soma_x_model_info is None:
            return ""
        info = self.soma_x_model_info
        return f"{info.display_name} | {info.path.name}"

    def _soma_x_model_tooltip(self):
        if self.soma_x_model_info is None:
            return ""
        info = self.soma_x_model_info
        return (
            f"{info.path}\n"
            f"Vertices: {info.vertex_count:,}\n"
            f"Joints: {info.joint_count}\n"
            f"Shape coefficients: {info.shape_coefficient_count}"
        )

    def _soma_x_progress_label(self):
        percent = getattr(self, "soma_x_conversion_percent", None)
        if percent is None:
            return ""
        return f"{max(0, min(100, int(percent)))}%"

    def _update_soma_x_progress_from_line(self, line):
        match = _SOMA_X_FRAME_PROGRESS_PATTERN.search(line)
        if match is None:
            return
        current = int(match.group("current"))
        total = int(match.group("total"))
        if total <= 0:
            return
        self.soma_x_conversion_percent = min(100, (current * 100) // total)

    def _draw_soma_x_model_info(self, ui):
        dependency_message = self._soma_x_dependency_message()
        error_message = getattr(self, "soma_x_error_message", None)
        summary = self._soma_x_model_summary()
        progress = self._soma_x_progress_label()
        text = (
            dependency_message
            or error_message
            or summary
            or _SOMA_X_SUPPORTED_MODEL_INFO
        )
        color = (
            _UI_SOMA_X_ERROR_INFO_COLOR
            if dependency_message or error_message
            else _UI_SOMA_X_MODEL_INFO_COLOR
        )
        if progress and not dependency_message and not error_message:
            text = f"{summary} | {progress}" if summary else progress
        ui.push_font(
            None,
            _UI_SOMA_X_MODEL_INFO_FONT_SIZE,
        )
        try:
            ui.text_colored(ui.ImVec4(*color), text)
        finally:
            ui.pop_font()
        if (
            summary
            and not dependency_message
            and not error_message
            and ui.is_item_hovered()
        ):
            ui.set_tooltip(self._soma_x_model_tooltip())

    def _preflight_soma_x_motion_model(self, source):
        try:
            smplx_motion_utils.load_smpl_motion(
                source,
                model_type=self.soma_x_model_info.model_type,
            )
        except smplx_motion_utils.HumanModelMotionMismatchError:
            self.soma_x_error_message = _SOMA_X_MODEL_MOTION_MISMATCH_INFO
            return False
        except Exception:
            # The converter keeps responsibility for all non-compatibility errors.
            pass
        return True

    @staticmethod
    def _is_soma_x_model_motion_mismatch(details):
        return "HumanModelMotionMismatchError" in details

    def start_soma_x_conversion(self, path):
        if self.soma_x_process is not None:
            return
        if self.soma_x_model_info is None or self.soma_x_model_path is None:
            self._show_gui_error(
                "SOMA-X human model required",
                "Select a licensed SMPL-family human model first.",
            )
            return
        self.soma_x_motion_path = pathlib.Path(path).expanduser().resolve()
        self.soma_x_output_path = None
        self.soma_x_retarget_ready = False
        self.soma_x_conversion_percent = None
        self.soma_x_error_message = None
        status = smplx_motion_utils.probe_soma_x_dependencies()
        self.soma_x_dependency_status = status
        if not status.available:
            self._show_gui_error(
                "SOMA-X is unavailable",
                f"{status.reason}\n\nInstall with:\n"
                f"{smplx_motion_utils.soma_x_install_command()}",
            )
            return

        try:
            source = pathlib.Path(path).expanduser().resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            if not self._preflight_soma_x_motion_model(source):
                return
            model = self.soma_x_model_path
            options = self._soma_x_conversion_options()
            options["model_type"] = self.soma_x_model_info.model_type
            output = smplx_motion_utils.temp_retarget_npz_path_for_smpl(
                source,
                model,
                options,
            )
            repo_root = pathlib.Path(__file__).resolve().parents[1]
            command = [
                sys.executable,
                str(repo_root / "tools/convert_smpl_to_retarget_npz.py"),
                "--input",
                str(source),
                "--output",
                str(output),
                "--model",
                str(model),
                "--model-type",
                self.soma_x_model_info.model_type,
                "--device",
                str(options["device_name"]),
                "--batch-size",
                str(options["batch_size"]),
                "--source-coordinate",
                str(options["source_coordinate"]),
                (
                    "--canonicalize-heading"
                    if options["canonicalize_heading"]
                    else "--no-canonicalize-heading"
                ),
                (
                    "--rebase-root-horizontal"
                    if options["rebase_root_horizontal"]
                    else "--no-rebase-root-horizontal"
                ),
            ]
            environment = os.environ.copy()
            python_path = environment.get("PYTHONPATH")
            environment["PYTHONPATH"] = (
                str(repo_root)
                if not python_path
                else f"{repo_root}{os.pathsep}{python_path}"
            )
            self.soma_x_motion_path = source
            self.soma_x_output_path = output
            self.soma_x_retarget_ready = False
            self.soma_x_conversion_percent = 0
            self.soma_x_error_message = None
            self.soma_x_log_lines.clear()
            self.soma_x_process = subprocess.Popen(
                command,
                cwd=repo_root,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self.soma_x_reader_thread = threading.Thread(
                target=self._read_soma_x_process_output,
                daemon=True,
            )
            self.soma_x_reader_thread.start()
        except Exception as exc:
            self.soma_x_process = None
            self.soma_x_retarget_ready = False
            self.soma_x_conversion_percent = None
            self._show_gui_error("SOMA-X conversion error", str(exc))

    def _read_soma_x_process_output(self):
        process = self.soma_x_process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            line = line.rstrip()
            if not line:
                continue
            print(line, flush=True)
            self.soma_x_log_lines.append(line)
            self._update_soma_x_progress_from_line(line)

    def _poll_soma_x_process(self):
        process = self.soma_x_process
        if process is None:
            return
        return_code = process.poll()
        if return_code is None:
            return
        if self.soma_x_reader_thread is not None:
            self.soma_x_reader_thread.join(timeout=0.2)
        self.soma_x_process = None
        if return_code == 0:
            try:
                self.load_npz_file(
                    self.soma_x_output_path,
                    motion_kind="soma_x",
                )
                self.soma_x_retarget_ready = True
                self.soma_x_conversion_percent = 100
                self.soma_x_error_message = None
            except Exception as exc:
                self.soma_x_retarget_ready = False
                self.soma_x_conversion_percent = None
                self._show_gui_error("SOMA-X load error", str(exc))
            return
        self.soma_x_retarget_ready = False
        self.soma_x_conversion_percent = None
        details = "\n".join(self.soma_x_log_lines)
        if self._is_soma_x_model_motion_mismatch(details):
            self.soma_x_error_message = _SOMA_X_MODEL_MOTION_MISMATCH_INFO
            return
        self._show_gui_error(
            "SOMA-X conversion failed",
            details or f"Converter exited with status {return_code}",
        )

    def _stop_soma_x_process(self):
        process = self.soma_x_process
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)
        self.soma_x_process = None
        self.soma_x_conversion_percent = None

    def compute_playback_total_time(self):
        bvh_max_time = 0.0
        for buffer in self.animation_buffers:
            if buffer is not None:
                bvh_max_time = max(bvh_max_time, buffer.num_frames * (1 / buffer.sample_rate))
        
        csv_max_time = 0.0
        for buffer in self.robot_csv_animation_buffers:
            if buffer is not None:
                csv_max_time = max(csv_max_time, buffer.num_frames * (1 / buffer.sample_rate))

        self.playback_total_time = max(bvh_max_time, csv_max_time)
        self.playback_time = wp.clamp(self.playback_time, 0.0, self.playback_total_time)

    def update_robot_states(self):
        for i in range(self.num_robots):
            robot_offset = self.robot_offsets[i]

            joint_q_offset = self.robot_joint_q_offsets[i]
            if self.robot_csv_animation_buffers[i] is not None:
                buffer = self.robot_csv_animation_buffers[i]
                # Apply visual offset
                prev_xform = wp.transform(buffer.xform)
                buffer.xform = robot_offset

                data = buffer.sample(self.playback_time)
                wp.copy(self.model.joint_q, wp.array(data, dtype=wp.float32), joint_q_offset, 0, self.robot_num_joint_q)
                buffer.xform = prev_xform
            else:
                root_tx = wp.mul(
                    robot_offset,
                    wp.mul(
                        self.robot_default_display_offsets[i],
                        wp.transform(*self.robot_default_joint_q_values[joint_q_offset:(joint_q_offset + 7)])))

                wp.copy(
                    self.model.joint_q,
                    wp.array(self.robot_default_joint_q_values[joint_q_offset:(joint_q_offset + self.robot_num_joint_q)], dtype=wp.float32),
                    joint_q_offset,
                    0, self.robot_num_joint_q)
                wp.copy(self.model.joint_q, wp.array(root_tx[0:7], dtype=wp.float32), joint_q_offset, 0, 7)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state, None)

    def step(self):
        self.time += self.frame_dt
        if self.is_playing:
            self.playback_time += self.frame_dt * self.playback_speed
            if self.playback_loop and self.playback_total_time > 0.0:
                self.playback_time %= self.playback_total_time
            else:
                self.playback_time = max(0.0, min(self.playback_time, self.playback_total_time))

        for i in range(len(self.animation_buffers)):
            self.skeleton_instances[i].set_local_transforms(self.animation_buffers[i].sample(self.playback_time))

        def clamp_gizmo_transform(tx: wp.transform):
            return wp.transform(
                wp.vec3(tx.p[0], tx.p[1], 0.0),
                math_utils.quat_twist(wp.vec3(0.0, 0.0, 1.0), tx.q))

        for i in range(len(self.robot_offsets)):
            self.robot_offsets[i] = clamp_gizmo_transform(self.robot_offsets[i])
        for i in range(len(self.animation_offsets)):
            self.animation_offsets[i] = clamp_gizmo_transform(self.animation_offsets[i])

        self.update_robot_states()

    def render(self):
        self.viewer.begin_frame(self.time)
        if len(self.animation_buffers) > 0:
            for i in range(len(self.skeleton_instances)):
                prev_xform = wp.transform(self.skeleton_instances[i].xform)
                self.skeleton_instances[i].xform = wp.mul(self.animation_offsets[i], self.skeleton_instances[i].xform)
                if self.show_skeleton:
                    self.skeleton_renderer.draw(self.viewer, self.skeleton_instances[i], i)
                if self.show_skeleton_joint_axes:
                    tx = self.skeleton_instances[i].compute_global_transforms()
                    self.coordinate_renderer.draw(self.viewer, tx, 0.1, i)
                if self.show_skeleton_mesh:
                    self.skeletal_mesh_renderer.draw(self.viewer, self.skeleton_instances[i], self.skeleton_instances[i].color, i)
                self.skeleton_instances[i].xform = prev_xform
        
        if self.show_gizmos:
            for i, offset in enumerate(self.robot_offsets):
                self.viewer.log_gizmo(f"robot_offset{i}", offset)
            for i, offset in enumerate(self.animation_offsets):
                self.viewer.log_gizmo(f"animation_offset{i}", offset)
        
        self.viewer.log_state(self.state)
        self.viewer.end_frame()

    def run(self):
        try:
            while self.viewer.is_running():
                with wp.ScopedTimer("step", active=False):
                    self.step()
                with wp.ScopedTimer("render", active=False):
                    self.render()
        finally:
            self._stop_soma_x_process()
            self.viewer.close()

    def retarget_motion(self):
        retarget_source = self.retarget_source_options[self.retarget_source_idx]
        retarget_target = self.retarget_target_options[self.retarget_target_idx]
        retarget_solver = self.retarget_solver_options[self.retarget_solver_idx]
        
        if (retarget_solver == 'Newton'):
            import soma_retargeter.pipelines.newton_pipeline as newton_pipeline
            retargeter_config = _load_retargeter_config(self.config)
            pipeline = newton_pipeline.NewtonPipeline(
                self.skeleton,
                retarget_source,
                retarget_target,
                retarget_config=retargeter_config)
        else:
            raise(ValueError(f"[ERROR]: Unknown retargeter solver [{retarget_solver}"))
        
        r_offsets = [wp.transform(wp.vec3(0,0,0), wp.quat(*s.xform[3:7])) for s in self.skeleton_instances]
        pipeline.add_input_motions(self.animation_buffers, r_offsets, True)
        if _is_ai_sapiens_target(retarget_target):
            target_orientation_yaw_deg = _float_config_or_env(
                self.config,
                "ai_sapiens_target_orientation_yaw_deg",
                "SOMA_RETARGETER_AI_SAPIENS_TARGET_ORIENTATION_YAW_DEG",
                0.0,
            )
            target_position_yaw_deg = _float_config_or_env(
                self.config,
                "ai_sapiens_target_position_yaw_deg",
                "SOMA_RETARGETER_AI_SAPIENS_TARGET_POSITION_YAW_DEG",
                target_orientation_yaw_deg,
            )
            target_yaw_pivot = _str_config_or_env(
                self.config,
                "ai_sapiens_target_yaw_pivot",
                "SOMA_RETARGETER_AI_SAPIENS_TARGET_YAW_PIVOT",
                "origin",
            )
            _apply_target_yaw_to_pipeline(
                pipeline,
                position_yaw_deg=target_position_yaw_deg,
                orientation_yaw_deg=target_orientation_yaw_deg,
                pivot_mode=target_yaw_pivot,
            )
        buffers = pipeline.execute()
        
        if buffers is not None:
            if _is_ai_sapiens_target(retarget_target):
                mjcf_config_value = _ai_sapiens_mjcf_config_value(self.config, retargeter_config)
                for buffer in buffers:
                    _apply_ai_sapiens_output_convention(self.config, buffer, mjcf_config_value)
            t_offsets = [wp.transform(wp.vec3(*s.xform[:3]), wp.quat_identity()) for s in self.skeleton_instances]
            for i, buffer in enumerate(buffers):
                buffer.xform = t_offsets[i]

        self.robot_csv_animation_buffers[0] = buffers[0]

    def ui_scene_options(self, ui):
        import tkinter as tk
        from tkinter import filedialog as tk_filedialog
        
        viewport = ui.get_main_viewport()

        panel_size = ui.ImVec2(320, 420)
        ui.set_next_window_pos(
            ui.ImVec2(
                viewport.size.x - _UI_NEWTON_PANEL_MARGIN - panel_size.x,
                viewport.size.y - _UI_NEWTON_PANEL_MARGIN - panel_size.y))
        
        ui.set_next_window_size(panel_size)
        ui.set_next_window_bg_alpha(_UI_NEWTON_PANEL_ALPHA)

        ui.begin("Scene Options", flags=(ui.WindowFlags_.no_collapse | ui.WindowFlags_.no_resize))
        ui.separator()

        # Motion options
        if ui.collapsing_header("Motion", flags=ui.TreeNodeFlags_.default_open):
            ui.separator()
            ui.align_text_to_frame_padding()
            ui.text("BVH Motion:")
            ui.same_line()
            
            ui.push_id(100)
            if ui.button("Load"):
                root = tk.Tk()
                root.withdraw()
                bvh_path = tk_filedialog.askopenfilename(
                    title='Load BVH File',
                    defaultextension=".bvh",
                    filetypes=[('BVH files', '*.bvh')])

                if bvh_path:
                    self.load_bvh_file(bvh_path)
            ui.pop_id()

            if (
                len(self.animation_buffers) == 0
                or self.loaded_motion_kind != "bvh"
            ):
                ui.begin_disabled()

            ui.same_line()
            if ui.button("Retarget"):
                self.retarget_motion()
            
            if (
                len(self.animation_buffers) == 0
                or self.loaded_motion_kind != "bvh"
            ):
                ui.end_disabled()

            ui.align_text_to_frame_padding()
            ui.text("NPZ Motion:")
            ui.same_line()

            ui.push_id(150)
            if ui.button("Load"):
                root = tk.Tk()
                root.withdraw()
                npz_path = tk_filedialog.askopenfilename(
                    title='Load Kimodo NPZ File',
                    defaultextension=".npz",
                    filetypes=[('NPZ files', '*.npz')])

                if npz_path:
                    try:
                        self.load_npz_file(npz_path)
                    except Exception as exc:
                        print(f"[ERROR]: Failed to load NPZ motion [{npz_path}]: {exc}")
            ui.pop_id()

            if (
                len(self.animation_buffers) == 0
                or self.loaded_motion_kind != "npz"
            ):
                ui.begin_disabled()

            ui.same_line()
            ui.push_id(151)
            if ui.button("Retarget"):
                self.retarget_motion()
            ui.pop_id()

            if (
                len(self.animation_buffers) == 0
                or self.loaded_motion_kind != "npz"
            ):
                ui.end_disabled()

            ui.align_text_to_frame_padding()
            ui.text("SOMA-X:")
            ui.same_line()

            soma_x_controls = self._soma_x_controls_enabled()
            ui.push_id(175)
            if not soma_x_controls["human_model"]:
                ui.begin_disabled()
            if ui.button("Human Model"):
                self.choose_soma_x_model()
            if not soma_x_controls["human_model"]:
                ui.end_disabled()
            ui.pop_id()

            ui.same_line()
            ui.push_id(176)
            if not soma_x_controls["motion"]:
                ui.begin_disabled()
            if ui.button("Motion"):
                self.choose_soma_x_motion()
            if not soma_x_controls["motion"]:
                ui.end_disabled()
            ui.pop_id()

            ui.same_line()
            ui.push_id(177)
            if not soma_x_controls["retarget"]:
                ui.begin_disabled()
            if ui.button("Retarget"):
                self.retarget_motion()
            if not soma_x_controls["retarget"]:
                ui.end_disabled()
            ui.pop_id()

            self._draw_soma_x_model_info(ui)

            ui.align_text_to_frame_padding()
            ui.text("CSV Motion:")
            ui.same_line()
            
            ui.push_id(200)
            if ui.button("Load"):
                root = tk.Tk()
                root.withdraw()
                csv_path = tk_filedialog.askopenfilename(
                    title='Load CSV File',
                    defaultextension=".csv",
                    filetypes=[('CSV files', '*.csv')])

                if csv_path:
                    self.load_csv_file(csv_path)

            if self.robot_csv_animation_buffers[0] is None:
                ui.begin_disabled()
            ui.pop_id()

            ui.same_line()
            if ui.button("Save"):
                root = tk.Tk()
                root.withdraw()

                save_path = tk_filedialog.asksaveasfilename(
                    title="Save CSV File",
                    defaultextension=".csv",
                    filetypes=[("CSV files", "*.csv")])
                if save_path:
                    csv_utils.save_csv(
                        save_path,
                        self.robot_csv_animation_buffers[0],
                        csv_utils.get_csv_config_for_target(
                            self.retarget_target_options[self.retarget_target_idx]))

            if self.robot_csv_animation_buffers[0] is None:
                ui.end_disabled()

        # Visibility options
        ui.spacing()
        if ui.collapsing_header("Visibility", flags=ui.TreeNodeFlags_.default_open):
            ui.separator()

            changed, self.show_skeleton_mesh = ui.checkbox("Show Mesh", self.show_skeleton_mesh)
            if changed and self.skeletal_mesh_renderer is not None:
                self.skeletal_mesh_renderer.clear(self.viewer)
            changed, self.show_skeleton = ui.checkbox("Show Skeleton", self.show_skeleton)
            if changed and self.skeleton_renderer is not None:
                self.skeleton_renderer.clear(self.viewer)
            changed, self.show_skeleton_joint_axes = ui.checkbox("Show Joint Axes", self.show_skeleton_joint_axes)
            if changed and self.coordinate_renderer is not None:
                self.coordinate_renderer.clear(self.viewer)
            _, self.show_gizmos = ui.checkbox("Show Gizmos", self.show_gizmos)
            ui.same_line()
            if ui.button("Reset"):
                self.robot_offsets = [wp.transform(wp.vec3(0.0, i - (self.num_robots - 1) / 2.0, 0.0), wp.quat_identity()) for i in range(self.num_robots)]
                self.animation_offsets = [wp.transform_identity()] * len(self.skeleton_instances)
        ui.end()

    def ui_playback_controls(self, ui):
        viewport = ui.get_main_viewport()
        
        panel_height = 105
        panel_width = viewport.size.x - 2 * (2 * _UI_NEWTON_PANEL_MARGIN + _UI_NEWTON_PANEL_WIDTH)
        
        ui.set_next_window_pos(ui.ImVec2(_UI_NEWTON_PANEL_WIDTH + _UI_NEWTON_PANEL_MARGIN, viewport.size.y - _UI_NEWTON_PANEL_MARGIN - panel_height))
        ui.set_next_window_size(ui.ImVec2(panel_width, panel_height))
        ui.set_next_window_bg_alpha(_UI_NEWTON_PANEL_ALPHA)

        ui.begin("Playback Controls", flags=(ui.WindowFlags_.no_collapse | ui.WindowFlags_.no_resize))
        # Time slider
        ui.align_text_to_frame_padding()
        ui.text("Time (s):")
        ui.same_line()
        ui.set_next_item_width(panel_width - 150)
        changed, new_time = ui.slider_float(
            "##TimeSlider",
            self.playback_time,
            0.0,
            self.playback_total_time,
            "%.2f")
        if changed:
            self.playback_time = wp.clamp(new_time, 0.0, self.playback_total_time)
        ui.same_line()
        ui.text_colored(ui.ImVec4(0.6, 0.8, 1.0, 1.0), f"{self.playback_total_time:.2f}s")
        
        self.is_playing = not ui.button("Pause") if self.is_playing else ui.button("Play ")
        ui.same_line()

        # Speed slider
        ui.align_text_to_frame_padding()
        ui.text("Speed")
        ui.same_line()
        ui.set_next_item_width(100)
        changed, new_speed = ui.slider_float(
            "##SpeedSlider",
            self.playback_speed,
            -2.0, 2.0,
            "%.2f"
        )
        if changed:
            self.playback_speed = new_speed
        ui.same_line()
        _, self.playback_loop = ui.checkbox("Loop", self.playback_loop)
        ui.end()

    def _retarget_bvh_jobs(self, bvh_jobs):
        if not bvh_jobs:
            raise ValueError("No BVH motions were prepared for retargeting")
        batch_size = int(self.config['batch_size'])
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        # Keep the existing largest-first batching policy while carrying each
        # source's explicit CSV destination through preprocessing.
        bvh_jobs = sorted(
            bvh_jobs,
            key=lambda item: item[0].stat().st_size,
            reverse=True,
        )
        batches = [
            bvh_jobs[i:i + batch_size]
            for i in range(0, len(bvh_jobs), batch_size)
        ]

        # All skeletons should be the same, load one as our reference.
        bvh_importer = bvh_utils.BVHImporter()
        bvh_skeleton, _ = bvh_importer.create_skeleton(batches[0][0][0])

        bvh_tx_converter = self.converter.transform(wp.transform_identity())
        expected_num_joints = bvh_skeleton.num_joints

        retarget_source = self.config['retarget_source']
        retarget_solver = self.config['retargeter']
        retarget_target = self.config["retarget_target"]
        retarget_pipeline = None
        retargeter_config = None
        if (retarget_solver == 'Newton'):
            import soma_retargeter.pipelines.newton_pipeline as newton_pipeline
            retargeter_config = _load_retargeter_config(self.config)
            retarget_pipeline = newton_pipeline.NewtonPipeline(
                bvh_skeleton,
                retarget_source,
                retarget_target,
                retarget_config=retargeter_config)
        if retarget_pipeline is None:
            print(f"[ERROR]: Invalid retarget solver selected [{retarget_solver}]. Use 'Newton'.")
            exit(-1)

        nb_retargeted_motions = 0
        start_time = time.time()

        for batch_index, batch in enumerate(batches):
            print(f"[INFO]: Processing batch {batch_index+1} of {len(batches)}")
            print(f"[INFO]: Loading {len(batch)} animations...")
            animations = []
            for file_path, _ in batch:
                _, animation = bvh_utils.load_bvh(file_path, bvh_skeleton)
                # All animations should be on the same skeleton
                assert expected_num_joints == animation.skeleton.num_joints, (
                    f"[ERROR]: Unexpected number of joints in input motion. Expected {expected_num_joints}, "
                    f"got {animation.skeleton.num_joints}")
                
                animations.append(animation)
            assert(len(animations) == len(batch))

            if (len(animations) > 0):
                print("[INFO]: Retargeting...")
                retarget_pipeline.clear()
                retarget_pipeline.add_input_motions(animations, [bvh_tx_converter] * len(animations), True)
                if _is_ai_sapiens_target(retarget_target):
                    target_orientation_yaw_deg = _float_config_or_env(
                        self.config,
                        "ai_sapiens_target_orientation_yaw_deg",
                        "SOMA_RETARGETER_AI_SAPIENS_TARGET_ORIENTATION_YAW_DEG",
                        0.0,
                    )
                    target_position_yaw_deg = _float_config_or_env(
                        self.config,
                        "ai_sapiens_target_position_yaw_deg",
                        "SOMA_RETARGETER_AI_SAPIENS_TARGET_POSITION_YAW_DEG",
                        target_orientation_yaw_deg,
                    )
                    target_yaw_pivot = _str_config_or_env(
                        self.config,
                        "ai_sapiens_target_yaw_pivot",
                        "SOMA_RETARGETER_AI_SAPIENS_TARGET_YAW_PIVOT",
                        "origin",
                    )
                    _apply_target_yaw_to_pipeline(
                        retarget_pipeline,
                        position_yaw_deg=target_position_yaw_deg,
                        orientation_yaw_deg=target_orientation_yaw_deg,
                        pivot_mode=target_yaw_pivot,
                    )
                csv_buffers = retarget_pipeline.execute()

                assert(len(csv_buffers) == len(animations))
                target_csv_config = csv_utils.get_csv_config_for_target(retarget_target)
                mjcf_config_value = _ai_sapiens_mjcf_config_value(self.config, retargeter_config)
                for output_index in trange(len(csv_buffers), desc="[INFO]: Exporting CSV Files"):
                    csv_buffer = csv_buffers[output_index]
                    if _is_ai_sapiens_target(retarget_target):
                        _apply_ai_sapiens_output_convention(self.config, csv_buffer, mjcf_config_value)
                    dst_path = batch[output_index][1]
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    csv_utils.save_csv(dst_path, csv_buffer, target_csv_config)

            nb_retargeted_motions += len(batch)

        elapsed_time = time.time() - start_time
        elapsed_str = f"{int(elapsed_time // 3600):02d}:{int((elapsed_time % 3600) // 60):02d}:{int(elapsed_time % 60):02d}"
        print(
            f"[INFO]: Retargeted {nb_retargeted_motions} animations successfully "
            f"in {elapsed_str} "
            f"[{(elapsed_time/nb_retargeted_motions):.2f}s per motion]!")

    def batched_retargeting(self):
        if not os.path.isdir(self.config['import_folder']):
            print(f"[ERROR]: Import folder does not exist {self.config['import_folder']}.")
            exit(-1)

        import_path = pathlib.Path(self.config['import_folder'])
        if len(self.config['export_folder']) == 0:
            print("[ERROR]: No export folder specified.")
            exit(-1)

        export_path = pathlib.Path(self.config['export_folder'])
        if not export_path.is_dir():
            print(f"[WARNING]: Export folder does not exist! Creating new folder at {str(export_path)}!")
            export_path.mkdir(parents=True, exist_ok=True)

        bvh_files = list(import_path.rglob("*.bvh"))
        if not bvh_files:
            print(f"[ERROR]: Import folder {str(import_path)}, does not contain any BVH files.")
            exit(-1)
        self._retarget_bvh_jobs(
            [
                (
                    path,
                    export_path / path.relative_to(import_path).with_suffix(".csv"),
                )
                for path in bvh_files
            ]
        )

    def _headless_soma_x_options(self, args, model_type):
        options = self._soma_x_conversion_options()
        options.update(
            {
                "model_type": model_type,
                "fps_override": args.input_fps,
                "heading_yaw_degrees": args.heading_yaw_degrees,
            }
        )
        if args.device is not None:
            options["device_name"] = str(args.device)
        if args.soma_x_batch_size is not None:
            options["batch_size"] = args.soma_x_batch_size
        if args.source_coordinate is not None:
            options["source_coordinate"] = args.source_coordinate
        if args.canonicalize_heading is not None:
            options["canonicalize_heading"] = args.canonicalize_heading
        if args.rebase_root_horizontal is not None:
            options["rebase_root_horizontal"] = args.rebase_root_horizontal
        return options

    def headless_retargeting(self, args):
        model_path = smplx_motion_utils.resolve_human_model_path(
            args.human_model,
            self.config,
            model_type="auto",
        )
        jobs = motion_input_utils.plan_motion_jobs(
            args.input,
            args.output_dir,
            human_model=model_path,
        )
        raw_jobs = [
            job
            for job in jobs
            if job.kind == motion_input_utils.MotionInputKind.RAW_SMPL_NPZ
        ]

        model_info = None
        conversion_options = None
        if raw_jobs:
            if model_path is None:
                raise FileNotFoundError(
                    "Raw SMPL-family input requires --human-model, "
                    "SOMA_RETARGETER_SOMA_X_HUMAN_MODEL, or "
                    "soma_x_human_model in the config."
                )
            model_info = smplx_motion_utils.inspect_human_model(model_path)
            smplx_motion_utils.require_soma_x_dependencies()
            conversion_options = self._headless_soma_x_options(
                args,
                model_info.model_type,
            )
            for job in raw_jobs:
                try:
                    smplx_motion_utils.load_smpl_motion(
                        job.source_path,
                        model_type=model_info.model_type,
                        fps_override=args.input_fps,
                    )
                except smplx_motion_utils.HumanModelMotionMismatchError as exc:
                    raise smplx_motion_utils.HumanModelMotionMismatchError(
                        f"Human model and motion do not match: {job.source_path}"
                    ) from exc

        template_bvh = (
            args.bvh_template
            or os.environ.get("SOMA_RETARGETER_KIMODO_NPZ_TEMPLATE_BVH")
            or self.config.get("kimodo_npz_template_bvh")
        )
        offsets = (
            args.bvh_offsets
            or os.environ.get("SOMA_RETARGETER_KIMODO_NPZ_OFFSETS")
            or self.config.get("kimodo_npz_offsets")
        )
        configured_fps = args.input_fps
        if configured_fps is None and os.environ.get("SOMA_RETARGETER_KIMODO_NPZ_FPS") is not None:
            configured_fps = float(os.environ["SOMA_RETARGETER_KIMODO_NPZ_FPS"])
        if configured_fps is None:
            configured_fps = self.config.get("kimodo_npz_fps")
        position_scale = (
            args.bvh_position_scale
            if args.bvh_position_scale is not None
            else _float_config_or_env(
                self.config,
                "kimodo_npz_position_scale",
                "SOMA_RETARGETER_KIMODO_NPZ_POSITION_SCALE",
                100.0,
            )
        )
        compare = _bool_config_or_env(
            self.config,
            "kimodo_npz_compare",
            "SOMA_RETARGETER_KIMODO_NPZ_COMPARE",
            False,
        )

        runtime_cache = {}
        prepared_jobs = []
        for job_index, job in enumerate(jobs, start=1):
            npz_path = job.source_path
            if job.kind == motion_input_utils.MotionInputKind.RAW_SMPL_NPZ:
                if job.soma_npz_path is None:
                    raise RuntimeError(f"Raw SMPL job has no SOMA output path: {job}")
                npz_path = job.soma_npz_path
                if npz_path.exists() and not args.force:
                    valid, reason = motion_input_utils.validate_existing_soma_output(
                        job.source_path,
                        npz_path,
                        model_path,
                        conversion_options,
                    )
                    if not valid:
                        raise FileExistsError(
                            f"Existing SOMA77 output is stale or invalid: "
                            f"{npz_path} ({reason}). Pass --force to replace it."
                        )
                    print(
                        f"[INFO]: [{job_index}/{len(jobs)}] Reusing SOMA77 NPZ "
                        f"[{npz_path}] ({reason})"
                    )
                else:
                    result = motion_input_utils.convert_raw_smpl_to_soma_npz(
                        job.source_path,
                        npz_path,
                        model_path,
                        conversion_options,
                        runtime_cache=runtime_cache,
                        progress=lambda current, total, index=job_index: print(
                            f"[{index}/{len(jobs)}] Processed "
                            f"{current}/{total} frames",
                            flush=True,
                        ),
                    )
                    print(json.dumps(result, indent=2), flush=True)

            if job.kind != motion_input_utils.MotionInputKind.BVH:
                bvh_result = motion_input_utils.convert_soma_npz_to_bvh(
                    npz_path,
                    job.bvh_path,
                    template_bvh=template_bvh,
                    offsets=offsets,
                    configured_fps=configured_fps,
                    position_scale=position_scale,
                    compare=compare,
                )
                if bvh_result["fps_source"] == "fallback_30":
                    print(
                        f"[WARNING]: No FPS metadata or sidecar BVH for "
                        f"{npz_path}; using 30 FPS. Use --input-fps to override."
                    )
                print(
                    f"[INFO]: [{job_index}/{len(jobs)}] Prepared BVH "
                    f"[{job.bvh_path}] frames={bvh_result['frames']} "
                    f"fps={bvh_result['fps']:.6g} "
                    f"source={bvh_result['fps_source']}"
                )
            prepared_jobs.append((job.bvh_path, job.csv_path))

        self._retarget_bvh_jobs(prepared_jobs)

def main():
    import newton.examples

    parser = newton.examples.create_parser()
    parser.set_defaults(viewer=("null"))
    parser.add_argument(
        "--config",
        type=lambda x: None if x == "None" else str(x),
        default="./assets/default_bvh_to_csv_converter_config.json",
        help="Input json config file.")
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        help="Headless BVH/NPZ motion file or recursively scanned directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        help=(
            "Headless output root containing soma_npz/, soma_bvh/, "
            "and retargeted_csv/."
        ),
    )
    parser.add_argument(
        "--human-model",
        type=pathlib.Path,
        help="Licensed SMPL/SMPL-H/SMPL-X model used only for raw NPZ input.",
    )
    parser.add_argument(
        "--input-fps",
        type=float,
        help="Override source FPS for raw conversion and NPZ-to-BVH output.",
    )
    parser.add_argument(
        "--soma-x-batch-size",
        type=int,
        help="Override the SOMA-X frame batch size for raw NPZ conversion.",
    )
    parser.add_argument(
        "--source-coordinate",
        choices=("auto", "amass", "kimodo"),
        help="Override the coordinate convention of raw SMPL-family input.",
    )
    parser.add_argument(
        "--canonicalize-heading",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable frame-zero heading canonicalization.",
    )
    parser.add_argument(
        "--heading-yaw-degrees",
        type=float,
        default=0.0,
        help="Additional heading rotation applied during raw conversion.",
    )
    parser.add_argument(
        "--rebase-root-horizontal",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable frame-zero horizontal root rebasing.",
    )
    parser.add_argument(
        "--bvh-template",
        type=pathlib.Path,
        help="Template BVH used when converting SOMA77 NPZ input.",
    )
    parser.add_argument(
        "--bvh-offsets",
        type=pathlib.Path,
        help="SOMA77 rest-pose offsets used for fixed-Euler BVH conversion.",
    )
    parser.add_argument(
        "--bvh-position-scale",
        type=float,
        help="Root-position scale used for generated BVH files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace stale or invalid generated preprocessing outputs.",
    )

    viewer, args = newton.examples.init(parser)
    if not pathlib.Path(args.config).exists():
        print(f"[ERROR]: Main config json file not found: {args.config}")
        exit(1)

    if (args.input is None) != (args.output_dir is None):
        parser.error("--input and --output-dir must be specified together")
    if args.input is not None and not isinstance(viewer, newton.viewer.ViewerNull):
        parser.error("--input and --output-dir are supported only with --viewer null")
    if args.soma_x_batch_size is not None and args.soma_x_batch_size <= 0:
        parser.error("--soma-x-batch-size must be positive")

    config = io_utils.load_json(args.config)
    with wp.ScopedDevice(args.device):
        app = Viewer(viewer, config)
        if not isinstance(viewer, newton.viewer.ViewerNull):
            app.run()
        elif args.input is not None:
            app.headless_retargeting(args)
        else:
            app.batched_retargeting()

if __name__ == "__main__":
    main()
