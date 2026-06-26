# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import newton

import json
import pathlib
import time
import warp as wp
import numpy as np
from scipy.spatial.transform import Rotation as R

import soma_retargeter.utils.math_utils as math_utils
import soma_retargeter.assets.bvh as bvh_utils
import soma_retargeter.assets.csv as csv_utils
import soma_retargeter.assets.ai_sapiens as ai_sapiens_assets
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

        self.animation_offsets = []
        self.animation_buffers = []
        self.skeleton_instances = []
        self.robot_csv_animation_buffers = [None for _ in range(self.num_robots)]

    def gui(self, ui):
        self.ui_playback_controls(ui)
        self.ui_scene_options(ui)

    def load_csv_file(self, path):
        self.robot_csv_animation_buffers[0] = csv_utils.load_csv(
            path,
            csv_config=csv_utils.get_csv_config_for_target(
                self.retarget_target_options[self.retarget_target_idx]))
        self.compute_playback_total_time()

    def load_bvh_file(self, path):
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
        self.compute_playback_total_time()

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
        while self.viewer.is_running():
            with wp.ScopedTimer("step", active=False):
                self.step()
            with wp.ScopedTimer("render", active=False):
                self.render()

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

        panel_size = ui.ImVec2(320, 320)
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

            if (len(self.animation_buffers) == 0):
                ui.begin_disabled()

            ui.same_line()
            if ui.button("Retarget"):
                self.retarget_motion()
            
            if (len(self.animation_buffers) == 0):
                ui.end_disabled()

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

        batch_size = self.config['batch_size']
        bvh_files = list(import_path.rglob("*.bvh"))
        if (len(bvh_files) == 0):
            print(f"[ERROR]: Import folder {str(import_path)}, does not contain any BVH files.")
            exit(-1)

        # Sort files based on size (largest first)
        bvh_files.sort(key=lambda p: p.stat().st_size, reverse=True)
        batches = [bvh_files[i:i + batch_size] for i in range(0, len(bvh_files), batch_size)]
        
        # All skeletons should be the same, load one as our reference
        bvh_importer = bvh_utils.BVHImporter()
        bvh_skeleton, _ = bvh_importer.create_skeleton(batches[0][0])

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

        for i, batch in enumerate(batches):
            print(f"[INFO]: Processing batch {i+1} of {len(batches)}")
            
            print(f"[INFO]: Loading {len(batch)} animations...")
            animations = []
            for file_path in batch:
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
                for i in trange(len(csv_buffers), desc="[INFO]: Exporting CSV Files"):
                    csv_buffer = csv_buffers[i]
                    if _is_ai_sapiens_target(retarget_target):
                        _apply_ai_sapiens_output_convention(self.config, csv_buffer, mjcf_config_value)
                    dst_path = export_path / pathlib.Path(batch[i]).relative_to(import_path).with_suffix(".csv")
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    csv_utils.save_csv(dst_path, csv_buffer, target_csv_config)

            nb_retargeted_motions += len(batch)

        elapsed_time = time.time() - start_time
        elapsed_str = f"{int(elapsed_time // 3600):02d}:{int((elapsed_time % 3600) // 60):02d}:{int(elapsed_time % 60):02d}"
        print(
            f"[INFO]: Retargeted {nb_retargeted_motions} animations successfully "
            f"in {elapsed_str} "
            f"[{(elapsed_time/nb_retargeted_motions):.2f}s per motion]!")

def main():
    import newton.examples

    parser = newton.examples.create_parser()
    parser.set_defaults(viewer=("null"))
    parser.add_argument(
        "--config",
        type=lambda x: None if x == "None" else str(x),
        default="./assets/default_bvh_to_csv_converter_config.json",
        help="Input json config file.")

    viewer, args = newton.examples.init(parser)
    if not pathlib.Path(args.config).exists():
        print(f"[ERROR]: Main config json file not found: {args.config}")
        exit(1)

    config = io_utils.load_json(args.config)
    with wp.ScopedDevice(args.device):
        app = Viewer(viewer, config)
        if not isinstance(viewer, newton.viewer.ViewerNull):
            app.run()
        else:
            app.batched_retargeting()

if __name__ == "__main__":
    main()
