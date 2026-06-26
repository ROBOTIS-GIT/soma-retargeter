# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warp as wp
import math
import numpy as np
import newton
import newton.ik as ik
from tqdm import trange

import soma_retargeter.assets.bvh as bvh_utils
import soma_retargeter.assets.ai_sapiens as ai_sapiens_assets
import soma_retargeter.utils.newton_utils as newton_utils
import soma_retargeter.utils.io_utils as io_utils
import soma_retargeter.pipelines.utils as pipeline_utils
from soma_retargeter.pipelines.ik_objectives import (
    IKArmBendAngleObjective,
    IKLimbPlaneNormalObjective,
    IKMaskedBodyPositionObjective,
    IKPointSegmentDistanceBarrier,
    IKSmoothJointFilter,
    IKTemporalJointReference,
)
from soma_retargeter.animation.skeleton import Skeleton, SkeletonInstance
from soma_retargeter.animation.animation_buffer import AnimationBuffer
from soma_retargeter.robotics.human_to_robot_scaler import HumanToRobotScaler
from soma_retargeter.robotics.csv_animation_buffer import CSVAnimationBuffer
from soma_retargeter.pipelines.feet_stabilizer import FeetStabilizer
from soma_retargeter.pipelines.joint_limit_clamper import JointLimitClamper

_DEFAULT_IK_SOLVER_ITERATIONS = 24
_DEFAULT_JOINT_LIMIT_OBJECTIVE_WEIGHT = 10.0
_DEFAULT_SMOOTH_JOINT_FILTER_OBJECTIVE_WEIGHT = 5.5
_DEFAULT_NUM_INITIALIZATION_FRAMES = 10
_DEFAULT_NUM_STABILIZATION_FRAMES = 5


class NewtonPipeline:
    """
    Newton-based motion retargeting pipeline.

    This pipeline retargets human motion captured on a common skeleton
    to a target robot (currently Unitree G1) using inverse kinematics (IK),
    custom objectives, and optional post-processing filters such as
    joint limit clamping and feet stabilization.
    """
    def __init__(self, skeleton: Skeleton, source_type='soma', robot_type='unitree_g1', retarget_config: dict = None):
        """
        Initialize the Newton retargeting pipeline.

        Args:
            skeleton: Common skeleton definition used by the input clips to be retargeted.
            source_type: Source skeleton type name. Currently only "soma" is supported.
            robot_type: Target robot type name. Currently only "unitree_g1" is supported.
            retarget_config: Optional configuration dictionary. If None, a
                configuration is loaded from disk based on the source/target
                types.

        Raises:
            ValueError: If the target robot type is not supported.
        """
        self.source_type = pipeline_utils.get_source_type_from_str(source_type)
        self.target_type = pipeline_utils.get_target_type_from_str(robot_type)
        self.input_targets = []
        self.input_sample_rates = []
        self.max_frames = -1

        if retarget_config is None:
            retargeter_config = pipeline_utils.get_retargeter_config(self.source_type, self.target_type)
        else:
            retargeter_config = retarget_config

        self.ik_iterations = retargeter_config.get('ik_iterations', _DEFAULT_IK_SOLVER_ITERATIONS)
        self.joint_limit_weight = retargeter_config.get('joint_limit_weight', _DEFAULT_JOINT_LIMIT_OBJECTIVE_WEIGHT)
        self.smooth_joint_filter_weight = retargeter_config.get('smooth_joint_filter_weight', _DEFAULT_SMOOTH_JOINT_FILTER_OBJECTIVE_WEIGHT)
        self.neutral_reference_weight = retargeter_config.get('neutral_reference_weight', 0.0)
        self.ai_sapiens_temporal_yaw_twist_reference_enabled = (
            self.target_type == pipeline_utils.TargetType.AI_SAPIENS
            and bool(retargeter_config.get("enable_ai_sapiens_temporal_yaw_twist_reference", False))
        )
        self.ai_sapiens_temporal_yaw_twist_reference_weight = float(
            retargeter_config.get("ai_sapiens_temporal_yaw_twist_reference_weight", 0.0)
        )
        self.ai_sapiens_temporal_yaw_twist_reference_body_masks = retargeter_config.get(
            "ai_sapiens_temporal_yaw_twist_reference_body_masks",
            {},
        )
        self.ai_sapiens_temporal_yaw_twist_reference_start_after_warmup = bool(
            retargeter_config.get("ai_sapiens_temporal_yaw_twist_reference_start_after_warmup", True)
        )
        self.ai_sapiens_arm_ik_temporal_reference_mode = str(
            retargeter_config.get("ai_sapiens_arm_ik_temporal_reference_mode", "previous")
        ).strip().lower()
        self.post_processing_enabled = retargeter_config.get('enable_post_processing', True)
        self.reset_solver_state_at_output_start = retargeter_config.get(
            'reset_solver_state_at_output_start',
            self.target_type == pipeline_utils.TargetType.AI_SAPIENS)
        self.warmup_neutral_joint_names = retargeter_config.get('warmup_neutral_joint_names', [])
        self.enable_self_penetration = False
        self.smooth_joint_filter_coord_masks = None
        self.neutral_reference_coord_masks = None
        self.ai_sapiens_temporal_yaw_twist_reference_coord_masks = None
        self.ai_sapiens_arm_ik_temporal_reference_coord_masks = None
        self.joint_limit_clamper = None
        self.warmup_neutral_q_indices = []
        self.last_internal_trace = None
        self.last_solver_stage_trace = None
        self.enable_solver_stage_trace = (
            self.target_type == pipeline_utils.TargetType.AI_SAPIENS
            and bool(retargeter_config.get("enable_solver_stage_trace", True))
        )
        self.ai_sapiens_projection_start_after_warmup = bool(
            retargeter_config.get("ai_sapiens_projection_start_after_warmup", True)
        )
        self.ai_sapiens_projection_skip_single_output_frame = bool(
            retargeter_config.get("ai_sapiens_projection_skip_single_output_frame", True)
        )
        self.ai_sapiens_arm_projection_enabled = (
            self.target_type == pipeline_utils.TargetType.AI_SAPIENS
            and bool(retargeter_config.get("enable_ai_sapiens_arm_projection", True))
        )
        self.ai_sapiens_arm_projection_anchor = str(
            retargeter_config.get("ai_sapiens_arm_projection_anchor", "shoulder")
        )
        self.ai_sapiens_arm_projection_max_reach_margin_m = float(
            retargeter_config.get("ai_sapiens_arm_projection_max_reach_margin_m", 0.015)
        )
        self.ai_sapiens_arm_projection_min_reach_margin_m = float(
            retargeter_config.get("ai_sapiens_arm_projection_min_reach_margin_m", 0.005)
        )
        self.ai_sapiens_arm_projection_blend = float(
            retargeter_config.get("ai_sapiens_arm_projection_blend", 1.0)
        )
        self.ai_sapiens_arm_projection_mode = str(
            retargeter_config.get("ai_sapiens_arm_projection_mode", "two_bone")
        )
        self.ai_sapiens_arm_projection_min_elbow_bend_deg = float(
            retargeter_config.get("ai_sapiens_arm_projection_min_elbow_bend_deg", 0.0)
        )
        self.ai_sapiens_arm_projection_ramp_output_frames = int(
            retargeter_config.get("ai_sapiens_arm_projection_ramp_output_frames", 0)
        )
        self.ai_sapiens_arm_projection_bend_mode = str(
            retargeter_config.get("ai_sapiens_arm_projection_bend_mode", "raw")
        )
        self.ai_sapiens_arm_projection_state_gate_enabled = (
            self.target_type == pipeline_utils.TargetType.AI_SAPIENS
            and bool(retargeter_config.get("enable_ai_sapiens_arm_projection_state_gate", False))
        )
        self.ai_sapiens_arm_projection_flip_bend_roots = set(
            str(name)
            for name in retargeter_config.get("ai_sapiens_arm_projection_flip_bend_roots", [])
        )
        self.ai_sapiens_arm_projection_body_chains = retargeter_config.get(
            "ai_sapiens_arm_projection_body_chains",
            None,
        )
        self.ai_sapiens_arm_projection_data = []
        self.ai_sapiens_source_body_frame_preservation_enabled = (
            self.target_type == pipeline_utils.TargetType.AI_SAPIENS
            and bool(retargeter_config.get("enable_ai_sapiens_source_body_frame_preservation", False))
        )
        self.ai_sapiens_source_body_frame_preserve_blend = float(
            retargeter_config.get("ai_sapiens_source_body_frame_preserve_blend", 1.0)
        )
        self.ai_sapiens_source_body_frame_upper_follow_blend = float(
            retargeter_config.get("ai_sapiens_source_body_frame_upper_follow_blend", 0.0)
        )
        self.ai_sapiens_source_body_frame_arm_follow_blend = float(
            retargeter_config.get("ai_sapiens_source_body_frame_arm_follow_blend", 0.0)
        )
        self.ai_sapiens_source_body_frame_max_chest_shift_m = float(
            retargeter_config.get("ai_sapiens_source_body_frame_max_chest_shift_m", 0.12)
        )
        self.ai_sapiens_source_body_frame_min_source_horizontal_m = float(
            retargeter_config.get("ai_sapiens_source_body_frame_min_source_horizontal_m", 0.0)
        )
        self.ai_sapiens_source_body_frame_shift_chest = bool(
            retargeter_config.get("ai_sapiens_source_body_frame_shift_chest", True)
        )
        self.ai_sapiens_source_body_frame_rotation_blend = float(
            retargeter_config.get("ai_sapiens_source_body_frame_rotation_blend", 0.0)
        )
        self.ai_sapiens_source_body_frame_rotate_hips = bool(
            retargeter_config.get("ai_sapiens_source_body_frame_rotate_hips", False)
        )
        self.ai_sapiens_source_body_frame_rotate_chest = bool(
            retargeter_config.get("ai_sapiens_source_body_frame_rotate_chest", False)
        )
        self.ai_sapiens_source_body_frame_rotate_upper = bool(
            retargeter_config.get("ai_sapiens_source_body_frame_rotate_upper", False)
        )
        self.ai_sapiens_source_whole_body_pose_enabled = (
            self.target_type == pipeline_utils.TargetType.AI_SAPIENS
            and bool(retargeter_config.get("enable_ai_sapiens_source_whole_body_pose", False))
        )
        self.ai_sapiens_source_whole_body_pose_rotation_blend = float(
            retargeter_config.get("ai_sapiens_source_whole_body_pose_rotation_blend", 0.0)
        )
        self.ai_sapiens_source_whole_body_pose_min_source_horizontal_m = float(
            retargeter_config.get("ai_sapiens_source_whole_body_pose_min_source_horizontal_m", 0.0)
        )
        self.ai_sapiens_source_whole_body_pose_max_target_shift_m = float(
            retargeter_config.get("ai_sapiens_source_whole_body_pose_max_target_shift_m", 0.0)
        )
        self.ai_sapiens_source_whole_body_pose_targets = list(
            retargeter_config.get(
                "ai_sapiens_source_whole_body_pose_targets",
                [
                    "Hips",
                    "Chest",
                    "LeftArm",
                    "LeftForeArm",
                    "LeftHand",
                    "RightArm",
                    "RightForeArm",
                    "RightHand",
                    "LeftLeg",
                    "LeftShin",
                    "LeftFoot",
                    "RightLeg",
                    "RightShin",
                    "RightFoot",
                ],
            )
        )
        self.ai_sapiens_source_whole_body_pose_rotate_orientations = bool(
            retargeter_config.get("ai_sapiens_source_whole_body_pose_rotate_orientations", True)
        )
        self.ai_sapiens_source_whole_body_pose_preserve_foot_z = bool(
            retargeter_config.get("ai_sapiens_source_whole_body_pose_preserve_foot_z", True)
        )
        self.ai_sapiens_source_body_chain_preservation_enabled = (
            self.target_type == pipeline_utils.TargetType.AI_SAPIENS
            and bool(retargeter_config.get("enable_ai_sapiens_source_body_chain_preservation", False))
        )
        self.ai_sapiens_source_body_chain_blend = float(
            retargeter_config.get("ai_sapiens_source_body_chain_blend", 1.0)
        )
        self.ai_sapiens_source_body_chain_vertical_blend = float(
            retargeter_config.get("ai_sapiens_source_body_chain_vertical_blend", 1.0)
        )
        self.ai_sapiens_source_body_chain_max_shift_m = float(
            retargeter_config.get("ai_sapiens_source_body_chain_max_shift_m", 0.08)
        )
        self.ai_sapiens_source_body_chain_min_source_horizontal_m = float(
            retargeter_config.get("ai_sapiens_source_body_chain_min_source_horizontal_m", 0.0)
        )
        self.ai_sapiens_source_body_chain_target_names = [
            str(name)
            for name in retargeter_config.get(
                "ai_sapiens_source_body_chain_target_names",
                [
                    "Chest",
                    "LeftArm",
                    "RightArm",
                    "LeftForeArm",
                    "RightForeArm",
                    "LeftHand",
                    "RightHand",
                ],
            )
        ]
        self.ai_sapiens_direct_body_chain_staged_solver_enabled = (
            self.target_type == pipeline_utils.TargetType.AI_SAPIENS
            and bool(retargeter_config.get("enable_ai_sapiens_direct_body_chain_staged_solver", False))
        )
        self.ai_sapiens_direct_body_chain_staged_start_after_warmup = bool(
            retargeter_config.get("ai_sapiens_direct_body_chain_staged_start_after_warmup", True)
        )
        self.ai_sapiens_direct_body_chain_stage1_iterations = int(
            retargeter_config.get("ai_sapiens_direct_body_chain_stage1_iterations", max(1, self.ik_iterations // 2))
        )
        self.ai_sapiens_direct_body_chain_stage2_iterations = int(
            retargeter_config.get("ai_sapiens_direct_body_chain_stage2_iterations", max(1, self.ik_iterations))
        )
        self.ai_sapiens_direct_body_chain_stage1_default_position_scale = float(
            retargeter_config.get("ai_sapiens_direct_body_chain_stage1_default_position_scale", 0.10)
        )
        self.ai_sapiens_direct_body_chain_stage1_default_rotation_scale = float(
            retargeter_config.get("ai_sapiens_direct_body_chain_stage1_default_rotation_scale", 0.05)
        )
        self.ai_sapiens_direct_body_chain_stage2_default_position_scale = float(
            retargeter_config.get("ai_sapiens_direct_body_chain_stage2_default_position_scale", 1.0)
        )
        self.ai_sapiens_direct_body_chain_stage2_default_rotation_scale = float(
            retargeter_config.get("ai_sapiens_direct_body_chain_stage2_default_rotation_scale", 1.0)
        )
        self.ai_sapiens_direct_body_chain_stage1_position_scales = {
            str(k): float(v)
            for k, v in retargeter_config.get(
                "ai_sapiens_direct_body_chain_stage1_position_scales",
                {
                    "Hips": 2.0,
                    "Chest": 4.0,
                    "LeftArm": 3.0,
                    "RightArm": 3.0,
                    "LeftForeArm": 0.35,
                    "RightForeArm": 0.35,
                    "LeftHand": 0.20,
                    "RightHand": 0.20,
                    "LeftFoot": 0.25,
                    "RightFoot": 0.25,
                },
            ).items()
        }
        self.ai_sapiens_direct_body_chain_stage1_rotation_scales = {
            str(k): float(v)
            for k, v in retargeter_config.get(
                "ai_sapiens_direct_body_chain_stage1_rotation_scales",
                {
                    "Hips": 1.0,
                    "Chest": 2.0,
                    "LeftArm": 0.35,
                    "RightArm": 0.35,
                    "LeftForeArm": 0.15,
                    "RightForeArm": 0.15,
                    "LeftHand": 0.05,
                    "RightHand": 0.05,
                },
            ).items()
        }
        self.ai_sapiens_direct_body_chain_stage2_position_scales = {
            str(k): float(v)
            for k, v in retargeter_config.get(
                "ai_sapiens_direct_body_chain_stage2_position_scales",
                {
                    "Hips": 1.4,
                    "Chest": 1.8,
                    "LeftArm": 1.5,
                    "RightArm": 1.5,
                    "LeftFoot": 1.2,
                    "RightFoot": 1.2,
                },
            ).items()
        }
        self.ai_sapiens_direct_body_chain_stage2_rotation_scales = {
            str(k): float(v)
            for k, v in retargeter_config.get(
                "ai_sapiens_direct_body_chain_stage2_rotation_scales",
                {
                    "Hips": 1.0,
                    "Chest": 1.4,
                    "LeftArm": 0.8,
                    "RightArm": 0.8,
                    "LeftFoot": 1.2,
                    "RightFoot": 1.2,
                },
            ).items()
        }
        self.ai_sapiens_adaptive_arm_objective_enabled = (
            self.target_type == pipeline_utils.TargetType.AI_SAPIENS
            and bool(retargeter_config.get("enable_ai_sapiens_adaptive_arm_objective", False))
        )
        self.ai_sapiens_arm_reach_ratio_start = float(
            retargeter_config.get("ai_sapiens_arm_reach_ratio_start", 0.95)
        )
        self.ai_sapiens_arm_reach_ratio_end = float(
            retargeter_config.get("ai_sapiens_arm_reach_ratio_end", 1.12)
        )
        self.ai_sapiens_hand_position_min_scale = float(
            retargeter_config.get("ai_sapiens_hand_position_min_scale", 0.55)
        )
        self.ai_sapiens_forearm_position_min_scale = float(
            retargeter_config.get("ai_sapiens_forearm_position_min_scale", 0.70)
        )
        self.ai_sapiens_hand_rotation_mode = str(
            retargeter_config.get("ai_sapiens_hand_rotation_mode", "off")
        )
        self.ai_sapiens_elbow_hint_weight_min = float(
            retargeter_config.get("ai_sapiens_elbow_hint_weight_min", 0.0)
        )
        self.ai_sapiens_elbow_hint_weight_max = float(
            retargeter_config.get("ai_sapiens_elbow_hint_weight_max", 0.35)
        )
        self.ai_sapiens_adaptive_arm_objective_apply_target_relaxation = bool(
            retargeter_config.get(
                "ai_sapiens_adaptive_arm_objective_apply_target_relaxation",
                False,
            )
        )
        self.ai_sapiens_adaptive_arm_objective_data = []
        self.input_adaptive_arm_objective_scales = []
        self.ai_sapiens_risk_window_forearm_priority_enabled = (
            self.target_type == pipeline_utils.TargetType.AI_SAPIENS
            and bool(retargeter_config.get("enable_ai_sapiens_risk_window_forearm_priority", False))
        )
        self.ai_sapiens_risk_window_capsule_threshold_m = float(
            retargeter_config.get("ai_sapiens_risk_window_capsule_threshold_m", 0.16)
        )
        self.ai_sapiens_risk_window_source_elbow_active_deg = float(
            retargeter_config.get("ai_sapiens_risk_window_source_elbow_active_deg", 155.0)
        )
        self.ai_sapiens_risk_window_near_max_margin_m = float(
            retargeter_config.get("ai_sapiens_risk_window_near_max_margin_m", 0.010)
        )
        self.ai_sapiens_risk_window_max_hand_hip_horizontal_m = float(
            retargeter_config.get("ai_sapiens_risk_window_max_hand_hip_horizontal_m", math.inf)
        )
        self.ai_sapiens_risk_window_min_hand_hip_vertical_drop_m = float(
            retargeter_config.get("ai_sapiens_risk_window_min_hand_hip_vertical_drop_m", 0.0)
        )
        self.ai_sapiens_risk_window_active_ratio_cap = float(
            retargeter_config.get("ai_sapiens_risk_window_active_ratio_cap", 0.25)
        )
        self.ai_sapiens_risk_window_smoothing_frames = int(
            retargeter_config.get("ai_sapiens_risk_window_smoothing_frames", 45)
        )
        self.ai_sapiens_risk_window_preactivation_frames = int(
            retargeter_config.get("ai_sapiens_risk_window_preactivation_frames", 0)
        )
        self.ai_sapiens_risk_window_postactivation_frames = int(
            retargeter_config.get("ai_sapiens_risk_window_postactivation_frames", 0)
        )
        self.ai_sapiens_risk_window_activation_mode = str(
            retargeter_config.get("ai_sapiens_risk_window_activation_mode", "score")
        ).strip().lower()
        self.ai_sapiens_risk_window_forearm_t_scale = float(
            retargeter_config.get("ai_sapiens_risk_window_forearm_t_scale", 1.25)
        )
        self.ai_sapiens_risk_window_hand_t_scale = float(
            retargeter_config.get("ai_sapiens_risk_window_hand_t_scale", 0.95)
        )
        self.ai_sapiens_risk_window_hand_r_scale = float(
            retargeter_config.get("ai_sapiens_risk_window_hand_r_scale", 0.50)
        )
        self.ai_sapiens_risk_window_arm_t_scale = float(
            retargeter_config.get("ai_sapiens_risk_window_arm_t_scale", 1.0)
        )
        self.ai_sapiens_risk_window_forearm_r_scale = float(
            retargeter_config.get("ai_sapiens_risk_window_forearm_r_scale", 1.0)
        )
        self.ai_sapiens_risk_window_torso_lock_enabled = (
            self.target_type == pipeline_utils.TargetType.AI_SAPIENS
            and bool(retargeter_config.get("enable_ai_sapiens_risk_window_torso_lock", False))
        )
        self.ai_sapiens_risk_window_chest_t_scale = float(
            retargeter_config.get("ai_sapiens_risk_window_chest_t_scale", 1.25)
        )
        self.ai_sapiens_risk_window_chest_r_scale = float(
            retargeter_config.get("ai_sapiens_risk_window_chest_r_scale", 1.15)
        )
        self.ai_sapiens_risk_window_hips_t_scale = float(
            retargeter_config.get("ai_sapiens_risk_window_hips_t_scale", 1.15)
        )
        self.ai_sapiens_risk_window_hips_r_scale = float(
            retargeter_config.get("ai_sapiens_risk_window_hips_r_scale", 1.0)
        )
        self.ai_sapiens_risk_window_absolute_weights_enabled = (
            self.target_type == pipeline_utils.TargetType.AI_SAPIENS
            and bool(retargeter_config.get("enable_ai_sapiens_risk_window_absolute_weights", False))
        )
        def _optional_risk_weight(key):
            value = retargeter_config.get(key, None)
            if value is None:
                return math.nan
            return float(value)
        self.ai_sapiens_risk_window_arm_t_weight = _optional_risk_weight("ai_sapiens_risk_window_arm_t_weight")
        self.ai_sapiens_risk_window_hand_t_weight = _optional_risk_weight("ai_sapiens_risk_window_hand_t_weight")
        self.ai_sapiens_risk_window_forearm_t_weight = _optional_risk_weight("ai_sapiens_risk_window_forearm_t_weight")
        self.ai_sapiens_risk_window_hand_r_weight = _optional_risk_weight("ai_sapiens_risk_window_hand_r_weight")
        self.ai_sapiens_risk_window_forearm_r_weight = _optional_risk_weight("ai_sapiens_risk_window_forearm_r_weight")
        self.ai_sapiens_risk_window_chest_t_weight = _optional_risk_weight("ai_sapiens_risk_window_chest_t_weight")
        self.ai_sapiens_risk_window_chest_r_weight = _optional_risk_weight("ai_sapiens_risk_window_chest_r_weight")
        self.ai_sapiens_risk_window_hips_t_weight = _optional_risk_weight("ai_sapiens_risk_window_hips_t_weight")
        self.ai_sapiens_risk_window_hips_r_weight = _optional_risk_weight("ai_sapiens_risk_window_hips_r_weight")
        self.ai_sapiens_risk_window_data = []
        self.input_risk_window_objective_scales = []
        self.ai_sapiens_elbow_branch_hint_objective_enabled = (
            self.target_type == pipeline_utils.TargetType.AI_SAPIENS
            and bool(retargeter_config.get("enable_ai_sapiens_elbow_branch_hint_objective", False))
        )
        self.ai_sapiens_elbow_branch_hint_weight = float(
            retargeter_config.get("ai_sapiens_elbow_branch_hint_weight", 0.0)
        )
        self.ai_sapiens_elbow_branch_hint_start_after_warmup = bool(
            retargeter_config.get("ai_sapiens_elbow_branch_hint_start_after_warmup", True)
        )
        self.ai_sapiens_elbow_branch_hint_max_reach_margin_m = float(
            retargeter_config.get("ai_sapiens_elbow_branch_hint_max_reach_margin_m", 0.020)
        )
        self.ai_sapiens_elbow_branch_hint_min_reach_margin_m = float(
            retargeter_config.get("ai_sapiens_elbow_branch_hint_min_reach_margin_m", 0.010)
        )
        self.ai_sapiens_elbow_branch_hint_data = []
        self.input_elbow_branch_hint_targets = []
        self.input_elbow_branch_hint_delta_traces = []
        self.ai_sapiens_bilateral_arm_bend_objective_enabled = (
            self.target_type == pipeline_utils.TargetType.AI_SAPIENS
            and bool(retargeter_config.get("enable_ai_sapiens_bilateral_arm_bend_objective", False))
        )
        self.ai_sapiens_arm_bend_source_active_deg = float(
            retargeter_config.get("ai_sapiens_arm_bend_source_active_deg", 155.0)
        )
        self.ai_sapiens_arm_bend_overextended_deg = float(
            retargeter_config.get("ai_sapiens_arm_bend_overextended_deg", 165.0)
        )
        self.ai_sapiens_arm_bend_error_trigger_deg = float(
            retargeter_config.get("ai_sapiens_arm_bend_error_trigger_deg", 20.0)
        )
        self.ai_sapiens_arm_bend_require_contact_risk = bool(
            retargeter_config.get("ai_sapiens_arm_bend_require_contact_risk", False)
        )
        self.ai_sapiens_arm_bend_contact_risk_distance_m = float(
            retargeter_config.get("ai_sapiens_arm_bend_contact_risk_distance_m", 0.12)
        )
        _bend_weight_cfg = retargeter_config.get("ai_sapiens_arm_bend_weight", 0.0)
        if isinstance(_bend_weight_cfg, (list, tuple)):
            _bend_weight_cfg = _bend_weight_cfg[0] if _bend_weight_cfg else 0.0
        self.ai_sapiens_arm_bend_weight = float(_bend_weight_cfg)
        self.ai_sapiens_soft_bend_wrist_reference_enabled = (
            self.target_type == pipeline_utils.TargetType.AI_SAPIENS
            and bool(retargeter_config.get("enable_ai_sapiens_soft_bend_wrist_reference", False))
        )
        self.ai_sapiens_soft_bend_wrist_reference_max_delta_m = float(
            retargeter_config.get("ai_sapiens_soft_bend_wrist_reference_max_delta_m", 0.0)
        )
        self.ai_sapiens_soft_bend_wrist_reference_weight = float(
            retargeter_config.get("ai_sapiens_soft_bend_wrist_reference_weight", 0.05)
        )
        self.ai_sapiens_contact_gated_elbow_midpoint_hint_enabled = (
            self.target_type == pipeline_utils.TargetType.AI_SAPIENS
            and bool(retargeter_config.get("enable_ai_sapiens_contact_gated_elbow_midpoint_hint", False))
        )
        self.ai_sapiens_contact_gated_elbow_hint_weight = float(
            retargeter_config.get("ai_sapiens_contact_gated_elbow_hint_weight", 0.0)
        )
        self.ai_sapiens_contact_gated_elbow_source_active_deg = float(
            retargeter_config.get("ai_sapiens_contact_gated_elbow_source_active_deg", 155.0)
        )
        self.ai_sapiens_contact_gated_elbow_near_max_margin_m = float(
            retargeter_config.get("ai_sapiens_contact_gated_elbow_near_max_margin_m", 0.010)
        )
        self.ai_sapiens_contact_gated_elbow_contact_risk_distance_m = float(
            retargeter_config.get("ai_sapiens_contact_gated_elbow_contact_risk_distance_m", 0.12)
        )
        self.ai_sapiens_contact_gated_elbow_max_mid_delta_m = float(
            retargeter_config.get("ai_sapiens_contact_gated_elbow_max_mid_delta_m", 0.030)
        )
        self.ai_sapiens_contact_micro_wrist_reference_enabled = (
            self.target_type == pipeline_utils.TargetType.AI_SAPIENS
            and bool(retargeter_config.get("enable_ai_sapiens_contact_micro_wrist_reference", False))
        )
        self.ai_sapiens_contact_micro_wrist_reference_weight = float(
            retargeter_config.get("ai_sapiens_contact_micro_wrist_reference_weight", 0.025)
        )
        self.ai_sapiens_contact_micro_wrist_reference_max_delta_m = float(
            retargeter_config.get("ai_sapiens_contact_micro_wrist_reference_max_delta_m", 0.0)
        )
        self.ai_sapiens_capsule_proxy_barrier_enabled = (
            self.target_type == pipeline_utils.TargetType.AI_SAPIENS
            and bool(retargeter_config.get("enable_ai_sapiens_capsule_proxy_barrier", False))
        )
        self.ai_sapiens_capsule_proxy_barrier_clearance_m = float(
            retargeter_config.get("ai_sapiens_capsule_proxy_barrier_clearance_m", 0.035)
        )
        self.ai_sapiens_capsule_proxy_barrier_weight = float(
            retargeter_config.get("ai_sapiens_capsule_proxy_barrier_weight", 0.0)
        )
        self.ai_sapiens_capsule_proxy_barrier_risk_distance_m = float(
            retargeter_config.get("ai_sapiens_capsule_proxy_barrier_risk_distance_m", 0.12)
        )
        self.ai_sapiens_capsule_proxy_barrier_data = []
        self.ai_sapiens_bilateral_arm_bend_data = []
        self.ai_sapiens_limb_bend_angle_objective_enabled = (
            self.target_type == pipeline_utils.TargetType.AI_SAPIENS
            and bool(retargeter_config.get("enable_ai_sapiens_limb_bend_angle_objective", False))
        )
        self.ai_sapiens_limb_bend_angle_weight = float(
            retargeter_config.get("ai_sapiens_limb_bend_angle_weight", 0.0)
        )
        self.ai_sapiens_limb_bend_angle_start_after_warmup = bool(
            retargeter_config.get("ai_sapiens_limb_bend_angle_start_after_warmup", True)
        )
        self.ai_sapiens_limb_bend_angle_skip_single_output_frame = bool(
            retargeter_config.get("ai_sapiens_limb_bend_angle_skip_single_output_frame", True)
        )
        self.ai_sapiens_limb_bend_angle_groups = [
            str(group)
            for group in retargeter_config.get("ai_sapiens_limb_bend_angle_groups", ["arm", "leg"])
        ]
        self.ai_sapiens_limb_bend_angle_body_chains = retargeter_config.get(
            "ai_sapiens_limb_bend_angle_body_chains",
            None,
        )
        self.ai_sapiens_limb_bend_angle_data = []
        self.ai_sapiens_limb_plane_normal_objective_enabled = (
            self.target_type == pipeline_utils.TargetType.AI_SAPIENS
            and bool(retargeter_config.get("enable_ai_sapiens_limb_plane_normal_objective", False))
        )
        self.ai_sapiens_limb_plane_normal_weight = float(
            retargeter_config.get("ai_sapiens_limb_plane_normal_weight", 0.0)
        )
        self.ai_sapiens_limb_plane_normal_start_after_warmup = bool(
            retargeter_config.get("ai_sapiens_limb_plane_normal_start_after_warmup", True)
        )
        self.ai_sapiens_limb_plane_normal_skip_single_output_frame = bool(
            retargeter_config.get("ai_sapiens_limb_plane_normal_skip_single_output_frame", True)
        )
        self.ai_sapiens_limb_plane_normal_groups = [
            str(group)
            for group in retargeter_config.get("ai_sapiens_limb_plane_normal_groups", ["arm", "leg"])
        ]
        self.ai_sapiens_limb_plane_normal_body_chains = retargeter_config.get(
            "ai_sapiens_limb_plane_normal_body_chains",
            None,
        )
        self.ai_sapiens_limb_plane_normal_data = []
        self.ai_sapiens_limb_midpoint_position_objective_enabled = (
            self.target_type == pipeline_utils.TargetType.AI_SAPIENS
            and bool(retargeter_config.get("enable_ai_sapiens_limb_midpoint_position_objective", False))
        )
        self.ai_sapiens_limb_midpoint_position_weight = float(
            retargeter_config.get("ai_sapiens_limb_midpoint_position_weight", 0.0)
        )
        self.ai_sapiens_limb_midpoint_position_group_weights = {
            str(group).lower(): float(weight)
            for group, weight in retargeter_config.get(
                "ai_sapiens_limb_midpoint_position_group_weights",
                {},
            ).items()
        }
        self.ai_sapiens_limb_midpoint_position_max_delta_m = float(
            retargeter_config.get("ai_sapiens_limb_midpoint_position_max_delta_m", 0.015)
        )
        self.ai_sapiens_limb_midpoint_position_start_after_warmup = bool(
            retargeter_config.get("ai_sapiens_limb_midpoint_position_start_after_warmup", True)
        )
        self.ai_sapiens_limb_midpoint_position_skip_single_output_frame = bool(
            retargeter_config.get("ai_sapiens_limb_midpoint_position_skip_single_output_frame", True)
        )
        self.ai_sapiens_limb_midpoint_position_suppress_contact_risk = bool(
            retargeter_config.get("ai_sapiens_limb_midpoint_position_suppress_contact_risk", False)
        )
        self.ai_sapiens_limb_midpoint_position_contact_risk_distance_m = float(
            retargeter_config.get("ai_sapiens_limb_midpoint_position_contact_risk_distance_m", 0.15)
        )
        self.ai_sapiens_limb_midpoint_position_contact_suppression_groups = [
            str(group)
            for group in retargeter_config.get(
                "ai_sapiens_limb_midpoint_position_contact_suppression_groups",
                ["arm"],
            )
        ]
        self.ai_sapiens_limb_midpoint_position_groups = [
            str(group)
            for group in retargeter_config.get("ai_sapiens_limb_midpoint_position_groups", ["leg"])
        ]
        self.ai_sapiens_limb_midpoint_position_body_chains = retargeter_config.get(
            "ai_sapiens_limb_midpoint_position_body_chains",
            None,
        )
        self.ai_sapiens_limb_midpoint_position_data = []
        self.ai_sapiens_torso_local_limb_midpoint_objective_enabled = (
            self.target_type == pipeline_utils.TargetType.AI_SAPIENS
            and bool(retargeter_config.get("enable_ai_sapiens_torso_local_limb_midpoint_objective", False))
        )
        self.ai_sapiens_torso_local_limb_midpoint_weight = float(
            retargeter_config.get("ai_sapiens_torso_local_limb_midpoint_weight", 0.0)
        )
        self.ai_sapiens_torso_local_limb_midpoint_group_weights = {
            str(group).lower(): float(weight)
            for group, weight in retargeter_config.get(
                "ai_sapiens_torso_local_limb_midpoint_group_weights",
                {},
            ).items()
        }
        self.ai_sapiens_torso_local_limb_midpoint_max_delta_m = float(
            retargeter_config.get("ai_sapiens_torso_local_limb_midpoint_max_delta_m", 0.012)
        )
        self.ai_sapiens_torso_local_limb_midpoint_offset_scale = float(
            retargeter_config.get("ai_sapiens_torso_local_limb_midpoint_offset_scale", 1.0)
        )
        self.ai_sapiens_torso_local_limb_midpoint_start_after_warmup = bool(
            retargeter_config.get("ai_sapiens_torso_local_limb_midpoint_start_after_warmup", True)
        )
        self.ai_sapiens_torso_local_limb_midpoint_skip_single_output_frame = bool(
            retargeter_config.get("ai_sapiens_torso_local_limb_midpoint_skip_single_output_frame", True)
        )
        self.ai_sapiens_torso_local_limb_midpoint_groups = [
            str(group)
            for group in retargeter_config.get("ai_sapiens_torso_local_limb_midpoint_groups", ["arm"])
        ]
        self.ai_sapiens_torso_local_limb_midpoint_body_chains = retargeter_config.get(
            "ai_sapiens_torso_local_limb_midpoint_body_chains",
            None,
        )
        self.ai_sapiens_torso_local_limb_midpoint_data = []
        self.input_arm_bend_objective_target_angles = []
        self.input_arm_bend_objective_active_masks = []
        self.input_arm_bend_objective_trace = []
        self.input_limb_bend_angle_target_angles = []
        self.input_limb_bend_angle_active_masks = []
        self.input_limb_bend_angle_trace = []
        self.input_limb_plane_normal_targets = []
        self.input_limb_plane_normal_active_masks = []
        self.input_limb_plane_normal_trace = []
        self.input_limb_midpoint_position_targets = []
        self.input_limb_midpoint_position_active_masks = []
        self.input_limb_midpoint_position_trace = []
        self.input_torso_local_limb_midpoint_offsets = []
        self.input_torso_local_limb_midpoint_active_masks = []
        self.input_torso_local_limb_midpoint_trace = []
        self.input_soft_bend_wrist_reference_targets = []
        self.input_soft_bend_wrist_reference_active_masks = []
        self.input_soft_bend_wrist_reference_trace = []
        self.input_contact_gated_elbow_hint_targets = []
        self.input_contact_gated_elbow_hint_active_masks = []
        self.input_contact_gated_elbow_hint_trace = []
        self.input_contact_micro_wrist_reference_targets = []
        self.input_contact_micro_wrist_reference_active_masks = []
        self.input_contact_micro_wrist_reference_trace = []
        self.input_capsule_proxy_barrier_active_masks = []
        self.input_capsule_proxy_barrier_trace = []
        self.input_risk_window_objective_scales = []
        self.ai_sapiens_arm_segment_direction_enabled = (
            self.target_type == pipeline_utils.TargetType.AI_SAPIENS
            and bool(retargeter_config.get("enable_ai_sapiens_arm_segment_direction_objective", False))
        )
        self.ai_sapiens_arm_upper_direction_weight = float(
            retargeter_config.get("ai_sapiens_arm_upper_direction_weight", 0.0)
        )
        self.ai_sapiens_arm_forearm_direction_weight = float(
            retargeter_config.get("ai_sapiens_arm_forearm_direction_weight", 0.0)
        )
        self.ai_sapiens_arm_direction_blend = float(
            retargeter_config.get("ai_sapiens_arm_direction_blend", 1.0)
        )
        self.ai_sapiens_arm_direction_apply_after_projection = bool(
            retargeter_config.get("ai_sapiens_arm_direction_apply_after_projection", True)
        )
        self.ai_sapiens_arm_direction_contact_risk_scale_enabled = bool(
            retargeter_config.get("ai_sapiens_arm_direction_contact_risk_scale_enabled", False)
        )
        self.ai_sapiens_arm_direction_contact_risk_distance_m = float(
            retargeter_config.get("ai_sapiens_arm_direction_contact_risk_distance_m", 0.15)
        )
        self.ai_sapiens_arm_direction_contact_risk_soft_range_m = float(
            retargeter_config.get("ai_sapiens_arm_direction_contact_risk_soft_range_m", 0.08)
        )
        self.ai_sapiens_arm_direction_contact_risk_min_scale = float(
            retargeter_config.get("ai_sapiens_arm_direction_contact_risk_min_scale", 0.35)
        )
        self.ai_sapiens_arm_segment_direction_chains = retargeter_config.get(
            "ai_sapiens_arm_segment_direction_chains",
            None,
        )
        self.ai_sapiens_arm_segment_direction_data = []
        self.ai_sapiens_leg_projection_enabled = (
            self.target_type == pipeline_utils.TargetType.AI_SAPIENS
            and bool(retargeter_config.get("enable_ai_sapiens_leg_projection", False))
        )
        self.ai_sapiens_leg_projection_anchor = str(
            retargeter_config.get("ai_sapiens_leg_projection_anchor", "foot")
        )
        self.ai_sapiens_leg_projection_max_reach_margin_m = float(
            retargeter_config.get("ai_sapiens_leg_projection_max_reach_margin_m", 0.010)
        )
        self.ai_sapiens_leg_projection_min_reach_margin_m = float(
            retargeter_config.get("ai_sapiens_leg_projection_min_reach_margin_m", 0.005)
        )
        self.ai_sapiens_leg_projection_blend = float(
            retargeter_config.get("ai_sapiens_leg_projection_blend", 1.0)
        )
        self.ai_sapiens_leg_projection_ramp_output_frames = int(
            retargeter_config.get("ai_sapiens_leg_projection_ramp_output_frames", 0)
        )
        self.ai_sapiens_leg_projection_bend_mode = str(
            retargeter_config.get("ai_sapiens_leg_projection_bend_mode", "raw")
        )
        self.ai_sapiens_leg_projection_flip_bend_roots = set(
            str(name)
            for name in retargeter_config.get("ai_sapiens_leg_projection_flip_bend_roots", [])
        )
        self.ai_sapiens_leg_projection_body_chains = retargeter_config.get(
            "ai_sapiens_leg_projection_body_chains",
            None,
        )
        self.ai_sapiens_leg_projection_data = []
        self.ai_sapiens_dynamic_lateral_correction_enabled = (
            self.target_type == pipeline_utils.TargetType.AI_SAPIENS
            and bool(retargeter_config.get("enable_ai_sapiens_dynamic_lateral_correction", False))
        )
        self.ai_sapiens_source_segment_direction_enabled = (
            self.target_type == pipeline_utils.TargetType.AI_SAPIENS
            and bool(retargeter_config.get("enable_ai_sapiens_source_segment_direction_correction", False))
        )
        self.ai_sapiens_source_segment_direction_start_after_warmup = bool(
            retargeter_config.get("ai_sapiens_source_segment_direction_start_after_warmup", True)
        )
        self.ai_sapiens_source_segment_direction_skip_single_output_frame = bool(
            retargeter_config.get("ai_sapiens_source_segment_direction_skip_single_output_frame", True)
        )
        self.ai_sapiens_source_segment_direction_chains = retargeter_config.get(
            "ai_sapiens_source_segment_direction_chains",
            [],
        )
        self.ai_sapiens_source_segment_direction_data = []
        self.ai_sapiens_source_foot_orientation_enabled = (
            self.target_type == pipeline_utils.TargetType.AI_SAPIENS
            and bool(retargeter_config.get("enable_ai_sapiens_source_foot_orientation_correction", False))
        )
        self.ai_sapiens_source_foot_orientation_start_after_warmup = bool(
            retargeter_config.get("ai_sapiens_source_foot_orientation_start_after_warmup", True)
        )
        self.ai_sapiens_source_foot_orientation_skip_single_output_frame = bool(
            retargeter_config.get("ai_sapiens_source_foot_orientation_skip_single_output_frame", True)
        )
        self.ai_sapiens_source_foot_orientation_blend = float(
            retargeter_config.get("ai_sapiens_source_foot_orientation_blend", 1.0)
        )
        self.ai_sapiens_source_foot_orientation_mode = str(
            retargeter_config.get(
                "ai_sapiens_source_foot_orientation_mode",
                "align_x_preserve_previous_up",
            )
        )
        self.ai_sapiens_source_foot_orientation_gate_mode = str(
            retargeter_config.get("ai_sapiens_source_foot_orientation_gate_mode", "none")
        )
        self.ai_sapiens_source_foot_orientation_gate_height_on_m = float(
            retargeter_config.get("ai_sapiens_source_foot_orientation_gate_height_on_m", 0.035)
        )
        self.ai_sapiens_source_foot_orientation_gate_height_off_m = float(
            retargeter_config.get("ai_sapiens_source_foot_orientation_gate_height_off_m", 0.090)
        )
        self.ai_sapiens_source_foot_orientation_gate_speed_on_mps = float(
            retargeter_config.get("ai_sapiens_source_foot_orientation_gate_speed_on_mps", 0.20)
        )
        self.ai_sapiens_source_foot_orientation_gate_speed_off_mps = float(
            retargeter_config.get("ai_sapiens_source_foot_orientation_gate_speed_off_mps", 0.80)
        )
        self.ai_sapiens_source_foot_orientation_targets = retargeter_config.get(
            "ai_sapiens_source_foot_orientation_targets",
            [
                {
                    "target": "LeftFoot",
                    "source_foot": "LeftFoot",
                    "source_toe_candidates": ["LeftToeBase", "LeftToeEnd", "LeftToe"],
                },
                {
                    "target": "RightFoot",
                    "source_foot": "RightFoot",
                    "source_toe_candidates": ["RightToeBase", "RightToeEnd", "RightToe"],
                },
            ],
        )
        self.ai_sapiens_chest_anchored_arm_root_enabled = (
            self.target_type == pipeline_utils.TargetType.AI_SAPIENS
            and bool(retargeter_config.get("enable_ai_sapiens_chest_anchored_arm_root_correction", False))
        )
        self.ai_sapiens_chest_anchored_arm_root_start_after_warmup = bool(
            retargeter_config.get("ai_sapiens_chest_anchored_arm_root_start_after_warmup", True)
        )
        self.ai_sapiens_chest_anchored_arm_root_skip_single_output_frame = bool(
            retargeter_config.get("ai_sapiens_chest_anchored_arm_root_skip_single_output_frame", True)
        )
        self.ai_sapiens_chest_anchored_arm_root_blend = float(
            retargeter_config.get("ai_sapiens_chest_anchored_arm_root_blend", 1.0)
        )
        self.ai_sapiens_chest_anchored_arm_root_chains = retargeter_config.get(
            "ai_sapiens_chest_anchored_arm_root_chains",
            [],
        )
        self.ai_sapiens_chest_anchored_arm_root_data = []
        self.ai_sapiens_dynamic_lateral_pairs = retargeter_config.get(
            "ai_sapiens_dynamic_lateral_pairs",
            [
                ["LeftLeg", "RightLeg"],
                ["LeftShin", "RightShin"],
                ["LeftFoot", "RightFoot"],
                ["LeftArm", "RightArm"],
            ],
        )
        self.ai_sapiens_dynamic_lateral_followers = retargeter_config.get(
            "ai_sapiens_dynamic_lateral_followers",
            {},
        )
        self.ai_sapiens_pair_span_correction_enabled = (
            self.target_type == pipeline_utils.TargetType.AI_SAPIENS
            and bool(retargeter_config.get("enable_ai_sapiens_pair_span_correction", False))
        )
        self.ai_sapiens_pair_span_start_after_warmup = bool(
            retargeter_config.get("ai_sapiens_pair_span_start_after_warmup", True)
        )
        self.ai_sapiens_pair_span_skip_single_output_frame = bool(
            retargeter_config.get("ai_sapiens_pair_span_skip_single_output_frame", True)
        )
        self.ai_sapiens_pair_span_stage = str(
            retargeter_config.get("ai_sapiens_pair_span_stage", "before_source_segment")
        )
        self.ai_sapiens_pair_span_pairs = retargeter_config.get(
            "ai_sapiens_pair_span_pairs",
            [],
        )
        self.ai_sapiens_pair_span_data = []
        self.ai_sapiens_output_joint_safety_margin_enabled = (
            self.target_type == pipeline_utils.TargetType.AI_SAPIENS
            and bool(retargeter_config.get("enable_ai_sapiens_output_joint_safety_margin", False))
        )
        self.ai_sapiens_output_joint_safety_margins_rad = retargeter_config.get(
            "ai_sapiens_output_joint_safety_margins_rad",
            {},
        )
        self.ai_sapiens_ik_joint_safety_margins_rad = retargeter_config.get(
            "ai_sapiens_ik_joint_safety_margins_rad",
            {},
        )
        self.ai_sapiens_output_joint_safety_margin_specs = []
        self.ai_sapiens_output_joint_step_limit_enabled = (
            self.target_type == pipeline_utils.TargetType.AI_SAPIENS
            and bool(retargeter_config.get("enable_ai_sapiens_output_joint_step_limit", False))
        )
        self.ai_sapiens_output_joint_step_limits_rad = retargeter_config.get(
            "ai_sapiens_output_joint_step_limits_rad",
            {},
        )
        self.ai_sapiens_output_joint_step_limit_specs = []
        self.ai_sapiens_root_orientation_step_limit_enabled = (
            self.target_type == pipeline_utils.TargetType.AI_SAPIENS
            and bool(retargeter_config.get("enable_ai_sapiens_root_orientation_step_limit", False))
        )
        self.ai_sapiens_root_orientation_max_step_deg = float(
            retargeter_config.get("ai_sapiens_root_orientation_max_step_deg", 0.0)
        )
        self.ai_sapiens_root_orientation_max_step_rad = np.deg2rad(
            max(0.0, self.ai_sapiens_root_orientation_max_step_deg)
        )
        self.ai_sapiens_arm_temporal_regularization_enabled = (
            self.target_type == pipeline_utils.TargetType.AI_SAPIENS
            and bool(retargeter_config.get("enable_ai_sapiens_arm_temporal_regularization", False))
        )
        self.ai_sapiens_arm_temporal_step_weight = float(
            retargeter_config.get("ai_sapiens_arm_temporal_step_weight", 0.15)
        )
        self.ai_sapiens_arm_temporal_acceleration_weight = float(
            retargeter_config.get("ai_sapiens_arm_temporal_acceleration_weight", 0.05)
        )
        self.ai_sapiens_arm_temporal_max_correction_rad = float(
            retargeter_config.get("ai_sapiens_arm_temporal_max_correction_rad", 0.03)
        )
        self.ai_sapiens_arm_temporal_joint_names = retargeter_config.get(
            "ai_sapiens_arm_temporal_joint_names",
            [
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_roll_joint",
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_roll_joint",
            ],
        )
        self.ai_sapiens_arm_temporal_q_indices = []
        self.ai_sapiens_arm_ik_temporal_reference_enabled = (
            self.target_type == pipeline_utils.TargetType.AI_SAPIENS
            and bool(retargeter_config.get("enable_ai_sapiens_arm_ik_temporal_reference", False))
        )
        self.ai_sapiens_arm_ik_temporal_reference_weight = float(
            retargeter_config.get("ai_sapiens_arm_ik_temporal_reference_weight", 0.0)
        )
        self.ai_sapiens_adaptive_arm_temporal_reference_enabled = (
            self.target_type == pipeline_utils.TargetType.AI_SAPIENS
            and bool(retargeter_config.get("enable_ai_sapiens_adaptive_arm_temporal_reference", False))
        )
        self.ai_sapiens_arm_temporal_reference_base_weight = float(
            retargeter_config.get(
                "ai_sapiens_arm_temporal_reference_base_weight",
                self.ai_sapiens_arm_ik_temporal_reference_weight,
            )
        )
        self.ai_sapiens_arm_temporal_reference_high_motion_weight = float(
            retargeter_config.get(
                "ai_sapiens_arm_temporal_reference_high_motion_weight",
                max(self.ai_sapiens_arm_temporal_reference_base_weight, self.ai_sapiens_arm_ik_temporal_reference_weight),
            )
        )
        self.ai_sapiens_arm_temporal_reference_trigger_l2_p95 = float(
            retargeter_config.get("ai_sapiens_arm_temporal_reference_trigger_l2_p95", 0.15)
        )
        self.ai_sapiens_arm_temporal_reference_hand_residual_guard_m = float(
            retargeter_config.get("ai_sapiens_arm_temporal_reference_hand_residual_guard_m", 0.01)
        )
        self.ai_sapiens_arm_ik_temporal_reference_body_masks = retargeter_config.get(
            "ai_sapiens_arm_ik_temporal_reference_body_masks",
            {},
        )
        self.ai_sapiens_arm_nullspace_temporal_reference_enabled = (
            self.target_type == pipeline_utils.TargetType.AI_SAPIENS
            and bool(retargeter_config.get("enable_ai_sapiens_arm_nullspace_temporal_reference", False))
        )
        self.ai_sapiens_wrist_roll_temporal_weight_base = float(
            retargeter_config.get("ai_sapiens_wrist_roll_temporal_weight_base", 0.025)
        )
        self.ai_sapiens_wrist_roll_temporal_weight_high = float(
            retargeter_config.get("ai_sapiens_wrist_roll_temporal_weight_high", 0.080)
        )
        self.ai_sapiens_arm_branch_temporal_weight_base = float(
            retargeter_config.get("ai_sapiens_arm_branch_temporal_weight_base", 0.010)
        )
        self.ai_sapiens_arm_branch_temporal_weight_high = float(
            retargeter_config.get("ai_sapiens_arm_branch_temporal_weight_high", 0.035)
        )
        self.ai_sapiens_nullspace_temporal_reach_trigger = float(
            retargeter_config.get("ai_sapiens_nullspace_temporal_reach_trigger", 1.05)
        )
        self.ai_sapiens_nullspace_temporal_step_trigger_rad = float(
            retargeter_config.get("ai_sapiens_nullspace_temporal_step_trigger_rad", 0.08)
        )
        self.ai_sapiens_wrist_roll_nullspace_joint_names = retargeter_config.get(
            "ai_sapiens_wrist_roll_nullspace_joint_names",
            ["left_wrist_roll_joint", "right_wrist_roll_joint"],
        )
        self.ai_sapiens_arm_branch_nullspace_joint_names = retargeter_config.get(
            "ai_sapiens_arm_branch_nullspace_joint_names",
            [
                "left_shoulder_pitch_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "right_shoulder_pitch_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
            ],
        )
        self.ai_sapiens_wrist_roll_nullspace_q_indices = []
        self.ai_sapiens_arm_branch_nullspace_q_indices = []
        self.ai_sapiens_left_wrist_roll_q_indices = []
        self.ai_sapiens_right_wrist_roll_q_indices = []
        self.ai_sapiens_left_arm_branch_q_indices = []
        self.ai_sapiens_right_arm_branch_q_indices = []
        self.ai_sapiens_wrist_roll_nullspace_coord_masks = None
        self.ai_sapiens_arm_branch_nullspace_coord_masks = None
        self.ai_sapiens_sparse_wrist_roll_resolve_enabled = (
            self.target_type == pipeline_utils.TargetType.AI_SAPIENS
            and bool(retargeter_config.get("enable_ai_sapiens_sparse_wrist_roll_resolve", False))
        )
        self.ai_sapiens_sparse_wrist_roll_trigger_deg = float(
            retargeter_config.get("ai_sapiens_sparse_wrist_roll_trigger_deg", 30.0)
        )
        self.ai_sapiens_sparse_wrist_roll_reject_deg = float(
            retargeter_config.get("ai_sapiens_sparse_wrist_roll_reject_deg", 45.0)
        )
        sparse_wrist_weights = retargeter_config.get(
            "ai_sapiens_sparse_wrist_roll_weights",
            [0.020],
        )
        if isinstance(sparse_wrist_weights, (int, float)):
            sparse_wrist_weights = [float(sparse_wrist_weights)]
        self.ai_sapiens_sparse_wrist_roll_weights = [
            float(w) for w in sparse_wrist_weights if float(w) >= 0.0
        ] or [0.020]
        self.ai_sapiens_sparse_wrist_repair_hand_residual_guard_m = float(
            retargeter_config.get("ai_sapiens_sparse_wrist_repair_hand_residual_guard_m", 0.003)
        )
        self.ai_sapiens_sparse_wrist_repair_iterations = int(
            retargeter_config.get(
                "ai_sapiens_sparse_wrist_repair_iterations",
                max(8, int(max(1, self.ik_iterations) // 2)),
            )
        )
        self.ai_sapiens_sparse_branch_resolve_enabled = (
            self.target_type == pipeline_utils.TargetType.AI_SAPIENS
            and bool(retargeter_config.get("enable_ai_sapiens_sparse_branch_resolve", False))
        )
        sparse_branch_weights = retargeter_config.get(
            "ai_sapiens_sparse_branch_weights",
            [0.006],
        )
        if isinstance(sparse_branch_weights, (int, float)):
            sparse_branch_weights = [float(sparse_branch_weights)]
        self.ai_sapiens_sparse_branch_weights = [
            float(w) for w in sparse_branch_weights if float(w) >= 0.0
        ] or [0.006]
        self.ai_sapiens_dynamic_lateral_start_after_warmup = bool(
            retargeter_config.get("ai_sapiens_dynamic_lateral_start_after_warmup", True)
        )
        self.ai_sapiens_dynamic_lateral_skip_single_output_frame = bool(
            retargeter_config.get("ai_sapiens_dynamic_lateral_skip_single_output_frame", True)
        )
        self.ai_sapiens_dynamic_lateral_data = []
        self.ai_sapiens_sparse_pelvis_local_corridor_enabled = (
            self.target_type == pipeline_utils.TargetType.AI_SAPIENS
            and bool(retargeter_config.get("enable_ai_sapiens_sparse_pelvis_local_corridor", False))
        )
        self.ai_sapiens_sparse_corridor_capsule_threshold_m = float(
            retargeter_config.get("ai_sapiens_sparse_corridor_capsule_threshold_m", 0.10)
        )
        self.ai_sapiens_sparse_corridor_max_hand_shift_m = float(
            retargeter_config.get("ai_sapiens_sparse_corridor_max_hand_shift_m", 0.0045)
        )
        self.ai_sapiens_sparse_corridor_active_ratio_cap = float(
            retargeter_config.get("ai_sapiens_sparse_corridor_active_ratio_cap", 0.25)
        )
        self.ai_sapiens_forearm_segment_corridor_enabled = (
            self.target_type == pipeline_utils.TargetType.AI_SAPIENS
            and bool(retargeter_config.get("enable_ai_sapiens_forearm_segment_corridor", False))
        )
        self.ai_sapiens_forearm_segment_corridor_threshold_m = float(
            retargeter_config.get("ai_sapiens_forearm_segment_corridor_capsule_threshold_m", 0.08)
        )
        self.ai_sapiens_forearm_segment_corridor_max_shift_m = float(
            retargeter_config.get("ai_sapiens_forearm_segment_corridor_max_shift_m", 0.003)
        )
        self.ai_sapiens_forearm_segment_corridor_gain = float(
            retargeter_config.get("ai_sapiens_forearm_segment_corridor_gain", 0.25)
        )
        self.ai_sapiens_forearm_segment_corridor_smooth_window = int(
            retargeter_config.get("ai_sapiens_forearm_segment_corridor_smooth_window", 9)
        )
        self.ai_sapiens_forearm_segment_corridor_active_ratio_cap = float(
            retargeter_config.get("ai_sapiens_forearm_segment_corridor_active_ratio_cap", 0.20)
        )
        self.ai_sapiens_window_endpoint_release_enabled = (
            self.target_type == pipeline_utils.TargetType.AI_SAPIENS
            and bool(retargeter_config.get("enable_ai_sapiens_window_endpoint_release", False))
        )
        self.ai_sapiens_window_endpoint_release_windows = list(
            retargeter_config.get("ai_sapiens_window_endpoint_release_windows", [])
            or []
        )
        self.ai_sapiens_window_endpoint_release_hand_budget_m = float(
            retargeter_config.get("ai_sapiens_window_endpoint_release_hand_budget_m", 0.0)
        )
        self.ai_sapiens_window_endpoint_release_forearm_budget_m = float(
            retargeter_config.get("ai_sapiens_window_endpoint_release_forearm_budget_m", 0.0)
        )
        self.ai_sapiens_window_endpoint_release_capsule_threshold_m = float(
            retargeter_config.get("ai_sapiens_window_endpoint_release_capsule_threshold_m", 0.16)
        )
        self.ai_sapiens_window_endpoint_release_gain = float(
            retargeter_config.get("ai_sapiens_window_endpoint_release_gain", 1.0)
        )
        self.ai_sapiens_window_endpoint_release_taper_frames = int(
            retargeter_config.get("ai_sapiens_window_endpoint_release_taper_frames", 10)
        )
        self.ai_sapiens_window_endpoint_release_direction_mode = str(
            retargeter_config.get(
                "ai_sapiens_window_endpoint_release_direction_mode",
                "pelvis_local_inward_only",
            )
        )
        self.ai_sapiens_hand_hip_target_clearance_enabled = (
            self.target_type == pipeline_utils.TargetType.AI_SAPIENS
            and (
                bool(retargeter_config.get("enable_ai_sapiens_hand_hip_target_clearance", False))
                or self.ai_sapiens_sparse_pelvis_local_corridor_enabled
            )
        )
        self.ai_sapiens_hand_hip_clearance_detect_m = float(
            retargeter_config.get(
                "ai_sapiens_hand_hip_clearance_detect_m",
                self.ai_sapiens_sparse_corridor_capsule_threshold_m
                if self.ai_sapiens_sparse_pelvis_local_corridor_enabled else 0.10,
            )
        )
        self.ai_sapiens_hand_hip_clearance_margin_m = float(
            retargeter_config.get("ai_sapiens_hand_hip_clearance_margin_m", 0.05)
        )
        self.ai_sapiens_hand_hip_clearance_gain = float(
            retargeter_config.get("ai_sapiens_hand_hip_clearance_gain", 0.25)
        )
        self.ai_sapiens_hand_hip_clearance_max_shift_m = float(
            retargeter_config.get(
                "ai_sapiens_hand_hip_clearance_max_shift_m",
                self.ai_sapiens_sparse_corridor_max_hand_shift_m
                if self.ai_sapiens_sparse_pelvis_local_corridor_enabled else 0.015,
            )
        )
        self.ai_sapiens_hand_hip_clearance_smooth_window = int(
            retargeter_config.get("ai_sapiens_hand_hip_clearance_smooth_window", 5)
        )
        self.ai_sapiens_hand_hip_clearance_ramp_output_frames = int(
            retargeter_config.get("ai_sapiens_hand_hip_clearance_ramp_output_frames", 0)
        )
        self.ai_sapiens_hand_hip_clearance_mode = str(
            retargeter_config.get(
                "ai_sapiens_hand_hip_clearance_mode",
                "guarded_pelvis_local_radial_corridor"
                if self.ai_sapiens_sparse_pelvis_local_corridor_enabled
                else "pelvis_local_same_side_hip_capsule",
            )
        )
        self.ai_sapiens_hand_hip_clearance_cross_side_drop_guard_m = float(
            retargeter_config.get("ai_sapiens_hand_hip_clearance_cross_side_drop_guard_m", 0.005)
        )
        self.ai_sapiens_hand_hip_clearance_min_same_side_gain_m = float(
            retargeter_config.get("ai_sapiens_hand_hip_clearance_min_same_side_gain_m", 1e-6)
        )
        self.ai_sapiens_hand_hip_clearance_sequence_min_applied_ratio_per_side = float(
            retargeter_config.get("ai_sapiens_hand_hip_clearance_sequence_min_applied_ratio_per_side", 0.0)
        )
        self.ai_sapiens_hand_hip_clearance_sequence_min_same_gain_p50_m = float(
            retargeter_config.get("ai_sapiens_hand_hip_clearance_sequence_min_same_gain_p50_m", 0.0)
        )
        self.ai_sapiens_hand_hip_clearance_window_guard_size_frames = int(
            retargeter_config.get("ai_sapiens_hand_hip_clearance_window_guard_size_frames", 0)
        )
        self.ai_sapiens_hand_hip_clearance_window_guard_stride_frames = int(
            retargeter_config.get("ai_sapiens_hand_hip_clearance_window_guard_stride_frames", 0)
        )
        self.ai_sapiens_hand_hip_clearance_window_min_applied_ratio_per_side = float(
            retargeter_config.get("ai_sapiens_hand_hip_clearance_window_min_applied_ratio_per_side", 0.0)
        )
        self.ai_sapiens_hand_hip_clearance_window_min_same_gain_p50_m = float(
            retargeter_config.get("ai_sapiens_hand_hip_clearance_window_min_same_gain_p50_m", 0.0)
        )
        self.input_target_preprocess_summaries = []
        self.input_target_stage_traces = []

        if (
            self.target_type == pipeline_utils.TargetType.UNITREE_G1
            or self.target_type == pipeline_utils.TargetType.AI_SAPIENS
        ):
            self.robot_builder = newton.ModelBuilder()
            if self.target_type == pipeline_utils.TargetType.UNITREE_G1:
                self.robot_builder.add_mjcf(
                    newton.utils.download_asset("unitree_g1") / "mjcf/g1_29dof_rev_1_0.xml")
            else:
                self.robot_builder.add_mjcf(
                    ai_sapiens_assets.resolve_ai_sapiens_mjcf_path(retargeter_config.get("robot_mjcf")))

            self.human_robot_scaler = HumanToRobotScaler(
                skeleton, retargeter_config['model_height'], io_utils.get_config_file(retargeter_config['human_robot_scaler_config']))

            self.num_body_count = self.robot_builder.body_count
            self.num_dofs = self.robot_builder.joint_dof_count
            self.ik_model = self._build_model(1)
            self._apply_ai_sapiens_ik_joint_safety_margins_to_model()
            self.warmup_neutral_q_indices = self._build_joint_q_indices(
                self.warmup_neutral_joint_names)
            self.ai_sapiens_arm_temporal_q_indices = self._build_joint_q_indices(
                self.ai_sapiens_arm_temporal_joint_names)
            self.ai_sapiens_wrist_roll_nullspace_q_indices = self._build_joint_q_indices(
                self.ai_sapiens_wrist_roll_nullspace_joint_names)
            self.ai_sapiens_arm_branch_nullspace_q_indices = self._build_joint_q_indices(
                self.ai_sapiens_arm_branch_nullspace_joint_names)
            self.ai_sapiens_left_wrist_roll_q_indices = self._build_joint_q_indices(
                ["left_wrist_roll_joint"])
            self.ai_sapiens_right_wrist_roll_q_indices = self._build_joint_q_indices(
                ["right_wrist_roll_joint"])
            self.ai_sapiens_left_arm_branch_q_indices = self._build_joint_q_indices(
                ["left_shoulder_pitch_joint", "left_shoulder_yaw_joint", "left_elbow_joint"])
            self.ai_sapiens_right_arm_branch_q_indices = self._build_joint_q_indices(
                ["right_shoulder_pitch_joint", "right_shoulder_yaw_joint", "right_elbow_joint"])
            if (
                self.ai_sapiens_arm_nullspace_temporal_reference_enabled
                or self.ai_sapiens_sparse_wrist_roll_resolve_enabled
            ):
                self.ai_sapiens_wrist_roll_nullspace_coord_masks = self._joint_coord_mask_from_q_indices(
                    self.ai_sapiens_wrist_roll_nullspace_q_indices)
            if (
                self.ai_sapiens_arm_nullspace_temporal_reference_enabled
                or self.ai_sapiens_sparse_branch_resolve_enabled
            ):
                self.ai_sapiens_arm_branch_nullspace_coord_masks = self._joint_coord_mask_from_q_indices(
                    self.ai_sapiens_arm_branch_nullspace_q_indices)

            (
                self.mapped_joints,
                self.mapped_joint_indices,
                self.mapped_body_link_pos_data,
                self.mapped_body_link_rot_data
            ) = self._build_target_mapping(
                self.ik_model,
                self.human_robot_scaler.skeleton,
                retargeter_config)

            if self.ai_sapiens_arm_projection_enabled:
                self.ai_sapiens_arm_projection_data = self._build_ai_sapiens_arm_projection_data()
            if (
                self.ai_sapiens_adaptive_arm_objective_enabled
                or self.ai_sapiens_arm_nullspace_temporal_reference_enabled
            ):
                self.ai_sapiens_adaptive_arm_objective_data = self._build_ai_sapiens_arm_projection_data()
            if self.ai_sapiens_risk_window_forearm_priority_enabled or self.ai_sapiens_arm_projection_state_gate_enabled:
                self.ai_sapiens_risk_window_data = self._build_ai_sapiens_arm_projection_data()
            if self.ai_sapiens_elbow_branch_hint_objective_enabled:
                self.ai_sapiens_elbow_branch_hint_data = self._build_ai_sapiens_arm_projection_data()
            if (
                self.ai_sapiens_bilateral_arm_bend_objective_enabled
                or self.ai_sapiens_soft_bend_wrist_reference_enabled
                or self.ai_sapiens_contact_gated_elbow_midpoint_hint_enabled
                or self.ai_sapiens_contact_micro_wrist_reference_enabled
            ):
                self.ai_sapiens_bilateral_arm_bend_data = self._build_ai_sapiens_arm_projection_data()
            if self.ai_sapiens_limb_bend_angle_objective_enabled:
                self.ai_sapiens_limb_bend_angle_data = self._build_ai_sapiens_limb_bend_angle_data()
            if self.ai_sapiens_limb_plane_normal_objective_enabled:
                self.ai_sapiens_limb_plane_normal_data = self._build_ai_sapiens_limb_plane_normal_data()
            if self.ai_sapiens_limb_midpoint_position_objective_enabled:
                self.ai_sapiens_limb_midpoint_position_data = (
                    self._build_ai_sapiens_limb_midpoint_position_data()
                )
            if self.ai_sapiens_torso_local_limb_midpoint_objective_enabled:
                self.ai_sapiens_torso_local_limb_midpoint_data = (
                    self._build_ai_sapiens_torso_local_limb_midpoint_data()
                )
            if self.ai_sapiens_capsule_proxy_barrier_enabled:
                self.ai_sapiens_capsule_proxy_barrier_data = self._build_ai_sapiens_capsule_proxy_barrier_data()
            if self.ai_sapiens_arm_segment_direction_enabled:
                self.ai_sapiens_arm_segment_direction_data = (
                    self._build_ai_sapiens_arm_segment_direction_data()
                )
            if self.ai_sapiens_leg_projection_enabled:
                self.ai_sapiens_leg_projection_data = self._build_ai_sapiens_leg_projection_data()
            if self.ai_sapiens_dynamic_lateral_correction_enabled:
                self.ai_sapiens_dynamic_lateral_data = self._build_ai_sapiens_dynamic_lateral_data()
            if self.ai_sapiens_source_segment_direction_enabled:
                self.ai_sapiens_source_segment_direction_data = (
                    self._build_ai_sapiens_source_segment_direction_data()
                )
            if self.ai_sapiens_chest_anchored_arm_root_enabled:
                self.ai_sapiens_chest_anchored_arm_root_data = (
                    self._build_ai_sapiens_chest_anchored_arm_root_data()
                )
            if self.ai_sapiens_pair_span_correction_enabled:
                self.ai_sapiens_pair_span_data = self._build_ai_sapiens_pair_span_data()

            smooth_joint_filter_objective_body_masks = retargeter_config.get('smooth_joint_filter_objective_body_masks', None)
            if smooth_joint_filter_objective_body_masks is not None:
                self.smooth_joint_filter_coord_masks = newton_utils.create_joint_coord_masks(
                    self.ik_model, smooth_joint_filter_objective_body_masks, 0.0)

            neutral_reference_objective_body_masks = retargeter_config.get('neutral_reference_objective_body_masks', None)
            if neutral_reference_objective_body_masks is not None:
                self.neutral_reference_coord_masks = newton_utils.create_joint_coord_masks(
                    self.ik_model, neutral_reference_objective_body_masks, 0.0)

            if (
                self.ai_sapiens_temporal_yaw_twist_reference_enabled
                and self.ai_sapiens_temporal_yaw_twist_reference_body_masks
            ):
                self.ai_sapiens_temporal_yaw_twist_reference_coord_masks = (
                    newton_utils.create_joint_coord_masks(
                        self.ik_model,
                        self.ai_sapiens_temporal_yaw_twist_reference_body_masks,
                        0.0,
                    )
                )

            if (
                self.ai_sapiens_arm_ik_temporal_reference_enabled
                and self.ai_sapiens_arm_ik_temporal_reference_body_masks
            ):
                self.ai_sapiens_arm_ik_temporal_reference_coord_masks = (
                    newton_utils.create_joint_coord_masks(
                        self.ik_model,
                        self.ai_sapiens_arm_ik_temporal_reference_body_masks,
                        0.0,
                    )
                )

            effector_names = self.human_robot_scaler.effector_names()
            self.target_effector_indices = [effector_names.index(name) for name in self.mapped_joints]
            self.feet_effector_indices = [
                self.mapped_joints.index("LeftFoot"),
                self.mapped_joints.index("RightFoot")]

            self.feet_stabilizer = FeetStabilizer(io_utils.get_config_file(retargeter_config['feet_stabilizer_config']))
            self.joint_limit_clamper = JointLimitClamper(self.ik_model)
            self.ai_sapiens_output_joint_safety_margin_specs = self._build_ai_sapiens_output_joint_safety_margin_specs()
            self.ai_sapiens_output_joint_step_limit_specs = self._build_ai_sapiens_output_joint_step_limit_specs()

            self.initialization_pose = None
            self.num_initialization_frames = 0
            self.num_stabilization_frames = 0
            if (retargeter_config['initialization_pose']):
                init_skel, init_anim = bvh_utils.load_bvh(io_utils.get_config_file(retargeter_config['initialization_pose']))
                self.initialization_pose = SkeletonInstance(init_skel, [0, 0, 0], wp.transform_identity())
                self.initialization_pose.set_local_transforms(init_anim.get_local_transforms(0))
                self.num_initialization_frames = retargeter_config.get('num_initialization_frames', _DEFAULT_NUM_INITIALIZATION_FRAMES)
                self.num_stabilization_frames = retargeter_config.get('num_stabilization_frames', _DEFAULT_NUM_STABILIZATION_FRAMES)
        else:
            raise ValueError("Unsupported robot type.")

    def clear(self):
        """
        Clear all accumulated input motions and reset internal state.

        This removes all previously added motions set for retargeting.
        It does not modify static configuration such as the robot model or IK settings.
        """
        self.input_targets = []
        self.input_sample_rates = []
        self.input_target_preprocess_summaries = []
        self.input_target_stage_traces = []
        self.input_adaptive_arm_objective_scales = []
        self.input_elbow_branch_hint_targets = []
        self.input_elbow_branch_hint_delta_traces = []
        self.input_arm_bend_objective_target_angles = []
        self.input_arm_bend_objective_active_masks = []
        self.input_arm_bend_objective_trace = []
        self.input_limb_bend_angle_target_angles = []
        self.input_limb_bend_angle_active_masks = []
        self.input_limb_bend_angle_trace = []
        self.input_limb_plane_normal_targets = []
        self.input_limb_plane_normal_active_masks = []
        self.input_limb_plane_normal_trace = []
        self.input_limb_midpoint_position_targets = []
        self.input_limb_midpoint_position_active_masks = []
        self.input_limb_midpoint_position_trace = []
        self.input_torso_local_limb_midpoint_offsets = []
        self.input_torso_local_limb_midpoint_active_masks = []
        self.input_torso_local_limb_midpoint_trace = []
        self.input_soft_bend_wrist_reference_targets = []
        self.input_soft_bend_wrist_reference_active_masks = []
        self.input_soft_bend_wrist_reference_trace = []
        self.input_contact_gated_elbow_hint_targets = []
        self.input_contact_gated_elbow_hint_active_masks = []
        self.input_contact_gated_elbow_hint_trace = []
        self.input_contact_micro_wrist_reference_targets = []
        self.input_contact_micro_wrist_reference_active_masks = []
        self.input_contact_micro_wrist_reference_trace = []
        self.input_capsule_proxy_barrier_active_masks = []
        self.input_capsule_proxy_barrier_trace = []
        self.input_risk_window_objective_scales = []
        self.last_internal_trace = None
        self.last_solver_stage_trace = None
        self.max_frames = -1

    def add_input_motions(self, buffers: list[AnimationBuffer], offsets: list[wp.transform], scale_animation: bool):
        """
        Add input motions to be retargeted.
        Each buffer is converted into IK targets using the human-to-robot scaler.

        Args:
            buffers: List of input animation buffers defined on the common skeleton.
            offsets: List of root transforms applied to each buffer. If the
                length does not match `buffers`, identity transforms are used
                for all.
            scale_animation: Whether to rescale the source motion using the
                configured HumanToRobotScaler.
        """
        offsets = offsets if len(offsets) == len(buffers) else [wp.transform_identity()] * len(buffers)
        for i in trange(len(buffers), desc="[INFO] Converting Motions for Newton"):
            buffer = buffers[i]
            if self.initialization_pose and self.num_initialization_frames > 0:
                buffer = newton_utils.create_buffer_with_initialization_frames(
                    self.initialization_pose, buffers[i], self.num_initialization_frames, self.num_stabilization_frames)

            self.max_frames = max(self.max_frames, buffer.num_frames)
            buffer_effectors = self.human_robot_scaler.compute_effectors_from_buffer(buffer, scale_animation, offsets[i])
            targets = np.asarray(buffer_effectors[:, self.target_effector_indices, :], dtype=np.float32).copy()
            target_raw_scaled = targets.copy()
            body_frame_summary = {
                "enabled": bool(self.ai_sapiens_source_body_frame_preservation_enabled),
                "corrected_frame_count": 0,
                "mean_chest_shift_m": 0.0,
                "max_chest_shift_m": 0.0,
            }
            body_frame_trace = np.full((targets.shape[0], 10), np.nan, dtype=np.float32)
            if self.ai_sapiens_source_body_frame_preservation_enabled:
                targets, body_frame_summary, body_frame_trace = (
                    self._apply_ai_sapiens_source_body_frame_preservation(
                        targets,
                        buffer,
                        offsets[i],
                    )
                )
            target_after_source_body_frame_preservation = targets.copy()
            whole_body_pose_summary = {
                "enabled": bool(self.ai_sapiens_source_whole_body_pose_enabled),
                "corrected_frame_count": 0,
                "mean_rotation_angle_deg": 0.0,
                "max_rotation_angle_deg": 0.0,
                "max_target_shift_m": 0.0,
            }
            whole_body_pose_trace = np.full((targets.shape[0], 12), np.nan, dtype=np.float32)
            if self.ai_sapiens_source_whole_body_pose_enabled:
                targets, whole_body_pose_summary, whole_body_pose_trace = (
                    self._apply_ai_sapiens_source_whole_body_pose(
                        targets,
                        buffer,
                        offsets[i],
                    )
                )
            target_after_source_whole_body_pose = targets.copy()
            body_chain_summary = {
                "enabled": bool(self.ai_sapiens_source_body_chain_preservation_enabled),
                "corrected_frame_count": 0,
                "mean_shift_m": 0.0,
                "max_shift_m": 0.0,
            }
            body_chain_trace = np.full((targets.shape[0], 12), np.nan, dtype=np.float32)
            if self.ai_sapiens_source_body_chain_preservation_enabled:
                targets, body_chain_summary, body_chain_trace = (
                    self._apply_ai_sapiens_source_body_chain_preservation(
                        targets,
                        buffer,
                        offsets[i],
                    )
                )
            target_after_source_body_chain_preservation = targets.copy()
            lateral_summary = {"enabled": False}
            if self.ai_sapiens_dynamic_lateral_correction_enabled:
                targets, lateral_summary = self._apply_ai_sapiens_dynamic_lateral_correction(
                    targets,
                    buffer,
                    offsets[i],
                )
            target_after_dynamic_lateral = targets.copy()
            pair_span_summary = {
                "enabled": bool(self.ai_sapiens_pair_span_correction_enabled),
                "pair_count": int(len(self.ai_sapiens_pair_span_data)),
                "stage": "per_pair",
                "before_source_segment": None,
                "after_source_segment": None,
            }
            if self.ai_sapiens_pair_span_correction_enabled:
                targets, before_pair_span_summary = self._apply_ai_sapiens_pair_span_correction(
                    targets,
                    buffer,
                    offsets[i],
                    stage="before_source_segment",
                )
                pair_span_summary["before_source_segment"] = before_pair_span_summary
            target_after_pair_span = targets.copy()
            source_segment_summary = {
                "enabled": bool(self.ai_sapiens_source_segment_direction_enabled),
                "chain_count": int(len(self.ai_sapiens_source_segment_direction_data)),
            }
            if self.ai_sapiens_source_segment_direction_enabled:
                targets, source_segment_summary = (
                    self._apply_ai_sapiens_source_segment_direction_correction(
                        targets,
                        buffer,
                        offsets[i],
                    )
                )
            target_after_source_segment_direction = targets.copy()
            source_foot_orientation_summary = {
                "enabled": bool(self.ai_sapiens_source_foot_orientation_enabled),
                "target_count": int(len(self.ai_sapiens_source_foot_orientation_targets)),
            }
            if self.ai_sapiens_source_foot_orientation_enabled:
                targets, source_foot_orientation_summary = (
                    self._apply_ai_sapiens_source_foot_orientation_correction(
                        targets,
                        buffer,
                        offsets[i],
                    )
                )
            target_after_source_foot_orientation = targets.copy()
            chest_anchor_summary = {
                "enabled": bool(self.ai_sapiens_chest_anchored_arm_root_enabled),
                "chain_count": int(len(self.ai_sapiens_chest_anchored_arm_root_data)),
            }
            if self.ai_sapiens_chest_anchored_arm_root_enabled:
                targets, chest_anchor_summary = (
                    self._apply_ai_sapiens_chest_anchored_arm_root_correction(
                        targets,
                        buffer,
                        offsets[i],
                    )
                )
            target_after_chest_anchored_arm_root = targets.copy()
            if self.ai_sapiens_pair_span_correction_enabled:
                targets, after_pair_span_summary = self._apply_ai_sapiens_pair_span_correction(
                    targets,
                    buffer,
                    offsets[i],
                    stage="after_source_segment",
                )
                pair_span_summary["after_source_segment"] = after_pair_span_summary
                if int(after_pair_span_summary.get("corrected_pair_count", 0)) > 0:
                    target_after_pair_span = targets.copy()
            leg_projection_summary = {
                "enabled": bool(self.ai_sapiens_leg_projection_enabled),
                "chain_count": int(len(self.ai_sapiens_leg_projection_data)),
            }
            if self.ai_sapiens_leg_projection_enabled:
                targets, leg_projection_summary = self._apply_ai_sapiens_leg_projection(targets)
            target_after_leg_projection = targets.copy()
            arm_projection_pre_summary = {
                "enabled": bool(self.ai_sapiens_arm_projection_enabled),
                "chain_count": int(len(self.ai_sapiens_arm_projection_data)),
                "stage": "pre_clearance",
            }
            if self.ai_sapiens_arm_projection_enabled:
                targets, arm_projection_pre_summary = self._apply_ai_sapiens_arm_projection(
                    targets,
                    stage="pre_clearance",
                )
            target_after_arm_feasible_projection_pre_clearance = targets.copy()
            hand_hip_clearance_summary = {
                "enabled": bool(self.ai_sapiens_hand_hip_target_clearance_enabled),
                "hand_count": 0,
            }
            if self.ai_sapiens_hand_hip_target_clearance_enabled:
                targets, hand_hip_clearance_summary = self._apply_ai_sapiens_hand_hip_target_clearance(targets)
            target_after_hand_hip_clearance = targets.copy()
            forearm_segment_corridor_summary = {
                "enabled": bool(self.ai_sapiens_forearm_segment_corridor_enabled),
                "arm_count": 0,
            }
            if self.ai_sapiens_forearm_segment_corridor_enabled:
                targets, forearm_segment_corridor_summary = (
                    self._apply_ai_sapiens_forearm_segment_corridor(targets)
                )
            target_after_forearm_segment_corridor = targets.copy()
            window_endpoint_release_summary = {
                "enabled": bool(self.ai_sapiens_window_endpoint_release_enabled),
                "window_count": int(len(self.ai_sapiens_window_endpoint_release_windows)),
                "applied_frame_count": 0,
            }
            window_endpoint_release_trace = np.full(
                (targets.shape[0], 2, 12),
                np.nan,
                dtype=np.float32,
            )
            if self.ai_sapiens_window_endpoint_release_enabled:
                (
                    targets,
                    window_endpoint_release_summary,
                    window_endpoint_release_trace,
                ) = self._apply_ai_sapiens_window_endpoint_release(targets)
            target_after_window_endpoint_release = targets.copy()
            arm_projection_final_summary = {
                "enabled": bool(self.ai_sapiens_arm_projection_enabled),
                "chain_count": int(len(self.ai_sapiens_arm_projection_data)),
                "stage": "final_after_clearance",
            }
            if self.ai_sapiens_arm_projection_enabled:
                targets, arm_projection_final_summary = self._apply_ai_sapiens_arm_projection(
                    targets,
                    stage="final_after_clearance",
                )
            target_after_arm_feasible_projection_final = targets.copy()
            arm_segment_direction_summary = {
                "enabled": bool(self.ai_sapiens_arm_segment_direction_enabled),
                "chain_count": int(len(self.ai_sapiens_arm_segment_direction_data)),
            }
            arm_projection_post_direction_summary = {
                "enabled": False,
                "chain_count": 0,
                "stage": "post_arm_segment_direction",
            }
            if (
                self.ai_sapiens_arm_segment_direction_enabled
                and self.ai_sapiens_arm_direction_apply_after_projection
            ):
                targets, arm_segment_direction_summary = (
                    self._apply_ai_sapiens_arm_segment_direction_objective(
                        targets,
                        buffer,
                        offsets[i],
                    )
                )
                if self.ai_sapiens_arm_projection_enabled:
                    targets, arm_projection_post_direction_summary = (
                        self._apply_ai_sapiens_arm_projection(
                            targets,
                            stage="post_arm_segment_direction",
                        )
                    )
            target_after_arm_segment_direction = targets.copy()
            target_after_arm_feasible_projection_post_direction = targets.copy()
            target_after_arm_projection = targets.copy()
            adaptive_arm_summary = {
                "enabled": bool(self.ai_sapiens_adaptive_arm_objective_enabled),
                "chain_count": int(len(self.ai_sapiens_adaptive_arm_objective_data)),
            }
            adaptive_arm_scale_trace = np.zeros(
                (
                    targets.shape[0],
                    max(1, len(self.ai_sapiens_adaptive_arm_objective_data)),
                    4,
                ),
                dtype=np.float32,
            )
            if (
                self.ai_sapiens_adaptive_arm_objective_enabled
                or self.ai_sapiens_arm_nullspace_temporal_reference_enabled
            ):
                targets, adaptive_arm_summary, adaptive_arm_scale_trace = (
                    self._apply_ai_sapiens_adaptive_arm_objective_targets(targets)
                )
            target_after_adaptive_arm_objective = targets.copy()
            elbow_branch_hint_summary = {
                "enabled": bool(self.ai_sapiens_elbow_branch_hint_objective_enabled),
                "chain_count": int(len(self.ai_sapiens_elbow_branch_hint_data)),
            }
            elbow_branch_hint_targets = np.full(
                (
                    targets.shape[0],
                    max(1, len(self.ai_sapiens_elbow_branch_hint_data)),
                    3,
                ),
                np.nan,
                dtype=np.float32,
            )
            elbow_branch_hint_delta_trace = np.full(
                (
                    targets.shape[0],
                    max(1, len(self.ai_sapiens_elbow_branch_hint_data)),
                    4,
                ),
                np.nan,
                dtype=np.float32,
            )
            if self.ai_sapiens_elbow_branch_hint_objective_enabled:
                (
                    elbow_branch_hint_targets,
                    elbow_branch_hint_summary,
                    elbow_branch_hint_delta_trace,
                ) = self._compute_ai_sapiens_elbow_branch_hint_targets(targets)
            arm_bend_summary = {
                "enabled": bool(self.ai_sapiens_bilateral_arm_bend_objective_enabled),
                "chain_count": int(len(self.ai_sapiens_bilateral_arm_bend_data)),
                "weight": float(self.ai_sapiens_arm_bend_weight),
            }
            soft_bend_summary = {
                "enabled": bool(self.ai_sapiens_soft_bend_wrist_reference_enabled),
                "chain_count": int(len(self.ai_sapiens_bilateral_arm_bend_data)),
                "weight": float(self.ai_sapiens_soft_bend_wrist_reference_weight),
                "max_delta_m": float(self.ai_sapiens_soft_bend_wrist_reference_max_delta_m),
            }
            (
                arm_bend_target_angles,
                arm_bend_active_masks,
                arm_bend_trace,
                soft_bend_wrist_targets,
                soft_bend_wrist_active_masks,
                soft_bend_wrist_trace,
                arm_bend_summary,
                soft_bend_summary,
            ) = self._compute_ai_sapiens_bilateral_arm_bend_objective_targets(targets)
            (
                limb_bend_target_angles,
                limb_bend_active_masks,
                limb_bend_trace,
                limb_bend_summary,
            ) = self._compute_ai_sapiens_limb_bend_angle_objective_targets(targets)
            (
                limb_plane_normal_targets,
                limb_plane_normal_active_masks,
                limb_plane_normal_trace,
                limb_plane_normal_summary,
            ) = self._compute_ai_sapiens_limb_plane_normal_objective_targets(targets)
            (
                limb_midpoint_targets,
                limb_midpoint_active_masks,
                limb_midpoint_trace,
                limb_midpoint_summary,
            ) = self._compute_ai_sapiens_limb_midpoint_position_objective_targets(targets)
            (
                torso_local_midpoint_offsets,
                torso_local_midpoint_active_masks,
                torso_local_midpoint_trace,
                torso_local_midpoint_summary,
            ) = self._compute_ai_sapiens_torso_local_limb_midpoint_offsets(targets)
            (
                contact_elbow_targets,
                contact_elbow_active_masks,
                contact_elbow_trace,
                contact_wrist_targets,
                contact_wrist_active_masks,
                contact_wrist_trace,
                contact_elbow_summary,
                contact_wrist_summary,
            ) = self._compute_ai_sapiens_contact_gated_arm_references(targets)
            (
                capsule_barrier_active_masks,
                capsule_barrier_trace,
                capsule_barrier_summary,
            ) = self._compute_ai_sapiens_capsule_proxy_barrier_masks(targets)
            (
                risk_window_scale_trace,
                risk_window_summary,
            ) = self._compute_ai_sapiens_risk_window_objective_scales(targets)

            self.input_targets.append(targets)
            self.input_adaptive_arm_objective_scales.append(adaptive_arm_scale_trace)
            self.input_risk_window_objective_scales.append(risk_window_scale_trace)
            self.input_elbow_branch_hint_targets.append(elbow_branch_hint_targets)
            self.input_elbow_branch_hint_delta_traces.append(elbow_branch_hint_delta_trace)
            self.input_arm_bend_objective_target_angles.append(arm_bend_target_angles)
            self.input_arm_bend_objective_active_masks.append(arm_bend_active_masks)
            self.input_arm_bend_objective_trace.append(arm_bend_trace)
            self.input_limb_bend_angle_target_angles.append(limb_bend_target_angles)
            self.input_limb_bend_angle_active_masks.append(limb_bend_active_masks)
            self.input_limb_bend_angle_trace.append(limb_bend_trace)
            self.input_limb_plane_normal_targets.append(limb_plane_normal_targets)
            self.input_limb_plane_normal_active_masks.append(limb_plane_normal_active_masks)
            self.input_limb_plane_normal_trace.append(limb_plane_normal_trace)
            self.input_limb_midpoint_position_targets.append(limb_midpoint_targets)
            self.input_limb_midpoint_position_active_masks.append(limb_midpoint_active_masks)
            self.input_limb_midpoint_position_trace.append(limb_midpoint_trace)
            self.input_torso_local_limb_midpoint_offsets.append(torso_local_midpoint_offsets)
            self.input_torso_local_limb_midpoint_active_masks.append(torso_local_midpoint_active_masks)
            self.input_torso_local_limb_midpoint_trace.append(torso_local_midpoint_trace)
            self.input_soft_bend_wrist_reference_targets.append(soft_bend_wrist_targets)
            self.input_soft_bend_wrist_reference_active_masks.append(soft_bend_wrist_active_masks)
            self.input_soft_bend_wrist_reference_trace.append(soft_bend_wrist_trace)
            self.input_contact_gated_elbow_hint_targets.append(contact_elbow_targets)
            self.input_contact_gated_elbow_hint_active_masks.append(contact_elbow_active_masks)
            self.input_contact_gated_elbow_hint_trace.append(contact_elbow_trace)
            self.input_contact_micro_wrist_reference_targets.append(contact_wrist_targets)
            self.input_contact_micro_wrist_reference_active_masks.append(contact_wrist_active_masks)
            self.input_contact_micro_wrist_reference_trace.append(contact_wrist_trace)
            self.input_capsule_proxy_barrier_active_masks.append(capsule_barrier_active_masks)
            self.input_capsule_proxy_barrier_trace.append(capsule_barrier_trace)
            self.input_sample_rates.append(buffers[i].sample_rate)
            self.input_target_stage_traces.append({
                "target_raw_scaled": target_raw_scaled,
                "target_after_source_body_frame_preservation": target_after_source_body_frame_preservation,
                "target_after_source_whole_body_pose": target_after_source_whole_body_pose,
                "target_after_source_body_chain_preservation": target_after_source_body_chain_preservation,
                "source_body_frame_preservation_trace": body_frame_trace,
                "source_whole_body_pose_trace": whole_body_pose_trace,
                "source_body_chain_preservation_trace": body_chain_trace,
                "target_after_dynamic_lateral": target_after_dynamic_lateral,
                "target_after_pair_span": target_after_pair_span,
                "target_after_source_segment_direction": target_after_source_segment_direction,
                "target_after_source_foot_orientation": target_after_source_foot_orientation,
                "target_after_chest_anchored_arm_root": target_after_chest_anchored_arm_root,
                "target_after_leg_projection": target_after_leg_projection,
                "target_after_arm_feasible_projection_pre_clearance": (
                    target_after_arm_feasible_projection_pre_clearance
                ),
                "target_after_arm_projection": target_after_arm_projection,
                "target_after_hand_hip_clearance": target_after_hand_hip_clearance,
                "target_after_forearm_segment_corridor": target_after_forearm_segment_corridor,
                "target_after_window_endpoint_release": target_after_window_endpoint_release,
                "target_after_arm_feasible_projection_final": (
                    target_after_arm_feasible_projection_final
                ),
                "target_after_arm_segment_direction": target_after_arm_segment_direction,
                "target_after_arm_feasible_projection_post_direction": (
                    target_after_arm_feasible_projection_post_direction
                ),
                "target_after_adaptive_arm_objective": (
                    target_after_adaptive_arm_objective
                ),
                "adaptive_arm_objective_scale_trace": adaptive_arm_scale_trace,
                "risk_window_objective_scale_trace": risk_window_scale_trace,
                "elbow_branch_hint_targets": elbow_branch_hint_targets,
                "elbow_branch_hint_delta_trace": elbow_branch_hint_delta_trace,
                "arm_bend_objective_target_angles": arm_bend_target_angles,
                "arm_bend_objective_active_mask": arm_bend_active_masks,
                "arm_bend_objective_trace": arm_bend_trace,
                "limb_bend_angle_objective_target_angles": limb_bend_target_angles,
                "limb_bend_angle_objective_active_mask": limb_bend_active_masks,
                "limb_bend_angle_objective_trace": limb_bend_trace,
                "limb_plane_normal_objective_targets": limb_plane_normal_targets,
                "limb_plane_normal_objective_active_mask": limb_plane_normal_active_masks,
                "limb_plane_normal_objective_trace": limb_plane_normal_trace,
                "limb_midpoint_position_objective_targets": limb_midpoint_targets,
                "limb_midpoint_position_objective_active_mask": limb_midpoint_active_masks,
                "limb_midpoint_position_objective_trace": limb_midpoint_trace,
                "torso_local_limb_midpoint_objective_offsets": torso_local_midpoint_offsets,
                "torso_local_limb_midpoint_objective_active_mask": torso_local_midpoint_active_masks,
                "torso_local_limb_midpoint_objective_trace": torso_local_midpoint_trace,
                "soft_bend_wrist_reference_targets": soft_bend_wrist_targets,
                "soft_bend_wrist_reference_active_mask": soft_bend_wrist_active_masks,
                "soft_bend_wrist_reference_trace": soft_bend_wrist_trace,
                "contact_gated_elbow_hint_targets": contact_elbow_targets,
                "contact_gated_elbow_hint_active_mask": contact_elbow_active_masks,
                "contact_gated_elbow_hint_trace": contact_elbow_trace,
                "contact_micro_wrist_reference_targets": contact_wrist_targets,
                "contact_micro_wrist_reference_active_mask": contact_wrist_active_masks,
                "contact_micro_wrist_reference_trace": contact_wrist_trace,
                "capsule_proxy_barrier_active_mask": capsule_barrier_active_masks,
                "capsule_proxy_barrier_trace": capsule_barrier_trace,
                "window_endpoint_release_trace": window_endpoint_release_trace,
            })
            self.input_target_preprocess_summaries.append({
                "ai_sapiens_source_body_frame_preservation": body_frame_summary,
                "ai_sapiens_source_whole_body_pose": whole_body_pose_summary,
                "ai_sapiens_source_body_chain_preservation": body_chain_summary,
                "ai_sapiens_dynamic_lateral_correction": lateral_summary,
                "ai_sapiens_pair_span_correction": pair_span_summary,
                "ai_sapiens_source_segment_direction_correction": source_segment_summary,
                "ai_sapiens_source_foot_orientation_correction": source_foot_orientation_summary,
                "ai_sapiens_chest_anchored_arm_root_correction": chest_anchor_summary,
                "ai_sapiens_leg_projection": leg_projection_summary,
                "ai_sapiens_arm_projection_pre_clearance": arm_projection_pre_summary,
                "ai_sapiens_arm_projection_final": arm_projection_final_summary,
                "ai_sapiens_arm_segment_direction_objective": arm_segment_direction_summary,
                "ai_sapiens_adaptive_arm_objective": adaptive_arm_summary,
                "ai_sapiens_risk_window_forearm_priority": risk_window_summary,
                "ai_sapiens_elbow_branch_hint_objective": elbow_branch_hint_summary,
                "ai_sapiens_bilateral_arm_bend_objective": arm_bend_summary,
                "ai_sapiens_limb_bend_angle_objective": limb_bend_summary,
                "ai_sapiens_limb_plane_normal_objective": limb_plane_normal_summary,
                "ai_sapiens_limb_midpoint_position_objective": limb_midpoint_summary,
                "ai_sapiens_torso_local_limb_midpoint_objective": torso_local_midpoint_summary,
                "ai_sapiens_soft_bend_wrist_reference": soft_bend_summary,
                "ai_sapiens_contact_gated_elbow_midpoint_hint": contact_elbow_summary,
                "ai_sapiens_contact_micro_wrist_reference": contact_wrist_summary,
                "ai_sapiens_capsule_proxy_barrier": capsule_barrier_summary,
                "ai_sapiens_arm_projection_post_direction": arm_projection_post_direction_summary,
                "ai_sapiens_arm_projection": arm_projection_final_summary,
                "ai_sapiens_hand_hip_target_clearance": hand_hip_clearance_summary,
                "ai_sapiens_forearm_segment_corridor": forearm_segment_corridor_summary,
                "ai_sapiens_window_endpoint_release": window_endpoint_release_summary,
                "ai_sapiens_output_joint_safety_margin": self._ai_sapiens_output_joint_safety_margin_summary(),
                "ai_sapiens_arm_temporal_regularization": self._ai_sapiens_arm_temporal_regularization_summary(),
            })

    def execute(self):
        """
        Run the retargeting pipeline on all added input motions.

        This method builds a multi-environment Newton model, sets up IK
        objectives, and performs frame-by-frame IK solving.

        Returns:
            list[CSVAnimationBuffer]: A list of retargeted robot motions, one per input motion.
        """
        num_envs = len(self.input_targets)
        if num_envs == 0:
            self.retargeted_motions = []
            return

        # Clamp objective weights to valid values
        self.ik_iterations = max(1, self.ik_iterations)
        self.joint_limit_weight = max(0.0, self.joint_limit_weight)
        self.smooth_joint_filter_weight = max(0.0, self.smooth_joint_filter_weight)
        self.neutral_reference_weight = max(0.0, self.neutral_reference_weight)
        self.ai_sapiens_temporal_yaw_twist_reference_weight = max(
            0.0,
            self.ai_sapiens_temporal_yaw_twist_reference_weight,
        )
        self.ai_sapiens_arm_ik_temporal_reference_weight = max(
            0.0,
            self.ai_sapiens_arm_ik_temporal_reference_weight,
        )
        self.ai_sapiens_limb_bend_angle_weight = max(
            0.0,
            self.ai_sapiens_limb_bend_angle_weight,
        )
        self.ai_sapiens_limb_midpoint_position_weight = max(
            0.0,
            self.ai_sapiens_limb_midpoint_position_weight,
        )
        self.ai_sapiens_torso_local_limb_midpoint_weight = max(
            0.0,
            self.ai_sapiens_torso_local_limb_midpoint_weight,
        )

        print("[INFO] Newton Retargeter Settings: ")
        print(f"[INFO]\t  Source Skeleton Type: {pipeline_utils.get_source_str_from_type(self.source_type)}")
        print(f"[INFO]\t  Target Robot Type: {pipeline_utils.get_target_str_from_type(self.target_type)}")
        print(f"[INFO]\t  Post-Processing Enabled: {self.post_processing_enabled}")
        print(f"[INFO]\t  Initialization Pose: {self.initialization_pose is not None}")
        print(f"[INFO]\t  Initialization Frame Count: {self.num_initialization_frames}")
        print(f"[INFO]\t  Constraint Stabilization Frame Count: {self.num_stabilization_frames}")
        print(f"[INFO]\t  Reset Solver State At Output Start: {self.reset_solver_state_at_output_start}")
        print(f"[INFO]\t  Warmup Neutral Joint Names: {self.warmup_neutral_joint_names}")
        print(f"[INFO]\t  IK Solver Iterations: {self.ik_iterations}")
        print(f"[INFO]\t  Joint Limit Objective Weight: {self.joint_limit_weight}")
        print(f"[INFO]\t  Smooth Joint Filter Objective Weight: {self.smooth_joint_filter_weight}")
        print(f"[INFO]\t  Neutral Reference Objective Weight: {self.neutral_reference_weight}")
        if self.target_type == pipeline_utils.TargetType.AI_SAPIENS:
            print(
                "[INFO]\t  AI Sapiens Temporal Yaw/Twist Reference: "
                f"{self.ai_sapiens_temporal_yaw_twist_reference_enabled} "
                f"(weight={self.ai_sapiens_temporal_yaw_twist_reference_weight:.6f})"
            )
            print(
                "[INFO]\t  AI Sapiens Arm IK Temporal Reference: "
                f"{self.ai_sapiens_arm_ik_temporal_reference_enabled} "
                f"(weight={self.ai_sapiens_arm_ik_temporal_reference_weight:.6f})"
            )
            print(
                "[INFO]\t  AI Sapiens Sparse Wrist Roll Resolve: "
                f"{self.ai_sapiens_sparse_wrist_roll_resolve_enabled} "
                f"(trigger={self.ai_sapiens_sparse_wrist_roll_trigger_deg:.3f} deg, "
                f"weights={self.ai_sapiens_sparse_wrist_roll_weights})"
            )
        if self.target_type == pipeline_utils.TargetType.AI_SAPIENS:
            print(
                "[INFO]\t  AI Sapiens Root Orientation Step Limit: "
                f"{self.ai_sapiens_root_orientation_step_limit_enabled} "
                f"({self.ai_sapiens_root_orientation_max_step_deg:.3f} deg)"
            )
            print(
                "[INFO]\t  AI Sapiens Direct Body Chain Staged Solver: "
                f"{self.ai_sapiens_direct_body_chain_staged_solver_enabled} "
                f"(stage1_iter={self.ai_sapiens_direct_body_chain_stage1_iterations}, "
                f"stage2_iter={self.ai_sapiens_direct_body_chain_stage2_iterations})"
            )
            print(
                "[INFO]\t  AI Sapiens Limb Bend Angle Objective: "
                f"{self.ai_sapiens_limb_bend_angle_objective_enabled} "
                f"(weight={self.ai_sapiens_limb_bend_angle_weight:.6f}, "
                f"groups={self.ai_sapiens_limb_bend_angle_groups})"
            )
            print(
                "[INFO]\t  AI Sapiens Limb Plane Normal Objective: "
                f"{self.ai_sapiens_limb_plane_normal_objective_enabled} "
                f"(weight={self.ai_sapiens_limb_plane_normal_weight:.6f}, "
                f"groups={self.ai_sapiens_limb_plane_normal_groups})"
            )
            print(
                "[INFO]\t  AI Sapiens Limb Midpoint Position Objective: "
                f"{self.ai_sapiens_limb_midpoint_position_objective_enabled} "
                f"(weight={self.ai_sapiens_limb_midpoint_position_weight:.6f}, "
                f"max_delta_m={self.ai_sapiens_limb_midpoint_position_max_delta_m:.6f}, "
                f"groups={self.ai_sapiens_limb_midpoint_position_groups}, "
                f"group_weights={self.ai_sapiens_limb_midpoint_position_group_weights}, "
                f"suppress_contact={self.ai_sapiens_limb_midpoint_position_suppress_contact_risk})"
            )
            print(
                "[INFO]\t  AI Sapiens Torso-Local Limb Midpoint Objective: "
                f"{self.ai_sapiens_torso_local_limb_midpoint_objective_enabled} "
                f"(weight={self.ai_sapiens_torso_local_limb_midpoint_weight:.6f}, "
                f"max_delta_m={self.ai_sapiens_torso_local_limb_midpoint_max_delta_m:.6f}, "
                f"offset_scale={self.ai_sapiens_torso_local_limb_midpoint_offset_scale:.6f}, "
                f"groups={self.ai_sapiens_torso_local_limb_midpoint_groups}, "
                f"group_weights={self.ai_sapiens_torso_local_limb_midpoint_group_weights})"
            )

        model = self._build_model(num_envs)
        state = model.state()

        if self.post_processing_enabled:
            self.feet_stabilizer.setup_num_envs(num_envs)
            env_feet_tx = np.empty((num_envs, len(self.feet_effector_indices), 7), dtype=np.float32)

        (
            position_objectives,
            rotation_objectives,
            joint_limit_objective,
            smooth_joint_filter_objective,
            neutral_reference_objective,
            temporal_yaw_twist_reference_objective,
            arm_ik_temporal_reference_objective,
            wrist_roll_nullspace_temporal_objective,
            arm_branch_nullspace_temporal_objective,
            elbow_branch_hint_objectives,
            arm_bend_angle_objective,
            limb_bend_angle_objective,
            limb_plane_normal_objective,
            limb_midpoint_position_objective,
            torso_local_limb_midpoint_objective,
            soft_bend_wrist_reference_objective,
            contact_gated_elbow_hint_objective,
            contact_micro_wrist_reference_objective,
            capsule_proxy_barrier_objective,
        ) = self._create_ik_objectives(num_envs, model, state)

        # Add optional objectives
        ik_solver_active_objectives = [*position_objectives, *rotation_objectives]
        if self.joint_limit_weight > 0.0:
            ik_solver_active_objectives.append(joint_limit_objective)
        if self.smooth_joint_filter_weight > 0.0:
            ik_solver_active_objectives.append(smooth_joint_filter_objective)
        if self.neutral_reference_weight > 0.0 and self.neutral_reference_coord_masks is not None:
            ik_solver_active_objectives.append(neutral_reference_objective)
        if (
            self.ai_sapiens_temporal_yaw_twist_reference_enabled
            and self.ai_sapiens_temporal_yaw_twist_reference_weight > 0.0
            and self.ai_sapiens_temporal_yaw_twist_reference_coord_masks is not None
        ):
            ik_solver_active_objectives.append(temporal_yaw_twist_reference_objective)
        if (
            self.ai_sapiens_arm_ik_temporal_reference_enabled
            and self.ai_sapiens_arm_ik_temporal_reference_weight > 0.0
            and self.ai_sapiens_arm_ik_temporal_reference_coord_masks is not None
        ):
            ik_solver_active_objectives.append(arm_ik_temporal_reference_objective)
        if (
            (
                self.ai_sapiens_arm_nullspace_temporal_reference_enabled
                or self.ai_sapiens_sparse_wrist_roll_resolve_enabled
            )
            and self.ai_sapiens_wrist_roll_nullspace_coord_masks is not None
            and max(
                self.ai_sapiens_wrist_roll_temporal_weight_base,
                self.ai_sapiens_wrist_roll_temporal_weight_high,
                *self.ai_sapiens_sparse_wrist_roll_weights,
            ) > 0.0
        ):
            ik_solver_active_objectives.append(wrist_roll_nullspace_temporal_objective)
        if (
            (
                self.ai_sapiens_arm_nullspace_temporal_reference_enabled
                or self.ai_sapiens_sparse_branch_resolve_enabled
            )
            and self.ai_sapiens_arm_branch_nullspace_coord_masks is not None
            and max(
                self.ai_sapiens_arm_branch_temporal_weight_base,
                self.ai_sapiens_arm_branch_temporal_weight_high,
                *self.ai_sapiens_sparse_branch_weights,
            ) > 0.0
        ):
            ik_solver_active_objectives.append(arm_branch_nullspace_temporal_objective)
        if (
            self.ai_sapiens_elbow_branch_hint_objective_enabled
            and self.ai_sapiens_elbow_branch_hint_weight > 0.0
            and elbow_branch_hint_objectives
        ):
            ik_solver_active_objectives.extend(elbow_branch_hint_objectives)
        if (
            self.ai_sapiens_bilateral_arm_bend_objective_enabled
            and self.ai_sapiens_arm_bend_weight > 0.0
            and arm_bend_angle_objective is not None
        ):
            ik_solver_active_objectives.append(arm_bend_angle_objective)
        if (
            self.ai_sapiens_limb_bend_angle_objective_enabled
            and self.ai_sapiens_limb_bend_angle_weight > 0.0
            and limb_bend_angle_objective is not None
        ):
            ik_solver_active_objectives.append(limb_bend_angle_objective)
        if (
            self.ai_sapiens_limb_plane_normal_objective_enabled
            and self.ai_sapiens_limb_plane_normal_weight > 0.0
            and limb_plane_normal_objective is not None
        ):
            ik_solver_active_objectives.append(limb_plane_normal_objective)
        if (
            self.ai_sapiens_limb_midpoint_position_objective_enabled
            and self.ai_sapiens_limb_midpoint_position_weight > 0.0
            and limb_midpoint_position_objective
        ):
            ik_solver_active_objectives.extend(limb_midpoint_position_objective)
        if (
            self.ai_sapiens_torso_local_limb_midpoint_objective_enabled
            and self.ai_sapiens_torso_local_limb_midpoint_weight > 0.0
            and torso_local_limb_midpoint_objective
        ):
            ik_solver_active_objectives.extend(torso_local_limb_midpoint_objective)
        if (
            self.ai_sapiens_soft_bend_wrist_reference_enabled
            and self.ai_sapiens_soft_bend_wrist_reference_weight > 0.0
            and soft_bend_wrist_reference_objective is not None
        ):
            ik_solver_active_objectives.append(soft_bend_wrist_reference_objective)
        if (
            self.ai_sapiens_contact_gated_elbow_midpoint_hint_enabled
            and self.ai_sapiens_contact_gated_elbow_hint_weight > 0.0
            and contact_gated_elbow_hint_objective is not None
        ):
            ik_solver_active_objectives.append(contact_gated_elbow_hint_objective)
        if (
            self.ai_sapiens_contact_micro_wrist_reference_enabled
            and self.ai_sapiens_contact_micro_wrist_reference_weight > 0.0
            and contact_micro_wrist_reference_objective is not None
        ):
            ik_solver_active_objectives.append(contact_micro_wrist_reference_objective)
        if (
            self.ai_sapiens_capsule_proxy_barrier_enabled
            and self.ai_sapiens_capsule_proxy_barrier_weight > 0.0
            and capsule_proxy_barrier_objective is not None
        ):
            ik_solver_active_objectives.append(capsule_proxy_barrier_objective)

        jacobian_mode = (
            ik.IKJacobianType.MIXED
            if (
                (
                    self.ai_sapiens_bilateral_arm_bend_objective_enabled
                    and arm_bend_angle_objective is not None
                )
                or (
                    self.ai_sapiens_limb_bend_angle_objective_enabled
                    and limb_bend_angle_objective is not None
                )
                or (
                    self.ai_sapiens_soft_bend_wrist_reference_enabled
                    and soft_bend_wrist_reference_objective is not None
                )
                or (
                    self.ai_sapiens_contact_gated_elbow_midpoint_hint_enabled
                    and contact_gated_elbow_hint_objective is not None
                )
                or (
                    self.ai_sapiens_contact_micro_wrist_reference_enabled
                    and contact_micro_wrist_reference_objective is not None
                )
                or (
                    self.ai_sapiens_capsule_proxy_barrier_enabled
                    and capsule_proxy_barrier_objective is not None
                )
            )
            else ik.IKJacobianType.ANALYTIC
        )
        ik_solver = ik.IKSolver(
            model=self.ik_model,
            n_problems=num_envs,
            objectives=ik_solver_active_objectives,
            lambda_initial=0.1,
            jacobian_mode=jacobian_mode)

        joint_q = wp.empty(shape=(num_envs, self.ik_model.joint_coord_count))
        wp.copy(joint_q, model.joint_q)
        default_joint_q = model.joint_q.numpy().reshape(num_envs, self.ik_model.joint_coord_count)
        position_objective_base_weights = [
            float(data[1]) for data in self.mapped_body_link_pos_data
        ]
        rotation_objective_base_weights = [
            float(w) for _, w in self.mapped_body_link_rot_data
        ]
        mapped_joint_to_objective_index = {
            str(name): idx for idx, name in enumerate(self.mapped_joints)
        }
        adaptive_position_target_indices = {
            "LeftHand": mapped_joint_to_objective_index.get("LeftHand"),
            "RightHand": mapped_joint_to_objective_index.get("RightHand"),
            "LeftForeArm": mapped_joint_to_objective_index.get("LeftForeArm"),
            "RightForeArm": mapped_joint_to_objective_index.get("RightForeArm"),
        }
        adaptive_rotation_target_indices = {
            "LeftHand": mapped_joint_to_objective_index.get("LeftHand"),
            "RightHand": mapped_joint_to_objective_index.get("RightHand"),
        }
        risk_window_position_target_indices = {
            "LeftArm": mapped_joint_to_objective_index.get("LeftArm"),
            "RightArm": mapped_joint_to_objective_index.get("RightArm"),
            "LeftHand": mapped_joint_to_objective_index.get("LeftHand"),
            "RightHand": mapped_joint_to_objective_index.get("RightHand"),
            "LeftForeArm": mapped_joint_to_objective_index.get("LeftForeArm"),
            "RightForeArm": mapped_joint_to_objective_index.get("RightForeArm"),
            "Chest": mapped_joint_to_objective_index.get("Chest"),
            "Hips": mapped_joint_to_objective_index.get("Hips"),
        }
        risk_window_rotation_target_indices = {
            "LeftHand": mapped_joint_to_objective_index.get("LeftHand"),
            "RightHand": mapped_joint_to_objective_index.get("RightHand"),
            "LeftForeArm": mapped_joint_to_objective_index.get("LeftForeArm"),
            "RightForeArm": mapped_joint_to_objective_index.get("RightForeArm"),
            "Chest": mapped_joint_to_objective_index.get("Chest"),
            "Hips": mapped_joint_to_objective_index.get("Hips"),
        }

        def _apply_direct_body_chain_stage_weights(
            position_scales,
            rotation_scales,
            default_position_scale,
            default_rotation_scale,
        ):
            for idx, objective in enumerate(position_objectives):
                objective.weight = (
                    position_objective_base_weights[idx]
                    * float(default_position_scale)
                )
            for idx, objective in enumerate(rotation_objectives):
                objective.weight = (
                    rotation_objective_base_weights[idx]
                    * float(default_rotation_scale)
                )
            for target_name, scale in position_scales.items():
                target_idx = mapped_joint_to_objective_index.get(str(target_name))
                if target_idx is not None and 0 <= target_idx < len(position_objectives):
                    position_objectives[target_idx].weight = (
                        position_objective_base_weights[target_idx] * float(scale)
                    )
            for target_name, scale in rotation_scales.items():
                target_idx = mapped_joint_to_objective_index.get(str(target_name))
                if target_idx is not None and 0 <= target_idx < len(rotation_objectives):
                    rotation_objectives[target_idx].weight = (
                        rotation_objective_base_weights[target_idx] * float(scale)
                    )

        def _restore_objective_weights(position_weights, rotation_weights):
            for idx, objective in enumerate(position_objectives):
                objective.weight = float(position_weights[idx])
            for idx, objective in enumerate(rotation_objectives):
                objective.weight = float(rotation_weights[idx])

        def _snapshot_objective_weights():
            return (
                [float(objective.weight) for objective in position_objectives],
                [float(objective.weight) for objective in rotation_objectives],
            )

        if self.neutral_reference_weight > 0.0 and self.neutral_reference_coord_masks is not None:
            neutral_reference_objective.set_target(default_joint_q)
        if (
            self.ai_sapiens_temporal_yaw_twist_reference_enabled
            and self.ai_sapiens_temporal_yaw_twist_reference_weight > 0.0
            and self.ai_sapiens_temporal_yaw_twist_reference_coord_masks is not None
        ):
            temporal_yaw_twist_reference_objective.set_target(default_joint_q)
        if (
            self.ai_sapiens_arm_ik_temporal_reference_enabled
            and self.ai_sapiens_arm_ik_temporal_reference_weight > 0.0
            and self.ai_sapiens_arm_ik_temporal_reference_coord_masks is not None
        ):
            arm_ik_temporal_reference_objective.set_target(default_joint_q)
        if (
            self.ai_sapiens_arm_nullspace_temporal_reference_enabled
            or self.ai_sapiens_sparse_wrist_roll_resolve_enabled
            or self.ai_sapiens_sparse_branch_resolve_enabled
        ):
            wrist_roll_nullspace_temporal_objective.set_target(default_joint_q)
            if not self.ai_sapiens_arm_nullspace_temporal_reference_enabled:
                wrist_roll_nullspace_temporal_objective.set_weight(0.0)
        if (
            self.ai_sapiens_arm_nullspace_temporal_reference_enabled
            or self.ai_sapiens_sparse_branch_resolve_enabled
        ):
            arm_branch_nullspace_temporal_objective.set_target(default_joint_q)
            if not self.ai_sapiens_arm_nullspace_temporal_reference_enabled:
                arm_branch_nullspace_temporal_objective.set_weight(0.0)

        # Solver initialization
        ik_solver.reset()

        graph_capture = None

        def single_step():
            ik_solver.step(joint_q, joint_q, iterations=self.ik_iterations)

        if (
            wp.get_device().is_cuda
            and not self.ai_sapiens_adaptive_arm_objective_enabled
            and not self.ai_sapiens_sparse_wrist_roll_resolve_enabled
            and not self.ai_sapiens_sparse_branch_resolve_enabled
            and not self.ai_sapiens_bilateral_arm_bend_objective_enabled
            and not self.ai_sapiens_limb_bend_angle_objective_enabled
            and not self.ai_sapiens_soft_bend_wrist_reference_enabled
            and not self.ai_sapiens_risk_window_forearm_priority_enabled
            and not self.ai_sapiens_direct_body_chain_staged_solver_enabled
        ):
            with wp.ScopedCapture() as cap:
                single_step()
            graph_capture = cap.graph
        else:
            ik_solver.step(joint_q, joint_q, iterations=self.ik_iterations)

        #import time
        num_frames_to_remove = self.num_initialization_frames + self.num_stabilization_frames
        joint_q_data = [np.full((len(self.input_targets[i]),), None) for i in range(num_envs)]
        solver_trace_arrays = None
        if self.enable_solver_stage_trace:
            q_trace_shape = (self.max_frames, num_envs, self.ik_model.joint_coord_count)
            solver_trace_arrays = {
                "q_before_solve": np.full(q_trace_shape, np.nan, dtype=np.float32),
                "q_after_direct_body_chain_stage1": np.full(q_trace_shape, np.nan, dtype=np.float32),
                "q_after_direct_body_chain_stage2": np.full(q_trace_shape, np.nan, dtype=np.float32),
                "q_after_solve": np.full(q_trace_shape, np.nan, dtype=np.float32),
                "q_after_sparse_wrist_repair": np.full(q_trace_shape, np.nan, dtype=np.float32),
                "q_after_warmup_neutral": np.full(q_trace_shape, np.nan, dtype=np.float32),
                "q_after_feet_stabilizer": np.full(q_trace_shape, np.nan, dtype=np.float32),
                "q_after_joint_limit": np.full(q_trace_shape, np.nan, dtype=np.float32),
                "q_after_output_safety_margin": np.full(q_trace_shape, np.nan, dtype=np.float32),
                "q_after_output_joint_step_limit": np.full(q_trace_shape, np.nan, dtype=np.float32),
                "q_after_root_orientation_step": np.full(q_trace_shape, np.nan, dtype=np.float32),
                "q_after_arm_ik_temporal_reference": np.full(q_trace_shape, np.nan, dtype=np.float32),
                "q_after_arm_temporal_regularization": np.full(q_trace_shape, np.nan, dtype=np.float32),
                "q_after_postprocess": np.full(q_trace_shape, np.nan, dtype=np.float32),
                "arm_ik_temporal_reference_weight": np.full((self.max_frames,), np.nan, dtype=np.float32),
                "nullspace_temporal_weight_trace": np.full((self.max_frames, 4), np.nan, dtype=np.float32),
                "sparse_wrist_repair_trace": np.full((self.max_frames, num_envs, 10), np.nan, dtype=np.float32),
                "arm_bend_objective_weight_trace": np.full((self.max_frames, 2), np.nan, dtype=np.float32),
                "limb_bend_angle_objective_weight_trace": np.full((self.max_frames, 2), np.nan, dtype=np.float32),
                "limb_plane_normal_objective_weight_trace": np.full((self.max_frames, 2), np.nan, dtype=np.float32),
                "limb_midpoint_position_objective_weight_trace": np.full((self.max_frames, 2), np.nan, dtype=np.float32),
                "torso_local_limb_midpoint_objective_weight_trace": np.full((self.max_frames, 2), np.nan, dtype=np.float32),
            }
        self.last_internal_trace = {
            "num_initialization_frames": int(self.num_initialization_frames),
            "num_stabilization_frames": int(self.num_stabilization_frames),
            "num_frames_to_remove": int(num_frames_to_remove),
            "reset_solver_state_at_output_start": bool(self.reset_solver_state_at_output_start),
            "warmup_neutral_joint_names": list(self.warmup_neutral_joint_names),
            "warmup_neutral_q_indices": [int(idx) for idx in self.warmup_neutral_q_indices],
            "qpos_schema": self._qpos_schema(),
            "warmup_qpos_by_env": [[] for _ in range(num_envs)],
        }
        for frame in trange(self.max_frames, desc="[INFO] Retargeting Motions"):
            if num_frames_to_remove > 0 and frame <= num_frames_to_remove:
                smooth_joint_filter_objective.set_weight(self.smooth_joint_filter_weight * (frame / float(num_frames_to_remove)))

            if (
                self.reset_solver_state_at_output_start
                and num_frames_to_remove > 0
                and frame == num_frames_to_remove
            ):
                wp.copy(joint_q, model.joint_q)
                ik_solver.reset()

            if (
                self.ai_sapiens_temporal_yaw_twist_reference_enabled
                and self.ai_sapiens_temporal_yaw_twist_reference_weight > 0.0
                and self.ai_sapiens_temporal_yaw_twist_reference_coord_masks is not None
            ):
                active_temporal_reference = (
                    (not self.ai_sapiens_temporal_yaw_twist_reference_start_after_warmup)
                    or frame >= num_frames_to_remove
                )
                temporal_yaw_twist_reference_objective.set_weight(
                    self.ai_sapiens_temporal_yaw_twist_reference_weight
                    if active_temporal_reference
                    else 0.0
                )
                temporal_reference_q = np.asarray(default_joint_q, dtype=np.float32).copy()
                if active_temporal_reference and frame > num_frames_to_remove:
                    for env in range(num_envs):
                        if frame > (len(self.input_targets[env]) - 1):
                            continue
                        previous = joint_q_data[env][frame - 1]
                        if previous is not None:
                            temporal_reference_q[env] = np.asarray(previous, dtype=np.float32)
                temporal_yaw_twist_reference_objective.set_target(temporal_reference_q)

            if (
                self.ai_sapiens_arm_ik_temporal_reference_enabled
                and self.ai_sapiens_arm_ik_temporal_reference_weight > 0.0
                and self.ai_sapiens_arm_ik_temporal_reference_coord_masks is not None
            ):
                active_arm_reference = frame > num_frames_to_remove
                arm_temporal_weight = float(self.ai_sapiens_arm_ik_temporal_reference_weight)
                if self.ai_sapiens_adaptive_arm_temporal_reference_enabled:
                    arm_temporal_weight = float(self.ai_sapiens_arm_temporal_reference_base_weight)
                    previous_step_l2_values = []
                    if active_arm_reference and frame > num_frames_to_remove + 1:
                        for env in range(num_envs):
                            if frame > (len(self.input_targets[env]) - 1):
                                continue
                            previous = joint_q_data[env][frame - 1]
                            previous2 = joint_q_data[env][frame - 2]
                            if previous is None or previous2 is None:
                                continue
                            prev = np.asarray(previous, dtype=np.float64)
                            prev2 = np.asarray(previous2, dtype=np.float64)
                            if self.ai_sapiens_arm_temporal_q_indices:
                                diff = prev[self.ai_sapiens_arm_temporal_q_indices] - prev2[self.ai_sapiens_arm_temporal_q_indices]
                            else:
                                diff = prev - prev2
                            diff = diff[np.isfinite(diff)]
                            if diff.size > 0:
                                previous_step_l2_values.append(float(np.linalg.norm(diff)))
                    if previous_step_l2_values:
                        step_p95 = float(np.percentile(previous_step_l2_values, 95))
                        if step_p95 > float(self.ai_sapiens_arm_temporal_reference_trigger_l2_p95):
                            arm_temporal_weight = float(
                                self.ai_sapiens_arm_temporal_reference_high_motion_weight
                            )
                arm_ik_temporal_reference_objective.set_weight(
                    arm_temporal_weight
                    if active_arm_reference
                    else 0.0
                )
                if solver_trace_arrays is not None:
                    solver_trace_arrays["arm_ik_temporal_reference_weight"][frame] = (
                        arm_temporal_weight if active_arm_reference else 0.0
                    )
                arm_reference_q = np.asarray(default_joint_q, dtype=np.float32).copy()
                if active_arm_reference and frame > num_frames_to_remove:
                    for env in range(num_envs):
                        if frame > (len(self.input_targets[env]) - 1):
                            continue
                        previous = joint_q_data[env][frame - 1]
                        if previous is not None:
                            previous_np = np.asarray(previous, dtype=np.float32)
                            if (
                                self.ai_sapiens_arm_ik_temporal_reference_mode
                                in {"linear_predict", "acceleration"}
                                and frame > num_frames_to_remove + 1
                            ):
                                previous2 = joint_q_data[env][frame - 2]
                                if previous2 is not None:
                                    previous2_np = np.asarray(previous2, dtype=np.float32)
                                    arm_reference_q[env] = (
                                        previous_np + (previous_np - previous2_np)
                                    )
                                else:
                                    arm_reference_q[env] = previous_np
                            else:
                                arm_reference_q[env] = previous_np
                arm_ik_temporal_reference_objective.set_target(arm_reference_q)

            if self.ai_sapiens_arm_nullspace_temporal_reference_enabled:
                active_nullspace_reference = frame > num_frames_to_remove
                nullspace_reference_q = np.asarray(default_joint_q, dtype=np.float32).copy()
                max_reach_ratio = 0.0
                max_prev_step_l2 = 0.0
                if active_nullspace_reference:
                    for env in range(num_envs):
                        if frame > (len(self.input_targets[env]) - 1):
                            continue
                        previous = joint_q_data[env][frame - 1]
                        if previous is not None:
                            nullspace_reference_q[env] = np.asarray(previous, dtype=np.float32)
                        if env < len(self.input_adaptive_arm_objective_scales):
                            scales = self.input_adaptive_arm_objective_scales[env]
                            if frame < len(scales):
                                reach_values = np.asarray(scales[frame, :, 0], dtype=np.float64)
                                reach_values = reach_values[np.isfinite(reach_values)]
                                if reach_values.size > 0:
                                    max_reach_ratio = max(max_reach_ratio, float(np.nanmax(reach_values)))
                        if frame > num_frames_to_remove + 1:
                            previous2 = joint_q_data[env][frame - 2]
                            if previous is not None and previous2 is not None:
                                prev = np.asarray(previous, dtype=np.float64)
                                prev2 = np.asarray(previous2, dtype=np.float64)
                                q_indices = sorted(set(
                                    self.ai_sapiens_wrist_roll_nullspace_q_indices
                                    + self.ai_sapiens_arm_branch_nullspace_q_indices
                                ))
                                if q_indices:
                                    diff = prev[q_indices] - prev2[q_indices]
                                    diff = diff[np.isfinite(diff)]
                                    if diff.size > 0:
                                        max_prev_step_l2 = max(max_prev_step_l2, float(np.linalg.norm(diff)))
                high_nullspace = (
                    active_nullspace_reference
                    and (
                        max_reach_ratio >= float(self.ai_sapiens_nullspace_temporal_reach_trigger)
                        or max_prev_step_l2 >= float(self.ai_sapiens_nullspace_temporal_step_trigger_rad)
                    )
                )
                wrist_weight = (
                    float(self.ai_sapiens_wrist_roll_temporal_weight_high)
                    if high_nullspace
                    else float(self.ai_sapiens_wrist_roll_temporal_weight_base)
                )
                branch_weight = (
                    float(self.ai_sapiens_arm_branch_temporal_weight_high)
                    if high_nullspace
                    else float(self.ai_sapiens_arm_branch_temporal_weight_base)
                )
                if not active_nullspace_reference:
                    wrist_weight = 0.0
                    branch_weight = 0.0
                wrist_roll_nullspace_temporal_objective.set_weight(wrist_weight)
                arm_branch_nullspace_temporal_objective.set_weight(branch_weight)
                wrist_roll_nullspace_temporal_objective.set_target(nullspace_reference_q)
                arm_branch_nullspace_temporal_objective.set_target(nullspace_reference_q)
                if solver_trace_arrays is not None:
                    solver_trace_arrays["nullspace_temporal_weight_trace"][frame] = np.asarray(
                        [max_reach_ratio, wrist_weight, branch_weight, max_prev_step_l2],
                        dtype=np.float32,
                    )

            #start_time = time.time()
            for i, objective in enumerate(position_objectives):
                objective.weight = position_objective_base_weights[i]
            for i, objective in enumerate(rotation_objectives):
                objective.weight = rotation_objective_base_weights[i]
            if self.ai_sapiens_adaptive_arm_objective_enabled and frame >= num_frames_to_remove:
                chain_scales = []
                for env in range(num_envs):
                    if env >= len(self.input_adaptive_arm_objective_scales):
                        continue
                    scales = self.input_adaptive_arm_objective_scales[env]
                    if frame >= len(scales):
                        continue
                    chain_scales.append(np.asarray(scales[frame], dtype=np.float64))
                if chain_scales:
                    mean_scale = np.nanmean(np.stack(chain_scales, axis=0), axis=0)
                    chain_map = [
                        ("LeftHand", "LeftForeArm", 0),
                        ("RightHand", "RightForeArm", 1),
                    ]
                    for hand_name, forearm_name, chain_idx in chain_map:
                        if chain_idx >= mean_scale.shape[0]:
                            continue
                        hand_scale = float(np.clip(mean_scale[chain_idx, 1], 0.0, 1.0))
                        forearm_scale = float(np.clip(mean_scale[chain_idx, 2], 0.0, 1.0))
                        hand_idx = adaptive_position_target_indices.get(hand_name)
                        if hand_idx is not None:
                            position_objectives[hand_idx].weight = (
                                position_objective_base_weights[hand_idx] * hand_scale
                            )
                        forearm_idx = adaptive_position_target_indices.get(forearm_name)
                        if forearm_idx is not None:
                            position_objectives[forearm_idx].weight = (
                                position_objective_base_weights[forearm_idx] * forearm_scale
                            )
                    if self.ai_sapiens_hand_rotation_mode == "off":
                        for target_idx in adaptive_rotation_target_indices.values():
                            if target_idx is not None:
                                rotation_objectives[target_idx].weight = 0.0
            if self.ai_sapiens_risk_window_forearm_priority_enabled and frame >= num_frames_to_remove:
                def _risk_weight(base_weight, scale_value, absolute_weight, active_value):
                    if (
                        self.ai_sapiens_risk_window_absolute_weights_enabled
                        and math.isfinite(float(absolute_weight))
                    ):
                        active = float(np.clip(active_value, 0.0, 1.0))
                        return float(base_weight) * (1.0 - active) + float(absolute_weight) * active
                    return float(base_weight) * float(scale_value)

                chain_scales = []
                for env in range(num_envs):
                    if env >= len(self.input_risk_window_objective_scales):
                        continue
                    scales = self.input_risk_window_objective_scales[env]
                    if frame >= len(scales):
                        continue
                    chain_scales.append(np.asarray(scales[frame], dtype=np.float64))
                if chain_scales:
                    mean_scale = np.nanmean(np.stack(chain_scales, axis=0), axis=0)
                    chain_map = [
                        ("LeftArm", "LeftHand", "LeftForeArm", 0),
                        ("RightArm", "RightHand", "RightForeArm", 1),
                    ]
                    for arm_name, hand_name, forearm_name, chain_idx in chain_map:
                        if chain_idx >= mean_scale.shape[0]:
                            continue
                        active_value = float(mean_scale[chain_idx, 0]) if mean_scale.shape[1] > 0 else 0.0
                        hand_scale = float(np.clip(mean_scale[chain_idx, 5], 0.0, 2.0))
                        forearm_scale = float(np.clip(mean_scale[chain_idx, 6], 0.0, 3.0))
                        hand_rot_scale = float(np.clip(mean_scale[chain_idx, 7], 0.0, 2.0))
                        arm_scale = float(np.clip(mean_scale[chain_idx, 10], 0.0, 3.0)) if mean_scale.shape[1] > 10 else 1.0
                        forearm_rot_scale = float(np.clip(mean_scale[chain_idx, 11], 0.0, 2.0)) if mean_scale.shape[1] > 11 else 1.0
                        arm_idx = risk_window_position_target_indices.get(arm_name)
                        if arm_idx is not None:
                            position_objectives[arm_idx].weight = (
                                _risk_weight(
                                    position_objective_base_weights[arm_idx],
                                    arm_scale,
                                    self.ai_sapiens_risk_window_arm_t_weight,
                                    active_value,
                                )
                            )
                        hand_idx = risk_window_position_target_indices.get(hand_name)
                        if hand_idx is not None:
                            position_objectives[hand_idx].weight = (
                                _risk_weight(
                                    position_objective_base_weights[hand_idx],
                                    hand_scale,
                                    self.ai_sapiens_risk_window_hand_t_weight,
                                    active_value,
                                )
                            )
                        forearm_idx = risk_window_position_target_indices.get(forearm_name)
                        if forearm_idx is not None:
                            position_objectives[forearm_idx].weight = (
                                _risk_weight(
                                    position_objective_base_weights[forearm_idx],
                                    forearm_scale,
                                    self.ai_sapiens_risk_window_forearm_t_weight,
                                    active_value,
                                )
                            )
                        hand_rot_idx = risk_window_rotation_target_indices.get(hand_name)
                        if hand_rot_idx is not None:
                            rotation_objectives[hand_rot_idx].weight = (
                                _risk_weight(
                                    rotation_objective_base_weights[hand_rot_idx],
                                    hand_rot_scale,
                                    self.ai_sapiens_risk_window_hand_r_weight,
                                    active_value,
                                )
                            )
                        forearm_rot_idx = risk_window_rotation_target_indices.get(forearm_name)
                        if forearm_rot_idx is not None:
                            rotation_objectives[forearm_rot_idx].weight = (
                                _risk_weight(
                                    rotation_objective_base_weights[forearm_rot_idx],
                                    forearm_rot_scale,
                                    self.ai_sapiens_risk_window_forearm_r_weight,
                                    active_value,
                                )
                            )
                    if self.ai_sapiens_risk_window_torso_lock_enabled:
                        chest_scale = float(np.clip(np.nanmax(mean_scale[:, 8]), 0.0, 3.0))
                        hips_scale = float(np.clip(np.nanmax(mean_scale[:, 9]), 0.0, 3.0))
                        torso_active = float(np.clip(np.nanmax(mean_scale[:, 0]), 0.0, 1.0))
                        chest_idx = risk_window_position_target_indices.get("Chest")
                        if chest_idx is not None:
                            position_objectives[chest_idx].weight = (
                                _risk_weight(
                                    position_objective_base_weights[chest_idx],
                                    chest_scale,
                                    self.ai_sapiens_risk_window_chest_t_weight,
                                    torso_active,
                                )
                            )
                            chest_rot_idx = risk_window_rotation_target_indices.get("Chest")
                            if chest_rot_idx is not None:
                                chest_rot_scale = 1.0 + (
                                    (float(self.ai_sapiens_risk_window_chest_r_scale) - 1.0)
                                    * max(0.0, min(1.0, chest_scale - 1.0))
                                    / max(1.0e-6, float(self.ai_sapiens_risk_window_chest_t_scale) - 1.0)
                                )
                                rotation_objectives[chest_rot_idx].weight = (
                                    _risk_weight(
                                        rotation_objective_base_weights[chest_rot_idx],
                                        chest_rot_scale,
                                        self.ai_sapiens_risk_window_chest_r_weight,
                                        torso_active,
                                    )
                                )
                        hips_idx = risk_window_position_target_indices.get("Hips")
                        if hips_idx is not None:
                            position_objectives[hips_idx].weight = (
                                _risk_weight(
                                    position_objective_base_weights[hips_idx],
                                    hips_scale,
                                    self.ai_sapiens_risk_window_hips_t_weight,
                                    torso_active,
                                )
                            )
                            hips_rot_idx = risk_window_rotation_target_indices.get("Hips")
                            if hips_rot_idx is not None:
                                hips_rot_scale = float(np.clip(np.nanmax(mean_scale[:, 12]), 0.0, 3.0)) if mean_scale.shape[1] > 12 else 1.0
                                rotation_objectives[hips_rot_idx].weight = (
                                    _risk_weight(
                                        rotation_objective_base_weights[hips_rot_idx],
                                        hips_rot_scale,
                                        self.ai_sapiens_risk_window_hips_r_weight,
                                        torso_active,
                                    )
                                )
            if self.ai_sapiens_elbow_branch_hint_objective_enabled and elbow_branch_hint_objectives:
                hint_weight = (
                    float(self.ai_sapiens_elbow_branch_hint_weight)
                    if (
                        not self.ai_sapiens_elbow_branch_hint_start_after_warmup
                        or frame >= num_frames_to_remove
                    )
                    else 0.0
                )
                for hint_idx, objective in enumerate(elbow_branch_hint_objectives):
                    objective.weight = hint_weight
                    for env in range(num_envs):
                        if env >= len(self.input_elbow_branch_hint_targets):
                            continue
                        hints = self.input_elbow_branch_hint_targets[env]
                        if (
                            frame >= hints.shape[0]
                            or hint_idx >= hints.shape[1]
                            or not np.isfinite(hints[frame, hint_idx]).all()
                        ):
                            continue
                        objective.set_target_position(
                            env,
                            wp.vec3(*hints[frame, hint_idx, 0:3]),
                        )
            if (
                self.ai_sapiens_bilateral_arm_bend_objective_enabled
                and arm_bend_angle_objective is not None
            ):
                chain_count = max(1, len(self.ai_sapiens_bilateral_arm_bend_data))
                target_angles = np.zeros((num_envs, chain_count), dtype=np.float32)
                active_masks = np.zeros((num_envs, chain_count), dtype=np.float32)
                active_bend_count = 0
                for env in range(num_envs):
                    if env >= len(self.input_arm_bend_objective_target_angles):
                        continue
                    angles = self.input_arm_bend_objective_target_angles[env]
                    masks = self.input_arm_bend_objective_active_masks[env]
                    if frame >= angles.shape[0]:
                        continue
                    count = min(chain_count, angles.shape[1])
                    target_angles[env, :count] = angles[frame, :count]
                    active_masks[env, :count] = masks[frame, :count]
                    active_bend_count += int(np.sum(masks[frame, :count] > 0.0))
                bend_weight = (
                    float(self.ai_sapiens_arm_bend_weight)
                    if frame >= num_frames_to_remove
                    else 0.0
                )
                arm_bend_angle_objective.set_weight(bend_weight)
                arm_bend_angle_objective.set_targets(target_angles, active_masks)
                if solver_trace_arrays is not None:
                    solver_trace_arrays["arm_bend_objective_weight_trace"][frame, 0] = bend_weight
                    solver_trace_arrays["arm_bend_objective_weight_trace"][frame, 1] = float(active_bend_count)
            if (
                self.ai_sapiens_limb_bend_angle_objective_enabled
                and limb_bend_angle_objective is not None
            ):
                chain_count = max(1, len(self.ai_sapiens_limb_bend_angle_data))
                target_angles = np.zeros((num_envs, chain_count), dtype=np.float32)
                active_masks = np.zeros((num_envs, chain_count), dtype=np.float32)
                active_limb_count = 0
                for env in range(num_envs):
                    if env >= len(self.input_limb_bend_angle_target_angles):
                        continue
                    angles = self.input_limb_bend_angle_target_angles[env]
                    masks = self.input_limb_bend_angle_active_masks[env]
                    if frame >= angles.shape[0]:
                        continue
                    count = min(chain_count, angles.shape[1])
                    target_angles[env, :count] = angles[frame, :count]
                    active_masks[env, :count] = masks[frame, :count]
                    active_limb_count += int(np.sum(masks[frame, :count] > 0.0))
                limb_weight = (
                    float(self.ai_sapiens_limb_bend_angle_weight)
                    if frame >= num_frames_to_remove
                    else 0.0
                )
                limb_bend_angle_objective.set_weight(limb_weight)
                limb_bend_angle_objective.set_targets(target_angles, active_masks)
                if solver_trace_arrays is not None:
                    solver_trace_arrays["limb_bend_angle_objective_weight_trace"][frame, 0] = limb_weight
                    solver_trace_arrays["limb_bend_angle_objective_weight_trace"][frame, 1] = float(active_limb_count)
            if (
                self.ai_sapiens_limb_plane_normal_objective_enabled
                and limb_plane_normal_objective is not None
            ):
                chain_count = max(1, len(self.ai_sapiens_limb_plane_normal_data))
                target_normals = np.zeros((num_envs, chain_count, 3), dtype=np.float32)
                active_masks = np.zeros((num_envs, chain_count), dtype=np.float32)
                active_plane_count = 0
                for env in range(num_envs):
                    if env >= len(self.input_limb_plane_normal_targets):
                        continue
                    normals = self.input_limb_plane_normal_targets[env]
                    masks = self.input_limb_plane_normal_active_masks[env]
                    if frame >= normals.shape[0]:
                        continue
                    count = min(chain_count, normals.shape[1])
                    target_normals[env, :count, :] = normals[frame, :count, :]
                    active_masks[env, :count] = masks[frame, :count]
                    active_plane_count += int(np.sum(masks[frame, :count] > 0.0))
                plane_weight = (
                    float(self.ai_sapiens_limb_plane_normal_weight)
                    if frame >= num_frames_to_remove
                    else 0.0
                )
                limb_plane_normal_objective.set_weight(plane_weight)
                limb_plane_normal_objective.set_targets(target_normals, active_masks)
                if solver_trace_arrays is not None:
                    solver_trace_arrays["limb_plane_normal_objective_weight_trace"][frame, 0] = plane_weight
                    solver_trace_arrays["limb_plane_normal_objective_weight_trace"][frame, 1] = float(active_plane_count)
            if (
                self.ai_sapiens_limb_midpoint_position_objective_enabled
                and limb_midpoint_position_objective
            ):
                current_joint_q = joint_q.numpy().astype(np.float32, copy=False)
                current_joint_q_flat = wp.array(current_joint_q.reshape(-1), dtype=wp.float32)
                newton.eval_fk(model, current_joint_q_flat, model.joint_qd, state)
                current_body_q = state.body_q.numpy()
                chain_count = min(
                    len(limb_midpoint_position_objective),
                    len(self.ai_sapiens_limb_midpoint_position_data),
                )
                active_midpoint_count = 0
                active_midpoint_weight_sum = 0.0
                for chain_idx in range(chain_count):
                    objective = limb_midpoint_position_objective[chain_idx]
                    chain = self.ai_sapiens_limb_midpoint_position_data[chain_idx]
                    mid_target_idx = int(chain["mid_idx"])
                    mid_link_idx = int(self.mapped_body_link_pos_data[mid_target_idx][0])
                    root_name = str(chain.get("root_name", ""))
                    if root_name in {"LeftArm", "RightArm"}:
                        group_name = "arm"
                    elif root_name in {"LeftLeg", "RightLeg"}:
                        group_name = "leg"
                    else:
                        group_name = "other"
                    chain_weight = (
                        float(
                            self.ai_sapiens_limb_midpoint_position_group_weights.get(
                                group_name,
                                self.ai_sapiens_limb_midpoint_position_weight,
                            )
                        )
                        if frame >= num_frames_to_remove
                        else 0.0
                    )
                    chain_weight = max(0.0, chain_weight)
                    chain_active_count = 0
                    for env in range(num_envs):
                        base = env * self.num_body_count
                        target = current_body_q[base + mid_link_idx][0:3]
                        active = False
                        if env < len(self.input_limb_midpoint_position_targets):
                            targets_ref = self.input_limb_midpoint_position_targets[env]
                            masks = self.input_limb_midpoint_position_active_masks[env]
                            if frame < targets_ref.shape[0] and chain_idx < targets_ref.shape[1]:
                                candidate_target = targets_ref[frame, chain_idx, :]
                                active = bool(
                                    np.isfinite(candidate_target).all()
                                    and masks[frame, chain_idx] > 0.0
                                )
                                if active:
                                    target = candidate_target
                        objective.set_target_position(env, wp.vec3(*target[0:3]))
                        if active:
                            chain_active_count += 1
                    objective.weight = chain_weight if chain_active_count > 0 else 0.0
                    active_midpoint_count += chain_active_count
                    active_midpoint_weight_sum += chain_weight * float(chain_active_count)
                if solver_trace_arrays is not None:
                    solver_trace_arrays["limb_midpoint_position_objective_weight_trace"][frame, 0] = (
                        active_midpoint_weight_sum / float(active_midpoint_count)
                        if active_midpoint_count > 0
                        else 0.0
                    )
                    solver_trace_arrays["limb_midpoint_position_objective_weight_trace"][frame, 1] = float(active_midpoint_count)
            if (
                self.ai_sapiens_torso_local_limb_midpoint_objective_enabled
                and torso_local_limb_midpoint_objective
            ):
                current_joint_q = joint_q.numpy().astype(np.float32, copy=False)
                current_joint_q_flat = wp.array(current_joint_q.reshape(-1), dtype=wp.float32)
                newton.eval_fk(model, current_joint_q_flat, model.joint_qd, state)
                current_body_q = state.body_q.numpy()
                chain_count = min(
                    len(torso_local_limb_midpoint_objective),
                    len(self.ai_sapiens_torso_local_limb_midpoint_data),
                )
                active_torso_local_count = 0
                active_torso_local_weight_sum = 0.0
                max_delta = max(0.0, float(self.ai_sapiens_torso_local_limb_midpoint_max_delta_m))
                offset_scale = float(self.ai_sapiens_torso_local_limb_midpoint_offset_scale)
                for chain_idx in range(chain_count):
                    objective = torso_local_limb_midpoint_objective[chain_idx]
                    chain = self.ai_sapiens_torso_local_limb_midpoint_data[chain_idx]
                    mid_target_idx = int(chain["mid_idx"])
                    torso_target_idx = int(chain["torso_idx"])
                    mid_link_idx = int(self.mapped_body_link_pos_data[mid_target_idx][0])
                    torso_link_idx = int(self.mapped_body_link_pos_data[torso_target_idx][0])
                    root_name = str(chain.get("root_name", ""))
                    if root_name in {"LeftArm", "RightArm"}:
                        group_name = "arm"
                    elif root_name in {"LeftLeg", "RightLeg"}:
                        group_name = "leg"
                    else:
                        group_name = "other"
                    chain_weight = (
                        float(
                            self.ai_sapiens_torso_local_limb_midpoint_group_weights.get(
                                group_name,
                                self.ai_sapiens_torso_local_limb_midpoint_weight,
                            )
                        )
                        if frame >= num_frames_to_remove
                        else 0.0
                    )
                    chain_weight = max(0.0, chain_weight)
                    chain_active_count = 0
                    for env in range(num_envs):
                        base = env * self.num_body_count
                        current_mid = np.asarray(current_body_q[base + mid_link_idx][0:3], dtype=np.float64)
                        target = current_mid.copy()
                        active = False
                        if env < len(self.input_torso_local_limb_midpoint_offsets):
                            offsets_ref = self.input_torso_local_limb_midpoint_offsets[env]
                            masks = self.input_torso_local_limb_midpoint_active_masks[env]
                            if frame < offsets_ref.shape[0] and chain_idx < offsets_ref.shape[1]:
                                local_offset = np.asarray(offsets_ref[frame, chain_idx, :], dtype=np.float64)
                                active = bool(
                                    np.isfinite(local_offset).all()
                                    and masks[frame, chain_idx] > 0.0
                                )
                                if active:
                                    torso_tf = current_body_q[base + torso_link_idx]
                                    torso_pos = np.asarray(torso_tf[0:3], dtype=np.float64)
                                    torso_quat = self._normalize_quat_np_xyzw(torso_tf[3:7])
                                    candidate = torso_pos + self._quat_rotate_np_xyzw(
                                        torso_quat,
                                        local_offset * offset_scale,
                                    )
                                    delta_vec = candidate - current_mid
                                    delta = float(np.linalg.norm(delta_vec))
                                    if max_delta > 0.0 and delta > max_delta:
                                        candidate = current_mid + delta_vec * (max_delta / max(delta, 1.0e-8))
                                    target = candidate
                        objective.set_target_position(env, wp.vec3(*target[0:3]))
                        if active:
                            chain_active_count += 1
                    objective.weight = chain_weight if chain_active_count > 0 else 0.0
                    active_torso_local_count += chain_active_count
                    active_torso_local_weight_sum += chain_weight * float(chain_active_count)
                if solver_trace_arrays is not None:
                    solver_trace_arrays["torso_local_limb_midpoint_objective_weight_trace"][frame, 0] = (
                        active_torso_local_weight_sum / float(active_torso_local_count)
                        if active_torso_local_count > 0
                        else 0.0
                    )
                    solver_trace_arrays["torso_local_limb_midpoint_objective_weight_trace"][frame, 1] = float(active_torso_local_count)
            if (
                self.ai_sapiens_soft_bend_wrist_reference_enabled
                and soft_bend_wrist_reference_objective is not None
            ):
                chain_count = max(1, len(self.ai_sapiens_bilateral_arm_bend_data))
                wrist_targets = np.zeros((num_envs, chain_count, 3), dtype=np.float32)
                wrist_masks = np.zeros((num_envs, chain_count), dtype=np.float32)
                active_wrist_count = 0
                for env in range(num_envs):
                    if env >= len(self.input_soft_bend_wrist_reference_targets):
                        continue
                    targets_ref = self.input_soft_bend_wrist_reference_targets[env]
                    masks = self.input_soft_bend_wrist_reference_active_masks[env]
                    if frame >= targets_ref.shape[0]:
                        continue
                    count = min(chain_count, targets_ref.shape[1])
                    frame_targets = targets_ref[frame, :count, :]
                    finite = np.isfinite(frame_targets).all(axis=1)
                    wrist_targets[env, :count, :] = np.where(
                        finite[:, None],
                        frame_targets,
                        0.0,
                    )
                    wrist_masks[env, :count] = np.where(finite, masks[frame, :count], 0.0)
                    active_wrist_count += int(np.sum(wrist_masks[env, :count] > 0.0))
                wrist_weight = (
                    float(self.ai_sapiens_soft_bend_wrist_reference_weight)
                    if frame >= num_frames_to_remove
                    else 0.0
                )
                soft_bend_wrist_reference_objective.set_weight(wrist_weight)
                soft_bend_wrist_reference_objective.set_targets(wrist_targets, wrist_masks)
                if solver_trace_arrays is not None:
                    # column 0 is bend-objective weight, column 1 active bend count;
                    # keep soft reference activity in the same compact trace by adding.
                    solver_trace_arrays["arm_bend_objective_weight_trace"][frame, 1] += float(active_wrist_count)
            if (
                self.ai_sapiens_contact_gated_elbow_midpoint_hint_enabled
                and contact_gated_elbow_hint_objective is not None
            ):
                chain_count = max(1, len(self.ai_sapiens_bilateral_arm_bend_data))
                elbow_targets = np.zeros((num_envs, chain_count, 3), dtype=np.float32)
                elbow_masks = np.zeros((num_envs, chain_count), dtype=np.float32)
                active_elbow_count = 0
                for env in range(num_envs):
                    if env >= len(self.input_contact_gated_elbow_hint_targets):
                        continue
                    targets_ref = self.input_contact_gated_elbow_hint_targets[env]
                    masks = self.input_contact_gated_elbow_hint_active_masks[env]
                    if frame >= targets_ref.shape[0]:
                        continue
                    count = min(chain_count, targets_ref.shape[1])
                    frame_targets = targets_ref[frame, :count, :]
                    finite = np.isfinite(frame_targets).all(axis=1)
                    elbow_targets[env, :count, :] = np.where(
                        finite[:, None],
                        frame_targets,
                        0.0,
                    )
                    elbow_masks[env, :count] = np.where(finite, masks[frame, :count], 0.0)
                    active_elbow_count += int(np.sum(elbow_masks[env, :count] > 0.0))
                elbow_weight = (
                    float(self.ai_sapiens_contact_gated_elbow_hint_weight)
                    if frame >= num_frames_to_remove
                    else 0.0
                )
                contact_gated_elbow_hint_objective.set_weight(elbow_weight)
                contact_gated_elbow_hint_objective.set_targets(elbow_targets, elbow_masks)
                if solver_trace_arrays is not None:
                    solver_trace_arrays["arm_bend_objective_weight_trace"][frame, 1] += float(active_elbow_count)
            if (
                self.ai_sapiens_contact_micro_wrist_reference_enabled
                and contact_micro_wrist_reference_objective is not None
            ):
                chain_count = max(1, len(self.ai_sapiens_bilateral_arm_bend_data))
                wrist_targets = np.zeros((num_envs, chain_count, 3), dtype=np.float32)
                wrist_masks = np.zeros((num_envs, chain_count), dtype=np.float32)
                active_wrist_count = 0
                for env in range(num_envs):
                    if env >= len(self.input_contact_micro_wrist_reference_targets):
                        continue
                    targets_ref = self.input_contact_micro_wrist_reference_targets[env]
                    masks = self.input_contact_micro_wrist_reference_active_masks[env]
                    if frame >= targets_ref.shape[0]:
                        continue
                    count = min(chain_count, targets_ref.shape[1])
                    frame_targets = targets_ref[frame, :count, :]
                    finite = np.isfinite(frame_targets).all(axis=1)
                    wrist_targets[env, :count, :] = np.where(
                        finite[:, None],
                        frame_targets,
                        0.0,
                    )
                    wrist_masks[env, :count] = np.where(finite, masks[frame, :count], 0.0)
                    active_wrist_count += int(np.sum(wrist_masks[env, :count] > 0.0))
                wrist_weight = (
                    float(self.ai_sapiens_contact_micro_wrist_reference_weight)
                    if frame >= num_frames_to_remove
                    else 0.0
                )
                contact_micro_wrist_reference_objective.set_weight(wrist_weight)
                contact_micro_wrist_reference_objective.set_targets(wrist_targets, wrist_masks)
                if solver_trace_arrays is not None:
                    solver_trace_arrays["arm_bend_objective_weight_trace"][frame, 1] += float(active_wrist_count)
            if (
                self.ai_sapiens_capsule_proxy_barrier_enabled
                and capsule_proxy_barrier_objective is not None
            ):
                pair_count = max(1, len(self.ai_sapiens_capsule_proxy_barrier_data))
                barrier_masks = np.zeros((num_envs, pair_count), dtype=np.float32)
                active_barrier_count = 0
                for env in range(num_envs):
                    if env >= len(self.input_capsule_proxy_barrier_active_masks):
                        continue
                    masks = self.input_capsule_proxy_barrier_active_masks[env]
                    if frame >= masks.shape[0]:
                        continue
                    count = min(pair_count, masks.shape[1])
                    barrier_masks[env, :count] = masks[frame, :count]
                    active_barrier_count += int(np.sum(masks[frame, :count] > 0.0))
                barrier_weight = (
                    float(self.ai_sapiens_capsule_proxy_barrier_weight)
                    if frame >= num_frames_to_remove
                    else 0.0
                )
                capsule_proxy_barrier_objective.set_weight(barrier_weight)
                capsule_proxy_barrier_objective.set_active_mask(barrier_masks)
                if solver_trace_arrays is not None:
                    solver_trace_arrays["arm_bend_objective_weight_trace"][frame, 1] += float(active_barrier_count)
            for env in range(num_envs):
                if frame > (len(self.input_targets[env])-1):
                    continue
                frame_targets = self.input_targets[env][frame]
                for i, target in enumerate(frame_targets):
                    position_objectives[i].set_target_position(env, wp.vec3(*target[0:3]))
                    rotation_objectives[i].set_target_rotation(env, wp.quat(*target[3:7]))

            if solver_trace_arrays is not None:
                solver_trace_arrays["q_before_solve"][frame] = joint_q.numpy()

            direct_body_chain_stage_active = (
                self.ai_sapiens_direct_body_chain_staged_solver_enabled
                and (
                    (not self.ai_sapiens_direct_body_chain_staged_start_after_warmup)
                    or frame >= num_frames_to_remove
                )
            )
            if direct_body_chain_stage_active:
                final_position_weights, final_rotation_weights = _snapshot_objective_weights()
                _apply_direct_body_chain_stage_weights(
                    self.ai_sapiens_direct_body_chain_stage1_position_scales,
                    self.ai_sapiens_direct_body_chain_stage1_rotation_scales,
                    self.ai_sapiens_direct_body_chain_stage1_default_position_scale,
                    self.ai_sapiens_direct_body_chain_stage1_default_rotation_scale,
                )
                ik_solver.step(
                    joint_q,
                    joint_q,
                    iterations=max(1, self.ai_sapiens_direct_body_chain_stage1_iterations),
                )
                if solver_trace_arrays is not None:
                    solver_trace_arrays["q_after_direct_body_chain_stage1"][frame] = joint_q.numpy()
                _apply_direct_body_chain_stage_weights(
                    self.ai_sapiens_direct_body_chain_stage2_position_scales,
                    self.ai_sapiens_direct_body_chain_stage2_rotation_scales,
                    self.ai_sapiens_direct_body_chain_stage2_default_position_scale,
                    self.ai_sapiens_direct_body_chain_stage2_default_rotation_scale,
                )
                ik_solver.step(
                    joint_q,
                    joint_q,
                    iterations=max(1, self.ai_sapiens_direct_body_chain_stage2_iterations),
                )
                if solver_trace_arrays is not None:
                    solver_trace_arrays["q_after_direct_body_chain_stage2"][frame] = joint_q.numpy()
                _restore_objective_weights(final_position_weights, final_rotation_weights)
            elif graph_capture is not None:
                wp.capture_launch(graph_capture)
            else:
                single_step()

            if solver_trace_arrays is not None:
                if not direct_body_chain_stage_active:
                    solver_trace_arrays["q_after_direct_body_chain_stage1"][frame] = joint_q.numpy()
                    solver_trace_arrays["q_after_direct_body_chain_stage2"][frame] = joint_q.numpy()
                solver_trace_arrays["q_after_solve"][frame] = joint_q.numpy()
                solver_trace_arrays["q_after_arm_ik_temporal_reference"][frame] = joint_q.numpy()

            if self.ai_sapiens_sparse_wrist_roll_resolve_enabled:
                sparse_trace = np.zeros((num_envs, 10), dtype=np.float32)
                sparse_trace[:, 4] = float(self.ai_sapiens_sparse_wrist_roll_weights[0])
                sparse_trace[:, 5] = (
                    float(self.ai_sapiens_sparse_branch_weights[0])
                    if self.ai_sapiens_sparse_branch_resolve_enabled
                    else 0.0
                )
                sparse_trace[:, 8] = float(max(1, self.ai_sapiens_sparse_wrist_repair_iterations))
                repair_active = frame > num_frames_to_remove
                current_before_repair = joint_q.numpy().astype(np.float32)
                if repair_active:
                    previous_q = np.full_like(current_before_repair, np.nan, dtype=np.float32)
                    for env in range(num_envs):
                        if frame > (len(self.input_targets[env]) - 1):
                            continue
                        previous = joint_q_data[env][frame - 1]
                        if previous is not None:
                            previous_q[env] = np.asarray(previous, dtype=np.float32)

                    trigger_rad = math.radians(max(0.0, float(self.ai_sapiens_sparse_wrist_roll_trigger_deg)))
                    reject_rad = math.radians(max(0.0, float(self.ai_sapiens_sparse_wrist_roll_reject_deg)))
                    side_specs = [
                        ("left", self.ai_sapiens_left_wrist_roll_q_indices, self.ai_sapiens_left_arm_branch_q_indices),
                        ("right", self.ai_sapiens_right_wrist_roll_q_indices, self.ai_sapiens_right_arm_branch_q_indices),
                    ]
                    target_q = current_before_repair.copy()
                    branch_target_q = current_before_repair.copy()
                    triggered_pairs = []
                    before_steps = [[] for _ in range(num_envs)]
                    for env in range(num_envs):
                        if not np.isfinite(previous_q[env]).all():
                            continue
                        for side, wrist_indices, branch_indices in side_specs:
                            if not wrist_indices:
                                continue
                            wrist_idx = int(wrist_indices[0])
                            step = abs(float(current_before_repair[env, wrist_idx] - previous_q[env, wrist_idx]))
                            if math.isfinite(step):
                                before_steps[env].append(step)
                                if step > reject_rad:
                                    sparse_trace[env, 1] += 1.0
                                if step > trigger_rad:
                                    triggered_pairs.append((env, side))
                                    sparse_trace[env, 0] += 1.0
                                    for q_idx in wrist_indices:
                                        target_q[env, int(q_idx)] = previous_q[env, int(q_idx)]
                                    if self.ai_sapiens_sparse_branch_resolve_enabled:
                                        for q_idx in branch_indices:
                                            branch_target_q[env, int(q_idx)] = previous_q[env, int(q_idx)]

                    for env, steps in enumerate(before_steps):
                        if steps:
                            sparse_trace[env, 2] = float(math.degrees(max(steps)))
                            sparse_trace[env, 3] = sparse_trace[env, 2]
                    if self.ai_sapiens_sparse_branch_resolve_enabled:
                        sparse_trace[:, 9] = sparse_trace[:, 0]

                    if triggered_pairs:
                        before_errors = self._sparse_hand_position_errors(
                            model,
                            state,
                            joint_q,
                            current_before_repair,
                            frame,
                            triggered_pairs,
                        )
                        wrist_roll_nullspace_temporal_objective.set_target(target_q)
                        wrist_roll_nullspace_temporal_objective.set_weight(
                            float(self.ai_sapiens_sparse_wrist_roll_weights[0])
                        )
                        if self.ai_sapiens_sparse_branch_resolve_enabled:
                            arm_branch_nullspace_temporal_objective.set_target(branch_target_q)
                            arm_branch_nullspace_temporal_objective.set_weight(
                                float(self.ai_sapiens_sparse_branch_weights[0])
                            )

                        ik_solver.step(
                            joint_q,
                            joint_q,
                            iterations=max(1, int(self.ai_sapiens_sparse_wrist_repair_iterations)),
                        )
                        repaired_q = joint_q.numpy().astype(np.float32)
                        after_errors = self._sparse_hand_position_errors(
                            model,
                            state,
                            joint_q,
                            repaired_q,
                            frame,
                            triggered_pairs,
                        )
                        after_steps = [[] for _ in range(num_envs)]
                        for env, side in triggered_pairs:
                            wrist_indices = (
                                self.ai_sapiens_left_wrist_roll_q_indices
                                if side == "left"
                                else self.ai_sapiens_right_wrist_roll_q_indices
                            )
                            if not wrist_indices or not np.isfinite(previous_q[env]).all():
                                continue
                            wrist_idx = int(wrist_indices[0])
                            after_steps[env].append(abs(float(repaired_q[env, wrist_idx] - previous_q[env, wrist_idx])))
                        for env, steps in enumerate(after_steps):
                            if steps:
                                sparse_trace[env, 3] = float(math.degrees(max(steps)))

                        guard = max(0.0, float(self.ai_sapiens_sparse_wrist_repair_hand_residual_guard_m))
                        guard_failed = False
                        for key, before_error in before_errors.items():
                            after_error = after_errors.get(key)
                            if after_error is None:
                                continue
                            if float(after_error) - float(before_error) > guard:
                                guard_failed = True
                                break
                        max_after_step = max((max(steps) for steps in after_steps if steps), default=0.0)
                        max_before_step = max((max(steps) for steps in before_steps if steps), default=0.0)
                        step_worse = bool(max_after_step > max_before_step + 1e-7)
                        if guard_failed or step_worse:
                            wp.copy(joint_q, wp.array(current_before_repair, dtype=wp.float32))
                            for env, _side in triggered_pairs:
                                sparse_trace[env, 6] = 0.0
                                sparse_trace[env, 7] = 1.0
                                sparse_trace[env, 3] = sparse_trace[env, 2]
                        else:
                            for env, _side in triggered_pairs:
                                sparse_trace[env, 6] += 1.0
                                sparse_trace[env, 7] = 0.0

                        wrist_roll_nullspace_temporal_objective.set_weight(0.0)
                        if self.ai_sapiens_sparse_branch_resolve_enabled:
                            arm_branch_nullspace_temporal_objective.set_weight(0.0)

                if solver_trace_arrays is not None:
                    solver_trace_arrays["sparse_wrist_repair_trace"][frame] = sparse_trace
                    solver_trace_arrays["q_after_sparse_wrist_repair"][frame] = joint_q.numpy()
            elif solver_trace_arrays is not None:
                solver_trace_arrays["q_after_sparse_wrist_repair"][frame] = joint_q.numpy()

            if frame < num_frames_to_remove and self.warmup_neutral_q_indices:
                self._apply_neutral_q_indices(joint_q, default_joint_q, self.warmup_neutral_q_indices)

            if solver_trace_arrays is not None:
                solver_trace_arrays["q_after_warmup_neutral"][frame] = joint_q.numpy()

            data = None
            if self.post_processing_enabled:
                self.feet_stabilizer.reset_state(joint_q)

                for env in range(num_envs):
                    if frame > (len(self.input_targets[env])-1):
                        env_feet_tx[env] = np.asarray(self.input_targets[env][-1][self.feet_effector_indices])
                    else:
                        env_feet_tx[env] = np.asarray(self.input_targets[env][frame][self.feet_effector_indices])

                self.feet_stabilizer.solve(env_feet_tx)
                feet_state = self.feet_stabilizer.current_state()
                if solver_trace_arrays is not None:
                    solver_trace_arrays["q_after_feet_stabilizer"][frame] = feet_state.numpy()
                data = self.joint_limit_clamper.apply(feet_state).numpy()
            else:
                data = self.joint_limit_clamper.apply(joint_q).numpy()

            if solver_trace_arrays is not None:
                solver_trace_arrays["q_after_joint_limit"][frame] = data

            if self.ai_sapiens_output_joint_safety_margin_specs:
                data = self._apply_ai_sapiens_output_joint_safety_margin(data)

            if solver_trace_arrays is not None:
                solver_trace_arrays["q_after_output_safety_margin"][frame] = data

            if self.ai_sapiens_output_joint_step_limit_specs:
                previous_data = np.full_like(data, np.nan, dtype=np.float32)
                for env in range(num_envs):
                    if frame <= 0:
                        continue
                    if frame > (len(self.input_targets[env]) - 1):
                        continue
                    if (
                        self.reset_solver_state_at_output_start
                        and num_frames_to_remove > 0
                        and frame == num_frames_to_remove
                    ):
                        continue
                    previous = joint_q_data[env][frame - 1]
                    if previous is not None:
                        previous_data[env] = np.asarray(previous, dtype=np.float32)
                data = self._apply_ai_sapiens_output_joint_step_limit(data, previous_data)

            if solver_trace_arrays is not None:
                solver_trace_arrays["q_after_output_joint_step_limit"][frame] = data

            if self.ai_sapiens_root_orientation_step_limit_enabled and self.ai_sapiens_root_orientation_max_step_rad > 0.0:
                previous_data = np.full_like(data, np.nan, dtype=np.float32)
                for env in range(num_envs):
                    if frame <= 0:
                        continue
                    if frame > (len(self.input_targets[env]) - 1):
                        continue
                    if (
                        self.reset_solver_state_at_output_start
                        and num_frames_to_remove > 0
                        and frame == num_frames_to_remove
                    ):
                        continue
                    previous = joint_q_data[env][frame - 1]
                    if previous is not None:
                        previous_data[env] = np.asarray(previous, dtype=np.float32)
                data = self._apply_ai_sapiens_root_orientation_step_limit(data, previous_data)

            if solver_trace_arrays is not None:
                solver_trace_arrays["q_after_root_orientation_step"][frame] = data

            if self.ai_sapiens_arm_temporal_regularization_enabled and self.ai_sapiens_arm_temporal_q_indices:
                previous_data = np.full_like(data, np.nan, dtype=np.float32)
                previous2_data = np.full_like(data, np.nan, dtype=np.float32)
                for env in range(num_envs):
                    if frame <= 0:
                        continue
                    if frame > (len(self.input_targets[env]) - 1):
                        continue
                    if (
                        self.reset_solver_state_at_output_start
                        and num_frames_to_remove > 0
                        and frame == num_frames_to_remove
                    ):
                        continue
                    previous = joint_q_data[env][frame - 1]
                    if previous is not None:
                        previous_data[env] = np.asarray(previous, dtype=np.float32)
                    if frame > 1:
                        previous2 = joint_q_data[env][frame - 2]
                        if previous2 is not None:
                            previous2_data[env] = np.asarray(previous2, dtype=np.float32)
                data = self._apply_ai_sapiens_arm_temporal_regularization(
                    data,
                    previous_data,
                    previous2_data,
                )

            if solver_trace_arrays is not None:
                solver_trace_arrays["q_after_arm_temporal_regularization"][frame] = data

            if solver_trace_arrays is not None:
                solver_trace_arrays["q_after_postprocess"][frame] = data

            for env in range(num_envs):
                if frame > (len(self.input_targets[env])-1):
                    continue

                joint_q_data[env][frame] = data[env]
                if frame <= num_frames_to_remove:
                    self.last_internal_trace["warmup_qpos_by_env"][env].append({
                        "internal_frame": int(frame),
                        "source_frame": int(frame - num_frames_to_remove),
                        "qpos": np.asarray(data[env], dtype=np.float64).tolist(),
                    })

            #end_time = time.time()
            #print(f"Time taken for frame {frame}: {end_time - start_time} seconds")

        if solver_trace_arrays is not None:
            self.last_solver_stage_trace = {
                **solver_trace_arrays,
                "target_names": np.asarray(self.mapped_joints),
                "joint_names": np.asarray(self._qpos_schema()),
                "frame_index_internal": np.arange(self.max_frames, dtype=np.int32),
                "frame_index_output": np.arange(self.max_frames, dtype=np.int32) - int(num_frames_to_remove),
                "num_frames_to_remove": np.asarray(num_frames_to_remove, dtype=np.int32),
            }
        else:
            self.last_solver_stage_trace = None

        return [
            CSVAnimationBuffer.create_from_raw_data(joint_q_data[i][num_frames_to_remove:], self.input_sample_rates[i])
            for i in range(num_envs)]

    def get_last_internal_trace_for_env(self, env_index: int):
        if self.last_internal_trace is None:
            return None
        trace = {
            key: value
            for key, value in self.last_internal_trace.items()
            if key != "warmup_qpos_by_env"
        }
        traces = self.last_internal_trace.get("warmup_qpos_by_env", [])
        trace["warmup_qpos"] = traces[env_index] if env_index < len(traces) else []
        return trace

    def get_target_preprocess_summary_for_env(self, env_index: int):
        if env_index < len(self.input_target_preprocess_summaries):
            return self.input_target_preprocess_summaries[env_index]
        return None

    def get_solver_stage_trace_for_env(self, env_index: int):
        if self.last_solver_stage_trace is None:
            return None
        if env_index >= len(self.input_targets):
            return None
        frame_count = len(self.input_targets[env_index])
        trace = {}
        for key, value in self.last_solver_stage_trace.items():
            if key.startswith("q_"):
                trace[key] = np.asarray(value[:frame_count, env_index, :], dtype=np.float32)
            elif key in {"frame_index_internal", "frame_index_output"}:
                trace[key] = np.asarray(value[:frame_count], dtype=np.int32)
            elif key == "sparse_wrist_repair_trace":
                arr = np.asarray(value)
                if arr.ndim == 3:
                    trace[key] = np.asarray(arr[:frame_count, env_index, :], dtype=np.float32)
                else:
                    trace[key] = np.asarray(arr[:frame_count], dtype=np.float32)
            else:
                trace[key] = value
        if env_index < len(self.input_target_stage_traces):
            for key, value in self.input_target_stage_traces[env_index].items():
                trace[key] = np.asarray(value, dtype=np.float32)
        return trace

    def _build_model(self, num_envs: int):
        builder = newton.ModelBuilder()
        for _ in range(num_envs):
            builder.add_builder(self.robot_builder, xform=wp.transform_identity())

        builder.add_ground_plane()
        model = builder.finalize(requires_grad=True)

        return model

    def _qpos_schema(self):
        if self.target_type == pipeline_utils.TargetType.AI_SAPIENS:
            return [
                "root_translateX",
                "root_translateY",
                "root_translateZ",
                "root_quatX",
                "root_quatY",
                "root_quatZ",
                "root_quatW",
                *ai_sapiens_assets.AI_SAPIENS_JOINT_NAMES,
            ]
        return []

    def _apply_ai_sapiens_ik_joint_safety_margins_to_model(self):
        if (
            self.target_type != pipeline_utils.TargetType.AI_SAPIENS
            or not self.ai_sapiens_ik_joint_safety_margins_rad
            or self.ik_model is None
        ):
            return

        qpos_schema = self._qpos_schema()
        lower = self.ik_model.joint_limit_lower.numpy().astype(np.float32).copy()
        upper = self.ik_model.joint_limit_upper.numpy().astype(np.float32).copy()
        changed = 0
        for joint_name, margin_value in self.ai_sapiens_ik_joint_safety_margins_rad.items():
            if joint_name not in qpos_schema:
                print(f"[WARNING]: AI Sapiens IK safety-margin joint not found: {joint_name}")
                continue
            q_idx = int(qpos_schema.index(joint_name))
            if q_idx < 7:
                print(f"[WARNING]: AI Sapiens IK safety-margin ignores root coordinate: {joint_name}")
                continue
            dof_idx = q_idx - 1
            if dof_idx < 0 or dof_idx >= lower.shape[0]:
                print(f"[WARNING]: AI Sapiens IK safety-margin dof out of range: {joint_name}")
                continue
            margin = max(0.0, float(margin_value))
            half_range = 0.5 * float(upper[dof_idx] - lower[dof_idx])
            effective_margin = min(margin, max(0.0, half_range - 1e-6))
            if effective_margin <= 0.0:
                continue
            lower[dof_idx] += effective_margin
            upper[dof_idx] -= effective_margin
            changed += 1

        if changed <= 0:
            return
        device = self.ik_model.joint_limit_lower.device
        wp.copy(
            self.ik_model.joint_limit_lower,
            wp.array(lower, dtype=wp.float32, device=device),
        )
        wp.copy(
            self.ik_model.joint_limit_upper,
            wp.array(upper, dtype=wp.float32, device=device),
        )

    def _build_joint_q_indices(self, joint_names):
        if not joint_names:
            return []

        label_to_joint_idx = {
            newton_utils.get_name_from_label(label): idx
            for idx, label in enumerate(self.robot_builder.joint_label)
        }
        q_starts = self.ik_model.joint_q_start.numpy()
        q_indices = []
        for joint_name in joint_names:
            if joint_name not in label_to_joint_idx:
                print(f"[WARNING]: Warmup neutral joint not found in robot model: {joint_name}")
                continue
            joint_idx = label_to_joint_idx[joint_name]
            q_dim = int(q_starts[joint_idx + 1] - q_starts[joint_idx])
            if q_dim <= 0:
                continue
            q_start = int(q_starts[joint_idx])
            q_indices.extend(q_start + offset for offset in range(q_dim))

        return sorted(set(q_indices))

    def _joint_coord_mask_from_q_indices(self, q_indices):
        mask = np.zeros((self.ik_model.joint_coord_count,), dtype=np.float32)
        for q_idx in q_indices:
            q_idx = int(q_idx)
            if 0 <= q_idx < self.ik_model.joint_coord_count:
                mask[q_idx] = 1.0
        return mask

    def _sparse_hand_position_errors(self, model, state, joint_q, q_np, frame, env_side_pairs):
        if not env_side_pairs:
            return {}
        q_np = np.asarray(q_np, dtype=np.float32)
        q_flat = wp.array(q_np.reshape(-1), dtype=wp.float32)
        newton.eval_fk(model, q_flat, model.joint_qd, state)
        body_q = state.body_q.numpy()
        target_index_by_name = {
            str(name): idx for idx, name in enumerate(self.mapped_joints)
        }
        out = {}
        for env, side in env_side_pairs:
            target_name = "LeftHand" if side == "left" else "RightHand"
            target_idx = target_index_by_name.get(target_name)
            if target_idx is None or target_idx >= len(self.mapped_body_link_pos_data):
                continue
            if env < 0 or env >= len(self.input_targets):
                continue
            if frame < 0 or frame >= len(self.input_targets[env]):
                continue
            body_idx = self.mapped_body_link_pos_data[target_idx][0]
            body_flat_idx = int(env) * int(self.num_body_count) + int(body_idx)
            if body_flat_idx < 0 or body_flat_idx >= body_q.shape[0]:
                continue
            target_pos = np.asarray(self.input_targets[env][frame][target_idx][0:3], dtype=np.float64)
            fk_pos = np.asarray(body_q[body_flat_idx][0:3], dtype=np.float64)
            out[(int(env), str(side))] = float(np.linalg.norm(fk_pos - target_pos))
        return out

    def _build_ai_sapiens_output_joint_safety_margin_specs(self):
        if (
            self.target_type != pipeline_utils.TargetType.AI_SAPIENS
            or not self.ai_sapiens_output_joint_safety_margin_enabled
            or not self.ai_sapiens_output_joint_safety_margins_rad
            or self.joint_limit_clamper is None
        ):
            return []

        coord_lower = np.full(self.ik_model.joint_coord_count, -np.inf, dtype=np.float64)
        coord_upper = np.full(self.ik_model.joint_coord_count, np.inf, dtype=np.float64)
        dof_to_coord = self.joint_limit_clamper.dof_to_coord.numpy()
        limit_lower = self.ik_model.joint_limit_lower.numpy()
        limit_upper = self.ik_model.joint_limit_upper.numpy()
        for dof_idx, coord_idx in enumerate(dof_to_coord):
            coord_idx = int(coord_idx)
            if coord_idx < 0:
                continue
            coord_lower[coord_idx] = float(limit_lower[dof_idx])
            coord_upper[coord_idx] = float(limit_upper[dof_idx])

        qpos_schema = self._qpos_schema()
        specs = []
        for joint_name, margin_value in self.ai_sapiens_output_joint_safety_margins_rad.items():
            if joint_name not in qpos_schema:
                print(f"[WARNING]: AI Sapiens output safety margin joint not found: {joint_name}")
                continue
            margin = margin_value
            if isinstance(margin_value, dict):
                margin = margin_value.get("margin_rad", 0.0)
            margin = max(0.0, float(margin))
            if margin <= 0.0:
                continue
            q_idx = int(qpos_schema.index(joint_name))
            lower = float(coord_lower[q_idx])
            upper = float(coord_upper[q_idx])
            if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
                print(f"[WARNING]: AI Sapiens output safety margin has invalid limits for joint: {joint_name}")
                continue
            half_range = 0.5 * (upper - lower)
            effective_margin = min(margin, max(0.0, half_range - 1e-6))
            if effective_margin <= 0.0:
                continue
            specs.append({
                "joint": str(joint_name),
                "q_index": q_idx,
                "lower": lower,
                "upper": upper,
                "margin_rad": float(effective_margin),
            })
        return specs

    def _apply_ai_sapiens_output_joint_safety_margin(self, q_data):
        if not self.ai_sapiens_output_joint_safety_margin_specs:
            return q_data
        out = np.asarray(q_data, dtype=np.float32).copy()
        for spec in self.ai_sapiens_output_joint_safety_margin_specs:
            q_idx = int(spec["q_index"])
            lower = float(spec["lower"]) + float(spec["margin_rad"])
            upper = float(spec["upper"]) - float(spec["margin_rad"])
            out[:, q_idx] = np.clip(out[:, q_idx], lower, upper)
        return out

    def _build_ai_sapiens_output_joint_step_limit_specs(self):
        if (
            self.target_type != pipeline_utils.TargetType.AI_SAPIENS
            or not self.ai_sapiens_output_joint_step_limit_enabled
            or not self.ai_sapiens_output_joint_step_limits_rad
        ):
            return []

        qpos_schema = self._qpos_schema()
        root_quat_names = {
            "root_quatX",
            "root_quatY",
            "root_quatZ",
            "root_quatW",
        }
        specs = []
        for joint_name, limit_value in self.ai_sapiens_output_joint_step_limits_rad.items():
            if joint_name not in qpos_schema:
                print(f"[WARNING]: AI Sapiens output step-limit joint not found: {joint_name}")
                continue
            if joint_name in root_quat_names:
                print(
                    "[WARNING]: AI Sapiens output step-limit ignores root quaternion scalar "
                    f"component {joint_name}; use enable_ai_sapiens_root_orientation_step_limit instead."
                )
                continue
            limit = limit_value
            if isinstance(limit_value, dict):
                limit = limit_value.get("max_step_rad", 0.0)
            limit = max(0.0, float(limit))
            if limit <= 0.0:
                continue
            specs.append({
                "joint": str(joint_name),
                "q_index": int(qpos_schema.index(joint_name)),
                "max_step_rad": float(limit),
            })
        return specs

    def _apply_ai_sapiens_output_joint_step_limit(self, q_data, previous_q_data):
        if not self.ai_sapiens_output_joint_step_limit_specs:
            return q_data
        out = np.asarray(q_data, dtype=np.float32).copy()
        previous = np.asarray(previous_q_data, dtype=np.float32)
        for spec in self.ai_sapiens_output_joint_step_limit_specs:
            q_idx = int(spec["q_index"])
            max_step = float(spec["max_step_rad"])
            prev = previous[:, q_idx]
            valid = np.isfinite(prev)
            if not np.any(valid):
                continue
            delta = out[valid, q_idx] - prev[valid]
            out[valid, q_idx] = prev[valid] + np.clip(delta, -max_step, max_step)
        return out

    @staticmethod
    def _normalize_quat_xyzw(quat):
        quat = np.asarray(quat, dtype=np.float64)
        norm = float(np.linalg.norm(quat))
        if not np.isfinite(norm) or norm <= 1e-12:
            return None
        return quat / norm

    @staticmethod
    def _slerp_quat_xyzw(q0, q1, t):
        q0 = np.asarray(q0, dtype=np.float64)
        q1 = np.asarray(q1, dtype=np.float64)
        dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        if dot > 0.9995:
            out = q0 + float(t) * (q1 - q0)
            norm = float(np.linalg.norm(out))
            return out / norm if norm > 1e-12 else q0
        theta_0 = float(np.arccos(dot))
        sin_theta_0 = float(np.sin(theta_0))
        theta = theta_0 * float(t)
        s0 = float(np.cos(theta) - dot * np.sin(theta) / sin_theta_0)
        s1 = float(np.sin(theta) / sin_theta_0)
        return s0 * q0 + s1 * q1

    def _apply_ai_sapiens_root_orientation_step_limit(self, q_data, previous_q_data):
        if (
            not self.ai_sapiens_root_orientation_step_limit_enabled
            or self.ai_sapiens_root_orientation_max_step_rad <= 0.0
        ):
            return q_data
        out = np.asarray(q_data, dtype=np.float32).copy()
        previous = np.asarray(previous_q_data, dtype=np.float32)
        max_step = float(self.ai_sapiens_root_orientation_max_step_rad)
        quat_slice = slice(3, 7)
        for env in range(out.shape[0]):
            prev_q = self._normalize_quat_xyzw(previous[env, quat_slice])
            curr_q = self._normalize_quat_xyzw(out[env, quat_slice])
            if prev_q is None or curr_q is None:
                continue
            dot = float(np.clip(np.dot(prev_q, curr_q), -1.0, 1.0))
            if dot < 0.0:
                curr_q = -curr_q
                dot = -dot
            angle = 2.0 * float(np.arccos(np.clip(dot, -1.0, 1.0)))
            if angle <= max_step or angle <= 1e-12:
                out[env, quat_slice] = curr_q.astype(np.float32)
                continue
            limited = self._slerp_quat_xyzw(prev_q, curr_q, max_step / angle)
            out[env, quat_slice] = limited.astype(np.float32)
        return out

    def _apply_ai_sapiens_arm_temporal_regularization(self, q_data, previous_q_data, previous2_q_data):
        if (
            not self.ai_sapiens_arm_temporal_regularization_enabled
            or not self.ai_sapiens_arm_temporal_q_indices
        ):
            return q_data
        out = np.asarray(q_data, dtype=np.float32).copy()
        previous = np.asarray(previous_q_data, dtype=np.float32)
        previous2 = np.asarray(previous2_q_data, dtype=np.float32)
        q_indices = np.asarray(self.ai_sapiens_arm_temporal_q_indices, dtype=np.int32)
        step_weight = float(np.clip(self.ai_sapiens_arm_temporal_step_weight, 0.0, 1.0))
        accel_weight = float(np.clip(self.ai_sapiens_arm_temporal_acceleration_weight, 0.0, 1.0))
        max_correction = max(0.0, float(self.ai_sapiens_arm_temporal_max_correction_rad))
        if max_correction <= 0.0 or (step_weight <= 0.0 and accel_weight <= 0.0):
            return out

        for env in range(out.shape[0]):
            curr = out[env, q_indices].astype(np.float64)
            candidate = curr.copy()
            prev = previous[env, q_indices].astype(np.float64)
            prev_valid = np.isfinite(prev).all()
            if prev_valid and step_weight > 0.0:
                candidate -= step_weight * (curr - prev)
            prev2 = previous2[env, q_indices].astype(np.float64)
            if prev_valid and np.isfinite(prev2).all() and accel_weight > 0.0:
                accel = curr - (2.0 * prev) + prev2
                candidate -= accel_weight * accel
            correction = np.clip(candidate - curr, -max_correction, max_correction)
            out[env, q_indices] = (curr + correction).astype(np.float32)
        return out

    def _ai_sapiens_arm_temporal_regularization_summary(self):
        return {
            "enabled": bool(self.ai_sapiens_arm_temporal_regularization_enabled),
            "joint_names": [str(name) for name in self.ai_sapiens_arm_temporal_joint_names],
            "q_indices": [int(idx) for idx in self.ai_sapiens_arm_temporal_q_indices],
            "step_weight": float(self.ai_sapiens_arm_temporal_step_weight),
            "acceleration_weight": float(self.ai_sapiens_arm_temporal_acceleration_weight),
            "max_correction_rad": float(self.ai_sapiens_arm_temporal_max_correction_rad),
            "hand_residual_guard_active": False,
        }

    def _ai_sapiens_output_joint_safety_margin_summary(self):
        return {
            "enabled": bool(self.ai_sapiens_output_joint_safety_margin_enabled),
            "joint_count": int(len(self.ai_sapiens_output_joint_safety_margin_specs)),
            "joints": [
                {
                    "joint": spec["joint"],
                    "q_index": int(spec["q_index"]),
                    "lower": float(spec["lower"]),
                    "upper": float(spec["upper"]),
                    "margin_rad": float(spec["margin_rad"]),
                }
                for spec in self.ai_sapiens_output_joint_safety_margin_specs
            ],
        }

    def _apply_ai_sapiens_hand_hip_target_clearance(self, targets):
        summary = {
            "enabled": True,
            "mode": str(self.ai_sapiens_hand_hip_clearance_mode),
            "start_frame": 0,
            "skipped": False,
            "skip_reason": None,
            "detect_m": float(self.ai_sapiens_hand_hip_clearance_detect_m),
            "margin_m": float(self.ai_sapiens_hand_hip_clearance_margin_m),
            "gain": float(self.ai_sapiens_hand_hip_clearance_gain),
            "max_shift_m": float(self.ai_sapiens_hand_hip_clearance_max_shift_m),
            "smooth_window": int(self.ai_sapiens_hand_hip_clearance_smooth_window),
            "ramp_output_frames": int(self.ai_sapiens_hand_hip_clearance_ramp_output_frames),
            "hand_count": 0,
            "applied_frame_count": 0,
            "guarded_candidate_frame_count": 0,
            "guarded_downscale_frame_count": 0,
            "guarded_cancel_frame_count": 0,
            "sequence_guard_enabled": False,
            "sequence_guard_pass": True,
            "sequence_guard_rolled_back": False,
            "sequence_applied_ratio_per_side": 0.0,
            "sequence_same_gain_p50_m": float("nan"),
            "window_guard_enabled": False,
            "window_guard_size_frames": 0,
            "window_guard_stride_frames": 0,
            "window_guard_accepted_count": 0,
            "window_guard_rolled_back_count": 0,
            "sparse_corridor_enabled": bool(self.ai_sapiens_sparse_pelvis_local_corridor_enabled),
            "sparse_active_ratio_cap": float(self.ai_sapiens_sparse_corridor_active_ratio_cap),
            "sparse_active_cap_removed_frame_count": 0,
            "shift_p95_m": float("nan"),
            "shift_max_m": float("nan"),
            "min_distance_before_m": float("nan"),
            "min_distance_after_m": float("nan"),
            "hands": [],
        }
        corrected = np.asarray(targets, dtype=np.float32).copy()
        original_corrected = corrected.copy()
        start_frame, should_skip, skip_reason = self._projection_start_and_skip(corrected)
        summary["start_frame"] = int(start_frame)
        if should_skip:
            summary["skipped"] = True
            summary["skip_reason"] = skip_reason
            return corrected, summary

        target_to_idx = {name: idx for idx, name in enumerate(self.mapped_joints)}
        pelvis_idx = target_to_idx.get("Hips")
        chest_idx = target_to_idx.get("Chest")
        left_leg_idx = target_to_idx.get("LeftLeg")
        right_leg_idx = target_to_idx.get("RightLeg")
        hand_specs = [
            ("LeftHand", target_to_idx.get("LeftHand"), left_leg_idx, 1.0),
            ("RightHand", target_to_idx.get("RightHand"), right_leg_idx, -1.0),
        ]
        hand_specs = [
            spec for spec in hand_specs
            if spec[1] is not None
            and spec[2] is not None
            and pelvis_idx is not None
            and left_leg_idx is not None
            and right_leg_idx is not None
        ]
        summary["hand_count"] = int(len(hand_specs))
        if not hand_specs:
            summary["skipped"] = True
            summary["skip_reason"] = "missing_hand_pelvis_or_leg_targets"
            return corrected, summary
        if self.ai_sapiens_hand_hip_clearance_mode not in {
            "pelvis_local_same_side_hip_capsule",
            "guarded_pelvis_local_same_side_hip_capsule",
            "pelvis_local_radial_corridor",
            "guarded_pelvis_local_radial_corridor",
            "legacy_xy",
        }:
            summary["skipped"] = True
            summary["skip_reason"] = (
                f"unsupported_hand_hip_clearance_mode:{self.ai_sapiens_hand_hip_clearance_mode}"
            )
            return corrected, summary

        detect = max(0.0, float(self.ai_sapiens_hand_hip_clearance_detect_m))
        margin = max(0.0, float(self.ai_sapiens_hand_hip_clearance_margin_m))
        gain = max(0.0, float(self.ai_sapiens_hand_hip_clearance_gain))
        max_shift = max(0.0, float(self.ai_sapiens_hand_hip_clearance_max_shift_m))
        ramp_frames = max(0, int(self.ai_sapiens_hand_hip_clearance_ramp_output_frames))
        guarded_mode = self.ai_sapiens_hand_hip_clearance_mode in {
            "guarded_pelvis_local_same_side_hip_capsule",
            "guarded_pelvis_local_radial_corridor",
        }
        radial_corridor_mode = self.ai_sapiens_hand_hip_clearance_mode in {
            "pelvis_local_radial_corridor",
            "guarded_pelvis_local_radial_corridor",
        }
        cross_drop_guard = max(0.0, float(self.ai_sapiens_hand_hip_clearance_cross_side_drop_guard_m))
        min_same_gain = max(0.0, float(self.ai_sapiens_hand_hip_clearance_min_same_side_gain_m))
        seq_min_ratio = max(0.0, float(self.ai_sapiens_hand_hip_clearance_sequence_min_applied_ratio_per_side))
        seq_min_same_gain = max(0.0, float(self.ai_sapiens_hand_hip_clearance_sequence_min_same_gain_p50_m))
        sequence_guard_enabled = seq_min_ratio > 0.0 or seq_min_same_gain > 0.0
        summary["sequence_guard_enabled"] = bool(sequence_guard_enabled)
        window_guard_size = max(0, int(self.ai_sapiens_hand_hip_clearance_window_guard_size_frames))
        window_guard_stride = max(1, int(self.ai_sapiens_hand_hip_clearance_window_guard_stride_frames or window_guard_size or 1))
        window_min_ratio = max(0.0, float(self.ai_sapiens_hand_hip_clearance_window_min_applied_ratio_per_side))
        window_min_same_gain = max(0.0, float(self.ai_sapiens_hand_hip_clearance_window_min_same_gain_p50_m))
        window_guard_enabled = window_guard_size > 0 and (window_min_ratio > 0.0 or window_min_same_gain > 0.0)
        summary["window_guard_enabled"] = bool(window_guard_enabled)
        summary["window_guard_size_frames"] = int(window_guard_size)
        summary["window_guard_stride_frames"] = int(window_guard_stride if window_guard_enabled else 0)
        if detect <= 0.0 or gain <= 0.0 or max_shift <= 0.0:
            summary["skipped"] = True
            summary["skip_reason"] = "zero_detect_gain_or_max_shift"
            return corrected, summary

        all_shift_norms = []
        all_before_distances = []
        all_after_distances = []
        total_applied = 0
        total_guarded_candidates = 0
        total_guarded_downscaled = 0
        total_guarded_canceled = 0
        all_applied_same_gains = []
        per_frame_applied_counts = np.zeros((corrected.shape[0],), dtype=np.int32)
        per_frame_same_gains = [[] for _ in range(corrected.shape[0])]
        for hand_name, hand_idx, hip_idx, side_sign in hand_specs:
            opposite_hip_idx = right_leg_idx if hip_idx == left_leg_idx else left_leg_idx
            raw_shifts = np.zeros((corrected.shape[0], 3), dtype=np.float64)
            before_distances = []
            for frame in range(start_frame, corrected.shape[0]):
                hand = np.asarray(corrected[frame, hand_idx, 0:3], dtype=np.float64)
                pelvis = np.asarray(corrected[frame, pelvis_idx, 0:3], dtype=np.float64)
                left_leg = np.asarray(corrected[frame, left_leg_idx, 0:3], dtype=np.float64)
                right_leg = np.asarray(corrected[frame, right_leg_idx, 0:3], dtype=np.float64)
                hip = np.asarray(corrected[frame, hip_idx, 0:3], dtype=np.float64)
                lateral = self._normalize_np(
                    left_leg - right_leg,
                    fallback=np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
                )
                if chest_idx is not None:
                    up_raw = np.asarray(corrected[frame, chest_idx, 0:3], dtype=np.float64) - pelvis
                else:
                    up_raw = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
                up_raw = up_raw - np.dot(up_raw, lateral) * lateral
                up = self._normalize_np(
                    up_raw,
                    fallback=np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
                )
                forward = self._normalize_np(
                    np.cross(lateral, up),
                    fallback=np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
                )
                up = self._normalize_np(
                    np.cross(forward, lateral),
                    fallback=up,
                )
                segment = hip - pelvis
                seg_len_sq = float(np.dot(segment, segment))
                if seg_len_sq > 1e-12:
                    t = float(np.clip(np.dot(hand - pelvis, segment) / seg_len_sq, 0.0, 1.0))
                    closest = pelvis + segment * t
                else:
                    closest = hip
                away = hand - closest
                distance = float(np.linalg.norm(away))
                before_distances.append(distance)
                if distance >= detect:
                    continue
                outward = lateral * float(side_sign)
                if radial_corridor_mode:
                    direction = self._normalize_np(
                        away,
                        fallback=outward,
                    )
                else:
                    direction = self._normalize_np(
                        0.80 * outward + 0.20 * up,
                        fallback=outward,
                    )
                shift_mag = min(max_shift, gain * max(0.0, (detect + margin) - distance))
                raw_shifts[frame] = direction * shift_mag

            window = max(1, int(self.ai_sapiens_hand_hip_clearance_smooth_window))
            if window > 1:
                kernel = np.ones(window, dtype=np.float64) / float(window)
                smoothed = np.zeros_like(raw_shifts)
                for axis in range(3):
                    smoothed[:, axis] = np.convolve(raw_shifts[:, axis], kernel, mode="same")
                shifts = smoothed
            else:
                shifts = raw_shifts

            sparse_removed_count = 0
            sparse_cap = float(self.ai_sapiens_sparse_corridor_active_ratio_cap)
            if self.ai_sapiens_sparse_pelvis_local_corridor_enabled and 0.0 < sparse_cap < 1.0:
                frame_count_for_cap = max(0, corrected.shape[0] - int(start_frame))
                if frame_count_for_cap > 0:
                    norms_for_cap = np.linalg.norm(shifts[start_frame:], axis=1)
                    active_local = np.flatnonzero(norms_for_cap > 1e-12)
                    keep_count = int(np.ceil(float(frame_count_for_cap) * sparse_cap))
                    keep_count = max(1, min(int(active_local.size), keep_count))
                    if active_local.size > keep_count:
                        order = np.argsort(norms_for_cap[active_local])
                        keep_local = set(int(v) for v in active_local[order[-keep_count:]])
                        for local_idx in active_local:
                            if int(local_idx) not in keep_local:
                                shifts[start_frame + int(local_idx)] = 0.0
                                sparse_removed_count += 1

            shift_norms = []
            after_distances = []
            applied_same_gains = []
            guarded_candidate_count = 0
            guarded_downscale_count = 0
            guarded_cancel_count = 0
            for frame in range(start_frame, corrected.shape[0]):
                shift = shifts[frame]
                if ramp_frames > 0:
                    ramp = min(1.0, max(0.0, (frame - start_frame + 1) / float(ramp_frames)))
                    shift = shift * ramp
                shift_norm = float(np.linalg.norm(shift))
                if shift_norm > max_shift:
                    shift = shift / shift_norm * max_shift
                    shift_norm = max_shift
                if guarded_mode and shift_norm > 1e-12:
                    guarded_candidate_count += 1
                    hand_before = np.asarray(corrected[frame, hand_idx, 0:3], dtype=np.float64)
                    pelvis = np.asarray(corrected[frame, pelvis_idx, 0:3], dtype=np.float64)
                    same_hip = np.asarray(corrected[frame, hip_idx, 0:3], dtype=np.float64)
                    cross_hip = np.asarray(corrected[frame, opposite_hip_idx, 0:3], dtype=np.float64)
                    same_before = self._point_to_segment_distance_np(hand_before, pelvis, same_hip)
                    cross_before = self._point_to_segment_distance_np(hand_before, pelvis, cross_hip)
                    selected_shift = None
                    selected_scale = 0.0
                    for scale in (1.0, 0.75, 0.50, 0.25):
                        candidate_shift = shift * scale
                        hand_candidate = hand_before + candidate_shift
                        same_after = self._point_to_segment_distance_np(hand_candidate, pelvis, same_hip)
                        cross_after = self._point_to_segment_distance_np(hand_candidate, pelvis, cross_hip)
                        if (
                            same_after > same_before + min_same_gain
                            and cross_after >= cross_before - cross_drop_guard
                        ):
                            selected_shift = candidate_shift
                            selected_scale = scale
                            break
                    if selected_shift is None:
                        shift = np.zeros(3, dtype=np.float64)
                        shift_norm = 0.0
                        guarded_cancel_count += 1
                    else:
                        if selected_scale < 0.999:
                            guarded_downscale_count += 1
                        shift = selected_shift
                        shift_norm = float(np.linalg.norm(shift))
                if shift_norm > 1e-12:
                    hand_before_for_gain = np.asarray(corrected[frame, hand_idx, 0:3], dtype=np.float64)
                    pelvis_for_gain = np.asarray(corrected[frame, pelvis_idx, 0:3], dtype=np.float64)
                    hip_for_gain = np.asarray(corrected[frame, hip_idx, 0:3], dtype=np.float64)
                    same_before_for_gain = self._point_to_segment_distance_np(
                        hand_before_for_gain, pelvis_for_gain, hip_for_gain
                    )
                    same_after_for_gain = self._point_to_segment_distance_np(
                        hand_before_for_gain + shift, pelvis_for_gain, hip_for_gain
                    )
                    applied_same_gains.append(float(same_after_for_gain - same_before_for_gain))
                    per_frame_applied_counts[frame] += 1
                    per_frame_same_gains[frame].append(float(same_after_for_gain - same_before_for_gain))
                    corrected[frame, hand_idx, 0:3] = (
                        np.asarray(corrected[frame, hand_idx, 0:3], dtype=np.float64) + shift
                    ).astype(np.float32)
                    total_applied += 1
                hand_after = np.asarray(corrected[frame, hand_idx, 0:3], dtype=np.float64)
                pelvis = np.asarray(corrected[frame, pelvis_idx, 0:3], dtype=np.float64)
                hip = np.asarray(corrected[frame, hip_idx, 0:3], dtype=np.float64)
                segment = hip - pelvis
                seg_len_sq = float(np.dot(segment, segment))
                if seg_len_sq > 1e-12:
                    t = float(np.clip(np.dot(hand_after - pelvis, segment) / seg_len_sq, 0.0, 1.0))
                    closest = pelvis + segment * t
                else:
                    closest = hip
                after_distances.append(float(np.linalg.norm(hand_after - closest)))
                shift_norms.append(shift_norm)

            all_shift_norms.extend(shift_norms)
            all_before_distances.extend(before_distances)
            all_after_distances.extend(after_distances)
            all_applied_same_gains.extend(applied_same_gains)
            total_guarded_candidates += guarded_candidate_count
            total_guarded_downscaled += guarded_downscale_count
            total_guarded_canceled += guarded_cancel_count
            summary["sparse_active_cap_removed_frame_count"] += int(sparse_removed_count)
            summary["hands"].append({
                "hand": hand_name,
                "hip_anchor": self.mapped_joints[hip_idx],
                "applied_frame_count": int(np.count_nonzero(np.asarray(shift_norms) > 1e-12)),
                "sparse_active_cap_removed_frame_count": int(sparse_removed_count),
                "guarded_candidate_frame_count": int(guarded_candidate_count),
                "guarded_downscale_frame_count": int(guarded_downscale_count),
                "guarded_cancel_frame_count": int(guarded_cancel_count),
                "shift_p95_m": self._safe_percentile(shift_norms, 95),
                "shift_max_m": float(np.max(shift_norms)) if shift_norms else float("nan"),
                "applied_same_gain_p50_m": self._safe_percentile(applied_same_gains, 50),
                "min_distance_before_m": float(np.min(before_distances)) if before_distances else float("nan"),
                "min_distance_after_m": float(np.min(after_distances)) if after_distances else float("nan"),
            })

        if window_guard_enabled:
            accepted_windows = 0
            rolled_back_windows = 0
            hand_indices = [spec[1] for spec in hand_specs if spec[1] is not None]
            for w_start in range(start_frame, corrected.shape[0], window_guard_stride):
                w_end = min(corrected.shape[0], w_start + window_guard_size)
                if w_end <= w_start:
                    continue
                window_gains = [
                    gain_value
                    for frame_gains in per_frame_same_gains[w_start:w_end]
                    for gain_value in frame_gains
                ]
                denom = max(1.0, float((w_end - w_start) * max(1, len(hand_specs))))
                window_ratio = float(np.sum(per_frame_applied_counts[w_start:w_end])) / denom
                window_gain_p50 = self._safe_percentile(window_gains, 50) if window_gains else 0.0
                window_pass = (
                    window_ratio >= window_min_ratio
                    and (window_gains or window_min_same_gain <= 0.0)
                    and window_gain_p50 >= window_min_same_gain
                )
                if window_pass:
                    accepted_windows += 1
                else:
                    rolled_back_windows += 1
                    for hand_idx in hand_indices:
                        corrected[w_start:w_end, hand_idx, 0:3] = original_corrected[w_start:w_end, hand_idx, 0:3]
            summary["window_guard_accepted_count"] = int(accepted_windows)
            summary["window_guard_rolled_back_count"] = int(rolled_back_windows)
            total_applied = 0
            all_shift_norms = []
            all_applied_same_gains = []
            for frame in range(start_frame, corrected.shape[0]):
                pelvis = np.asarray(original_corrected[frame, pelvis_idx, 0:3], dtype=np.float64)
                for _hand_name, hand_idx, hip_idx, _side_sign in hand_specs:
                    before = np.asarray(original_corrected[frame, hand_idx, 0:3], dtype=np.float64)
                    after = np.asarray(corrected[frame, hand_idx, 0:3], dtype=np.float64)
                    shift_norm = float(np.linalg.norm(after - before))
                    all_shift_norms.append(shift_norm)
                    if shift_norm > 1e-12:
                        total_applied += 1
                        hip = np.asarray(original_corrected[frame, hip_idx, 0:3], dtype=np.float64)
                        same_before = self._point_to_segment_distance_np(before, pelvis, hip)
                        same_after = self._point_to_segment_distance_np(after, pelvis, hip)
                        all_applied_same_gains.append(float(same_after - same_before))

        frame_count = max(0, corrected.shape[0] - int(start_frame))
        denom = max(1.0, float(frame_count * max(1, len(hand_specs))))
        applied_ratio_per_side = float(total_applied) / denom
        same_gain_p50 = self._safe_percentile(all_applied_same_gains, 50)
        sequence_pass = True
        if sequence_guard_enabled:
            if applied_ratio_per_side < seq_min_ratio:
                sequence_pass = False
            if all_applied_same_gains and same_gain_p50 < seq_min_same_gain:
                sequence_pass = False
            if not all_applied_same_gains and seq_min_same_gain > 0.0:
                sequence_pass = False

        summary["applied_frame_count"] = int(total_applied)
        summary["guarded_candidate_frame_count"] = int(total_guarded_candidates)
        summary["guarded_downscale_frame_count"] = int(total_guarded_downscaled)
        summary["guarded_cancel_frame_count"] = int(total_guarded_canceled)
        summary["sequence_applied_ratio_per_side"] = float(applied_ratio_per_side)
        summary["sequence_same_gain_p50_m"] = float(same_gain_p50)
        summary["sequence_guard_pass"] = bool(sequence_pass)
        summary["shift_p95_m"] = self._safe_percentile(all_shift_norms, 95)
        summary["shift_max_m"] = float(np.max(all_shift_norms)) if all_shift_norms else float("nan")
        summary["min_distance_before_m"] = float(np.min(all_before_distances)) if all_before_distances else float("nan")
        summary["min_distance_after_m"] = float(np.min(all_after_distances)) if all_after_distances else float("nan")
        if sequence_guard_enabled and not sequence_pass:
            summary["sequence_guard_rolled_back"] = True
            summary["applied_frame_count_before_sequence_guard"] = int(total_applied)
            summary["applied_frame_count"] = 0
            summary["shift_p95_m_before_sequence_guard"] = summary["shift_p95_m"]
            summary["shift_max_m_before_sequence_guard"] = summary["shift_max_m"]
            summary["shift_p95_m"] = 0.0
            summary["shift_max_m"] = 0.0
            return original_corrected, summary
        return corrected, summary

    def _apply_ai_sapiens_window_endpoint_release(self, targets):
        summary = {
            "enabled": True,
            "start_frame": 0,
            "skipped": False,
            "skip_reason": None,
            "window_count": int(len(self.ai_sapiens_window_endpoint_release_windows)),
            "hand_budget_m": float(self.ai_sapiens_window_endpoint_release_hand_budget_m),
            "forearm_budget_m": float(self.ai_sapiens_window_endpoint_release_forearm_budget_m),
            "capsule_threshold_m": float(self.ai_sapiens_window_endpoint_release_capsule_threshold_m),
            "gain": float(self.ai_sapiens_window_endpoint_release_gain),
            "taper_frames": int(self.ai_sapiens_window_endpoint_release_taper_frames),
            "direction_mode": str(self.ai_sapiens_window_endpoint_release_direction_mode),
            "applied_frame_count": 0,
            "applied_hand_frame_count": 0,
            "applied_forearm_frame_count": 0,
            "guard_cancel_count": 0,
            "shift_hand_p95_m": float("nan"),
            "shift_forearm_p95_m": float("nan"),
            "windows": [],
        }
        corrected = np.asarray(targets, dtype=np.float32).copy()
        trace = np.full((corrected.shape[0], 2, 12), np.nan, dtype=np.float32)
        start_frame, should_skip, skip_reason = self._projection_start_and_skip(corrected)
        summary["start_frame"] = int(start_frame)
        if should_skip:
            summary["skipped"] = True
            summary["skip_reason"] = skip_reason
            return corrected, summary, trace
        if not self.ai_sapiens_window_endpoint_release_windows:
            summary["skipped"] = True
            summary["skip_reason"] = "no_windows"
            return corrected, summary, trace

        hand_budget = max(0.0, float(self.ai_sapiens_window_endpoint_release_hand_budget_m))
        forearm_budget = max(
            0.0,
            min(
                float(self.ai_sapiens_window_endpoint_release_forearm_budget_m),
                0.5 * hand_budget if hand_budget > 0.0 else float(self.ai_sapiens_window_endpoint_release_forearm_budget_m),
            ),
        )
        threshold = max(0.0, float(self.ai_sapiens_window_endpoint_release_capsule_threshold_m))
        gain = max(0.0, float(self.ai_sapiens_window_endpoint_release_gain))
        taper_frames = max(0, int(self.ai_sapiens_window_endpoint_release_taper_frames))
        if threshold <= 0.0 or gain <= 0.0 or max(hand_budget, forearm_budget) <= 0.0:
            summary["skipped"] = True
            summary["skip_reason"] = "zero_threshold_gain_or_budget"
            return corrected, summary, trace

        target_to_idx = {name: idx for idx, name in enumerate(self.mapped_joints)}
        pelvis_idx = target_to_idx.get("Hips")
        chest_idx = target_to_idx.get("Chest")
        left_leg_idx = target_to_idx.get("LeftLeg")
        right_leg_idx = target_to_idx.get("RightLeg")
        arm_specs = [
            ("left", "LeftHand", target_to_idx.get("LeftHand"), "LeftForeArm", target_to_idx.get("LeftForeArm"), left_leg_idx, 1.0, 0),
            ("right", "RightHand", target_to_idx.get("RightHand"), "RightForeArm", target_to_idx.get("RightForeArm"), right_leg_idx, -1.0, 1),
        ]
        arm_specs = [
            spec for spec in arm_specs
            if pelvis_idx is not None
            and left_leg_idx is not None
            and right_leg_idx is not None
            and spec[2] is not None
            and spec[4] is not None
            and spec[5] is not None
        ]
        if not arm_specs:
            summary["skipped"] = True
            summary["skip_reason"] = "missing_required_targets"
            return corrected, summary, trace

        windows_by_side = {"left": [], "right": []}
        for raw_window in self.ai_sapiens_window_endpoint_release_windows:
            if not isinstance(raw_window, dict):
                continue
            side = str(raw_window.get("side", "")).strip().lower()
            if side not in windows_by_side:
                continue
            try:
                start_out = int(float(raw_window.get("window_start", 0)))
                end_out = int(float(raw_window.get("window_end", start_out)))
            except Exception:
                continue
            if end_out < start_out:
                continue
            windows_by_side[side].append({
                "start_out": start_out,
                "end_out": end_out,
                "class": str(raw_window.get("window_feasibility_class", "")),
                "source": str(raw_window.get("source", "")),
            })

        all_hand_shifts = []
        all_forearm_shifts = []
        applied_frames = set()
        guard_cancel_count = 0

        def _taper(output_frame, start_out, end_out):
            if taper_frames <= 0:
                return 1.0
            left = (int(output_frame) - int(start_out) + 1) / float(taper_frames)
            right = (int(end_out) - int(output_frame) + 1) / float(taper_frames)
            return float(np.clip(min(left, right), 0.0, 1.0))

        for side, hand_name, hand_idx, forearm_name, forearm_idx, hip_idx, side_sign, chain_idx in arm_specs:
            opposite_hip_idx = right_leg_idx if hip_idx == left_leg_idx else left_leg_idx
            side_windows = windows_by_side.get(side, [])
            side_applied = 0
            for win in side_windows:
                start_out = int(win["start_out"])
                end_out = int(win["end_out"])
                internal_start = max(start_frame, start_frame + start_out)
                internal_end = min(corrected.shape[0] - 1, start_frame + end_out)
                if internal_end < internal_start:
                    continue
                for frame in range(internal_start, internal_end + 1):
                    output_frame = frame - start_frame
                    hand_before = np.asarray(corrected[frame, hand_idx, 0:3], dtype=np.float64)
                    forearm_before = np.asarray(corrected[frame, forearm_idx, 0:3], dtype=np.float64)
                    pelvis = np.asarray(corrected[frame, pelvis_idx, 0:3], dtype=np.float64)
                    same_hip = np.asarray(corrected[frame, hip_idx, 0:3], dtype=np.float64)
                    cross_hip = np.asarray(corrected[frame, opposite_hip_idx, 0:3], dtype=np.float64)
                    left_leg = np.asarray(corrected[frame, left_leg_idx, 0:3], dtype=np.float64)
                    right_leg = np.asarray(corrected[frame, right_leg_idx, 0:3], dtype=np.float64)
                    lateral = self._normalize_np(
                        left_leg - right_leg,
                        fallback=np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
                    )
                    outward = lateral * float(side_sign)
                    arm_closest, hip_closest = self._closest_points_between_segments_np(
                        forearm_before,
                        hand_before,
                        pelvis,
                        same_hip,
                    )
                    segment_release_dir = self._normalize_np(
                        arm_closest - hip_closest,
                        fallback=outward,
                    )
                    direction_mode = str(self.ai_sapiens_window_endpoint_release_direction_mode)
                    if direction_mode == "forearm_segment_closest_point":
                        hand_release_dir = segment_release_dir
                        forearm_release_dir = segment_release_dir
                    elif direction_mode == "hybrid_pelvis_segment":
                        hand_release_dir = outward
                        forearm_release_dir = segment_release_dir
                    else:
                        hand_release_dir = outward
                        forearm_release_dir = outward
                    if chest_idx is not None:
                        up_raw = np.asarray(corrected[frame, chest_idx, 0:3], dtype=np.float64) - pelvis
                        up_raw = up_raw - np.dot(up_raw, lateral) * lateral
                        _ = self._normalize_np(up_raw, fallback=np.asarray([0.0, 0.0, 1.0], dtype=np.float64))

                    hand_before_dist = self._point_to_segment_distance_np(hand_before, pelvis, same_hip)
                    forearm_mid_before = 0.5 * (forearm_before + hand_before)
                    forearm_before_dist = self._point_to_segment_distance_np(forearm_mid_before, pelvis, same_hip)
                    risk_distance = min(hand_before_dist, forearm_before_dist)
                    trace[frame, chain_idx, 0] = np.float32(output_frame)
                    trace[frame, chain_idx, 1] = np.float32(risk_distance)
                    trace[frame, chain_idx, 2] = np.float32(hand_before_dist)
                    trace[frame, chain_idx, 3] = np.float32(forearm_before_dist)
                    trace[frame, chain_idx, 10] = np.float32(hand_budget)
                    trace[frame, chain_idx, 11] = np.float32(forearm_budget)
                    if risk_distance >= threshold:
                        trace[frame, chain_idx, 8] = np.float32(0.0)
                        continue

                    taper = _taper(output_frame, start_out, end_out)
                    hand_shift_mag = min(hand_budget, gain * max(0.0, threshold - hand_before_dist)) * taper
                    forearm_shift_mag = min(forearm_budget, gain * max(0.0, threshold - forearm_before_dist)) * taper
                    hand_shift = hand_release_dir * hand_shift_mag
                    forearm_shift = forearm_release_dir * forearm_shift_mag

                    same_after = self._point_to_segment_distance_np(hand_before + hand_shift, pelvis, same_hip)
                    cross_before = self._point_to_segment_distance_np(hand_before, pelvis, cross_hip)
                    cross_after = self._point_to_segment_distance_np(hand_before + hand_shift, pelvis, cross_hip)
                    if (
                        np.linalg.norm(hand_shift) > 1e-12
                        and (same_after <= hand_before_dist + 1e-9 or cross_after < cross_before - 0.005)
                    ):
                        hand_shift = np.zeros(3, dtype=np.float64)
                        guard_cancel_count += 1

                    mid_after = 0.5 * (forearm_before + forearm_shift + hand_before + hand_shift)
                    forearm_after_dist = self._point_to_segment_distance_np(mid_after, pelvis, same_hip)
                    if np.linalg.norm(forearm_shift) > 1e-12 and forearm_after_dist <= forearm_before_dist + 1e-9:
                        forearm_shift = np.zeros(3, dtype=np.float64)
                        guard_cancel_count += 1

                    hand_shift_norm = float(np.linalg.norm(hand_shift))
                    forearm_shift_norm = float(np.linalg.norm(forearm_shift))
                    if hand_shift_norm > 1e-12:
                        corrected[frame, hand_idx, 0:3] = (hand_before + hand_shift).astype(np.float32)
                    if forearm_shift_norm > 1e-12:
                        corrected[frame, forearm_idx, 0:3] = (forearm_before + forearm_shift).astype(np.float32)
                    if max(hand_shift_norm, forearm_shift_norm) > 1e-12:
                        applied_frames.add(int(frame))
                        side_applied += 1
                    all_hand_shifts.append(hand_shift_norm)
                    all_forearm_shifts.append(forearm_shift_norm)
                    hand_after = np.asarray(corrected[frame, hand_idx, 0:3], dtype=np.float64)
                    forearm_after = np.asarray(corrected[frame, forearm_idx, 0:3], dtype=np.float64)
                    trace[frame, chain_idx, 4] = np.float32(hand_shift_norm)
                    trace[frame, chain_idx, 5] = np.float32(forearm_shift_norm)
                    trace[frame, chain_idx, 6] = np.float32(self._point_to_segment_distance_np(hand_after, pelvis, same_hip))
                    trace[frame, chain_idx, 7] = np.float32(self._point_to_segment_distance_np(0.5 * (forearm_after + hand_after), pelvis, same_hip))
                    trace[frame, chain_idx, 8] = np.float32(1.0 if max(hand_shift_norm, forearm_shift_norm) > 1e-12 else 0.0)
                    trace[frame, chain_idx, 9] = np.float32(taper)
            summary["windows"].append({
                "side": side,
                "hand": hand_name,
                "forearm": forearm_name,
                "window_count": int(len(side_windows)),
                "applied_frame_count": int(side_applied),
            })

        summary["applied_frame_count"] = int(len(applied_frames))
        summary["applied_hand_frame_count"] = int(np.count_nonzero(np.asarray(all_hand_shifts) > 1e-12)) if all_hand_shifts else 0
        summary["applied_forearm_frame_count"] = int(np.count_nonzero(np.asarray(all_forearm_shifts) > 1e-12)) if all_forearm_shifts else 0
        summary["guard_cancel_count"] = int(guard_cancel_count)
        summary["shift_hand_p95_m"] = self._safe_percentile(all_hand_shifts, 95)
        summary["shift_forearm_p95_m"] = self._safe_percentile(all_forearm_shifts, 95)
        return corrected, summary, trace

    def _apply_ai_sapiens_forearm_segment_corridor(self, targets):
        summary = {
            "enabled": True,
            "start_frame": 0,
            "skipped": False,
            "skip_reason": None,
            "threshold_m": float(self.ai_sapiens_forearm_segment_corridor_threshold_m),
            "gain": float(self.ai_sapiens_forearm_segment_corridor_gain),
            "max_shift_m": float(self.ai_sapiens_forearm_segment_corridor_max_shift_m),
            "smooth_window": int(self.ai_sapiens_forearm_segment_corridor_smooth_window),
            "active_ratio_cap": float(self.ai_sapiens_forearm_segment_corridor_active_ratio_cap),
            "arm_count": 0,
            "applied_frame_count": 0,
            "active_cap_removed_frame_count": 0,
            "guarded_cancel_frame_count": 0,
            "guarded_downscale_frame_count": 0,
            "shift_p95_m": float("nan"),
            "shift_max_m": float("nan"),
            "min_distance_before_m": float("nan"),
            "min_distance_after_m": float("nan"),
            "arms": [],
        }
        corrected = np.asarray(targets, dtype=np.float32).copy()
        start_frame, should_skip, skip_reason = self._projection_start_and_skip(corrected)
        summary["start_frame"] = int(start_frame)
        if should_skip:
            summary["skipped"] = True
            summary["skip_reason"] = skip_reason
            return corrected, summary

        target_to_idx = {name: idx for idx, name in enumerate(self.mapped_joints)}
        pelvis_idx = target_to_idx.get("Hips")
        chest_idx = target_to_idx.get("Chest")
        left_leg_idx = target_to_idx.get("LeftLeg")
        right_leg_idx = target_to_idx.get("RightLeg")
        arm_specs = [
            ("LeftForeArm", target_to_idx.get("LeftForeArm"), target_to_idx.get("LeftHand"), left_leg_idx, 1.0),
            ("RightForeArm", target_to_idx.get("RightForeArm"), target_to_idx.get("RightHand"), right_leg_idx, -1.0),
        ]
        arm_specs = [
            spec for spec in arm_specs
            if spec[1] is not None
            and spec[2] is not None
            and spec[3] is not None
            and pelvis_idx is not None
            and left_leg_idx is not None
            and right_leg_idx is not None
        ]
        summary["arm_count"] = int(len(arm_specs))
        if not arm_specs:
            summary["skipped"] = True
            summary["skip_reason"] = "missing_forearm_hand_pelvis_or_leg_targets"
            return corrected, summary

        threshold = max(0.0, float(self.ai_sapiens_forearm_segment_corridor_threshold_m))
        gain = max(0.0, float(self.ai_sapiens_forearm_segment_corridor_gain))
        max_shift = max(0.0, float(self.ai_sapiens_forearm_segment_corridor_max_shift_m))
        if threshold <= 0.0 or gain <= 0.0 or max_shift <= 0.0:
            summary["skipped"] = True
            summary["skip_reason"] = "zero_threshold_gain_or_max_shift"
            return corrected, summary

        all_shift_norms = []
        all_before_distances = []
        all_after_distances = []
        total_applied = 0
        total_removed = 0
        total_canceled = 0
        total_downscaled = 0

        for arm_name, forearm_idx, hand_idx, hip_idx, side_sign in arm_specs:
            opposite_hip_idx = right_leg_idx if hip_idx == left_leg_idx else left_leg_idx
            raw_shifts = np.zeros((corrected.shape[0], 3), dtype=np.float64)
            before_distances = []
            for frame in range(start_frame, corrected.shape[0]):
                forearm = np.asarray(corrected[frame, forearm_idx, 0:3], dtype=np.float64)
                hand = np.asarray(corrected[frame, hand_idx, 0:3], dtype=np.float64)
                midpoint = 0.5 * (forearm + hand)
                pelvis = np.asarray(corrected[frame, pelvis_idx, 0:3], dtype=np.float64)
                left_leg = np.asarray(corrected[frame, left_leg_idx, 0:3], dtype=np.float64)
                right_leg = np.asarray(corrected[frame, right_leg_idx, 0:3], dtype=np.float64)
                hip = np.asarray(corrected[frame, hip_idx, 0:3], dtype=np.float64)
                lateral = self._normalize_np(
                    left_leg - right_leg,
                    fallback=np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
                )
                if chest_idx is not None:
                    up_raw = np.asarray(corrected[frame, chest_idx, 0:3], dtype=np.float64) - pelvis
                else:
                    up_raw = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
                up_raw = up_raw - np.dot(up_raw, lateral) * lateral
                up = self._normalize_np(up_raw, fallback=np.asarray([0.0, 0.0, 1.0], dtype=np.float64))
                segment = hip - pelvis
                seg_len_sq = float(np.dot(segment, segment))
                if seg_len_sq > 1e-12:
                    t = float(np.clip(np.dot(midpoint - pelvis, segment) / seg_len_sq, 0.0, 1.0))
                    closest = pelvis + segment * t
                else:
                    closest = hip
                away = midpoint - closest
                distance = float(np.linalg.norm(away))
                before_distances.append(distance)
                if distance >= threshold:
                    continue
                outward = lateral * float(side_sign)
                direction = self._normalize_np(
                    0.85 * away + 0.15 * up,
                    fallback=outward,
                )
                shift_mag = min(max_shift, gain * max(0.0, threshold - distance))
                raw_shifts[frame] = direction * shift_mag

            window = max(1, int(self.ai_sapiens_forearm_segment_corridor_smooth_window))
            if window > 1:
                kernel = np.ones(window, dtype=np.float64) / float(window)
                shifts = np.zeros_like(raw_shifts)
                for axis in range(3):
                    shifts[:, axis] = np.convolve(raw_shifts[:, axis], kernel, mode="same")
            else:
                shifts = raw_shifts

            removed_count = 0
            cap = float(self.ai_sapiens_forearm_segment_corridor_active_ratio_cap)
            if 0.0 < cap < 1.0:
                frame_count_for_cap = max(0, corrected.shape[0] - int(start_frame))
                if frame_count_for_cap > 0:
                    norms_for_cap = np.linalg.norm(shifts[start_frame:], axis=1)
                    active_local = np.flatnonzero(norms_for_cap > 1e-12)
                    keep_count = int(np.ceil(float(frame_count_for_cap) * cap))
                    keep_count = max(1, min(int(active_local.size), keep_count))
                    if active_local.size > keep_count:
                        order = np.argsort(norms_for_cap[active_local])
                        keep_local = set(int(v) for v in active_local[order[-keep_count:]])
                        for local_idx in active_local:
                            if int(local_idx) not in keep_local:
                                shifts[start_frame + int(local_idx)] = 0.0
                                removed_count += 1

            shift_norms = []
            after_distances = []
            canceled_count = 0
            downscaled_count = 0
            applied_count = 0
            for frame in range(start_frame, corrected.shape[0]):
                shift = shifts[frame]
                shift_norm = float(np.linalg.norm(shift))
                if shift_norm > max_shift:
                    shift = shift / shift_norm * max_shift
                    shift_norm = max_shift
                if shift_norm > 1e-12:
                    forearm_before = np.asarray(corrected[frame, forearm_idx, 0:3], dtype=np.float64)
                    hand = np.asarray(corrected[frame, hand_idx, 0:3], dtype=np.float64)
                    pelvis = np.asarray(corrected[frame, pelvis_idx, 0:3], dtype=np.float64)
                    same_hip = np.asarray(corrected[frame, hip_idx, 0:3], dtype=np.float64)
                    cross_hip = np.asarray(corrected[frame, opposite_hip_idx, 0:3], dtype=np.float64)
                    mid_before = 0.5 * (forearm_before + hand)
                    same_before = self._point_to_segment_distance_np(mid_before, pelvis, same_hip)
                    cross_before = self._point_to_segment_distance_np(mid_before, pelvis, cross_hip)
                    selected_shift = None
                    selected_scale = 0.0
                    for scale in (1.0, 0.75, 0.50, 0.25):
                        candidate_shift = shift * scale
                        mid_after = 0.5 * (forearm_before + candidate_shift + hand)
                        same_after = self._point_to_segment_distance_np(mid_after, pelvis, same_hip)
                        cross_after = self._point_to_segment_distance_np(mid_after, pelvis, cross_hip)
                        if same_after > same_before + 1e-6 and cross_after >= cross_before - 0.005:
                            selected_shift = candidate_shift
                            selected_scale = scale
                            break
                    if selected_shift is None:
                        shift = np.zeros(3, dtype=np.float64)
                        shift_norm = 0.0
                        canceled_count += 1
                    else:
                        if selected_scale < 0.999:
                            downscaled_count += 1
                        shift = selected_shift
                        shift_norm = float(np.linalg.norm(shift))
                if shift_norm > 1e-12:
                    corrected[frame, forearm_idx, 0:3] = (
                        np.asarray(corrected[frame, forearm_idx, 0:3], dtype=np.float64) + shift
                    ).astype(np.float32)
                    applied_count += 1
                    total_applied += 1
                forearm_after = np.asarray(corrected[frame, forearm_idx, 0:3], dtype=np.float64)
                hand_after = np.asarray(corrected[frame, hand_idx, 0:3], dtype=np.float64)
                midpoint_after = 0.5 * (forearm_after + hand_after)
                pelvis = np.asarray(corrected[frame, pelvis_idx, 0:3], dtype=np.float64)
                hip = np.asarray(corrected[frame, hip_idx, 0:3], dtype=np.float64)
                after_distances.append(self._point_to_segment_distance_np(midpoint_after, pelvis, hip))
                shift_norms.append(shift_norm)

            total_removed += removed_count
            total_canceled += canceled_count
            total_downscaled += downscaled_count
            all_shift_norms.extend(shift_norms)
            all_before_distances.extend(before_distances)
            all_after_distances.extend(after_distances)
            summary["arms"].append({
                "forearm": arm_name,
                "hand": self.mapped_joints[hand_idx],
                "hip_anchor": self.mapped_joints[hip_idx],
                "applied_frame_count": int(applied_count),
                "active_cap_removed_frame_count": int(removed_count),
                "guarded_cancel_frame_count": int(canceled_count),
                "guarded_downscale_frame_count": int(downscaled_count),
                "shift_p95_m": self._safe_percentile(shift_norms, 95),
                "shift_max_m": float(np.max(shift_norms)) if shift_norms else float("nan"),
                "min_distance_before_m": float(np.min(before_distances)) if before_distances else float("nan"),
                "min_distance_after_m": float(np.min(after_distances)) if after_distances else float("nan"),
            })

        summary["applied_frame_count"] = int(total_applied)
        summary["active_cap_removed_frame_count"] = int(total_removed)
        summary["guarded_cancel_frame_count"] = int(total_canceled)
        summary["guarded_downscale_frame_count"] = int(total_downscaled)
        summary["shift_p95_m"] = self._safe_percentile(all_shift_norms, 95)
        summary["shift_max_m"] = float(np.max(all_shift_norms)) if all_shift_norms else float("nan")
        summary["min_distance_before_m"] = float(np.min(all_before_distances)) if all_before_distances else float("nan")
        summary["min_distance_after_m"] = float(np.min(all_after_distances)) if all_after_distances else float("nan")
        return corrected, summary

    @staticmethod
    def _apply_neutral_q_indices(joint_q, default_joint_q, q_indices):
        joint_q_np = joint_q.numpy()
        joint_q_np[:, q_indices] = default_joint_q[:, q_indices]
        wp.copy(joint_q, wp.array(joint_q_np, dtype=wp.float32))

    @staticmethod
    def _normalize_np(vec, fallback=None, eps=1e-9):
        vec = np.asarray(vec, dtype=np.float64)
        norm = float(np.linalg.norm(vec))
        if norm > eps:
            return vec / norm
        if fallback is not None:
            fallback = np.asarray(fallback, dtype=np.float64)
            fallback_norm = float(np.linalg.norm(fallback))
            if fallback_norm > eps:
                return fallback / fallback_norm
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)

    @staticmethod
    def _point_to_segment_distance_np(point, start, end):
        point = np.asarray(point, dtype=np.float64)
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        segment = end - start
        seg_len_sq = float(np.dot(segment, segment))
        if seg_len_sq > 1e-12:
            t = float(np.clip(np.dot(point - start, segment) / seg_len_sq, 0.0, 1.0))
            closest = start + segment * t
        else:
            closest = end
        return float(np.linalg.norm(point - closest))

    @staticmethod
    def _closest_points_between_segments_np(a0, a1, b0, b1):
        a0 = np.asarray(a0, dtype=np.float64)
        a1 = np.asarray(a1, dtype=np.float64)
        b0 = np.asarray(b0, dtype=np.float64)
        b1 = np.asarray(b1, dtype=np.float64)

        def _point_segment(point, start, end):
            seg = end - start
            denom = float(np.dot(seg, seg))
            if denom <= 1e-12:
                return start
            t = float(np.clip(np.dot(point - start, seg) / denom, 0.0, 1.0))
            return start + t * seg

        candidates = []
        for point in (a0, a1):
            closest = _point_segment(point, b0, b1)
            candidates.append((point, closest))
        for point in (b0, b1):
            closest = _point_segment(point, a0, a1)
            candidates.append((closest, point))
        return min(
            candidates,
            key=lambda pair: float(np.dot(pair[0] - pair[1], pair[0] - pair[1])),
        )

    @staticmethod
    def _normalize_quat_np_xyzw(quat, fallback=None, eps=1e-9):
        quat = np.asarray(quat, dtype=np.float64)
        norm = float(np.linalg.norm(quat))
        if norm > eps:
            return quat / norm
        if fallback is not None:
            fallback = np.asarray(fallback, dtype=np.float64)
            fallback_norm = float(np.linalg.norm(fallback))
            if fallback_norm > eps:
                return fallback / fallback_norm
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    @staticmethod
    def _quat_multiply_np_xyzw(left, right):
        lx, ly, lz, lw = np.asarray(left, dtype=np.float64)
        rx, ry, rz, rw = np.asarray(right, dtype=np.float64)
        return np.asarray([
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ], dtype=np.float64)

    @staticmethod
    def _quat_conjugate_np_xyzw(quat):
        quat = np.asarray(quat, dtype=np.float64)
        return np.asarray([-quat[0], -quat[1], -quat[2], quat[3]], dtype=np.float64)

    def _quat_rotate_np_xyzw(self, quat, vec):
        quat = self._normalize_quat_np_xyzw(quat)
        vec_quat = np.asarray([vec[0], vec[1], vec[2], 0.0], dtype=np.float64)
        rotated = self._quat_multiply_np_xyzw(
            self._quat_multiply_np_xyzw(quat, vec_quat),
            self._quat_conjugate_np_xyzw(quat),
        )
        return rotated[0:3]

    @staticmethod
    def _quat_from_matrix_np_xyzw(matrix):
        matrix = np.asarray(matrix, dtype=np.float64)
        trace = float(np.trace(matrix))
        if trace > 0.0:
            s = float(np.sqrt(trace + 1.0) * 2.0)
            qw = 0.25 * s
            qx = (matrix[2, 1] - matrix[1, 2]) / s
            qy = (matrix[0, 2] - matrix[2, 0]) / s
            qz = (matrix[1, 0] - matrix[0, 1]) / s
        elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
            s = float(np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0)
            qw = (matrix[2, 1] - matrix[1, 2]) / s
            qx = 0.25 * s
            qy = (matrix[0, 1] + matrix[1, 0]) / s
            qz = (matrix[0, 2] + matrix[2, 0]) / s
        elif matrix[1, 1] > matrix[2, 2]:
            s = float(np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0)
            qw = (matrix[0, 2] - matrix[2, 0]) / s
            qx = (matrix[0, 1] + matrix[1, 0]) / s
            qy = 0.25 * s
            qz = (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = float(np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0)
            qw = (matrix[1, 0] - matrix[0, 1]) / s
            qx = (matrix[0, 2] + matrix[2, 0]) / s
            qy = (matrix[1, 2] + matrix[2, 1]) / s
            qz = 0.25 * s
        quat = np.asarray([qx, qy, qz, qw], dtype=np.float64)
        norm = float(np.linalg.norm(quat))
        return quat / norm if norm > 1e-12 else np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    def _quat_align_x_preserve_up_np_xyzw(self, forward, up_candidate, fallback_quat):
        x_axis = self._normalize_np(forward, fallback=[1.0, 0.0, 0.0])
        up_candidate = np.asarray(up_candidate, dtype=np.float64)
        z_axis = up_candidate - np.dot(up_candidate, x_axis) * x_axis
        if float(np.linalg.norm(z_axis)) <= 1e-9:
            world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
            z_axis = world_up - np.dot(world_up, x_axis) * x_axis
        if float(np.linalg.norm(z_axis)) <= 1e-9:
            fallback_quat = self._normalize_quat_np_xyzw(fallback_quat)
            z_axis = self._quat_rotate_np_xyzw(fallback_quat, [0.0, 0.0, 1.0])
            z_axis = z_axis - np.dot(z_axis, x_axis) * x_axis
        z_axis = self._normalize_np(z_axis, fallback=[0.0, 0.0, 1.0])
        y_axis = self._normalize_np(np.cross(z_axis, x_axis), fallback=[0.0, 1.0, 0.0])
        z_axis = self._normalize_np(np.cross(x_axis, y_axis), fallback=z_axis)
        matrix = np.column_stack([x_axis, y_axis, z_axis])
        return self._quat_from_matrix_np_xyzw(matrix)

    def _quat_from_two_vectors_np_xyzw(self, source_vec, target_vec):
        source_vec = self._normalize_np(source_vec, fallback=[0.0, 0.0, 1.0])
        target_vec = self._normalize_np(target_vec, fallback=source_vec)
        dot = float(np.clip(np.dot(source_vec, target_vec), -1.0, 1.0))
        if dot > 1.0 - 1.0e-8:
            return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        if dot < -1.0 + 1.0e-8:
            axis = np.cross(source_vec, np.asarray([1.0, 0.0, 0.0], dtype=np.float64))
            if float(np.linalg.norm(axis)) < 1.0e-8:
                axis = np.cross(source_vec, np.asarray([0.0, 1.0, 0.0], dtype=np.float64))
            axis = self._normalize_np(axis, fallback=[0.0, 0.0, 1.0])
            return np.asarray([axis[0], axis[1], axis[2], 0.0], dtype=np.float64)
        axis = np.cross(source_vec, target_vec)
        return self._normalize_quat_np_xyzw(
            np.asarray([axis[0], axis[1], axis[2], 1.0 + dot], dtype=np.float64)
        )

    def _quat_yaw_align_x_preserve_tilt_np_xyzw(self, forward, fallback_quat):
        """Yaw-rotate a quaternion so its local +X follows a world-XY direction."""
        fallback_quat = self._normalize_quat_np_xyzw(fallback_quat)
        source_xy = np.asarray([forward[0], forward[1], 0.0], dtype=np.float64)
        source_norm = float(np.linalg.norm(source_xy))
        if source_norm <= 1e-9:
            return fallback_quat
        source_xy /= source_norm

        current_forward = self._quat_rotate_np_xyzw(fallback_quat, [1.0, 0.0, 0.0])
        current_xy = np.asarray([current_forward[0], current_forward[1], 0.0], dtype=np.float64)
        current_norm = float(np.linalg.norm(current_xy))
        if current_norm <= 1e-9:
            return fallback_quat
        current_xy /= current_norm

        cross_z = float(current_xy[0] * source_xy[1] - current_xy[1] * source_xy[0])
        dot = float(np.clip(np.dot(current_xy, source_xy), -1.0, 1.0))
        half_angle = 0.5 * float(np.arctan2(cross_z, dot))
        yaw_delta = np.asarray([0.0, 0.0, np.sin(half_angle), np.cos(half_angle)], dtype=np.float64)
        return self._normalize_quat_np_xyzw(
            self._quat_multiply_np_xyzw(yaw_delta, fallback_quat),
            fallback=fallback_quat,
        )

    @staticmethod
    def _angle_between_np_deg(a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        an = float(np.linalg.norm(a))
        bn = float(np.linalg.norm(b))
        if an <= 1e-12 or bn <= 1e-12:
            return 0.0
        dot = float(np.clip(np.dot(a / an, b / bn), -1.0, 1.0))
        return float(np.degrees(np.arccos(dot)))

    def _build_ai_sapiens_limb_projection_data(self, chains):
        state = self.ik_model.state()
        newton.eval_fk(self.ik_model, self.ik_model.joint_q, self.ik_model.joint_qd, state)
        body_q = state.body_q.numpy()
        body_names = [
            newton_utils.get_name_from_label(label)
            for label in self.robot_builder.body_label
        ]
        body_name_to_idx = {name: idx for idx, name in enumerate(body_names)}
        target_name_to_idx = {name: idx for idx, name in enumerate(self.mapped_joints)}

        projection_data = []
        for root_name, mid_name, end_name, root_body, mid_body, end_body in chains:
            if not all(name in target_name_to_idx for name in (root_name, mid_name, end_name)):
                continue
            if not all(name in body_name_to_idx for name in (root_body, mid_body, end_body)):
                continue
            root_qpos0 = np.asarray(body_q[body_name_to_idx[root_body]][0:3], dtype=np.float64)
            mid_qpos0 = np.asarray(body_q[body_name_to_idx[mid_body]][0:3], dtype=np.float64)
            end_qpos0 = np.asarray(body_q[body_name_to_idx[end_body]][0:3], dtype=np.float64)
            upper_len = float(np.linalg.norm(mid_qpos0 - root_qpos0))
            lower_len = float(np.linalg.norm(end_qpos0 - mid_qpos0))
            if upper_len <= 1e-9 or lower_len <= 1e-9:
                continue
            fallback_end_dir = self._normalize_np(end_qpos0 - root_qpos0)
            fallback_bend_dir = (
                mid_qpos0
                - root_qpos0
                - np.dot(mid_qpos0 - root_qpos0, fallback_end_dir) * fallback_end_dir
            )
            fallback_bend_dir = self._normalize_np(fallback_bend_dir, fallback=[0.0, 0.0, -1.0])
            projection_data.append({
                "root_name": root_name,
                "mid_name": mid_name,
                "end_name": end_name,
                "root_idx": target_name_to_idx[root_name],
                "mid_idx": target_name_to_idx[mid_name],
                "end_idx": target_name_to_idx[end_name],
                "upper_len": float(upper_len),
                "lower_len": float(lower_len),
                "fallback_end_dir": fallback_end_dir,
                "fallback_bend_dir": fallback_bend_dir,
            })
        return projection_data

    def _build_ai_sapiens_source_segment_direction_data(self):
        chains = self.ai_sapiens_source_segment_direction_chains or []
        return self._build_ai_sapiens_segment_direction_data(chains)

    def _build_ai_sapiens_arm_segment_direction_data(self):
        chains = self.ai_sapiens_arm_segment_direction_chains
        if chains is None:
            chains = [
                {
                    "name": "left_arm_segment_direction",
                    "targets": ["LeftArm", "LeftForeArm", "LeftHand"],
                    "source_joints": ["LeftArm", "LeftForeArm", "LeftHand"],
                    "bodies": [
                        "left_shoulder_roll_g1_proxy",
                        "left_elbow_g1_proxy",
                        "left_wrist_yaw_g1_proxy",
                    ],
                    "anchor": "root",
                },
                {
                    "name": "right_arm_segment_direction",
                    "targets": ["RightArm", "RightForeArm", "RightHand"],
                    "source_joints": ["RightArm", "RightForeArm", "RightHand"],
                    "bodies": [
                        "right_shoulder_roll_g1_proxy",
                        "right_elbow_g1_proxy",
                        "right_wrist_yaw_g1_proxy",
                    ],
                    "anchor": "root",
                },
            ]
        return self._build_ai_sapiens_segment_direction_data(chains)

    def _build_ai_sapiens_segment_direction_data(self, chains):
        chains = chains or []
        state = self.ik_model.state()
        newton.eval_fk(self.ik_model, self.ik_model.joint_q, self.ik_model.joint_qd, state)
        body_q = state.body_q.numpy()
        body_names = [
            newton_utils.get_name_from_label(label)
            for label in self.robot_builder.body_label
        ]
        body_name_to_idx = {name: idx for idx, name in enumerate(body_names)}
        target_name_to_idx = {name: idx for idx, name in enumerate(self.mapped_joints)}

        data = []
        for chain in chains:
            targets = [str(name) for name in chain.get("targets", [])]
            bodies = [str(name) for name in chain.get("bodies", [])]
            source_names = [
                str(name)
                for name in chain.get("source_joints", targets)
            ]
            if len(targets) != 3 or len(bodies) != 3 or len(source_names) != 3:
                continue
            if not all(name in target_name_to_idx for name in targets):
                continue
            if not all(name in body_name_to_idx for name in bodies):
                continue
            source_indices = [
                self.human_robot_scaler.skeleton.joint_index(name)
                for name in source_names
            ]
            if any(idx < 0 for idx in source_indices):
                continue

            root_qpos0 = np.asarray(body_q[body_name_to_idx[bodies[0]]][0:3], dtype=np.float64)
            mid_qpos0 = np.asarray(body_q[body_name_to_idx[bodies[1]]][0:3], dtype=np.float64)
            end_qpos0 = np.asarray(body_q[body_name_to_idx[bodies[2]]][0:3], dtype=np.float64)
            upper_vec = mid_qpos0 - root_qpos0
            lower_vec = end_qpos0 - mid_qpos0
            upper_len = float(np.linalg.norm(upper_vec))
            lower_len = float(np.linalg.norm(lower_vec))
            if upper_len <= 1e-9 or lower_len <= 1e-9:
                continue

            data.append({
                "name": str(chain.get("name", "->".join(targets))),
                "targets": targets,
                "bodies": bodies,
                "source_joints": source_names,
                "target_indices": [int(target_name_to_idx[name]) for name in targets],
                "source_indices": [int(idx) for idx in source_indices],
                "anchor": str(chain.get("anchor", "root")),
                "upper_len": upper_len,
                "lower_len": lower_len,
                "fallback_upper_dir": self._normalize_np(upper_vec),
                "fallback_lower_dir": self._normalize_np(lower_vec),
            })
        return data

    def _source_positions_for_segment_direction(self, buffer, root_tx, direction_data):
        if not direction_data:
            return None
        needed_indices = sorted({
            idx
            for chain in direction_data
            for idx in chain["source_indices"]
        })
        source_positions = np.zeros(
            (buffer.num_frames, self.human_robot_scaler.skeleton.num_joints, 3),
            dtype=np.float64,
        )
        for frame in range(buffer.num_frames):
            global_tx = buffer.compute_global_transforms(frame, root_tx)
            for joint_idx in needed_indices:
                source_positions[frame, joint_idx] = np.asarray(
                    global_tx[joint_idx][0:3],
                    dtype=np.float64,
                )
        return source_positions

    def _source_positions_for_source_segment_direction(self, buffer, root_tx):
        return self._source_positions_for_segment_direction(
            buffer,
            root_tx,
            self.ai_sapiens_source_segment_direction_data,
        )

    def _source_positions_for_arm_segment_direction(self, buffer, root_tx):
        return self._source_positions_for_segment_direction(
            buffer,
            root_tx,
            self.ai_sapiens_arm_segment_direction_data,
        )

    def _apply_ai_sapiens_source_body_frame_preservation(self, targets, buffer, root_tx):
        summary = {
            "enabled": bool(self.ai_sapiens_source_body_frame_preservation_enabled),
            "corrected_frame_count": 0,
            "mean_chest_shift_m": 0.0,
            "max_chest_shift_m": 0.0,
            "blend": float(self.ai_sapiens_source_body_frame_preserve_blend),
            "upper_follow_blend": float(self.ai_sapiens_source_body_frame_upper_follow_blend),
            "arm_follow_blend": float(self.ai_sapiens_source_body_frame_arm_follow_blend),
            "max_chest_shift_m_config": float(self.ai_sapiens_source_body_frame_max_chest_shift_m),
            "min_source_horizontal_m": float(self.ai_sapiens_source_body_frame_min_source_horizontal_m),
            "shift_chest": bool(self.ai_sapiens_source_body_frame_shift_chest),
            "rotation_blend": float(self.ai_sapiens_source_body_frame_rotation_blend),
            "rotate_hips": bool(self.ai_sapiens_source_body_frame_rotate_hips),
            "rotate_chest": bool(self.ai_sapiens_source_body_frame_rotate_chest),
            "rotate_upper": bool(self.ai_sapiens_source_body_frame_rotate_upper),
        }
        trace = np.full((targets.shape[0], 10), np.nan, dtype=np.float32)
        if not self.ai_sapiens_source_body_frame_preservation_enabled:
            return targets, summary, trace

        target_to_idx = {name: idx for idx, name in enumerate(self.mapped_joints)}
        if "Hips" not in target_to_idx or "Chest" not in target_to_idx:
            summary["reason"] = "missing_hips_or_chest_target"
            return targets, summary, trace

        source_hips_idx = self.human_robot_scaler.skeleton.joint_index("Hips")
        source_chest_idx = self.human_robot_scaler.skeleton.joint_index("Chest")
        if source_hips_idx < 0 or source_chest_idx < 0:
            summary["reason"] = "missing_hips_or_chest_source"
            return targets, summary, trace

        hips_idx = target_to_idx["Hips"]
        chest_idx = target_to_idx["Chest"]
        blend = float(np.clip(self.ai_sapiens_source_body_frame_preserve_blend, 0.0, 1.0))
        upper_follow_blend = float(np.clip(self.ai_sapiens_source_body_frame_upper_follow_blend, 0.0, 1.0))
        arm_follow_blend = float(np.clip(self.ai_sapiens_source_body_frame_arm_follow_blend, 0.0, 1.0))
        max_shift = max(0.0, float(self.ai_sapiens_source_body_frame_max_chest_shift_m))
        min_source_horizontal = max(0.0, float(self.ai_sapiens_source_body_frame_min_source_horizontal_m))
        rotation_blend = float(np.clip(self.ai_sapiens_source_body_frame_rotation_blend, 0.0, 1.0))
        corrected = 0
        shifts: list[float] = []
        for frame in range(targets.shape[0]):
            global_tx = buffer.compute_global_transforms(frame, root_tx)
            source_hips = np.asarray(global_tx[source_hips_idx][0:3], dtype=np.float64)
            source_chest = np.asarray(global_tx[source_chest_idx][0:3], dtype=np.float64)
            source_vec = source_chest - source_hips
            source_len = float(np.linalg.norm(source_vec))
            source_horizontal = float(np.linalg.norm(source_vec[:2]))

            hips = np.asarray(targets[frame, hips_idx, :3], dtype=np.float64)
            chest = np.asarray(targets[frame, chest_idx, :3], dtype=np.float64)
            target_vec = chest - hips
            target_len = float(np.linalg.norm(target_vec))
            target_horizontal = float(np.linalg.norm(target_vec[:2]))
            target_vertical = float(target_vec[2])

            trace[frame, 0] = source_horizontal
            trace[frame, 1] = target_horizontal
            trace[frame, 3] = float(source_vec[2])
            trace[frame, 4] = target_vertical
            trace[frame, 7] = upper_follow_blend
            trace[frame, 8] = arm_follow_blend
            trace[frame, 9] = 0.0

            if (
                source_len <= 1.0e-9
                or target_len <= 1.0e-9
                or blend <= 0.0
                or source_horizontal < min_source_horizontal
            ):
                trace[frame, 2] = target_horizontal
                trace[frame, 5] = target_vertical
                trace[frame, 6] = 0.0
                continue

            desired_vec = source_vec / source_len * target_len
            rotation_delta = None
            frame_blend = blend
            if rotation_blend > 0.0:
                rotation_delta = self._quat_from_two_vectors_np_xyzw(target_vec, desired_vec)
                rotation_delta = self._slerp_quat_xyzw(
                    np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
                    rotation_delta,
                    rotation_blend,
                )
            delta = (hips + desired_vec - chest) * frame_blend
            shift_norm = float(np.linalg.norm(delta))
            if max_shift > 0.0 and shift_norm > max_shift:
                delta *= max_shift / max(shift_norm, 1.0e-9)
                shift_norm = max_shift
            if shift_norm <= 1.0e-9:
                trace[frame, 2] = target_horizontal
                trace[frame, 5] = target_vertical
                trace[frame, 6] = 0.0
                continue

            applied_chest_shift = 0.0
            if self.ai_sapiens_source_body_frame_shift_chest:
                targets[frame, chest_idx, :3] = (chest + delta).astype(np.float32)
                applied_chest_shift = shift_norm
            if rotation_delta is not None:
                if self.ai_sapiens_source_body_frame_rotate_hips:
                    hips_quat = self._normalize_quat_np_xyzw(targets[frame, hips_idx, 3:7])
                    targets[frame, hips_idx, 3:7] = self._quat_multiply_np_xyzw(
                        rotation_delta,
                        hips_quat,
                    ).astype(np.float32)
                if self.ai_sapiens_source_body_frame_rotate_chest:
                    chest_quat = self._normalize_quat_np_xyzw(targets[frame, chest_idx, 3:7])
                    targets[frame, chest_idx, 3:7] = self._quat_multiply_np_xyzw(
                        rotation_delta,
                        chest_quat,
                    ).astype(np.float32)
            for name in ("LeftArm", "RightArm"):
                idx = target_to_idx.get(name)
                if idx is not None and arm_follow_blend > 0.0:
                    targets[frame, idx, :3] = (
                        np.asarray(targets[frame, idx, :3], dtype=np.float64)
                        + delta * arm_follow_blend
                    ).astype(np.float32)
                if idx is not None and rotation_delta is not None and self.ai_sapiens_source_body_frame_rotate_upper:
                    quat = self._normalize_quat_np_xyzw(targets[frame, idx, 3:7])
                    targets[frame, idx, 3:7] = self._quat_multiply_np_xyzw(
                        rotation_delta,
                        quat,
                    ).astype(np.float32)
            for name in ("LeftForeArm", "LeftHand", "RightForeArm", "RightHand"):
                idx = target_to_idx.get(name)
                if idx is not None and upper_follow_blend > 0.0:
                    targets[frame, idx, :3] = (
                        np.asarray(targets[frame, idx, :3], dtype=np.float64)
                        + delta * upper_follow_blend
                    ).astype(np.float32)
                if idx is not None and rotation_delta is not None and self.ai_sapiens_source_body_frame_rotate_upper:
                    quat = self._normalize_quat_np_xyzw(targets[frame, idx, 3:7])
                    targets[frame, idx, 3:7] = self._quat_multiply_np_xyzw(
                        rotation_delta,
                        quat,
                    ).astype(np.float32)

            after_vec = np.asarray(targets[frame, chest_idx, :3], dtype=np.float64) - hips
            trace[frame, 2] = float(np.linalg.norm(after_vec[:2]))
            trace[frame, 5] = float(after_vec[2])
            trace[frame, 6] = applied_chest_shift
            trace[frame, 9] = 1.0
            corrected += 1
            shifts.append(applied_chest_shift)

        summary["corrected_frame_count"] = int(corrected)
        if shifts:
            summary["mean_chest_shift_m"] = float(np.mean(shifts))
            summary["max_chest_shift_m"] = float(np.max(shifts))
        return targets, summary, trace

    def _apply_ai_sapiens_source_whole_body_pose(self, targets, buffer, root_tx):
        summary = {
            "enabled": bool(self.ai_sapiens_source_whole_body_pose_enabled),
            "corrected_frame_count": 0,
            "mean_rotation_angle_deg": 0.0,
            "max_rotation_angle_deg": 0.0,
            "max_target_shift_m": 0.0,
            "rotation_blend": float(self.ai_sapiens_source_whole_body_pose_rotation_blend),
            "min_source_horizontal_m": float(self.ai_sapiens_source_whole_body_pose_min_source_horizontal_m),
            "max_target_shift_m_config": float(self.ai_sapiens_source_whole_body_pose_max_target_shift_m),
            "target_names": list(self.ai_sapiens_source_whole_body_pose_targets),
            "rotate_orientations": bool(self.ai_sapiens_source_whole_body_pose_rotate_orientations),
            "preserve_foot_z": bool(self.ai_sapiens_source_whole_body_pose_preserve_foot_z),
        }
        trace = np.full((targets.shape[0], 12), np.nan, dtype=np.float32)
        if not self.ai_sapiens_source_whole_body_pose_enabled:
            return targets, summary, trace

        target_to_idx = {name: idx for idx, name in enumerate(self.mapped_joints)}
        if "Hips" not in target_to_idx or "Chest" not in target_to_idx:
            summary["reason"] = "missing_hips_or_chest_target"
            return targets, summary, trace

        source_hips_idx = self.human_robot_scaler.skeleton.joint_index("Hips")
        source_chest_idx = self.human_robot_scaler.skeleton.joint_index("Chest")
        if source_hips_idx < 0 or source_chest_idx < 0:
            summary["reason"] = "missing_hips_or_chest_source"
            return targets, summary, trace

        hips_idx = target_to_idx["Hips"]
        chest_idx = target_to_idx["Chest"]
        rot_blend = float(np.clip(self.ai_sapiens_source_whole_body_pose_rotation_blend, 0.0, 1.0))
        min_source_horizontal = max(0.0, float(self.ai_sapiens_source_whole_body_pose_min_source_horizontal_m))
        max_target_shift = max(0.0, float(self.ai_sapiens_source_whole_body_pose_max_target_shift_m))
        target_indices = [
            target_to_idx[name]
            for name in self.ai_sapiens_source_whole_body_pose_targets
            if name in target_to_idx
        ]
        if not target_indices or rot_blend <= 0.0:
            summary["reason"] = "no_targets_or_zero_blend"
            return targets, summary, trace

        rotate_orientations = bool(self.ai_sapiens_source_whole_body_pose_rotate_orientations)
        preserve_foot_z = bool(self.ai_sapiens_source_whole_body_pose_preserve_foot_z)
        foot_indices = {
            idx
            for name, idx in target_to_idx.items()
            if name in {"LeftFoot", "RightFoot", "LeftToe", "RightToe", "LeftToeBase", "RightToeBase"}
        }
        corrected = 0
        angles_deg: list[float] = []
        shift_norms: list[float] = []
        for frame in range(targets.shape[0]):
            global_tx = buffer.compute_global_transforms(frame, root_tx)
            source_hips = np.asarray(global_tx[source_hips_idx][0:3], dtype=np.float64)
            source_chest = np.asarray(global_tx[source_chest_idx][0:3], dtype=np.float64)
            source_vec = source_chest - source_hips
            source_len = float(np.linalg.norm(source_vec))
            source_horizontal = float(np.linalg.norm(source_vec[:2]))

            hips = np.asarray(targets[frame, hips_idx, :3], dtype=np.float64)
            chest = np.asarray(targets[frame, chest_idx, :3], dtype=np.float64)
            target_vec = chest - hips
            target_len = float(np.linalg.norm(target_vec))
            target_horizontal = float(np.linalg.norm(target_vec[:2]))
            trace[frame, 0] = source_horizontal
            trace[frame, 1] = target_horizontal
            trace[frame, 9] = 0.0

            if (
                source_len <= 1.0e-9
                or target_len <= 1.0e-9
                or source_horizontal < min_source_horizontal
            ):
                trace[frame, 2] = target_horizontal
                trace[frame, 3] = 0.0
                trace[frame, 4] = 0.0
                continue

            desired_vec = source_vec / source_len * target_len
            rotation_delta = self._quat_from_two_vectors_np_xyzw(target_vec, desired_vec)
            rotation_delta = self._slerp_quat_xyzw(
                np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
                rotation_delta,
                rot_blend,
            )
            rotation_delta = self._normalize_quat_np_xyzw(rotation_delta)
            rot_w = float(np.clip(rotation_delta[3], -1.0, 1.0))
            angle_deg = float(np.rad2deg(2.0 * np.arccos(abs(rot_w))))
            if angle_deg <= 1.0e-9:
                trace[frame, 2] = target_horizontal
                trace[frame, 3] = 0.0
                trace[frame, 4] = 0.0
                continue

            frame_max_shift = 0.0
            for idx in target_indices:
                before = np.asarray(targets[frame, idx, :3], dtype=np.float64)
                relative = before - hips
                rotated = hips + self._quat_rotate_np_xyzw(rotation_delta, relative)
                if preserve_foot_z and idx in foot_indices:
                    rotated[2] = before[2]
                shift = rotated - before
                shift_norm = float(np.linalg.norm(shift))
                if max_target_shift > 0.0 and shift_norm > max_target_shift:
                    shift *= max_target_shift / max(shift_norm, 1.0e-9)
                    shift_norm = max_target_shift
                    rotated = before + shift
                targets[frame, idx, :3] = rotated.astype(np.float32)
                frame_max_shift = max(frame_max_shift, shift_norm)
                if rotate_orientations:
                    quat = self._normalize_quat_np_xyzw(targets[frame, idx, 3:7])
                    targets[frame, idx, 3:7] = self._quat_multiply_np_xyzw(
                        rotation_delta,
                        quat,
                    ).astype(np.float32)

            after_vec = np.asarray(targets[frame, chest_idx, :3], dtype=np.float64) - hips
            trace[frame, 2] = float(np.linalg.norm(after_vec[:2]))
            trace[frame, 3] = float(angle_deg)
            trace[frame, 4] = float(frame_max_shift)
            trace[frame, 5] = float(len(target_indices))
            trace[frame, 9] = 1.0
            corrected += 1
            angles_deg.append(angle_deg)
            shift_norms.append(frame_max_shift)

        summary["corrected_frame_count"] = int(corrected)
        if angles_deg:
            summary["mean_rotation_angle_deg"] = float(np.mean(angles_deg))
            summary["max_rotation_angle_deg"] = float(np.max(angles_deg))
        if shift_norms:
            summary["max_target_shift_m"] = float(np.max(shift_norms))
        return targets, summary, trace

    def _apply_ai_sapiens_source_body_chain_preservation(self, targets, buffer, root_tx):
        summary = {
            "enabled": bool(self.ai_sapiens_source_body_chain_preservation_enabled),
            "corrected_frame_count": 0,
            "mean_shift_m": 0.0,
            "max_shift_m": 0.0,
            "blend": float(self.ai_sapiens_source_body_chain_blend),
            "vertical_blend": float(self.ai_sapiens_source_body_chain_vertical_blend),
            "max_shift_m_config": float(self.ai_sapiens_source_body_chain_max_shift_m),
            "min_source_horizontal_m": float(self.ai_sapiens_source_body_chain_min_source_horizontal_m),
            "target_names": list(self.ai_sapiens_source_body_chain_target_names),
        }
        trace = np.full((targets.shape[0], 12), np.nan, dtype=np.float32)
        if not self.ai_sapiens_source_body_chain_preservation_enabled:
            return targets, summary, trace

        target_to_idx = {name: idx for idx, name in enumerate(self.mapped_joints)}
        if "Hips" not in target_to_idx or "Chest" not in target_to_idx:
            summary["reason"] = "missing_hips_or_chest_target"
            return targets, summary, trace

        source_hips_idx = self.human_robot_scaler.skeleton.joint_index("Hips")
        source_chest_idx = self.human_robot_scaler.skeleton.joint_index("Chest")
        if source_hips_idx < 0 or source_chest_idx < 0:
            summary["reason"] = "missing_hips_or_chest_source"
            return targets, summary, trace

        target_names = [
            name
            for name in self.ai_sapiens_source_body_chain_target_names
            if name in target_to_idx and self.human_robot_scaler.skeleton.joint_index(name) >= 0
        ]
        if not target_names:
            summary["reason"] = "missing_chain_targets"
            return targets, summary, trace

        trace_slots = {
            "Chest": 5,
            "LeftArm": 6,
            "LeftForeArm": 7,
            "LeftHand": 8,
            "RightArm": 9,
            "RightForeArm": 10,
            "RightHand": 11,
        }
        hips_idx = target_to_idx["Hips"]
        chest_idx = target_to_idx["Chest"]
        blend = float(np.clip(self.ai_sapiens_source_body_chain_blend, 0.0, 1.0))
        vertical_blend = float(np.clip(self.ai_sapiens_source_body_chain_vertical_blend, 0.0, 1.0))
        max_shift = max(0.0, float(self.ai_sapiens_source_body_chain_max_shift_m))
        min_source_horizontal = max(0.0, float(self.ai_sapiens_source_body_chain_min_source_horizontal_m))

        corrected = 0
        shifts: list[float] = []
        world_z = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        for frame in range(targets.shape[0]):
            global_tx = buffer.compute_global_transforms(frame, root_tx)
            source_hips = np.asarray(global_tx[source_hips_idx][0:3], dtype=np.float64)
            source_chest = np.asarray(global_tx[source_chest_idx][0:3], dtype=np.float64)
            source_torso = source_chest - source_hips
            source_horizontal_vec = np.asarray(
                [source_torso[0], source_torso[1], 0.0],
                dtype=np.float64,
            )
            source_horizontal = float(np.linalg.norm(source_horizontal_vec))

            hips = np.asarray(targets[frame, hips_idx, :3], dtype=np.float64)
            chest = np.asarray(targets[frame, chest_idx, :3], dtype=np.float64)
            raw_torso = chest - hips
            raw_horizontal = float(np.linalg.norm(raw_torso[:2]))
            trace[frame, 0] = source_horizontal
            trace[frame, 1] = raw_horizontal
            trace[frame, 2] = 0.0
            trace[frame, 3] = 0.0
            trace[frame, 4] = 0.0

            if source_horizontal < min_source_horizontal or source_horizontal <= 1.0e-9 or blend <= 0.0:
                continue

            x_axis = source_horizontal_vec / source_horizontal
            y_axis = np.cross(world_z, x_axis)
            y_norm = float(np.linalg.norm(y_axis))
            if y_norm <= 1.0e-9:
                continue
            y_axis /= y_norm

            frame_shifts: list[float] = []
            for name in target_names:
                source_idx = self.human_robot_scaler.skeleton.joint_index(name)
                target_idx = target_to_idx[name]
                source_pos = np.asarray(global_tx[source_idx][0:3], dtype=np.float64)
                source_vec = source_pos - source_hips
                source_len = float(np.linalg.norm(source_vec))
                raw_vec = np.asarray(targets[frame, target_idx, :3], dtype=np.float64) - hips
                raw_len = float(np.linalg.norm(raw_vec))
                if source_len <= 1.0e-9 or raw_len <= 1.0e-9:
                    continue

                scale = raw_len / source_len
                local_x = float(np.dot(source_vec, x_axis))
                local_y = float(np.dot(source_vec, y_axis))
                local_z = float(source_vec[2])
                desired_vec = (
                    x_axis * (local_x * scale)
                    + y_axis * (local_y * scale)
                    + world_z * (local_z * scale * vertical_blend + raw_vec[2] * (1.0 - vertical_blend))
                )
                desired = hips + desired_vec
                current = np.asarray(targets[frame, target_idx, :3], dtype=np.float64)
                delta = (desired - current) * blend
                shift = float(np.linalg.norm(delta))
                if max_shift > 0.0 and shift > max_shift:
                    delta *= max_shift / max(shift, 1.0e-9)
                    shift = max_shift
                if shift <= 1.0e-9:
                    continue
                targets[frame, target_idx, :3] = (current + delta).astype(np.float32)
                frame_shifts.append(shift)
                shifts.append(shift)
                slot = trace_slots.get(name)
                if slot is not None:
                    trace[frame, slot] = shift

            if frame_shifts:
                corrected += 1
                trace[frame, 2] = 1.0
                trace[frame, 3] = float(np.mean(frame_shifts))
                trace[frame, 4] = float(np.max(frame_shifts))

        summary["corrected_frame_count"] = int(corrected)
        if shifts:
            summary["mean_shift_m"] = float(np.mean(shifts))
            summary["max_shift_m"] = float(np.max(shifts))
        return targets, summary, trace

    def _build_ai_sapiens_chest_anchored_arm_root_data(self):
        chains = self.ai_sapiens_chest_anchored_arm_root_chains
        if not chains:
            chains = [
                {
                    "name": "left_arm_chest_anchor",
                    "chest_target": "Chest",
                    "targets": ["LeftArm", "LeftForeArm", "LeftHand"],
                    "source_joints": ["LeftArm", "LeftForeArm", "LeftHand"],
                    "bodies": [
                        "chest_g1_torso_proxy",
                        "left_shoulder_roll_g1_proxy",
                        "left_elbow_g1_proxy",
                        "left_wrist_yaw_g1_proxy",
                    ],
                },
                {
                    "name": "right_arm_chest_anchor",
                    "chest_target": "Chest",
                    "targets": ["RightArm", "RightForeArm", "RightHand"],
                    "source_joints": ["RightArm", "RightForeArm", "RightHand"],
                    "bodies": [
                        "chest_g1_torso_proxy",
                        "right_shoulder_roll_g1_proxy",
                        "right_elbow_g1_proxy",
                        "right_wrist_yaw_g1_proxy",
                    ],
                },
            ]

        state = self.ik_model.state()
        newton.eval_fk(self.ik_model, self.ik_model.joint_q, self.ik_model.joint_qd, state)
        body_q = state.body_q.numpy()
        body_names = [
            newton_utils.get_name_from_label(label)
            for label in self.robot_builder.body_label
        ]
        body_name_to_idx = {name: idx for idx, name in enumerate(body_names)}
        target_name_to_idx = {name: idx for idx, name in enumerate(self.mapped_joints)}

        data = []
        for chain in chains:
            chest_target = str(chain.get("chest_target", "Chest"))
            targets = [str(name) for name in chain.get("targets", [])]
            source_names = [
                str(name)
                for name in chain.get("source_joints", targets)
            ]
            bodies = [str(name) for name in chain.get("bodies", [])]
            if len(targets) != 3 or len(source_names) != 3 or len(bodies) != 4:
                continue
            if chest_target not in target_name_to_idx:
                continue
            if not all(name in target_name_to_idx for name in targets):
                continue
            if not all(name in body_name_to_idx for name in bodies):
                continue
            source_indices = [
                self.human_robot_scaler.skeleton.joint_index(name)
                for name in source_names
            ]
            if any(idx < 0 for idx in source_indices):
                continue

            chest_body, root_body, mid_body, end_body = bodies
            chest_qpos0 = np.asarray(body_q[body_name_to_idx[chest_body]], dtype=np.float64)
            root_qpos0 = np.asarray(body_q[body_name_to_idx[root_body]], dtype=np.float64)
            mid_qpos0 = np.asarray(body_q[body_name_to_idx[mid_body]], dtype=np.float64)
            end_qpos0 = np.asarray(body_q[body_name_to_idx[end_body]], dtype=np.float64)

            upper_vec = mid_qpos0[0:3] - root_qpos0[0:3]
            lower_vec = end_qpos0[0:3] - mid_qpos0[0:3]
            upper_len = float(np.linalg.norm(upper_vec))
            lower_len = float(np.linalg.norm(lower_vec))
            if upper_len <= 1e-9 or lower_len <= 1e-9:
                continue

            chest_q0 = self._normalize_quat_np_xyzw(chest_qpos0[3:7])
            chest_to_root_local = self._quat_rotate_np_xyzw(
                self._quat_conjugate_np_xyzw(chest_q0),
                root_qpos0[0:3] - chest_qpos0[0:3],
            )

            data.append({
                "name": str(chain.get("name", "->".join(targets))),
                "chest_target": chest_target,
                "targets": targets,
                "source_joints": source_names,
                "bodies": bodies,
                "chest_idx": int(target_name_to_idx[chest_target]),
                "target_indices": [int(target_name_to_idx[name]) for name in targets],
                "source_indices": [int(idx) for idx in source_indices],
                "chest_to_root_local": chest_to_root_local,
                "upper_len": upper_len,
                "lower_len": lower_len,
                "fallback_upper_dir": self._normalize_np(upper_vec),
                "fallback_lower_dir": self._normalize_np(lower_vec),
            })
        return data

    def _source_positions_for_chest_anchored_arm_root(self, buffer, root_tx):
        if not self.ai_sapiens_chest_anchored_arm_root_data:
            return None
        needed_indices = sorted({
            idx
            for chain in self.ai_sapiens_chest_anchored_arm_root_data
            for idx in chain["source_indices"]
        })
        source_positions = np.zeros(
            (buffer.num_frames, self.human_robot_scaler.skeleton.num_joints, 3),
            dtype=np.float64,
        )
        for frame in range(buffer.num_frames):
            global_tx = buffer.compute_global_transforms(frame, root_tx)
            for joint_idx in needed_indices:
                source_positions[frame, joint_idx] = np.asarray(
                    global_tx[joint_idx][0:3],
                    dtype=np.float64,
                )
        return source_positions

    def _apply_ai_sapiens_chest_anchored_arm_root_correction(self, targets, buffer, root_tx):
        summary = {
            "enabled": bool(self.ai_sapiens_chest_anchored_arm_root_enabled),
            "chain_count": int(len(self.ai_sapiens_chest_anchored_arm_root_data)),
            "corrected_chain_count": 0,
            "start_frame": 0,
            "skipped": False,
            "skip_reason": None,
            "mode": "chest_target_pose_plus_qpos0_chest_to_shoulder_offset_with_source_segment_directions",
            "blend": float(np.clip(self.ai_sapiens_chest_anchored_arm_root_blend, 0.0, 1.0)),
            "chains": [],
        }
        if not self.ai_sapiens_chest_anchored_arm_root_data:
            summary["skipped"] = True
            summary["skip_reason"] = "no_valid_chains"
            return targets, summary

        corrected = np.asarray(targets, dtype=np.float32).copy()
        raw_targets = np.asarray(targets, dtype=np.float64)
        source_positions = self._source_positions_for_chest_anchored_arm_root(buffer, root_tx)
        if source_positions is None:
            summary["skipped"] = True
            summary["skip_reason"] = "source_positions_unavailable"
            return corrected, summary

        start_frame = 0
        if self.ai_sapiens_chest_anchored_arm_root_start_after_warmup:
            start_frame = int(self.num_initialization_frames + self.num_stabilization_frames)
        summary["start_frame"] = int(start_frame)
        output_frame_count = max(0, int(corrected.shape[0]) - start_frame)
        if (
            self.ai_sapiens_chest_anchored_arm_root_skip_single_output_frame
            and output_frame_count <= 1
        ):
            summary["skipped"] = True
            summary["skip_reason"] = "single_output_frame_static_gate_guard"
            return corrected, summary

        blend = float(np.clip(self.ai_sapiens_chest_anchored_arm_root_blend, 0.0, 1.0))
        corrected_chain_count = 0
        for chain in self.ai_sapiens_chest_anchored_arm_root_data:
            chest_idx = int(chain["chest_idx"])
            root_idx, mid_idx, end_idx = chain["target_indices"]
            src_root_idx, src_mid_idx, src_end_idx = chain["source_indices"]
            upper_len = float(chain["upper_len"])
            lower_len = float(chain["lower_len"])
            chest_to_root_local = np.asarray(chain["chest_to_root_local"], dtype=np.float64)
            fallback_upper = np.asarray(chain["fallback_upper_dir"], dtype=np.float64)
            fallback_lower = np.asarray(chain["fallback_lower_dir"], dtype=np.float64)

            root_delta_values = []
            mid_delta_values = []
            end_delta_values = []
            upper_angle_before_values = []
            lower_angle_before_values = []
            upper_angle_after_values = []
            lower_angle_after_values = []
            root_distance_to_chest_values = []
            for frame in range(start_frame, corrected.shape[0]):
                chest_pos = np.asarray(corrected[frame, chest_idx, 0:3], dtype=np.float64)
                chest_quat = self._normalize_quat_np_xyzw(corrected[frame, chest_idx, 3:7])
                anchored_root = chest_pos + self._quat_rotate_np_xyzw(
                    chest_quat,
                    chest_to_root_local,
                )

                src_root = source_positions[frame, src_root_idx]
                src_mid = source_positions[frame, src_mid_idx]
                src_end = source_positions[frame, src_end_idx]
                upper_dir = self._normalize_np(src_mid - src_root, fallback_upper)
                lower_dir = self._normalize_np(src_end - src_mid, fallback_lower)

                old_root = np.asarray(corrected[frame, root_idx, 0:3], dtype=np.float64)
                old_mid = np.asarray(corrected[frame, mid_idx, 0:3], dtype=np.float64)
                old_end = np.asarray(corrected[frame, end_idx, 0:3], dtype=np.float64)
                old_upper = self._normalize_np(old_mid - old_root, fallback_upper)
                old_lower = self._normalize_np(old_end - old_mid, fallback_lower)

                root = old_root + (anchored_root - old_root) * blend
                mid = root + upper_dir * upper_len
                end = mid + lower_dir * lower_len

                new_upper = self._normalize_np(mid - root, fallback_upper)
                new_lower = self._normalize_np(end - mid, fallback_lower)
                upper_angle_before_values.append(
                    float(np.degrees(np.arccos(np.clip(np.dot(old_upper, upper_dir), -1.0, 1.0))))
                )
                lower_angle_before_values.append(
                    float(np.degrees(np.arccos(np.clip(np.dot(old_lower, lower_dir), -1.0, 1.0))))
                )
                upper_angle_after_values.append(
                    float(np.degrees(np.arccos(np.clip(np.dot(new_upper, upper_dir), -1.0, 1.0))))
                )
                lower_angle_after_values.append(
                    float(np.degrees(np.arccos(np.clip(np.dot(new_lower, lower_dir), -1.0, 1.0))))
                )

                corrected[frame, root_idx, 0:3] = root.astype(np.float32)
                corrected[frame, mid_idx, 0:3] = mid.astype(np.float32)
                corrected[frame, end_idx, 0:3] = end.astype(np.float32)

                root_delta_values.append(float(np.linalg.norm(root - raw_targets[frame, root_idx, 0:3])))
                mid_delta_values.append(float(np.linalg.norm(mid - raw_targets[frame, mid_idx, 0:3])))
                end_delta_values.append(float(np.linalg.norm(end - raw_targets[frame, end_idx, 0:3])))
                root_distance_to_chest_values.append(float(np.linalg.norm(root - chest_pos)))

            corrected_chain_count += 1
            summary["chains"].append({
                "name": str(chain["name"]),
                "chest_target": str(chain["chest_target"]),
                "targets": list(chain["targets"]),
                "source_joints": list(chain["source_joints"]),
                "bodies": list(chain["bodies"]),
                "upper_len_m": upper_len,
                "lower_len_m": lower_len,
                "root_delta_p95_m": self._safe_percentile(root_delta_values, 95),
                "mid_delta_p95_m": self._safe_percentile(mid_delta_values, 95),
                "end_delta_p95_m": self._safe_percentile(end_delta_values, 95),
                "root_distance_to_chest_p95_m": self._safe_percentile(root_distance_to_chest_values, 95),
                "upper_angle_before_p95_deg": self._safe_percentile(upper_angle_before_values, 95),
                "upper_angle_after_p95_deg": self._safe_percentile(upper_angle_after_values, 95),
                "lower_angle_before_p95_deg": self._safe_percentile(lower_angle_before_values, 95),
                "lower_angle_after_p95_deg": self._safe_percentile(lower_angle_after_values, 95),
            })

        summary["corrected_chain_count"] = int(corrected_chain_count)
        return corrected, summary

    def _apply_ai_sapiens_source_segment_direction_correction(self, targets, buffer, root_tx):
        summary = {
            "enabled": bool(self.ai_sapiens_source_segment_direction_enabled),
            "chain_count": int(len(self.ai_sapiens_source_segment_direction_data)),
            "corrected_chain_count": 0,
            "start_frame": 0,
            "skipped": False,
            "skip_reason": None,
            "mode": "source_3d_segment_direction_with_robot_qpos0_lengths",
            "chains": [],
        }
        if not self.ai_sapiens_source_segment_direction_data:
            summary["skipped"] = True
            summary["skip_reason"] = "no_valid_chains"
            return targets, summary

        corrected = np.asarray(targets, dtype=np.float32).copy()
        source_positions = self._source_positions_for_source_segment_direction(buffer, root_tx)
        if source_positions is None:
            summary["skipped"] = True
            summary["skip_reason"] = "source_positions_unavailable"
            return corrected, summary

        start_frame = 0
        if self.ai_sapiens_source_segment_direction_start_after_warmup:
            start_frame = int(self.num_initialization_frames + self.num_stabilization_frames)
        summary["start_frame"] = int(start_frame)
        output_frame_count = max(0, int(corrected.shape[0]) - start_frame)
        if self.ai_sapiens_source_segment_direction_skip_single_output_frame and output_frame_count <= 1:
            summary["skipped"] = True
            summary["skip_reason"] = "single_output_frame_static_gate_guard"
            return corrected, summary

        corrected_chain_count = 0
        for chain in self.ai_sapiens_source_segment_direction_data:
            root_idx, mid_idx, end_idx = chain["target_indices"]
            src_root_idx, src_mid_idx, src_end_idx = chain["source_indices"]
            upper_len = float(chain["upper_len"])
            lower_len = float(chain["lower_len"])
            fallback_upper = np.asarray(chain["fallback_upper_dir"], dtype=np.float64)
            fallback_lower = np.asarray(chain["fallback_lower_dir"], dtype=np.float64)
            anchor = str(chain.get("anchor", "root"))
            if anchor not in {"root", "end"}:
                anchor = "root"

            root_delta_values = []
            mid_delta_values = []
            end_delta_values = []
            upper_angle_before_values = []
            upper_angle_after_values = []
            lower_angle_before_values = []
            lower_angle_after_values = []
            for frame in range(start_frame, corrected.shape[0]):
                src_root = source_positions[frame, src_root_idx]
                src_mid = source_positions[frame, src_mid_idx]
                src_end = source_positions[frame, src_end_idx]
                upper_dir = self._normalize_np(src_mid - src_root, fallback_upper)
                lower_dir = self._normalize_np(src_end - src_mid, fallback_lower)

                old_root = np.asarray(corrected[frame, root_idx, 0:3], dtype=np.float64)
                old_mid = np.asarray(corrected[frame, mid_idx, 0:3], dtype=np.float64)
                old_end = np.asarray(corrected[frame, end_idx, 0:3], dtype=np.float64)
                old_upper = self._normalize_np(old_mid - old_root, fallback_upper)
                old_lower = self._normalize_np(old_end - old_mid, fallback_lower)

                if anchor == "end":
                    end = old_end
                    mid = end - lower_dir * lower_len
                    root = mid - upper_dir * upper_len
                else:
                    root = old_root
                    mid = root + upper_dir * upper_len
                    end = mid + lower_dir * lower_len

                new_upper = self._normalize_np(mid - root, fallback_upper)
                new_lower = self._normalize_np(end - mid, fallback_lower)
                upper_angle_before_values.append(
                    float(np.degrees(np.arccos(np.clip(np.dot(old_upper, upper_dir), -1.0, 1.0))))
                )
                upper_angle_after_values.append(
                    float(np.degrees(np.arccos(np.clip(np.dot(new_upper, upper_dir), -1.0, 1.0))))
                )
                lower_angle_before_values.append(
                    float(np.degrees(np.arccos(np.clip(np.dot(old_lower, lower_dir), -1.0, 1.0))))
                )
                lower_angle_after_values.append(
                    float(np.degrees(np.arccos(np.clip(np.dot(new_lower, lower_dir), -1.0, 1.0))))
                )

                corrected[frame, root_idx, 0:3] = root.astype(np.float32)
                corrected[frame, mid_idx, 0:3] = mid.astype(np.float32)
                corrected[frame, end_idx, 0:3] = end.astype(np.float32)
                root_delta_values.append(float(np.linalg.norm(root - old_root)))
                mid_delta_values.append(float(np.linalg.norm(mid - old_mid)))
                end_delta_values.append(float(np.linalg.norm(end - old_end)))

            corrected_chain_count += 1
            summary["chains"].append({
                "name": str(chain["name"]),
                "targets": list(chain["targets"]),
                "source_joints": list(chain["source_joints"]),
                "anchor": anchor,
                "upper_len_m": upper_len,
                "lower_len_m": lower_len,
                "root_delta_p95_m": self._safe_percentile(root_delta_values, 95),
                "mid_delta_p95_m": self._safe_percentile(mid_delta_values, 95),
                "end_delta_p95_m": self._safe_percentile(end_delta_values, 95),
                "upper_angle_before_p95_deg": self._safe_percentile(upper_angle_before_values, 95),
                "upper_angle_after_p95_deg": self._safe_percentile(upper_angle_after_values, 95),
                "lower_angle_before_p95_deg": self._safe_percentile(lower_angle_before_values, 95),
                "lower_angle_after_p95_deg": self._safe_percentile(lower_angle_after_values, 95),
            })

        summary["corrected_chain_count"] = int(corrected_chain_count)
        return corrected, summary

    def _apply_ai_sapiens_arm_segment_direction_objective(self, targets, buffer, root_tx):
        summary = {
            "enabled": bool(self.ai_sapiens_arm_segment_direction_enabled),
            "chain_count": int(len(self.ai_sapiens_arm_segment_direction_data)),
            "corrected_chain_count": 0,
            "start_frame": 0,
            "skipped": False,
            "skip_reason": None,
            "mode": "weak_source_arm_segment_direction_after_feasible_projection",
            "upper_direction_weight": float(self.ai_sapiens_arm_upper_direction_weight),
            "forearm_direction_weight": float(self.ai_sapiens_arm_forearm_direction_weight),
            "blend": float(self.ai_sapiens_arm_direction_blend),
            "contact_risk_scale_enabled": bool(self.ai_sapiens_arm_direction_contact_risk_scale_enabled),
            "contact_risk_distance_m": float(self.ai_sapiens_arm_direction_contact_risk_distance_m),
            "contact_risk_soft_range_m": float(self.ai_sapiens_arm_direction_contact_risk_soft_range_m),
            "contact_risk_min_scale": float(self.ai_sapiens_arm_direction_contact_risk_min_scale),
            "chains": [],
        }
        if not self.ai_sapiens_arm_segment_direction_data:
            summary["skipped"] = True
            summary["skip_reason"] = "no_valid_chains"
            return targets, summary

        corrected = np.asarray(targets, dtype=np.float32).copy()
        source_positions = self._source_positions_for_arm_segment_direction(buffer, root_tx)
        if source_positions is None:
            summary["skipped"] = True
            summary["skip_reason"] = "source_positions_unavailable"
            return corrected, summary

        start_frame, should_skip, skip_reason = self._projection_start_and_skip(corrected)
        summary["start_frame"] = int(start_frame)
        if should_skip:
            summary["skipped"] = True
            summary["skip_reason"] = skip_reason
            return corrected, summary

        upper_weight = float(np.clip(self.ai_sapiens_arm_upper_direction_weight, 0.0, 1.0))
        forearm_weight = float(np.clip(self.ai_sapiens_arm_forearm_direction_weight, 0.0, 1.0))
        blend = float(np.clip(self.ai_sapiens_arm_direction_blend, 0.0, 1.0))
        contact_scale_enabled = bool(self.ai_sapiens_arm_direction_contact_risk_scale_enabled)
        contact_distance = max(0.0, float(self.ai_sapiens_arm_direction_contact_risk_distance_m))
        contact_soft_range = max(
            1.0e-6,
            float(self.ai_sapiens_arm_direction_contact_risk_soft_range_m),
        )
        contact_min_scale = float(
            np.clip(self.ai_sapiens_arm_direction_contact_risk_min_scale, 0.0, 1.0)
        )
        target_name_to_idx = {str(name): idx for idx, name in enumerate(self.mapped_joints)}
        hips_idx = target_name_to_idx.get("Hips")
        if blend <= 0.0 or (upper_weight <= 0.0 and forearm_weight <= 0.0):
            summary["skipped"] = True
            summary["skip_reason"] = "zero_direction_weight_or_blend"
            return corrected, summary

        corrected_chain_count = 0
        for chain in self.ai_sapiens_arm_segment_direction_data:
            root_idx, mid_idx, end_idx = chain["target_indices"]
            src_root_idx, src_mid_idx, src_end_idx = chain["source_indices"]
            upper_len = float(chain["upper_len"])
            lower_len = float(chain["lower_len"])
            fallback_upper = np.asarray(chain["fallback_upper_dir"], dtype=np.float64)
            fallback_lower = np.asarray(chain["fallback_lower_dir"], dtype=np.float64)

            upper_before_values = []
            upper_after_values = []
            lower_before_values = []
            lower_after_values = []
            mid_delta_values = []
            end_delta_values = []
            contact_risk_values = []
            contact_scale_values = []
            targets_lower = [str(name).lower() for name in chain.get("targets", [])]
            chain_name_lower = str(chain.get("name", "")).lower()
            if "left" in chain_name_lower or any("left" in name for name in targets_lower):
                same_side_leg_idx = target_name_to_idx.get("LeftLeg")
            elif "right" in chain_name_lower or any("right" in name for name in targets_lower):
                same_side_leg_idx = target_name_to_idx.get("RightLeg")
            else:
                same_side_leg_idx = None
            risk_target_indices = [
                int(idx)
                for idx in (hips_idx, same_side_leg_idx)
                if idx is not None
            ]
            for frame in range(start_frame, corrected.shape[0]):
                src_root = source_positions[frame, src_root_idx]
                src_mid = source_positions[frame, src_mid_idx]
                src_end = source_positions[frame, src_end_idx]
                source_upper_dir = self._normalize_np(src_mid - src_root, fallback_upper)
                source_lower_dir = self._normalize_np(src_end - src_mid, fallback_lower)

                root = np.asarray(corrected[frame, root_idx, 0:3], dtype=np.float64)
                old_mid = np.asarray(corrected[frame, mid_idx, 0:3], dtype=np.float64)
                old_end = np.asarray(corrected[frame, end_idx, 0:3], dtype=np.float64)
                old_upper_dir = self._normalize_np(old_mid - root, fallback_upper)
                old_lower_dir = self._normalize_np(old_end - old_mid, fallback_lower)

                contact_scale = 1.0
                risk_distance = float("inf")
                if contact_scale_enabled and risk_target_indices:
                    distances = []
                    for risk_idx in risk_target_indices:
                        risk_point = np.asarray(corrected[frame, risk_idx, 0:3], dtype=np.float64)
                        distances.append(float(np.linalg.norm(old_mid - risk_point)))
                        distances.append(float(np.linalg.norm(old_end - risk_point)))
                    if distances:
                        risk_distance = float(min(distances))
                        t = float(np.clip((risk_distance - contact_distance) / contact_soft_range, 0.0, 1.0))
                        contact_scale = contact_min_scale + (1.0 - contact_min_scale) * t
                contact_risk_values.append(risk_distance)
                contact_scale_values.append(contact_scale)

                desired_mid = root + source_upper_dir * upper_len
                mid = old_mid + (desired_mid - old_mid) * (upper_weight * blend * contact_scale)
                desired_end = mid + source_lower_dir * lower_len
                end = old_end + (desired_end - old_end) * (forearm_weight * blend * contact_scale)

                new_upper_dir = self._normalize_np(mid - root, fallback_upper)
                new_lower_dir = self._normalize_np(end - mid, fallback_lower)
                upper_before_values.append(self._angle_between_np_deg(old_upper_dir, source_upper_dir))
                upper_after_values.append(self._angle_between_np_deg(new_upper_dir, source_upper_dir))
                lower_before_values.append(self._angle_between_np_deg(old_lower_dir, source_lower_dir))
                lower_after_values.append(self._angle_between_np_deg(new_lower_dir, source_lower_dir))
                mid_delta_values.append(float(np.linalg.norm(mid - old_mid)))
                end_delta_values.append(float(np.linalg.norm(end - old_end)))

                corrected[frame, mid_idx, 0:3] = mid.astype(np.float32)
                corrected[frame, end_idx, 0:3] = end.astype(np.float32)

            corrected_chain_count += 1
            summary["chains"].append({
                "name": str(chain["name"]),
                "targets": list(chain["targets"]),
                "source_joints": list(chain["source_joints"]),
                "upper_len_m": upper_len,
                "lower_len_m": lower_len,
                "upper_angle_before_p95_deg": self._safe_percentile(upper_before_values, 95),
                "upper_angle_after_p95_deg": self._safe_percentile(upper_after_values, 95),
                "lower_angle_before_p95_deg": self._safe_percentile(lower_before_values, 95),
                "lower_angle_after_p95_deg": self._safe_percentile(lower_after_values, 95),
                "mid_direction_delta_p95_m": self._safe_percentile(mid_delta_values, 95),
                "end_direction_delta_p95_m": self._safe_percentile(end_delta_values, 95),
                "contact_risk_distance_p05_m": self._safe_percentile(contact_risk_values, 5),
                "contact_risk_scale_p05": self._safe_percentile(contact_scale_values, 5),
            })

        summary["corrected_chain_count"] = int(corrected_chain_count)
        return corrected, summary

    def _apply_ai_sapiens_source_foot_orientation_correction(self, targets, buffer, root_tx):
        summary = {
            "enabled": bool(self.ai_sapiens_source_foot_orientation_enabled),
            "target_count": int(len(self.ai_sapiens_source_foot_orientation_targets)),
            "corrected_target_count": 0,
            "start_frame": 0,
            "skipped": False,
            "skip_reason": None,
            "mode": str(self.ai_sapiens_source_foot_orientation_mode),
            "blend": float(self.ai_sapiens_source_foot_orientation_blend),
            "gate_mode": str(self.ai_sapiens_source_foot_orientation_gate_mode),
            "gate_height_on_m": float(self.ai_sapiens_source_foot_orientation_gate_height_on_m),
            "gate_height_off_m": float(self.ai_sapiens_source_foot_orientation_gate_height_off_m),
            "gate_speed_on_mps": float(self.ai_sapiens_source_foot_orientation_gate_speed_on_mps),
            "gate_speed_off_mps": float(self.ai_sapiens_source_foot_orientation_gate_speed_off_mps),
            "targets": [],
        }
        if not self.ai_sapiens_source_foot_orientation_targets:
            summary["skipped"] = True
            summary["skip_reason"] = "no_targets"
            return targets, summary

        target_name_to_idx = {name: idx for idx, name in enumerate(self.mapped_joints)}
        specs = []
        source_indices_needed = set()
        for spec in self.ai_sapiens_source_foot_orientation_targets:
            target_name = str(spec.get("target", ""))
            source_foot = str(spec.get("source_foot", target_name))
            source_toe_candidates = [
                str(name)
                for name in spec.get(
                    "source_toe_candidates",
                    [source_foot.replace("Foot", "ToeBase"), source_foot.replace("Foot", "ToeEnd")],
                )
            ]
            if target_name not in target_name_to_idx:
                continue
            source_foot_idx = self.human_robot_scaler.skeleton.joint_index(source_foot)
            source_toe_idx = -1
            source_toe_name = None
            for candidate in source_toe_candidates:
                idx = self.human_robot_scaler.skeleton.joint_index(candidate)
                if idx >= 0:
                    source_toe_idx = idx
                    source_toe_name = candidate
                    break
            if source_foot_idx < 0 or source_toe_idx < 0:
                continue
            source_indices_needed.add(int(source_foot_idx))
            source_indices_needed.add(int(source_toe_idx))
            specs.append({
                "target": target_name,
                "target_idx": int(target_name_to_idx[target_name]),
                "source_foot": source_foot,
                "source_foot_idx": int(source_foot_idx),
                "source_toe": str(source_toe_name),
                "source_toe_idx": int(source_toe_idx),
            })

        if not specs:
            summary["skipped"] = True
            summary["skip_reason"] = "no_valid_targets"
            return targets, summary

        start_frame = 0
        if self.ai_sapiens_source_foot_orientation_start_after_warmup:
            start_frame = int(self.num_initialization_frames + self.num_stabilization_frames)
        summary["start_frame"] = int(start_frame)
        output_frame_count = max(0, int(targets.shape[0]) - start_frame)
        if self.ai_sapiens_source_foot_orientation_skip_single_output_frame and output_frame_count <= 1:
            summary["skipped"] = True
            summary["skip_reason"] = "single_output_frame_static_gate_guard"
            return targets, summary

        corrected = np.asarray(targets, dtype=np.float32).copy()
        source_positions = np.zeros(
            (buffer.num_frames, self.human_robot_scaler.skeleton.num_joints, 3),
            dtype=np.float64,
        )
        needed_indices = sorted(source_indices_needed)
        for frame in range(buffer.num_frames):
            global_tx = buffer.compute_global_transforms(frame, root_tx)
            for joint_idx in needed_indices:
                source_positions[frame, joint_idx] = np.asarray(
                    global_tx[joint_idx][0:3],
                    dtype=np.float64,
                )

        blend = float(np.clip(self.ai_sapiens_source_foot_orientation_blend, 0.0, 1.0))
        fps = float(getattr(buffer, "sample_rate", 120.0) or 120.0)
        gate_mode = str(self.ai_sapiens_source_foot_orientation_gate_mode)
        height_on = float(self.ai_sapiens_source_foot_orientation_gate_height_on_m)
        height_off = float(self.ai_sapiens_source_foot_orientation_gate_height_off_m)
        speed_on = float(self.ai_sapiens_source_foot_orientation_gate_speed_on_mps)
        speed_off = float(self.ai_sapiens_source_foot_orientation_gate_speed_off_mps)

        def _linear_gate(values, on_value, off_value):
            values = np.asarray(values, dtype=np.float64)
            if off_value <= on_value:
                return (values <= on_value).astype(np.float64)
            return np.clip((off_value - values) / (off_value - on_value), 0.0, 1.0)

        corrected_target_count = 0
        for spec in specs:
            target_idx = int(spec["target_idx"])
            foot_positions = source_positions[:, int(spec["source_foot_idx"])]
            relative_height = foot_positions[:, 2] - float(np.nanmin(foot_positions[:, 2]))
            frame_delta = np.zeros_like(foot_positions)
            if foot_positions.shape[0] > 1:
                frame_delta[1:] = foot_positions[1:] - foot_positions[:-1]
                frame_delta[0] = frame_delta[1]
            horizontal_speed = np.linalg.norm(frame_delta[:, 0:2], axis=1) * fps
            height_gate = _linear_gate(relative_height, height_on, height_off)
            speed_gate = _linear_gate(horizontal_speed, speed_on, speed_off)
            if gate_mode == "source_stance":
                gate_values = height_gate * speed_gate
            elif gate_mode == "source_low":
                gate_values = height_gate
            elif gate_mode == "source_swing":
                gate_values = 1.0 - (height_gate * speed_gate)
            else:
                gate_values = np.ones((corrected.shape[0],), dtype=np.float64)

            before_angle_values = []
            after_angle_values = []
            quat_delta_values = []
            applied_blend_values = []
            for frame in range(start_frame, corrected.shape[0]):
                source_forward = (
                    source_positions[frame, int(spec["source_toe_idx"])]
                    - source_positions[frame, int(spec["source_foot_idx"])]
                )
                previous_quat = self._normalize_quat_np_xyzw(corrected[frame, target_idx, 3:7])
                previous_forward = self._quat_rotate_np_xyzw(previous_quat, [1.0, 0.0, 0.0])
                previous_up = self._quat_rotate_np_xyzw(previous_quat, [0.0, 0.0, 1.0])
                if self.ai_sapiens_source_foot_orientation_mode in {
                    "yaw_only_preserve_current_tilt",
                    "yaw_only",
                }:
                    desired_quat = self._quat_yaw_align_x_preserve_tilt_np_xyzw(
                        source_forward,
                        previous_quat,
                    )
                else:
                    desired_quat = self._quat_align_x_preserve_up_np_xyzw(
                        source_forward,
                        previous_up,
                        previous_quat,
                    )
                frame_blend = float(np.clip(blend * gate_values[frame], 0.0, 1.0))
                if frame_blend < 1.0:
                    new_quat = self._slerp_quat_xyzw(previous_quat, desired_quat, frame_blend)
                else:
                    new_quat = desired_quat
                new_forward = self._quat_rotate_np_xyzw(new_quat, [1.0, 0.0, 0.0])
                before_angle_values.append(self._angle_between_np_deg(source_forward, previous_forward))
                after_angle_values.append(self._angle_between_np_deg(source_forward, new_forward))
                quat_dot = float(np.clip(np.dot(previous_quat, new_quat), -1.0, 1.0))
                quat_delta_values.append(float(2.0 * np.degrees(np.arccos(abs(quat_dot)))))
                applied_blend_values.append(frame_blend)
                corrected[frame, target_idx, 3:7] = new_quat.astype(np.float32)

            corrected_target_count += 1
            summary["targets"].append({
                "target": str(spec["target"]),
                "source_segment": f"{spec['source_foot']}->{spec['source_toe']}",
                "gate_mean": self._safe_mean(gate_values[start_frame:]),
                "gate_p50": self._safe_percentile(gate_values[start_frame:], 50),
                "gate_p95": self._safe_percentile(gate_values[start_frame:], 95),
                "gate_active_fraction_gt_0p5": float(np.mean(gate_values[start_frame:] > 0.5))
                if gate_values[start_frame:].size > 0
                else float("nan"),
                "applied_blend_mean": self._safe_mean(applied_blend_values),
                "applied_blend_p95": self._safe_percentile(applied_blend_values, 95),
                "source_forward_vs_target_plus_x_before_p95_deg": self._safe_percentile(before_angle_values, 95),
                "source_forward_vs_target_plus_x_after_p95_deg": self._safe_percentile(after_angle_values, 95),
                "target_quat_delta_p95_deg": self._safe_percentile(quat_delta_values, 95),
            })

        summary["corrected_target_count"] = int(corrected_target_count)
        return corrected, summary

    def _build_ai_sapiens_arm_projection_data(self):
        chains = self.ai_sapiens_arm_projection_body_chains
        if chains is None:
            chains = [
                {
                    "targets": ["LeftArm", "LeftForeArm", "LeftHand"],
                    "bodies": [
                        "left_shoulder_roll_g1_proxy",
                        "left_elbow_g1_proxy",
                        "left_wrist_yaw_g1_proxy",
                    ],
                },
                {
                    "targets": ["RightArm", "RightForeArm", "RightHand"],
                    "bodies": [
                        "right_shoulder_roll_g1_proxy",
                        "right_elbow_g1_proxy",
                        "right_wrist_yaw_g1_proxy",
                    ],
                },
            ]
        return self._build_ai_sapiens_limb_projection_data([
            (
                str(chain["targets"][0]),
                str(chain["targets"][1]),
                str(chain["targets"][2]),
                str(chain["bodies"][0]),
                str(chain["bodies"][1]),
                str(chain["bodies"][2]),
            )
            for chain in chains
        ])

    def _build_ai_sapiens_leg_projection_data(self):
        chains = self.ai_sapiens_leg_projection_body_chains
        if chains is None:
            chains = [
                {
                    "targets": ["LeftLeg", "LeftShin", "LeftFoot"],
                    "bodies": [
                        "left_hip_roll_link",
                        "left_knee_link",
                        "left_ankle_roll_link",
                    ],
                },
                {
                    "targets": ["RightLeg", "RightShin", "RightFoot"],
                    "bodies": [
                        "right_hip_roll_link",
                        "right_knee_link",
                        "right_ankle_roll_link",
                    ],
                },
            ]
        return self._build_ai_sapiens_limb_projection_data([
            (
                str(chain["targets"][0]),
                str(chain["targets"][1]),
                str(chain["targets"][2]),
                str(chain["bodies"][0]),
                str(chain["bodies"][1]),
                str(chain["bodies"][2]),
            )
            for chain in chains
        ])

    def _build_ai_sapiens_limb_bend_angle_data(self):
        chains = self.ai_sapiens_limb_bend_angle_body_chains
        if chains is None:
            groups = {str(group).lower() for group in self.ai_sapiens_limb_bend_angle_groups}
            chains = []
            if "arm" in groups or "arms" in groups or "upper" in groups:
                chains.extend([
                    {
                        "targets": ["LeftArm", "LeftForeArm", "LeftHand"],
                        "bodies": [
                            "left_shoulder_roll_g1_proxy",
                            "left_elbow_g1_proxy",
                            "left_wrist_yaw_g1_proxy",
                        ],
                    },
                    {
                        "targets": ["RightArm", "RightForeArm", "RightHand"],
                        "bodies": [
                            "right_shoulder_roll_g1_proxy",
                            "right_elbow_g1_proxy",
                            "right_wrist_yaw_g1_proxy",
                        ],
                    },
                ])
            if "leg" in groups or "legs" in groups or "lower" in groups:
                chains.extend([
                    {
                        "targets": ["LeftLeg", "LeftShin", "LeftFoot"],
                        "bodies": [
                            "left_hip_roll_link",
                            "left_knee_link",
                            "left_ankle_roll_link",
                        ],
                    },
                    {
                        "targets": ["RightLeg", "RightShin", "RightFoot"],
                        "bodies": [
                            "right_hip_roll_link",
                            "right_knee_link",
                            "right_ankle_roll_link",
                        ],
                    },
                ])
        return self._build_ai_sapiens_limb_projection_data([
            (
                str(chain["targets"][0]),
                str(chain["targets"][1]),
                str(chain["targets"][2]),
                str(chain["bodies"][0]),
                str(chain["bodies"][1]),
                str(chain["bodies"][2]),
            )
            for chain in chains
        ])

    def _build_ai_sapiens_limb_plane_normal_data(self):
        chains = self.ai_sapiens_limb_plane_normal_body_chains
        if chains is None:
            groups = {str(group).lower() for group in self.ai_sapiens_limb_plane_normal_groups}
            chains = []
            if "arm" in groups or "arms" in groups or "upper" in groups:
                chains.extend([
                    {
                        "targets": ["LeftArm", "LeftForeArm", "LeftHand"],
                        "bodies": [
                            "left_shoulder_roll_g1_proxy",
                            "left_elbow_g1_proxy",
                            "left_wrist_yaw_g1_proxy",
                        ],
                    },
                    {
                        "targets": ["RightArm", "RightForeArm", "RightHand"],
                        "bodies": [
                            "right_shoulder_roll_g1_proxy",
                            "right_elbow_g1_proxy",
                            "right_wrist_yaw_g1_proxy",
                        ],
                    },
                ])
            if "leg" in groups or "legs" in groups or "lower" in groups:
                chains.extend([
                    {
                        "targets": ["LeftLeg", "LeftShin", "LeftFoot"],
                        "bodies": [
                            "left_hip_roll_link",
                            "left_knee_link",
                            "left_ankle_roll_link",
                        ],
                    },
                    {
                        "targets": ["RightLeg", "RightShin", "RightFoot"],
                        "bodies": [
                            "right_hip_roll_link",
                            "right_knee_link",
                            "right_ankle_roll_link",
                        ],
                    },
                ])
        return self._build_ai_sapiens_limb_projection_data([
            (
                str(chain["targets"][0]),
                str(chain["targets"][1]),
                str(chain["targets"][2]),
                str(chain["bodies"][0]),
                str(chain["bodies"][1]),
                str(chain["bodies"][2]),
            )
            for chain in chains
        ])

    def _build_ai_sapiens_limb_midpoint_position_data(self):
        chains = self.ai_sapiens_limb_midpoint_position_body_chains
        if chains is None:
            groups = {str(group).lower() for group in self.ai_sapiens_limb_midpoint_position_groups}
            chains = []
            if "arm" in groups or "arms" in groups or "upper" in groups:
                chains.extend([
                    {
                        "targets": ["LeftArm", "LeftForeArm", "LeftHand"],
                        "bodies": [
                            "left_shoulder_roll_g1_proxy",
                            "left_elbow_g1_proxy",
                            "left_wrist_yaw_g1_proxy",
                        ],
                    },
                    {
                        "targets": ["RightArm", "RightForeArm", "RightHand"],
                        "bodies": [
                            "right_shoulder_roll_g1_proxy",
                            "right_elbow_g1_proxy",
                            "right_wrist_yaw_g1_proxy",
                        ],
                    },
                ])
            if "leg" in groups or "legs" in groups or "lower" in groups:
                chains.extend([
                    {
                        "targets": ["LeftLeg", "LeftShin", "LeftFoot"],
                        "bodies": [
                            "left_hip_roll_link",
                            "left_knee_link",
                            "left_ankle_roll_link",
                        ],
                    },
                    {
                        "targets": ["RightLeg", "RightShin", "RightFoot"],
                        "bodies": [
                            "right_hip_roll_link",
                            "right_knee_link",
                            "right_ankle_roll_link",
                        ],
                    },
                ])
        return self._build_ai_sapiens_limb_projection_data([
            (
                str(chain["targets"][0]),
                str(chain["targets"][1]),
                str(chain["targets"][2]),
                str(chain["bodies"][0]),
                str(chain["bodies"][1]),
                str(chain["bodies"][2]),
            )
            for chain in chains
        ])

    def _build_ai_sapiens_torso_local_limb_midpoint_data(self):
        chains = self.ai_sapiens_torso_local_limb_midpoint_body_chains
        if chains is None:
            groups = {str(group).lower() for group in self.ai_sapiens_torso_local_limb_midpoint_groups}
            chains = []
            if "arm" in groups or "arms" in groups or "upper" in groups:
                chains.extend([
                    {
                        "targets": ["LeftArm", "LeftForeArm", "LeftHand"],
                        "bodies": [
                            "left_shoulder_roll_g1_proxy",
                            "left_elbow_g1_proxy",
                            "left_wrist_yaw_g1_proxy",
                        ],
                        "torso": "Chest",
                    },
                    {
                        "targets": ["RightArm", "RightForeArm", "RightHand"],
                        "bodies": [
                            "right_shoulder_roll_g1_proxy",
                            "right_elbow_g1_proxy",
                            "right_wrist_yaw_g1_proxy",
                        ],
                        "torso": "Chest",
                    },
                ])
            if "leg" in groups or "legs" in groups or "lower" in groups:
                chains.extend([
                    {
                        "targets": ["LeftLeg", "LeftShin", "LeftFoot"],
                        "bodies": [
                            "left_hip_roll_link",
                            "left_knee_link",
                            "left_ankle_roll_link",
                        ],
                        "torso": "Hips",
                    },
                    {
                        "targets": ["RightLeg", "RightShin", "RightFoot"],
                        "bodies": [
                            "right_hip_roll_link",
                            "right_knee_link",
                            "right_ankle_roll_link",
                        ],
                        "torso": "Hips",
                    },
                ])
        projection_data = self._build_ai_sapiens_limb_projection_data([
            (
                str(chain["targets"][0]),
                str(chain["targets"][1]),
                str(chain["targets"][2]),
                str(chain["bodies"][0]),
                str(chain["bodies"][1]),
                str(chain["bodies"][2]),
            )
            for chain in chains
        ])
        target_name_to_idx = {name: idx for idx, name in enumerate(self.mapped_joints)}
        chain_by_root = {str(chain["targets"][0]): chain for chain in chains}
        filtered = []
        for item in projection_data:
            root_name = str(item["root_name"])
            torso_name = str(chain_by_root.get(root_name, {}).get("torso", "Chest"))
            if torso_name not in target_name_to_idx:
                continue
            item = dict(item)
            item["torso_name"] = torso_name
            item["torso_idx"] = int(target_name_to_idx[torso_name])
            filtered.append(item)
        return filtered

    def _build_ai_sapiens_dynamic_lateral_data(self):
        target_name_to_idx = {name: idx for idx, name in enumerate(self.mapped_joints)}
        data = []
        for pair in self.ai_sapiens_dynamic_lateral_pairs:
            if len(pair) != 2:
                continue
            left_name, right_name = pair
            if left_name not in target_name_to_idx or right_name not in target_name_to_idx:
                continue
            left_source_idx = self.human_robot_scaler.skeleton.joint_index(left_name)
            right_source_idx = self.human_robot_scaler.skeleton.joint_index(right_name)
            if left_source_idx < 0 or right_source_idx < 0:
                continue
            data.append({
                "left_name": left_name,
                "right_name": right_name,
                "left_target_idx": target_name_to_idx[left_name],
                "right_target_idx": target_name_to_idx[right_name],
                "left_source_idx": int(left_source_idx),
                "right_source_idx": int(right_source_idx),
                "left_follower_indices": [
                    int(target_name_to_idx[name])
                    for name in self.ai_sapiens_dynamic_lateral_followers.get(left_name, [])
                    if name in target_name_to_idx
                ],
                "right_follower_indices": [
                    int(target_name_to_idx[name])
                    for name in self.ai_sapiens_dynamic_lateral_followers.get(right_name, [])
                    if name in target_name_to_idx
                ],
            })
        return data

    def _build_ai_sapiens_pair_span_data(self):
        target_name_to_idx = {name: idx for idx, name in enumerate(self.mapped_joints)}
        body_names = [
            newton_utils.get_name_from_label(label)
            for label in self.robot_builder.body_label
        ]
        body_name_to_idx = {name: idx for idx, name in enumerate(body_names)}
        state = self.ik_model.state()
        newton.eval_fk(self.ik_model, self.ik_model.joint_q, self.ik_model.joint_qd, state)
        body_q = state.body_q.numpy()

        data = []
        for spec in self.ai_sapiens_pair_span_pairs:
            if isinstance(spec, dict):
                left_name = str(spec.get("left", ""))
                right_name = str(spec.get("right", ""))
                left_body = str(spec.get("left_body", ""))
                right_body = str(spec.get("right_body", ""))
                left_source_name = str(spec.get("left_source", left_name))
                right_source_name = str(spec.get("right_source", right_name))
                left_followers = list(spec.get("left_followers", []))
                right_followers = list(spec.get("right_followers", []))
                span_scale = float(spec.get("span_scale", 1.0))
                span_m = spec.get("span_m", None)
                blend = float(spec.get("blend", 1.0))
                stage = str(spec.get("stage", self.ai_sapiens_pair_span_stage))
            elif isinstance(spec, (list, tuple)) and len(spec) >= 4:
                left_name, right_name, left_body, right_body = [str(value) for value in spec[:4]]
                left_source_name = left_name
                right_source_name = right_name
                left_followers = []
                right_followers = []
                span_scale = 1.0
                span_m = None
                blend = 1.0
                stage = self.ai_sapiens_pair_span_stage
            else:
                continue

            if left_name not in target_name_to_idx or right_name not in target_name_to_idx:
                continue
            if left_body not in body_name_to_idx or right_body not in body_name_to_idx:
                continue
            left_source_idx = self.human_robot_scaler.skeleton.joint_index(left_source_name)
            right_source_idx = self.human_robot_scaler.skeleton.joint_index(right_source_name)
            if left_source_idx < 0 or right_source_idx < 0:
                continue

            left_qpos0 = np.asarray(body_q[body_name_to_idx[left_body]][0:3], dtype=np.float64)
            right_qpos0 = np.asarray(body_q[body_name_to_idx[right_body]][0:3], dtype=np.float64)
            reference_span = float(np.linalg.norm(left_qpos0 - right_qpos0))
            if span_m is not None:
                reference_span = float(span_m)
            reference_span *= span_scale
            if not np.isfinite(reference_span) or reference_span <= 1e-9:
                continue

            data.append({
                "left_name": left_name,
                "right_name": right_name,
                "left_target_idx": int(target_name_to_idx[left_name]),
                "right_target_idx": int(target_name_to_idx[right_name]),
                "left_source_idx": int(left_source_idx),
                "right_source_idx": int(right_source_idx),
                "left_follower_indices": [
                    int(target_name_to_idx[name])
                    for name in left_followers
                    if name in target_name_to_idx
                ],
                "right_follower_indices": [
                    int(target_name_to_idx[name])
                    for name in right_followers
                    if name in target_name_to_idx
                ],
                "reference_span_m": float(reference_span),
                "blend": float(np.clip(blend, 0.0, 1.0)),
                "stage": stage,
                "left_body": left_body,
                "right_body": right_body,
            })
        return data

    def _source_positions_for_dynamic_lateral(self, buffer, root_tx):
        if not self.ai_sapiens_dynamic_lateral_data:
            return None
        needed_indices = sorted({
            idx
            for pair in self.ai_sapiens_dynamic_lateral_data
            for idx in (pair["left_source_idx"], pair["right_source_idx"])
        })
        source_positions = np.zeros(
            (buffer.num_frames, self.human_robot_scaler.skeleton.num_joints, 3),
            dtype=np.float64,
        )
        for frame in range(buffer.num_frames):
            global_tx = buffer.compute_global_transforms(frame, root_tx)
            for joint_idx in needed_indices:
                source_positions[frame, joint_idx] = np.asarray(
                    global_tx[joint_idx][0:3],
                    dtype=np.float64,
                )
        return source_positions

    def _apply_ai_sapiens_dynamic_lateral_correction(self, targets, buffer, root_tx):
        summary = {
            "enabled": bool(self.ai_sapiens_dynamic_lateral_correction_enabled),
            "pair_count": int(len(self.ai_sapiens_dynamic_lateral_data)),
            "corrected_pair_count": 0,
            "start_frame": 0,
            "skipped": False,
            "skip_reason": None,
            "mode": "source_xy_direction_around_raw_target_midpoint",
        }
        if not self.ai_sapiens_dynamic_lateral_data:
            summary["skipped"] = True
            summary["skip_reason"] = "no_valid_pairs"
            return targets, summary

        corrected = np.asarray(targets, dtype=np.float32).copy()
        source_positions = self._source_positions_for_dynamic_lateral(buffer, root_tx)
        if source_positions is None:
            summary["skipped"] = True
            summary["skip_reason"] = "source_positions_unavailable"
            return corrected, summary

        start_frame = 0
        if self.ai_sapiens_dynamic_lateral_start_after_warmup:
            start_frame = int(self.num_initialization_frames + self.num_stabilization_frames)
        summary["start_frame"] = int(start_frame)
        output_frame_count = max(0, int(corrected.shape[0]) - start_frame)
        if self.ai_sapiens_dynamic_lateral_skip_single_output_frame and output_frame_count <= 1:
            summary["skipped"] = True
            summary["skip_reason"] = "single_output_frame_static_gate_guard"
            return corrected, summary

        eps = 1e-9
        corrected_pairs = 0
        for pair in self.ai_sapiens_dynamic_lateral_data:
            left_t_idx = pair["left_target_idx"]
            right_t_idx = pair["right_target_idx"]
            left_s_idx = pair["left_source_idx"]
            right_s_idx = pair["right_source_idx"]
            pair_corrected = False
            for frame in range(start_frame, corrected.shape[0]):
                source_vec_xy = (
                    source_positions[frame, left_s_idx, 0:2]
                    - source_positions[frame, right_s_idx, 0:2]
                )
                source_norm = float(np.linalg.norm(source_vec_xy))
                if source_norm <= eps:
                    continue
                source_dir_xy = source_vec_xy / source_norm

                left_xy = np.asarray(corrected[frame, left_t_idx, 0:2], dtype=np.float64)
                right_xy = np.asarray(corrected[frame, right_t_idx, 0:2], dtype=np.float64)
                target_vec_xy = left_xy - right_xy
                target_norm = float(np.linalg.norm(target_vec_xy))
                if target_norm <= eps:
                    continue
                midpoint_xy = 0.5 * (left_xy + right_xy)
                half_vec_xy = source_dir_xy * (0.5 * target_norm)
                new_left_xy = midpoint_xy + half_vec_xy
                new_right_xy = midpoint_xy - half_vec_xy
                left_delta_xy = new_left_xy - left_xy
                right_delta_xy = new_right_xy - right_xy
                corrected[frame, left_t_idx, 0:2] = new_left_xy.astype(np.float32)
                corrected[frame, right_t_idx, 0:2] = new_right_xy.astype(np.float32)
                for follower_idx in pair.get("left_follower_indices", []):
                    corrected[frame, follower_idx, 0:2] = (
                        np.asarray(corrected[frame, follower_idx, 0:2], dtype=np.float64)
                        + left_delta_xy
                    ).astype(np.float32)
                for follower_idx in pair.get("right_follower_indices", []):
                    corrected[frame, follower_idx, 0:2] = (
                        np.asarray(corrected[frame, follower_idx, 0:2], dtype=np.float64)
                        + right_delta_xy
                    ).astype(np.float32)
                pair_corrected = True
            if pair_corrected:
                corrected_pairs += 1

        summary["corrected_pair_count"] = int(corrected_pairs)
        summary["pairs"] = [
            f"{pair['left_name']}/{pair['right_name']}"
            for pair in self.ai_sapiens_dynamic_lateral_data
        ]
        summary["followers"] = {
            f"{pair['left_name']}/{pair['right_name']}": {
                "left_follower_count": int(len(pair.get("left_follower_indices", []))),
                "right_follower_count": int(len(pair.get("right_follower_indices", []))),
            }
            for pair in self.ai_sapiens_dynamic_lateral_data
        }
        return corrected, summary

    def _source_positions_for_pair_span(self, buffer, root_tx):
        if not self.ai_sapiens_pair_span_data:
            return None
        needed_indices = sorted({
            idx
            for pair in self.ai_sapiens_pair_span_data
            for idx in (pair["left_source_idx"], pair["right_source_idx"])
        })
        source_positions = np.zeros(
            (buffer.num_frames, self.human_robot_scaler.skeleton.num_joints, 3),
            dtype=np.float64,
        )
        for frame in range(buffer.num_frames):
            global_tx = buffer.compute_global_transforms(frame, root_tx)
            for joint_idx in needed_indices:
                source_positions[frame, joint_idx] = np.asarray(
                    global_tx[joint_idx][0:3],
                    dtype=np.float64,
                )
        return source_positions

    def _apply_ai_sapiens_pair_span_correction(self, targets, buffer, root_tx, stage=None):
        stage = str(stage or self.ai_sapiens_pair_span_stage)
        stage_pairs = [
            pair
            for pair in self.ai_sapiens_pair_span_data
            if str(pair.get("stage", self.ai_sapiens_pair_span_stage)) == stage
        ]
        summary = {
            "enabled": bool(self.ai_sapiens_pair_span_correction_enabled),
            "pair_count": int(len(stage_pairs)),
            "corrected_pair_count": 0,
            "start_frame": 0,
            "skipped": False,
            "skip_reason": None,
            "mode": "source_xy_direction_with_robot_qpos0_pair_span",
            "stage": stage,
        }
        if not stage_pairs:
            summary["skipped"] = True
            summary["skip_reason"] = "no_valid_pairs_for_stage"
            return targets, summary

        corrected = np.asarray(targets, dtype=np.float32).copy()
        source_positions = self._source_positions_for_pair_span(buffer, root_tx)
        if source_positions is None:
            summary["skipped"] = True
            summary["skip_reason"] = "source_positions_unavailable"
            return corrected, summary

        start_frame = 0
        if self.ai_sapiens_pair_span_start_after_warmup:
            start_frame = int(self.num_initialization_frames + self.num_stabilization_frames)
        summary["start_frame"] = int(start_frame)
        output_frame_count = max(0, int(corrected.shape[0]) - start_frame)
        if self.ai_sapiens_pair_span_skip_single_output_frame and output_frame_count <= 1:
            summary["skipped"] = True
            summary["skip_reason"] = "single_output_frame_static_gate_guard"
            return corrected, summary

        eps = 1e-9
        corrected_pairs = 0
        pair_summaries = []
        for pair in stage_pairs:
            left_t_idx = pair["left_target_idx"]
            right_t_idx = pair["right_target_idx"]
            left_s_idx = pair["left_source_idx"]
            right_s_idx = pair["right_source_idx"]
            reference_span = float(pair["reference_span_m"])
            blend = float(pair.get("blend", 1.0))
            deltas = []
            spans_before = []
            pair_corrected = False
            for frame in range(start_frame, corrected.shape[0]):
                source_vec_xy = (
                    source_positions[frame, left_s_idx, 0:2]
                    - source_positions[frame, right_s_idx, 0:2]
                )
                source_norm = float(np.linalg.norm(source_vec_xy))
                if source_norm > eps:
                    direction_xy = source_vec_xy / source_norm
                else:
                    target_vec_xy = (
                        np.asarray(corrected[frame, left_t_idx, 0:2], dtype=np.float64)
                        - np.asarray(corrected[frame, right_t_idx, 0:2], dtype=np.float64)
                    )
                    target_norm = float(np.linalg.norm(target_vec_xy))
                    if target_norm <= eps:
                        continue
                    direction_xy = target_vec_xy / target_norm

                left_xy = np.asarray(corrected[frame, left_t_idx, 0:2], dtype=np.float64)
                right_xy = np.asarray(corrected[frame, right_t_idx, 0:2], dtype=np.float64)
                span_before = float(np.linalg.norm(left_xy - right_xy))
                midpoint_xy = 0.5 * (left_xy + right_xy)
                half_vec_xy = direction_xy * (0.5 * reference_span)
                new_left_xy = midpoint_xy + half_vec_xy
                new_right_xy = midpoint_xy - half_vec_xy
                if blend < 1.0:
                    new_left_xy = left_xy + (new_left_xy - left_xy) * blend
                    new_right_xy = right_xy + (new_right_xy - right_xy) * blend
                left_delta_xy = new_left_xy - left_xy
                right_delta_xy = new_right_xy - right_xy

                corrected[frame, left_t_idx, 0:2] = new_left_xy.astype(np.float32)
                corrected[frame, right_t_idx, 0:2] = new_right_xy.astype(np.float32)
                for follower_idx in pair.get("left_follower_indices", []):
                    corrected[frame, follower_idx, 0:2] = (
                        np.asarray(corrected[frame, follower_idx, 0:2], dtype=np.float64)
                        + left_delta_xy
                    ).astype(np.float32)
                for follower_idx in pair.get("right_follower_indices", []):
                    corrected[frame, follower_idx, 0:2] = (
                        np.asarray(corrected[frame, follower_idx, 0:2], dtype=np.float64)
                        + right_delta_xy
                    ).astype(np.float32)

                deltas.append(float(max(np.linalg.norm(left_delta_xy), np.linalg.norm(right_delta_xy))))
                spans_before.append(span_before)
                pair_corrected = True
            if pair_corrected:
                corrected_pairs += 1
            pair_summaries.append({
                "pair": f"{pair['left_name']}/{pair['right_name']}",
                "left_body": pair["left_body"],
                "right_body": pair["right_body"],
                "reference_span_m": reference_span,
                "blend": blend,
                "span_before_mean_m": float(np.mean(spans_before)) if spans_before else float("nan"),
                "span_before_p95_m": self._safe_percentile(spans_before, 95) if spans_before else float("nan"),
                "correction_delta_mean_m": float(np.mean(deltas)) if deltas else float("nan"),
                "correction_delta_max_m": float(np.max(deltas)) if deltas else float("nan"),
            })

        summary["corrected_pair_count"] = int(corrected_pairs)
        summary["pairs"] = pair_summaries
        return corrected, summary

    def _projection_start_and_skip(self, targets):
        start_frame = 0
        if self.ai_sapiens_projection_start_after_warmup:
            start_frame = int(self.num_initialization_frames + self.num_stabilization_frames)
        output_frame_count = max(0, int(targets.shape[0]) - start_frame)
        if self.ai_sapiens_projection_skip_single_output_frame and output_frame_count <= 1:
            return start_frame, True, "single_output_frame_static_gate_guard"
        if start_frame >= int(targets.shape[0]):
            return start_frame, True, "no_output_frames"
        return start_frame, False, None

    @staticmethod
    def _safe_percentile(values, percentile):
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return float("nan")
        return float(np.percentile(values, percentile))

    @staticmethod
    def _safe_mean(values):
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return float("nan")
        return float(np.mean(values))

    def _apply_ai_sapiens_two_link_projection(
        self,
        targets,
        projection_data,
        *,
        kind: str,
        anchor: str,
        bend_mode: str,
        flip_bend_roots,
        max_reach_margin_m: float,
        min_reach_margin_m: float,
        blend: float = 1.0,
        ramp_output_frames: int = 0,
        projection_mode: str = "two_bone",
        min_elbow_bend_deg: float = 0.0,
    ):
        blend = float(np.clip(blend, 0.0, 1.0))
        ramp_output_frames = max(0, int(ramp_output_frames))
        projection_mode = str(projection_mode)
        min_elbow_bend_deg = max(0.0, float(min_elbow_bend_deg))
        summary = {
            "enabled": True,
            "kind": str(kind),
            "projection_mode": projection_mode,
            "anchor": str(anchor),
            "bend_mode": str(bend_mode),
            "flip_bend_roots": sorted(str(name) for name in flip_bend_roots),
            "chain_count": int(len(projection_data)),
            "start_frame": 0,
            "skipped": False,
            "skip_reason": None,
            "max_reach_margin_m": float(max_reach_margin_m),
            "min_reach_margin_m": float(min_reach_margin_m),
            "blend": float(blend),
            "ramp_output_frames": int(ramp_output_frames),
            "min_elbow_bend_deg": float(min_elbow_bend_deg),
            "chains": [],
        }
        if not projection_data:
            summary["skipped"] = True
            summary["skip_reason"] = "no_valid_chains"
            return targets, summary

        if anchor not in {"shoulder", "root", "foot", "end"}:
            summary["skipped"] = True
            summary["skip_reason"] = f"unsupported_anchor:{anchor}"
            return targets, summary
        if bend_mode not in {"raw", "qpos0"}:
            summary["skipped"] = True
            summary["skip_reason"] = f"unsupported_bend_mode:{bend_mode}"
            return targets, summary
        if projection_mode not in {"two_bone", "soft_two_bone_feasible"}:
            summary["skipped"] = True
            summary["skip_reason"] = f"unsupported_projection_mode:{projection_mode}"
            return targets, summary

        projected = np.asarray(targets, dtype=np.float32).copy()
        raw_targets = np.asarray(targets, dtype=np.float64)
        start_frame, should_skip, skip_reason = self._projection_start_and_skip(projected)
        summary["start_frame"] = int(start_frame)
        if should_skip:
            summary["skipped"] = True
            summary["skip_reason"] = skip_reason
            return projected, summary

        eps = 1e-8
        root_anchor = anchor in {"shoulder", "root"}
        for chain in projection_data:
            root_idx = int(chain["root_idx"])
            mid_idx = int(chain["mid_idx"])
            end_idx = int(chain["end_idx"])
            upper_len = float(chain["upper_len"])
            lower_len = float(chain["lower_len"])
            natural_min = abs(upper_len - lower_len)
            natural_max = upper_len + lower_len
            bend_limited_max = natural_max
            if min_elbow_bend_deg > 0.0:
                bend_rad = np.deg2rad(min(min_elbow_bend_deg, 179.0))
                bend_limited_max = np.sqrt(
                    max(
                        0.0,
                        upper_len * upper_len
                        + lower_len * lower_len
                        + 2.0 * upper_len * lower_len * np.cos(bend_rad),
                    )
                )
            min_reach = min(natural_max - eps, natural_min + max(0.0, float(min_reach_margin_m)))
            max_reach = max(
                min_reach + eps,
                min(
                    natural_max - max(0.0, float(max_reach_margin_m)),
                    bend_limited_max,
                ),
            )
            fallback_end_dir = np.asarray(chain["fallback_end_dir"], dtype=np.float64)
            fallback_bend_dir = np.asarray(chain["fallback_bend_dir"], dtype=np.float64)

            raw_distance_values = []
            projected_distance_values = []
            root_deltas = []
            mid_deltas = []
            end_deltas = []
            raw_over_frames = 0
            projected_over_frames = 0
            raw_under_frames = 0
            projected_under_frames = 0

            for frame in range(start_frame, projected.shape[0]):
                raw_root = np.asarray(projected[frame, root_idx, 0:3], dtype=np.float64)
                raw_mid = np.asarray(projected[frame, mid_idx, 0:3], dtype=np.float64)
                raw_end = np.asarray(projected[frame, end_idx, 0:3], dtype=np.float64)
                raw_distance = float(np.linalg.norm(raw_end - raw_root))
                distance = float(np.clip(raw_distance, min_reach, max_reach))

                if root_anchor:
                    root = raw_root
                    axis = self._normalize_np(raw_end - raw_root, fallback_end_dir)
                    end = root + axis * distance
                else:
                    end = raw_end
                    root_dir_from_end = self._normalize_np(raw_root - raw_end, -fallback_end_dir)
                    root = end + root_dir_from_end * distance
                    axis = self._normalize_np(end - root, fallback_end_dir)

                fallback_bend = fallback_bend_dir - np.dot(fallback_bend_dir, axis) * axis
                if bend_mode == "qpos0":
                    bend_dir = self._normalize_np(fallback_bend, fallback=fallback_bend_dir)
                else:
                    bend_raw = raw_mid - root
                    bend_dir = bend_raw - np.dot(bend_raw, axis) * axis
                    bend_dir = self._normalize_np(bend_dir, fallback=fallback_bend)
                if str(chain["root_name"]) in flip_bend_roots:
                    bend_dir = -bend_dir

                x = (upper_len * upper_len - lower_len * lower_len + distance * distance) / (2.0 * max(distance, eps))
                h_sq = max(0.0, upper_len * upper_len - x * x)
                mid = root + axis * x + bend_dir * np.sqrt(h_sq)

                frame_blend = blend
                if ramp_output_frames > 0:
                    frame_blend *= min(
                        1.0,
                        max(0.0, (frame - start_frame + 1) / float(ramp_output_frames)),
                    )
                if frame_blend < 1.0:
                    root = raw_root + (root - raw_root) * frame_blend
                    mid = raw_mid + (mid - raw_mid) * frame_blend
                    end = raw_end + (end - raw_end) * frame_blend

                projected[frame, root_idx, 0:3] = root.astype(np.float32)
                projected[frame, mid_idx, 0:3] = mid.astype(np.float32)
                projected[frame, end_idx, 0:3] = end.astype(np.float32)

                projected_distance = float(np.linalg.norm(end - root))
                raw_distance_values.append(raw_distance)
                projected_distance_values.append(projected_distance)
                root_deltas.append(float(np.linalg.norm(root - raw_targets[frame, root_idx, 0:3])))
                mid_deltas.append(float(np.linalg.norm(mid - raw_targets[frame, mid_idx, 0:3])))
                end_deltas.append(float(np.linalg.norm(end - raw_targets[frame, end_idx, 0:3])))
                raw_over_frames += int(raw_distance > max_reach + 1e-7)
                projected_over_frames += int(projected_distance > max_reach + 1e-7)
                raw_under_frames += int(raw_distance < min_reach - 1e-7)
                projected_under_frames += int(projected_distance < min_reach - 1e-7)

            summary["chains"].append({
                "root": str(chain["root_name"]),
                "mid": str(chain["mid_name"]),
                "end": str(chain["end_name"]),
                "upper_len_m": upper_len,
                "lower_len_m": lower_len,
                "natural_max_reach_m": float(natural_max),
                "bend_limited_max_reach_m": float(bend_limited_max),
                "min_reach_m": float(min_reach),
                "max_reach_m": float(max_reach),
                "raw_overreach_frames": int(raw_over_frames),
                "projected_overreach_frames": int(projected_over_frames),
                "raw_underreach_frames": int(raw_under_frames),
                "projected_underreach_frames": int(projected_under_frames),
                "raw_root_end_distance_p95_m": self._safe_percentile(raw_distance_values, 95),
                "raw_root_end_distance_max_m": float(np.max(raw_distance_values)) if raw_distance_values else float("nan"),
                "projected_root_end_distance_p95_m": self._safe_percentile(projected_distance_values, 95),
                "projected_root_end_distance_max_m": float(np.max(projected_distance_values)) if projected_distance_values else float("nan"),
                "root_projection_delta_max_m": float(np.max(root_deltas)) if root_deltas else float("nan"),
                "mid_projection_delta_max_m": float(np.max(mid_deltas)) if mid_deltas else float("nan"),
                "end_projection_delta_max_m": float(np.max(end_deltas)) if end_deltas else float("nan"),
                "root_projection_delta_mean_m": float(np.mean(root_deltas)) if root_deltas else float("nan"),
                "mid_projection_delta_mean_m": float(np.mean(mid_deltas)) if mid_deltas else float("nan"),
                "end_projection_delta_mean_m": float(np.mean(end_deltas)) if end_deltas else float("nan"),
            })

        return projected, summary

    def _apply_ai_sapiens_leg_projection(self, targets):
        return self._apply_ai_sapiens_two_link_projection(
            targets,
            self.ai_sapiens_leg_projection_data,
            kind="leg",
            anchor=self.ai_sapiens_leg_projection_anchor,
            bend_mode=self.ai_sapiens_leg_projection_bend_mode,
            flip_bend_roots=self.ai_sapiens_leg_projection_flip_bend_roots,
            max_reach_margin_m=self.ai_sapiens_leg_projection_max_reach_margin_m,
            min_reach_margin_m=self.ai_sapiens_leg_projection_min_reach_margin_m,
            blend=self.ai_sapiens_leg_projection_blend,
            ramp_output_frames=self.ai_sapiens_leg_projection_ramp_output_frames,
        )

    def _apply_ai_sapiens_arm_projection(self, targets, stage: str = "default"):
        projected, summary = self._apply_ai_sapiens_two_link_projection(
            targets,
            self.ai_sapiens_arm_projection_data,
            kind="arm",
            anchor=self.ai_sapiens_arm_projection_anchor,
            bend_mode=self.ai_sapiens_arm_projection_bend_mode,
            flip_bend_roots=self.ai_sapiens_arm_projection_flip_bend_roots,
            max_reach_margin_m=self.ai_sapiens_arm_projection_max_reach_margin_m,
            min_reach_margin_m=self.ai_sapiens_arm_projection_min_reach_margin_m,
            blend=self.ai_sapiens_arm_projection_blend,
            ramp_output_frames=self.ai_sapiens_arm_projection_ramp_output_frames,
            projection_mode=self.ai_sapiens_arm_projection_mode,
            min_elbow_bend_deg=self.ai_sapiens_arm_projection_min_elbow_bend_deg,
        )
        summary["stage"] = str(stage)
        summary["state_gate_enabled"] = bool(self.ai_sapiens_arm_projection_state_gate_enabled)
        if self.ai_sapiens_arm_projection_state_gate_enabled:
            scale_trace, gate_summary = self._compute_ai_sapiens_risk_window_objective_scales(targets)
            summary["state_gate_summary"] = gate_summary
            raw_targets = np.asarray(targets, dtype=np.float32)
            gated = np.asarray(raw_targets, dtype=np.float32).copy()
            active_frames_total = 0
            for chain_idx, chain in enumerate(self.ai_sapiens_arm_projection_data):
                if chain_idx >= scale_trace.shape[1]:
                    continue
                alpha = np.asarray(scale_trace[:, chain_idx, 0], dtype=np.float32)
                alpha = np.clip(alpha, 0.0, 1.0)
                active_frames_total += int(np.count_nonzero(alpha > 1.0e-6))
                blend = alpha[:, None]
                for key in ("root_idx", "mid_idx", "end_idx"):
                    target_idx = int(chain[key])
                    gated[:, target_idx, 0:3] = (
                        raw_targets[:, target_idx, 0:3] * (1.0 - blend)
                        + projected[:, target_idx, 0:3] * blend
                    )
            projected = gated
            summary["state_gate_active_frame_total"] = int(active_frames_total)
        return projected, summary

    def _apply_ai_sapiens_adaptive_arm_objective_targets(self, targets):
        """Compute adaptive arm reach scales and optional target relaxation."""
        adjusted = np.asarray(targets, dtype=np.float32).copy()
        chain_count = len(self.ai_sapiens_adaptive_arm_objective_data)
        scale_trace = np.zeros((adjusted.shape[0], max(1, chain_count), 4), dtype=np.float32)
        summary = {
            "enabled": True,
            "mode": "adaptive_target_relaxation_not_hard_projection",
            "chain_count": int(chain_count),
            "reach_ratio_start": float(self.ai_sapiens_arm_reach_ratio_start),
            "reach_ratio_end": float(self.ai_sapiens_arm_reach_ratio_end),
            "hand_position_min_scale": float(self.ai_sapiens_hand_position_min_scale),
            "forearm_position_min_scale": float(self.ai_sapiens_forearm_position_min_scale),
            "elbow_hint_weight_min": float(self.ai_sapiens_elbow_hint_weight_min),
            "elbow_hint_weight_max": float(self.ai_sapiens_elbow_hint_weight_max),
            "target_relaxation_applied": bool(
                self.ai_sapiens_adaptive_arm_objective_apply_target_relaxation
            ),
            "chains": [],
            "skipped": False,
            "skip_reason": None,
        }
        if not self.ai_sapiens_adaptive_arm_objective_data:
            summary["skipped"] = True
            summary["skip_reason"] = "no_valid_arm_chains"
            return adjusted, summary, scale_trace

        start = float(self.ai_sapiens_arm_reach_ratio_start)
        end = float(self.ai_sapiens_arm_reach_ratio_end)
        if end <= start:
            end = start + 1e-6
        hand_min_scale = float(np.clip(self.ai_sapiens_hand_position_min_scale, 0.0, 1.0))
        forearm_min_scale = float(np.clip(self.ai_sapiens_forearm_position_min_scale, 0.0, 1.0))
        hint_min = float(np.clip(self.ai_sapiens_elbow_hint_weight_min, 0.0, 1.0))
        hint_max = float(np.clip(self.ai_sapiens_elbow_hint_weight_max, 0.0, 1.0))

        # Compute the soft two-bone midpoint once, then only use its elbow point
        # as a weak hint.  End/wrist target is not clamped to this projection.
        projected, _ = self._apply_ai_sapiens_two_link_projection(
            targets,
            self.ai_sapiens_adaptive_arm_objective_data,
            kind="adaptive_arm_objective_hint",
            anchor="shoulder",
            bend_mode="raw",
            flip_bend_roots=set(),
            max_reach_margin_m=0.020,
            min_reach_margin_m=0.010,
            blend=1.0,
            ramp_output_frames=0,
            projection_mode="soft_two_bone_feasible",
            min_elbow_bend_deg=0.0,
        )

        start_frame, should_skip, skip_reason = self._projection_start_and_skip(adjusted)
        summary["start_frame"] = int(start_frame)
        if should_skip:
            summary["skipped"] = True
            summary["skip_reason"] = skip_reason
            return adjusted, summary, scale_trace

        for chain_idx, chain in enumerate(self.ai_sapiens_adaptive_arm_objective_data):
            root_idx = int(chain["root_idx"])
            mid_idx = int(chain["mid_idx"])
            end_idx = int(chain["end_idx"])
            max_reach = max(
                1e-8,
                float(chain["upper_len"]) + float(chain["lower_len"]) - 0.020,
            )
            reach_ratios = []
            hand_scales = []
            forearm_scales = []
            hint_weights = []
            hand_delta_values = []
            mid_delta_values = []
            high_reach_frames = 0
            for frame in range(start_frame, adjusted.shape[0]):
                raw_root = np.asarray(targets[frame, root_idx, 0:3], dtype=np.float64)
                raw_mid = np.asarray(targets[frame, mid_idx, 0:3], dtype=np.float64)
                raw_end = np.asarray(targets[frame, end_idx, 0:3], dtype=np.float64)
                root_to_end = raw_end - raw_root
                root_to_mid = raw_mid - raw_root
                distance = float(np.linalg.norm(root_to_end))
                reach_ratio = distance / max_reach
                alpha = float(np.clip((reach_ratio - start) / (end - start), 0.0, 1.0))
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                hand_scale = 1.0 - alpha * (1.0 - hand_min_scale)
                forearm_scale = 1.0 - alpha * (1.0 - forearm_min_scale)
                hint_weight = hint_min + alpha * (hint_max - hint_min)

                relaxed_end = raw_root + root_to_end * hand_scale
                relaxed_mid = raw_root + root_to_mid * forearm_scale
                hint_mid = np.asarray(projected[frame, mid_idx, 0:3], dtype=np.float64)
                new_mid = relaxed_mid + (hint_mid - relaxed_mid) * hint_weight

                adjusted[frame, end_idx, 0:3] = relaxed_end.astype(np.float32)
                adjusted[frame, mid_idx, 0:3] = new_mid.astype(np.float32)
                scale_trace[frame, chain_idx, 0] = np.float32(reach_ratio)
                scale_trace[frame, chain_idx, 1] = np.float32(hand_scale)
                scale_trace[frame, chain_idx, 2] = np.float32(forearm_scale)
                scale_trace[frame, chain_idx, 3] = np.float32(hint_weight)

                reach_ratios.append(reach_ratio)
                hand_scales.append(hand_scale)
                forearm_scales.append(forearm_scale)
                hint_weights.append(hint_weight)
                hand_delta_values.append(float(np.linalg.norm(relaxed_end - raw_end)))
                mid_delta_values.append(float(np.linalg.norm(new_mid - raw_mid)))
                high_reach_frames += int(reach_ratio > start)

            summary["chains"].append({
                "root": str(chain["root_name"]),
                "mid": str(chain["mid_name"]),
                "end": str(chain["end_name"]),
                "max_reach_m": float(max_reach),
                "reach_ratio_p95": self._safe_percentile(reach_ratios, 95),
                "reach_ratio_max": float(np.max(reach_ratios)) if reach_ratios else float("nan"),
                "high_reach_frames": int(high_reach_frames),
                "hand_scale_min": float(np.min(hand_scales)) if hand_scales else float("nan"),
                "forearm_scale_min": float(np.min(forearm_scales)) if forearm_scales else float("nan"),
                "elbow_hint_weight_max": float(np.max(hint_weights)) if hint_weights else float("nan"),
                "hand_target_delta_p95_m": self._safe_percentile(hand_delta_values, 95),
                "mid_target_delta_p95_m": self._safe_percentile(mid_delta_values, 95),
            })
        if self.ai_sapiens_adaptive_arm_objective_apply_target_relaxation:
            return adjusted, summary, scale_trace
        return np.asarray(targets, dtype=np.float32).copy(), summary, scale_trace

    def _compute_ai_sapiens_elbow_branch_hint_targets(self, targets):
        """Build weak elbow branch hint targets without modifying IK targets."""
        chain_count = len(self.ai_sapiens_elbow_branch_hint_data)
        hint_targets = np.full(
            (targets.shape[0], max(1, chain_count), 3),
            np.nan,
            dtype=np.float32,
        )
        delta_trace = np.full(
            (targets.shape[0], max(1, chain_count), 4),
            np.nan,
            dtype=np.float32,
        )
        summary = {
            "enabled": True,
            "chain_count": int(chain_count),
            "weight": float(self.ai_sapiens_elbow_branch_hint_weight),
            "max_reach_margin_m": float(self.ai_sapiens_elbow_branch_hint_max_reach_margin_m),
            "min_reach_margin_m": float(self.ai_sapiens_elbow_branch_hint_min_reach_margin_m),
            "chains": [],
            "skipped": False,
            "skip_reason": None,
        }
        if not self.ai_sapiens_elbow_branch_hint_data:
            summary["skipped"] = True
            summary["skip_reason"] = "no_valid_arm_chains"
            return hint_targets, summary, delta_trace

        projected, projection_summary = self._apply_ai_sapiens_two_link_projection(
            targets,
            self.ai_sapiens_elbow_branch_hint_data,
            kind="elbow_branch_hint_objective",
            anchor="shoulder",
            bend_mode="raw",
            flip_bend_roots=set(),
            max_reach_margin_m=self.ai_sapiens_elbow_branch_hint_max_reach_margin_m,
            min_reach_margin_m=self.ai_sapiens_elbow_branch_hint_min_reach_margin_m,
            blend=1.0,
            ramp_output_frames=0,
            projection_mode="soft_two_bone_feasible",
            min_elbow_bend_deg=0.0,
        )
        summary["projection"] = projection_summary

        raw = np.asarray(targets, dtype=np.float64)
        projected = np.asarray(projected, dtype=np.float64)
        start_frame = int(projection_summary.get("start_frame", 0) or 0)
        for chain_idx, chain in enumerate(self.ai_sapiens_elbow_branch_hint_data):
            root_idx = int(chain["root_idx"])
            mid_idx = int(chain["mid_idx"])
            end_idx = int(chain["end_idx"])
            max_reach = max(
                1e-8,
                float(chain["upper_len"])
                + float(chain["lower_len"])
                - float(self.ai_sapiens_elbow_branch_hint_max_reach_margin_m),
            )
            mid_delta_values = []
            reach_ratios = []
            for frame in range(start_frame, raw.shape[0]):
                raw_root = raw[frame, root_idx, 0:3]
                raw_mid = raw[frame, mid_idx, 0:3]
                raw_end = raw[frame, end_idx, 0:3]
                hint_mid = projected[frame, mid_idx, 0:3]
                reach_ratio = float(np.linalg.norm(raw_end - raw_root) / max_reach)
                mid_delta = float(np.linalg.norm(hint_mid - raw_mid))
                hint_targets[frame, chain_idx, :] = hint_mid.astype(np.float32)
                delta_trace[frame, chain_idx, 0] = np.float32(reach_ratio)
                delta_trace[frame, chain_idx, 1] = np.float32(mid_delta)
                delta_trace[frame, chain_idx, 2] = np.float32(max_reach)
                delta_trace[frame, chain_idx, 3] = np.float32(self.ai_sapiens_elbow_branch_hint_weight)
                reach_ratios.append(reach_ratio)
                mid_delta_values.append(mid_delta)
            summary["chains"].append({
                "root": str(chain["root_name"]),
                "mid": str(chain["mid_name"]),
                "end": str(chain["end_name"]),
                "reach_ratio_p95": self._safe_percentile(reach_ratios, 95),
                "reach_ratio_max": float(np.max(reach_ratios)) if reach_ratios else float("nan"),
                "hint_mid_delta_p95_m": self._safe_percentile(mid_delta_values, 95),
                "hint_mid_delta_max_m": float(np.max(mid_delta_values)) if mid_delta_values else float("nan"),
            })
        return hint_targets, summary, delta_trace

    def _compute_ai_sapiens_limb_bend_angle_objective_targets(self, targets):
        """Build general limb bend-angle targets without modifying IK targets."""
        chain_count = len(self.ai_sapiens_limb_bend_angle_data)
        frames = int(targets.shape[0])
        chain_dim = max(1, chain_count)
        target_angles = np.zeros((frames, chain_dim), dtype=np.float32)
        active_masks = np.zeros((frames, chain_dim), dtype=np.float32)
        trace = np.full((frames, chain_dim, 5), np.nan, dtype=np.float32)
        summary = {
            "enabled": bool(self.ai_sapiens_limb_bend_angle_objective_enabled),
            "chain_count": int(chain_count),
            "weight": float(self.ai_sapiens_limb_bend_angle_weight),
            "groups": list(self.ai_sapiens_limb_bend_angle_groups),
            "start_after_warmup": bool(self.ai_sapiens_limb_bend_angle_start_after_warmup),
            "chains": [],
            "skipped": False,
            "skip_reason": None,
        }
        if not self.ai_sapiens_limb_bend_angle_data:
            summary["skipped"] = True
            summary["skip_reason"] = "no_valid_limb_chains"
            return target_angles, active_masks, trace, summary

        raw = np.asarray(targets, dtype=np.float64)
        start_frame = 0
        if self.ai_sapiens_limb_bend_angle_start_after_warmup:
            start_frame = int(self.num_initialization_frames + self.num_stabilization_frames)
        summary["start_frame"] = int(start_frame)
        output_frame_count = max(0, int(raw.shape[0]) - start_frame)
        if self.ai_sapiens_limb_bend_angle_skip_single_output_frame and output_frame_count <= 1:
            summary["skipped"] = True
            summary["skip_reason"] = "single_output_frame_static_gate_guard"
            return target_angles, active_masks, trace, summary
        if start_frame >= int(raw.shape[0]):
            summary["skipped"] = True
            summary["skip_reason"] = "no_output_frames"
            return target_angles, active_masks, trace, summary

        eps = 1.0e-8
        for chain_idx, chain in enumerate(self.ai_sapiens_limb_bend_angle_data):
            root_idx = int(chain["root_idx"])
            mid_idx = int(chain["mid_idx"])
            end_idx = int(chain["end_idx"])
            source_angles = []
            root_end_distances = []
            active_count = 0
            for frame in range(frames):
                root = raw[frame, root_idx, 0:3]
                mid = raw[frame, mid_idx, 0:3]
                end = raw[frame, end_idx, 0:3]
                v0 = root - mid
                v1 = end - mid
                n0 = float(np.linalg.norm(v0))
                n1 = float(np.linalg.norm(v1))
                if n0 <= eps or n1 <= eps:
                    angle_rad = np.pi
                    angle_deg = 180.0
                else:
                    c = float(np.clip(np.dot(v0, v1) / (n0 * n1), -1.0, 1.0))
                    angle_rad = float(np.arccos(c))
                    angle_deg = float(np.degrees(angle_rad))
                active = frame >= int(start_frame)
                target_angles[frame, chain_idx] = np.float32(angle_rad)
                active_masks[frame, chain_idx] = np.float32(1.0 if active else 0.0)
                root_end_distance = float(np.linalg.norm(end - root))
                trace[frame, chain_idx, 0] = np.float32(angle_deg)
                trace[frame, chain_idx, 1] = np.float32(root_end_distance)
                trace[frame, chain_idx, 2] = np.float32(1.0 if active else 0.0)
                trace[frame, chain_idx, 3] = np.float32(float(chain["upper_len"]))
                trace[frame, chain_idx, 4] = np.float32(float(chain["lower_len"]))
                if active:
                    active_count += 1
                    source_angles.append(angle_deg)
                    root_end_distances.append(root_end_distance)
            summary["chains"].append({
                "root": str(chain["root_name"]),
                "mid": str(chain["mid_name"]),
                "end": str(chain["end_name"]),
                "active_frames": int(active_count),
                "source_angle_p50_deg": self._safe_percentile(source_angles, 50),
                "source_angle_p95_deg": self._safe_percentile(source_angles, 95),
                "root_end_distance_p95_m": self._safe_percentile(root_end_distances, 95),
                "upper_len_m": float(chain["upper_len"]),
                "lower_len_m": float(chain["lower_len"]),
            })
        return target_angles, active_masks, trace, summary

    def _compute_ai_sapiens_limb_plane_normal_objective_targets(self, targets):
        """Build source limb bend-plane normal targets without editing IK targets."""
        chain_count = len(self.ai_sapiens_limb_plane_normal_data)
        frames = int(targets.shape[0])
        chain_dim = max(1, chain_count)
        target_normals = np.zeros((frames, chain_dim, 3), dtype=np.float32)
        active_masks = np.zeros((frames, chain_dim), dtype=np.float32)
        trace = np.full((frames, chain_dim, 8), np.nan, dtype=np.float32)
        summary = {
            "enabled": bool(self.ai_sapiens_limb_plane_normal_objective_enabled),
            "chain_count": int(chain_count),
            "weight": float(self.ai_sapiens_limb_plane_normal_weight),
            "groups": list(self.ai_sapiens_limb_plane_normal_groups),
            "start_after_warmup": bool(self.ai_sapiens_limb_plane_normal_start_after_warmup),
            "chains": [],
            "skipped": False,
            "skip_reason": None,
        }
        if not self.ai_sapiens_limb_plane_normal_data:
            summary["skipped"] = True
            summary["skip_reason"] = "no_valid_limb_chains"
            return target_normals, active_masks, trace, summary

        raw = np.asarray(targets, dtype=np.float64)
        start_frame = 0
        if self.ai_sapiens_limb_plane_normal_start_after_warmup:
            start_frame = int(self.num_initialization_frames + self.num_stabilization_frames)
        summary["start_frame"] = int(start_frame)
        output_frame_count = max(0, int(raw.shape[0]) - start_frame)
        if self.ai_sapiens_limb_plane_normal_skip_single_output_frame and output_frame_count <= 1:
            summary["skipped"] = True
            summary["skip_reason"] = "single_output_frame_static_gate_guard"
            return target_normals, active_masks, trace, summary
        if start_frame >= int(raw.shape[0]):
            summary["skipped"] = True
            summary["skip_reason"] = "no_output_frames"
            return target_normals, active_masks, trace, summary

        eps = 1.0e-8
        for chain_idx, chain in enumerate(self.ai_sapiens_limb_plane_normal_data):
            root_idx = int(chain["root_idx"])
            mid_idx = int(chain["mid_idx"])
            end_idx = int(chain["end_idx"])
            active_count = 0
            plane_norms = []
            source_angles = []
            for frame in range(frames):
                root = raw[frame, root_idx, 0:3]
                mid = raw[frame, mid_idx, 0:3]
                end = raw[frame, end_idx, 0:3]
                upper = mid - root
                lower = end - mid
                normal_raw = np.cross(upper, lower)
                normal_len = float(np.linalg.norm(normal_raw))
                active = bool(frame >= int(start_frame) and normal_len > eps)
                if active:
                    normal = normal_raw / normal_len
                    target_normals[frame, chain_idx, :] = normal.astype(np.float32)
                    active_masks[frame, chain_idx] = np.float32(1.0)
                    active_count += 1
                    plane_norms.append(normal_len)
                else:
                    normal = np.zeros(3, dtype=np.float64)

                n0 = float(np.linalg.norm(upper))
                n1 = float(np.linalg.norm(lower))
                if n0 <= eps or n1 <= eps:
                    source_angle_deg = 180.0
                else:
                    c = float(np.clip(np.dot(-upper, lower) / (n0 * n1), -1.0, 1.0))
                    source_angle_deg = float(np.degrees(np.arccos(c)))
                if active:
                    source_angles.append(source_angle_deg)
                trace[frame, chain_idx, 0:3] = normal.astype(np.float32)
                trace[frame, chain_idx, 3] = np.float32(normal_len)
                trace[frame, chain_idx, 4] = np.float32(1.0 if active else 0.0)
                trace[frame, chain_idx, 5] = np.float32(source_angle_deg)
                trace[frame, chain_idx, 6] = np.float32(float(chain["upper_len"]))
                trace[frame, chain_idx, 7] = np.float32(float(chain["lower_len"]))
            summary["chains"].append({
                "root": str(chain["root_name"]),
                "mid": str(chain["mid_name"]),
                "end": str(chain["end_name"]),
                "active_frames": int(active_count),
                "plane_norm_p05": self._safe_percentile(plane_norms, 5),
                "plane_norm_p50": self._safe_percentile(plane_norms, 50),
                "source_angle_p50_deg": self._safe_percentile(source_angles, 50),
                "source_angle_p95_deg": self._safe_percentile(source_angles, 95),
                "upper_len_m": float(chain["upper_len"]),
                "lower_len_m": float(chain["lower_len"]),
            })
        return target_normals, active_masks, trace, summary

    def _compute_ai_sapiens_limb_midpoint_position_objective_targets(self, targets):
        """Build generic elbow/knee midpoint position hints without editing IK targets."""
        chain_count = len(self.ai_sapiens_limb_midpoint_position_data)
        frames = int(targets.shape[0])
        chain_dim = max(1, chain_count)
        midpoint_targets = np.full((frames, chain_dim, 3), np.nan, dtype=np.float32)
        active_masks = np.zeros((frames, chain_dim), dtype=np.float32)
        trace = np.full((frames, chain_dim, 10), np.nan, dtype=np.float32)
        summary = {
            "enabled": bool(self.ai_sapiens_limb_midpoint_position_objective_enabled),
            "chain_count": int(chain_count),
            "weight": float(self.ai_sapiens_limb_midpoint_position_weight),
            "max_delta_m": float(self.ai_sapiens_limb_midpoint_position_max_delta_m),
            "groups": list(self.ai_sapiens_limb_midpoint_position_groups),
            "group_weights": dict(self.ai_sapiens_limb_midpoint_position_group_weights),
            "suppress_contact_risk": bool(self.ai_sapiens_limb_midpoint_position_suppress_contact_risk),
            "contact_risk_distance_m": float(self.ai_sapiens_limb_midpoint_position_contact_risk_distance_m),
            "contact_suppression_groups": list(
                self.ai_sapiens_limb_midpoint_position_contact_suppression_groups
            ),
            "start_after_warmup": bool(self.ai_sapiens_limb_midpoint_position_start_after_warmup),
            "chains": [],
            "skipped": False,
            "skip_reason": None,
        }
        if not self.ai_sapiens_limb_midpoint_position_data:
            summary["skipped"] = True
            summary["skip_reason"] = "no_valid_limb_chains"
            return midpoint_targets, active_masks, trace, summary

        raw = np.asarray(targets, dtype=np.float64)
        start_frame = 0
        if self.ai_sapiens_limb_midpoint_position_start_after_warmup:
            start_frame = int(self.num_initialization_frames + self.num_stabilization_frames)
        summary["start_frame"] = int(start_frame)
        output_frame_count = max(0, int(raw.shape[0]) - start_frame)
        if self.ai_sapiens_limb_midpoint_position_skip_single_output_frame and output_frame_count <= 1:
            summary["skipped"] = True
            summary["skip_reason"] = "single_output_frame_static_gate_guard"
            return midpoint_targets, active_masks, trace, summary
        if start_frame >= int(raw.shape[0]):
            summary["skipped"] = True
            summary["skip_reason"] = "no_output_frames"
            return midpoint_targets, active_masks, trace, summary

        eps = 1.0e-8
        max_delta = max(0.0, float(self.ai_sapiens_limb_midpoint_position_max_delta_m))
        suppress_contact = bool(self.ai_sapiens_limb_midpoint_position_suppress_contact_risk)
        contact_distance = max(
            0.0,
            float(self.ai_sapiens_limb_midpoint_position_contact_risk_distance_m),
        )
        suppression_groups = {
            str(group).lower()
            for group in self.ai_sapiens_limb_midpoint_position_contact_suppression_groups
        }
        target_name_to_idx = {name: idx for idx, name in enumerate(self.mapped_joints)}
        hips_idx = target_name_to_idx.get("Hips")
        left_leg_idx = target_name_to_idx.get("LeftLeg")
        right_leg_idx = target_name_to_idx.get("RightLeg")
        for chain_idx, chain in enumerate(self.ai_sapiens_limb_midpoint_position_data):
            root_idx = int(chain["root_idx"])
            mid_idx = int(chain["mid_idx"])
            end_idx = int(chain["end_idx"])
            root_name = str(chain["root_name"])
            is_left = root_name.lower().startswith("left")
            is_arm = root_name in {"LeftArm", "RightArm"}
            is_leg = root_name in {"LeftLeg", "RightLeg"}
            group_name = "arm" if is_arm else "leg" if is_leg else "other"
            chain_weight = float(
                self.ai_sapiens_limb_midpoint_position_group_weights.get(
                    group_name,
                    self.ai_sapiens_limb_midpoint_position_weight,
                )
            )
            risk_indices = []
            if hips_idx is not None and is_arm:
                risk_indices.append(int(hips_idx))
            side_leg_idx = left_leg_idx if is_left else right_leg_idx
            if side_leg_idx is not None and is_arm:
                risk_indices.append(int(side_leg_idx))
            upper_len = float(chain["upper_len"])
            lower_len = float(chain["lower_len"])
            max_reach = max(eps, upper_len + lower_len)
            min_reach = max(eps, abs(upper_len - lower_len) + 0.010)
            fallback_bend_dir = np.asarray(chain["fallback_bend_dir"], dtype=np.float64)
            fallback_bend_dir = fallback_bend_dir / max(float(np.linalg.norm(fallback_bend_dir)), eps)
            source_angles = []
            delta_values = []
            risk_distances = []
            active_count = 0
            suppressed_count = 0
            for frame in range(frames):
                root = raw[frame, root_idx, 0:3]
                mid = raw[frame, mid_idx, 0:3]
                end = raw[frame, end_idx, 0:3]
                root_to_end = end - root
                raw_distance = float(np.linalg.norm(root_to_end))
                if raw_distance <= eps:
                    end_dir = np.asarray(chain["fallback_end_dir"], dtype=np.float64)
                    end_dir = end_dir / max(float(np.linalg.norm(end_dir)), eps)
                else:
                    end_dir = root_to_end / raw_distance

                bend_vec = mid - root - np.dot(mid - root, end_dir) * end_dir
                bend_norm = float(np.linalg.norm(bend_vec))
                bend_dir = bend_vec / bend_norm if bend_norm > eps else fallback_bend_dir
                bend_dir = bend_dir / max(float(np.linalg.norm(bend_dir)), eps)

                d = float(np.clip(raw_distance, min_reach, max_reach - 1.0e-6))
                a = (upper_len * upper_len - lower_len * lower_len + d * d) / max(2.0 * d, eps)
                h = float(np.sqrt(max(0.0, upper_len * upper_len - a * a)))
                hint_mid = root + end_dir * a + bend_dir * h
                delta_vec = hint_mid - mid
                delta = float(np.linalg.norm(delta_vec))
                if max_delta > 0.0 and delta > max_delta:
                    hint_mid = mid + delta_vec * (max_delta / max(delta, eps))
                    delta = max_delta

                v0 = root - mid
                v1 = end - mid
                n0 = float(np.linalg.norm(v0))
                n1 = float(np.linalg.norm(v1))
                if n0 <= eps or n1 <= eps:
                    source_angle_deg = 180.0
                else:
                    c = float(np.clip(np.dot(v0, v1) / (n0 * n1), -1.0, 1.0))
                    source_angle_deg = float(np.degrees(np.arccos(c)))

                risk_distance = float("inf")
                if risk_indices:
                    risk_positions = raw[frame, risk_indices, 0:3]
                    mid_distances = np.linalg.norm(risk_positions - mid[None, :], axis=1)
                    end_distances = np.linalg.norm(risk_positions - end[None, :], axis=1)
                    risk_distance = float(min(np.min(mid_distances), np.min(end_distances)))
                contact_suppressed = bool(
                    suppress_contact
                    and group_name in suppression_groups
                    and risk_distance <= contact_distance
                )
                active = bool(frame >= int(start_frame) and not contact_suppressed)
                midpoint_targets[frame, chain_idx, :] = hint_mid.astype(np.float32)
                active_masks[frame, chain_idx] = np.float32(1.0 if active else 0.0)
                trace[frame, chain_idx, 0] = np.float32(source_angle_deg)
                trace[frame, chain_idx, 1] = np.float32(raw_distance)
                trace[frame, chain_idx, 2] = np.float32(max_reach)
                trace[frame, chain_idx, 3] = np.float32(delta)
                trace[frame, chain_idx, 4] = np.float32(1.0 if active else 0.0)
                trace[frame, chain_idx, 5] = np.float32(upper_len)
                trace[frame, chain_idx, 6] = np.float32(lower_len)
                trace[frame, chain_idx, 7] = np.float32(bend_norm)
                trace[frame, chain_idx, 8] = np.float32(risk_distance)
                trace[frame, chain_idx, 9] = np.float32(1.0 if contact_suppressed else 0.0)
                if np.isfinite(risk_distance):
                    risk_distances.append(risk_distance)
                if contact_suppressed:
                    suppressed_count += 1
                if active:
                    active_count += 1
                    source_angles.append(source_angle_deg)
                    delta_values.append(delta)
            summary["chains"].append({
                "root": str(chain["root_name"]),
                "mid": str(chain["mid_name"]),
                "end": str(chain["end_name"]),
                "group": group_name,
                "weight": float(chain_weight),
                "active_frames": int(active_count),
                "contact_suppressed_frames": int(suppressed_count),
                "contact_risk_distance_p05_m": self._safe_percentile(risk_distances, 5),
                "source_angle_p50_deg": self._safe_percentile(source_angles, 50),
                "source_angle_p95_deg": self._safe_percentile(source_angles, 95),
                "mid_delta_p95_m": self._safe_percentile(delta_values, 95),
                "upper_len_m": float(upper_len),
                "lower_len_m": float(lower_len),
            })
        return midpoint_targets, active_masks, trace, summary

    def _target_torso_frame_matrix(self, raw, frame, torso_name):
        target_name_to_idx = {name: idx for idx, name in enumerate(self.mapped_joints)}
        hips_idx = target_name_to_idx.get("Hips")
        chest_idx = target_name_to_idx.get("Chest")
        if hips_idx is None or chest_idx is None:
            return np.eye(3, dtype=np.float64)

        hips = raw[frame, int(hips_idx), 0:3]
        chest = raw[frame, int(chest_idx), 0:3]
        up = self._normalize_np(chest - hips, fallback=[0.0, 0.0, 1.0])

        lateral = None
        torso_key = str(torso_name)
        if torso_key == "Chest":
            left_idx = target_name_to_idx.get("LeftArm")
            right_idx = target_name_to_idx.get("RightArm")
            if left_idx is not None and right_idx is not None:
                lateral = raw[frame, int(left_idx), 0:3] - raw[frame, int(right_idx), 0:3]
        else:
            left_idx = target_name_to_idx.get("LeftLeg")
            right_idx = target_name_to_idx.get("RightLeg")
            if left_idx is not None and right_idx is not None:
                lateral = raw[frame, int(left_idx), 0:3] - raw[frame, int(right_idx), 0:3]
        if lateral is None or float(np.linalg.norm(lateral)) <= 1.0e-9:
            lateral = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        lateral = lateral - np.dot(lateral, up) * up
        lateral = self._normalize_np(lateral, fallback=[0.0, 1.0, 0.0])
        forward = self._normalize_np(np.cross(lateral, up), fallback=[1.0, 0.0, 0.0])
        lateral = self._normalize_np(np.cross(up, forward), fallback=lateral)
        return np.column_stack([forward, lateral, up])

    def _compute_ai_sapiens_torso_local_limb_midpoint_offsets(self, targets):
        """Build elbow/knee midpoint offsets in a source torso-local frame."""
        chain_count = len(self.ai_sapiens_torso_local_limb_midpoint_data)
        frames = int(targets.shape[0])
        chain_dim = max(1, chain_count)
        local_offsets = np.full((frames, chain_dim, 3), np.nan, dtype=np.float32)
        active_masks = np.zeros((frames, chain_dim), dtype=np.float32)
        trace = np.full((frames, chain_dim, 8), np.nan, dtype=np.float32)
        summary = {
            "enabled": bool(self.ai_sapiens_torso_local_limb_midpoint_objective_enabled),
            "chain_count": int(chain_count),
            "weight": float(self.ai_sapiens_torso_local_limb_midpoint_weight),
            "max_delta_m": float(self.ai_sapiens_torso_local_limb_midpoint_max_delta_m),
            "offset_scale": float(self.ai_sapiens_torso_local_limb_midpoint_offset_scale),
            "groups": list(self.ai_sapiens_torso_local_limb_midpoint_groups),
            "group_weights": dict(self.ai_sapiens_torso_local_limb_midpoint_group_weights),
            "start_after_warmup": bool(self.ai_sapiens_torso_local_limb_midpoint_start_after_warmup),
            "chains": [],
            "skipped": False,
            "skip_reason": None,
        }
        if not self.ai_sapiens_torso_local_limb_midpoint_data:
            summary["skipped"] = True
            summary["skip_reason"] = "no_valid_limb_chains"
            return local_offsets, active_masks, trace, summary

        raw = np.asarray(targets, dtype=np.float64)
        start_frame = 0
        if self.ai_sapiens_torso_local_limb_midpoint_start_after_warmup:
            start_frame = int(self.num_initialization_frames + self.num_stabilization_frames)
        summary["start_frame"] = int(start_frame)
        output_frame_count = max(0, int(raw.shape[0]) - start_frame)
        if self.ai_sapiens_torso_local_limb_midpoint_skip_single_output_frame and output_frame_count <= 1:
            summary["skipped"] = True
            summary["skip_reason"] = "single_output_frame_static_gate_guard"
            return local_offsets, active_masks, trace, summary
        if start_frame >= int(raw.shape[0]):
            summary["skipped"] = True
            summary["skip_reason"] = "no_output_frames"
            return local_offsets, active_masks, trace, summary

        local_norms = []
        for chain_idx, chain in enumerate(self.ai_sapiens_torso_local_limb_midpoint_data):
            mid_idx = int(chain["mid_idx"])
            torso_idx = int(chain["torso_idx"])
            root_name = str(chain["root_name"])
            torso_name = str(chain["torso_name"])
            if root_name in {"LeftArm", "RightArm"}:
                group_name = "arm"
            elif root_name in {"LeftLeg", "RightLeg"}:
                group_name = "leg"
            else:
                group_name = "other"
            active_count = 0
            chain_norms = []
            for frame in range(frames):
                torso = raw[frame, torso_idx, 0:3]
                mid = raw[frame, mid_idx, 0:3]
                frame_matrix = self._target_torso_frame_matrix(raw, frame, torso_name)
                offset_world = mid - torso
                local = frame_matrix.T @ offset_world
                norm = float(np.linalg.norm(local))
                active = bool(frame >= int(start_frame) and np.isfinite(local).all())
                if active:
                    local_offsets[frame, chain_idx, :] = local.astype(np.float32)
                    active_masks[frame, chain_idx] = np.float32(1.0)
                    active_count += 1
                    chain_norms.append(norm)
                    local_norms.append(norm)
                trace[frame, chain_idx, 0:3] = local.astype(np.float32)
                trace[frame, chain_idx, 3] = np.float32(norm)
                trace[frame, chain_idx, 4] = np.float32(1.0 if active else 0.0)
                trace[frame, chain_idx, 5] = np.float32(float(chain["upper_len"]))
                trace[frame, chain_idx, 6] = np.float32(float(chain["lower_len"]))
                trace[frame, chain_idx, 7] = np.float32(0.0 if group_name == "arm" else 1.0)
            summary["chains"].append({
                "root": root_name,
                "mid": str(chain["mid_name"]),
                "end": str(chain["end_name"]),
                "torso": torso_name,
                "group": group_name,
                "active_frames": int(active_count),
                "local_offset_norm_p50_m": self._safe_percentile(chain_norms, 50),
                "local_offset_norm_p95_m": self._safe_percentile(chain_norms, 95),
            })
        summary["local_offset_norm_p95_m"] = self._safe_percentile(local_norms, 95)
        return local_offsets, active_masks, trace, summary

    def _compute_ai_sapiens_bilateral_arm_bend_objective_targets(self, targets):
        """Build source-state gated bend-angle and weak wrist-reference traces."""
        chain_count = len(self.ai_sapiens_bilateral_arm_bend_data)
        frames = int(targets.shape[0])
        chain_dim = max(1, chain_count)
        target_angles = np.zeros((frames, chain_dim), dtype=np.float32)
        active_masks = np.zeros((frames, chain_dim), dtype=np.float32)
        bend_trace = np.full((frames, chain_dim, 8), np.nan, dtype=np.float32)
        wrist_targets = np.full((frames, chain_dim, 3), np.nan, dtype=np.float32)
        wrist_active = np.zeros((frames, chain_dim), dtype=np.float32)
        wrist_trace = np.full((frames, chain_dim, 7), np.nan, dtype=np.float32)
        bend_summary = {
            "enabled": bool(self.ai_sapiens_bilateral_arm_bend_objective_enabled),
            "chain_count": int(chain_count),
            "weight": float(self.ai_sapiens_arm_bend_weight),
            "source_active_deg": float(self.ai_sapiens_arm_bend_source_active_deg),
            "overextended_deg": float(self.ai_sapiens_arm_bend_overextended_deg),
            "error_trigger_deg": float(self.ai_sapiens_arm_bend_error_trigger_deg),
            "require_contact_risk": bool(self.ai_sapiens_arm_bend_require_contact_risk),
            "contact_risk_distance_m": float(self.ai_sapiens_arm_bend_contact_risk_distance_m),
            "chains": [],
            "skipped": False,
            "skip_reason": None,
        }
        wrist_summary = {
            "enabled": bool(self.ai_sapiens_soft_bend_wrist_reference_enabled),
            "chain_count": int(chain_count),
            "weight": float(self.ai_sapiens_soft_bend_wrist_reference_weight),
            "max_delta_m": float(self.ai_sapiens_soft_bend_wrist_reference_max_delta_m),
            "chains": [],
            "skipped": False,
            "skip_reason": None,
        }
        if not self.ai_sapiens_bilateral_arm_bend_data:
            bend_summary["skipped"] = True
            bend_summary["skip_reason"] = "no_valid_arm_chains"
            wrist_summary["skipped"] = True
            wrist_summary["skip_reason"] = "no_valid_arm_chains"
            return (
                target_angles,
                active_masks,
                bend_trace,
                wrist_targets,
                wrist_active,
                wrist_trace,
                bend_summary,
                wrist_summary,
            )

        raw = np.asarray(targets, dtype=np.float64)
        start_frame, should_skip, skip_reason = self._projection_start_and_skip(raw)
        if should_skip:
            bend_summary["skipped"] = True
            bend_summary["skip_reason"] = skip_reason
            wrist_summary["skipped"] = True
            wrist_summary["skip_reason"] = skip_reason
            return (
                target_angles,
                active_masks,
                bend_trace,
                wrist_targets,
                wrist_active,
                wrist_trace,
                bend_summary,
                wrist_summary,
            )

        source_active_deg = float(self.ai_sapiens_arm_bend_source_active_deg)
        contact_risk_required = bool(self.ai_sapiens_arm_bend_require_contact_risk)
        contact_risk_distance = max(0.0, float(self.ai_sapiens_arm_bend_contact_risk_distance_m))
        target_name_to_idx = {name: idx for idx, name in enumerate(self.mapped_joints)}
        hips_idx = target_name_to_idx.get("Hips")
        max_delta_m = max(0.0, float(self.ai_sapiens_soft_bend_wrist_reference_max_delta_m))
        eps = 1.0e-8
        for chain_idx, chain in enumerate(self.ai_sapiens_bilateral_arm_bend_data):
            root_idx = int(chain["root_idx"])
            mid_idx = int(chain["mid_idx"])
            end_idx = int(chain["end_idx"])
            upper_len = float(chain["upper_len"])
            lower_len = float(chain["lower_len"])
            max_reach = max(eps, upper_len + lower_len)
            root_name = str(chain["root_name"])
            side = "left" if root_name.lower().startswith("left") else "right"
            side_leg_idx = target_name_to_idx.get("LeftLeg" if side == "left" else "RightLeg")
            risk_indices = [idx for idx in (hips_idx, side_leg_idx) if idx is not None]
            source_angles = []
            raw_distances = []
            contact_distances = []
            active_count = 0
            wrist_active_count = 0
            wrist_delta_values = []
            for frame in range(frames):
                root = raw[frame, root_idx, 0:3]
                mid = raw[frame, mid_idx, 0:3]
                end = raw[frame, end_idx, 0:3]
                v0 = root - mid
                v1 = end - mid
                n0 = float(np.linalg.norm(v0))
                n1 = float(np.linalg.norm(v1))
                if n0 <= eps or n1 <= eps:
                    source_angle_deg = 180.0
                    source_angle_rad = np.pi
                else:
                    c = float(np.clip(np.dot(v0, v1) / (n0 * n1), -1.0, 1.0))
                    source_angle_rad = float(np.arccos(c))
                    source_angle_deg = float(np.degrees(source_angle_rad))
                target_angles[frame, chain_idx] = np.float32(source_angle_rad)
                raw_distance = float(np.linalg.norm(end - root))
                risk_distance = float("inf")
                if risk_indices:
                    risk_positions = raw[frame, risk_indices, 0:3]
                    risk_distance = float(np.nanmin(np.linalg.norm(risk_positions - end[None, :], axis=1)))
                source_active = (
                    frame >= int(start_frame)
                    and source_angle_deg <= source_active_deg
                )
                near_max = raw_distance >= (max_reach - 0.010)
                contact_risk = risk_distance <= contact_risk_distance
                active = bool(source_active and near_max and (contact_risk or not contact_risk_required))
                if active:
                    active_count += 1
                    active_masks[frame, chain_idx] = np.float32(1.0)

                d_bend = float(
                    np.sqrt(
                        max(
                            0.0,
                            upper_len * upper_len
                            + lower_len * lower_len
                            - 2.0 * upper_len * lower_len * np.cos(source_angle_rad),
                        )
                    )
                )
                desired = end.copy()
                wrist_delta = 0.0
                wrist_ref_active = bool(active and max_delta_m > 0.0)
                if wrist_ref_active:
                    direction = end - root
                    norm = float(np.linalg.norm(direction))
                    if norm <= eps:
                        fallback = np.asarray(chain["fallback_end_dir"], dtype=np.float64)
                        direction = fallback
                        norm = float(np.linalg.norm(direction))
                    direction = direction / max(norm, eps)
                    unconstrained = root + direction * d_bend
                    delta_vec = unconstrained - end
                    delta_norm = float(np.linalg.norm(delta_vec))
                    if delta_norm > max_delta_m:
                        delta_vec *= max_delta_m / max(delta_norm, eps)
                    desired = end + delta_vec
                    wrist_delta = float(np.linalg.norm(delta_vec))
                    wrist_active[frame, chain_idx] = np.float32(1.0)
                    wrist_active_count += 1
                    wrist_delta_values.append(wrist_delta)
                wrist_targets[frame, chain_idx, :] = desired.astype(np.float32)
                bend_trace[frame, chain_idx, 0] = np.float32(source_angle_deg)
                bend_trace[frame, chain_idx, 1] = np.float32(raw_distance)
                bend_trace[frame, chain_idx, 2] = np.float32(max_reach)
                bend_trace[frame, chain_idx, 3] = np.float32(1.0 if near_max else 0.0)
                bend_trace[frame, chain_idx, 4] = np.float32(1.0 if active else 0.0)
                bend_trace[frame, chain_idx, 5] = np.float32(d_bend)
                bend_trace[frame, chain_idx, 6] = np.float32(wrist_delta)
                bend_trace[frame, chain_idx, 7] = np.float32(1.0 if wrist_ref_active else 0.0)
                wrist_trace[frame, chain_idx, 0] = np.float32(source_angle_deg)
                wrist_trace[frame, chain_idx, 1] = np.float32(raw_distance)
                wrist_trace[frame, chain_idx, 2] = np.float32(max_reach)
                wrist_trace[frame, chain_idx, 3] = np.float32(d_bend)
                wrist_trace[frame, chain_idx, 4] = np.float32(wrist_delta)
                wrist_trace[frame, chain_idx, 5] = np.float32(1.0 if wrist_ref_active else 0.0)
                wrist_trace[frame, chain_idx, 6] = np.float32(max_delta_m)
                if frame >= int(start_frame):
                    source_angles.append(source_angle_deg)
                    raw_distances.append(raw_distance)
                    contact_distances.append(risk_distance)
            bend_summary["chains"].append({
                "root": str(chain["root_name"]),
                "mid": str(chain["mid_name"]),
                "end": str(chain["end_name"]),
                "active_frames": int(active_count),
                "source_angle_p50_deg": self._safe_percentile(source_angles, 50),
                "source_angle_p95_deg": self._safe_percentile(source_angles, 95),
                "raw_root_end_distance_p95_m": self._safe_percentile(raw_distances, 95),
                "max_reach_m": float(max_reach),
                "contact_risk_distance_p05_m": self._safe_percentile(contact_distances, 5),
            })
            wrist_summary["chains"].append({
                "root": str(chain["root_name"]),
                "mid": str(chain["mid_name"]),
                "end": str(chain["end_name"]),
                "active_frames": int(wrist_active_count),
                "wrist_delta_p95_m": self._safe_percentile(wrist_delta_values, 95),
                "wrist_delta_max_m": (
                    float(np.max(wrist_delta_values)) if wrist_delta_values else float("nan")
                ),
            })
        return (
            target_angles,
            active_masks,
            bend_trace,
            wrist_targets,
            wrist_active,
            wrist_trace,
            bend_summary,
            wrist_summary,
        )

    def _compute_ai_sapiens_contact_gated_arm_references(self, targets):
        """Build contact-risk gated elbow midpoint hints and wrist micro references.

        These are weak auxiliary IK objectives. They do not rewrite raw
        hand/forearm targets; the frame mask is based only on source/target
        state available before IK solve.
        """
        chain_count = len(self.ai_sapiens_bilateral_arm_bend_data)
        frames = int(targets.shape[0])
        chain_dim = max(1, chain_count)
        elbow_targets = np.full((frames, chain_dim, 3), np.nan, dtype=np.float32)
        elbow_active = np.zeros((frames, chain_dim), dtype=np.float32)
        elbow_trace = np.full((frames, chain_dim, 10), np.nan, dtype=np.float32)
        wrist_targets = np.full((frames, chain_dim, 3), np.nan, dtype=np.float32)
        wrist_active = np.zeros((frames, chain_dim), dtype=np.float32)
        wrist_trace = np.full((frames, chain_dim, 8), np.nan, dtype=np.float32)
        elbow_summary = {
            "enabled": bool(self.ai_sapiens_contact_gated_elbow_midpoint_hint_enabled),
            "chain_count": int(chain_count),
            "weight": float(self.ai_sapiens_contact_gated_elbow_hint_weight),
            "source_active_deg": float(self.ai_sapiens_contact_gated_elbow_source_active_deg),
            "near_max_margin_m": float(self.ai_sapiens_contact_gated_elbow_near_max_margin_m),
            "contact_risk_distance_m": float(self.ai_sapiens_contact_gated_elbow_contact_risk_distance_m),
            "max_mid_delta_m": float(self.ai_sapiens_contact_gated_elbow_max_mid_delta_m),
            "chains": [],
            "skipped": False,
            "skip_reason": None,
        }
        wrist_summary = {
            "enabled": bool(self.ai_sapiens_contact_micro_wrist_reference_enabled),
            "chain_count": int(chain_count),
            "weight": float(self.ai_sapiens_contact_micro_wrist_reference_weight),
            "max_delta_m": float(self.ai_sapiens_contact_micro_wrist_reference_max_delta_m),
            "chains": [],
            "skipped": False,
            "skip_reason": None,
        }
        if not self.ai_sapiens_bilateral_arm_bend_data:
            elbow_summary["skipped"] = True
            elbow_summary["skip_reason"] = "no_valid_arm_chains"
            wrist_summary["skipped"] = True
            wrist_summary["skip_reason"] = "no_valid_arm_chains"
            return (
                elbow_targets,
                elbow_active,
                elbow_trace,
                wrist_targets,
                wrist_active,
                wrist_trace,
                elbow_summary,
                wrist_summary,
            )

        raw = np.asarray(targets, dtype=np.float64)
        start_frame, should_skip, skip_reason = self._projection_start_and_skip(raw)
        if should_skip:
            elbow_summary["skipped"] = True
            elbow_summary["skip_reason"] = skip_reason
            wrist_summary["skipped"] = True
            wrist_summary["skip_reason"] = skip_reason
            return (
                elbow_targets,
                elbow_active,
                elbow_trace,
                wrist_targets,
                wrist_active,
                wrist_trace,
                elbow_summary,
                wrist_summary,
            )

        target_name_to_idx = {name: idx for idx, name in enumerate(self.mapped_joints)}
        hips_idx = target_name_to_idx.get("Hips")
        eps = 1.0e-8
        source_active_deg = float(self.ai_sapiens_contact_gated_elbow_source_active_deg)
        near_max_margin = max(0.0, float(self.ai_sapiens_contact_gated_elbow_near_max_margin_m))
        contact_risk_distance = max(0.0, float(self.ai_sapiens_contact_gated_elbow_contact_risk_distance_m))
        max_mid_delta = max(0.0, float(self.ai_sapiens_contact_gated_elbow_max_mid_delta_m))
        max_wrist_delta = max(0.0, float(self.ai_sapiens_contact_micro_wrist_reference_max_delta_m))

        for chain_idx, chain in enumerate(self.ai_sapiens_bilateral_arm_bend_data):
            root_idx = int(chain["root_idx"])
            mid_idx = int(chain["mid_idx"])
            end_idx = int(chain["end_idx"])
            root_name = str(chain["root_name"])
            side = "left" if root_name.lower().startswith("left") else "right"
            side_leg_idx = target_name_to_idx.get("LeftLeg" if side == "left" else "RightLeg")
            risk_indices = [idx for idx in (hips_idx, side_leg_idx) if idx is not None]
            upper_len = float(chain["upper_len"])
            lower_len = float(chain["lower_len"])
            max_reach = max(eps, upper_len + lower_len)
            min_reach = max(eps, abs(upper_len - lower_len) + 0.010)
            fallback_bend_dir = np.asarray(chain["fallback_bend_dir"], dtype=np.float64)
            active_count = 0
            wrist_active_count = 0
            source_angles = []
            contact_distances = []
            mid_delta_values = []
            wrist_delta_values = []
            for frame in range(frames):
                root = raw[frame, root_idx, 0:3]
                mid = raw[frame, mid_idx, 0:3]
                end = raw[frame, end_idx, 0:3]
                v0 = root - mid
                v1 = end - mid
                n0 = float(np.linalg.norm(v0))
                n1 = float(np.linalg.norm(v1))
                if n0 <= eps or n1 <= eps:
                    source_angle_deg = 180.0
                else:
                    c = float(np.clip(np.dot(v0, v1) / (n0 * n1), -1.0, 1.0))
                    source_angle_deg = float(np.degrees(np.arccos(c)))

                root_to_end = end - root
                raw_distance = float(np.linalg.norm(root_to_end))
                end_dir = root_to_end / max(raw_distance, eps)
                if raw_distance <= eps:
                    end_dir = np.asarray(chain["fallback_end_dir"], dtype=np.float64)
                    end_dir = end_dir / max(float(np.linalg.norm(end_dir)), eps)

                bend_vec = mid - root - np.dot(mid - root, end_dir) * end_dir
                bend_norm = float(np.linalg.norm(bend_vec))
                bend_dir = bend_vec / bend_norm if bend_norm > eps else fallback_bend_dir
                bend_dir = bend_dir / max(float(np.linalg.norm(bend_dir)), eps)

                risk_distance = float("inf")
                nearest_risk = None
                if risk_indices:
                    risk_positions = raw[frame, risk_indices, 0:3]
                    dists = np.linalg.norm(risk_positions - end[None, :], axis=1)
                    nearest = int(np.argmin(dists))
                    risk_distance = float(dists[nearest])
                    nearest_risk = risk_positions[nearest]

                source_bent = source_angle_deg <= source_active_deg
                near_max = raw_distance >= (max_reach - near_max_margin)
                contact_risk = risk_distance <= contact_risk_distance
                active = bool(frame >= int(start_frame) and source_bent and near_max and contact_risk)

                # Two-bone elbow point with AI Sapiens lengths and source bend branch.
                d = float(np.clip(raw_distance, min_reach, max_reach - 1.0e-6))
                a = (upper_len * upper_len - lower_len * lower_len + d * d) / max(2.0 * d, eps)
                h = float(np.sqrt(max(0.0, upper_len * upper_len - a * a)))
                hint_mid = root + end_dir * a + bend_dir * h
                mid_delta_vec = hint_mid - mid
                mid_delta = float(np.linalg.norm(mid_delta_vec))
                if max_mid_delta > 0.0 and mid_delta > max_mid_delta:
                    hint_mid = mid + mid_delta_vec * (max_mid_delta / max(mid_delta, eps))
                    mid_delta = max_mid_delta

                wrist_target = end.copy()
                wrist_delta = 0.0
                if active and max_wrist_delta > 0.0 and nearest_risk is not None:
                    away = end - nearest_risk
                    away_norm = float(np.linalg.norm(away))
                    if away_norm <= eps:
                        away = end_dir
                        away_norm = float(np.linalg.norm(away))
                    away = away / max(away_norm, eps)
                    wrist_target = end + away * max_wrist_delta
                    wrist_delta = float(max_wrist_delta)

                elbow_targets[frame, chain_idx, :] = hint_mid.astype(np.float32)
                wrist_targets[frame, chain_idx, :] = wrist_target.astype(np.float32)
                if active:
                    elbow_active[frame, chain_idx] = np.float32(1.0)
                    active_count += 1
                    mid_delta_values.append(mid_delta)
                    if max_wrist_delta > 0.0:
                        wrist_active[frame, chain_idx] = np.float32(1.0)
                        wrist_active_count += 1
                        wrist_delta_values.append(wrist_delta)
                elbow_trace[frame, chain_idx, 0] = np.float32(source_angle_deg)
                elbow_trace[frame, chain_idx, 1] = np.float32(raw_distance)
                elbow_trace[frame, chain_idx, 2] = np.float32(max_reach)
                elbow_trace[frame, chain_idx, 3] = np.float32(risk_distance)
                elbow_trace[frame, chain_idx, 4] = np.float32(1.0 if source_bent else 0.0)
                elbow_trace[frame, chain_idx, 5] = np.float32(1.0 if near_max else 0.0)
                elbow_trace[frame, chain_idx, 6] = np.float32(1.0 if contact_risk else 0.0)
                elbow_trace[frame, chain_idx, 7] = np.float32(1.0 if active else 0.0)
                elbow_trace[frame, chain_idx, 8] = np.float32(mid_delta)
                elbow_trace[frame, chain_idx, 9] = np.float32(max_mid_delta)
                wrist_trace[frame, chain_idx, 0] = np.float32(source_angle_deg)
                wrist_trace[frame, chain_idx, 1] = np.float32(raw_distance)
                wrist_trace[frame, chain_idx, 2] = np.float32(max_reach)
                wrist_trace[frame, chain_idx, 3] = np.float32(risk_distance)
                wrist_trace[frame, chain_idx, 4] = np.float32(1.0 if active else 0.0)
                wrist_trace[frame, chain_idx, 5] = np.float32(wrist_delta)
                wrist_trace[frame, chain_idx, 6] = np.float32(max_wrist_delta)
                wrist_trace[frame, chain_idx, 7] = np.float32(1.0 if wrist_active[frame, chain_idx] > 0.0 else 0.0)
                if frame >= int(start_frame):
                    source_angles.append(source_angle_deg)
                    contact_distances.append(risk_distance)

            elbow_summary["chains"].append({
                "root": str(chain["root_name"]),
                "mid": str(chain["mid_name"]),
                "end": str(chain["end_name"]),
                "active_frames": int(active_count),
                "source_angle_p50_deg": self._safe_percentile(source_angles, 50),
                "source_angle_p95_deg": self._safe_percentile(source_angles, 95),
                "contact_risk_distance_p05_m": self._safe_percentile(contact_distances, 5),
                "mid_delta_p95_m": self._safe_percentile(mid_delta_values, 95),
                "mid_delta_max_m": float(np.max(mid_delta_values)) if mid_delta_values else float("nan"),
            })
            wrist_summary["chains"].append({
                "root": str(chain["root_name"]),
                "mid": str(chain["mid_name"]),
                "end": str(chain["end_name"]),
                "active_frames": int(wrist_active_count),
                "wrist_delta_p95_m": self._safe_percentile(wrist_delta_values, 95),
                "wrist_delta_max_m": float(np.max(wrist_delta_values)) if wrist_delta_values else float("nan"),
            })

        return (
            elbow_targets,
            elbow_active,
            elbow_trace,
            wrist_targets,
            wrist_active,
            wrist_trace,
            elbow_summary,
            wrist_summary,
        )

    def _build_ai_sapiens_capsule_proxy_barrier_data(self):
        body_names = [
            newton_utils.get_name_from_label(label)
            for label in self.robot_builder.body_label
        ]
        body_name_to_idx = {name: idx for idx, name in enumerate(body_names)}
        target_name_to_idx = {name: idx for idx, name in enumerate(self.mapped_joints)}
        specs = [
            {
                "name": "left_wrist_to_pelvis_left_hip",
                "side": "left",
                "point_body": "left_wrist_roll_rubber_hand",
                "segment_a_body": "pelvis",
                "segment_b_body": "left_hip_roll_link",
                "point_target": "LeftHand",
                "segment_a_target": "Hips",
                "segment_b_target": "LeftLeg",
            },
            {
                "name": "left_elbow_to_pelvis_left_hip",
                "side": "left",
                "point_body": "left_elbow_link",
                "segment_a_body": "pelvis",
                "segment_b_body": "left_hip_roll_link",
                "point_target": "LeftForeArm",
                "segment_a_target": "Hips",
                "segment_b_target": "LeftLeg",
            },
            {
                "name": "right_wrist_to_pelvis_right_hip",
                "side": "right",
                "point_body": "right_wrist_roll_rubber_hand",
                "segment_a_body": "pelvis",
                "segment_b_body": "right_hip_roll_link",
                "point_target": "RightHand",
                "segment_a_target": "Hips",
                "segment_b_target": "RightLeg",
            },
            {
                "name": "right_elbow_to_pelvis_right_hip",
                "side": "right",
                "point_body": "right_elbow_link",
                "segment_a_body": "pelvis",
                "segment_b_body": "right_hip_roll_link",
                "point_target": "RightForeArm",
                "segment_a_target": "Hips",
                "segment_b_target": "RightLeg",
            },
        ]
        data = []
        for spec in specs:
            body_keys = ("point_body", "segment_a_body", "segment_b_body")
            target_keys = ("point_target", "segment_a_target", "segment_b_target")
            if not all(spec[key] in body_name_to_idx for key in body_keys):
                continue
            if not all(spec[key] in target_name_to_idx for key in target_keys):
                continue
            data.append({
                **spec,
                "point_body_idx": int(body_name_to_idx[spec["point_body"]]),
                "segment_a_body_idx": int(body_name_to_idx[spec["segment_a_body"]]),
                "segment_b_body_idx": int(body_name_to_idx[spec["segment_b_body"]]),
                "point_target_idx": int(target_name_to_idx[spec["point_target"]]),
                "segment_a_target_idx": int(target_name_to_idx[spec["segment_a_target"]]),
                "segment_b_target_idx": int(target_name_to_idx[spec["segment_b_target"]]),
            })
        return data

    def _compute_ai_sapiens_capsule_proxy_barrier_masks(self, targets):
        pair_count = len(self.ai_sapiens_capsule_proxy_barrier_data)
        frames = int(targets.shape[0])
        pair_dim = max(1, pair_count)
        active_masks = np.zeros((frames, pair_dim), dtype=np.float32)
        trace = np.full((frames, pair_dim, 5), np.nan, dtype=np.float32)
        summary = {
            "enabled": bool(self.ai_sapiens_capsule_proxy_barrier_enabled),
            "pair_count": int(pair_count),
            "weight": float(self.ai_sapiens_capsule_proxy_barrier_weight),
            "clearance_m": float(self.ai_sapiens_capsule_proxy_barrier_clearance_m),
            "risk_distance_m": float(self.ai_sapiens_capsule_proxy_barrier_risk_distance_m),
            "pairs": [],
            "skipped": False,
            "skip_reason": None,
        }
        if not self.ai_sapiens_capsule_proxy_barrier_data:
            summary["skipped"] = True
            summary["skip_reason"] = "no_valid_proxy_pairs"
            return active_masks, trace, summary
        raw = np.asarray(targets, dtype=np.float64)
        start_frame, should_skip, skip_reason = self._projection_start_and_skip(raw)
        if should_skip:
            summary["skipped"] = True
            summary["skip_reason"] = skip_reason
            return active_masks, trace, summary
        risk_distance_m = max(0.0, float(self.ai_sapiens_capsule_proxy_barrier_risk_distance_m))
        for pair_idx, pair in enumerate(self.ai_sapiens_capsule_proxy_barrier_data):
            distances = []
            active_count = 0
            for frame in range(frames):
                p = raw[frame, int(pair["point_target_idx"]), 0:3]
                a = raw[frame, int(pair["segment_a_target_idx"]), 0:3]
                b = raw[frame, int(pair["segment_b_target_idx"]), 0:3]
                distance = self._point_to_segment_distance_np(p, a, b)
                active = bool(frame >= int(start_frame) and distance <= risk_distance_m)
                if active:
                    active_masks[frame, pair_idx] = np.float32(1.0)
                    active_count += 1
                distances.append(distance)
                trace[frame, pair_idx, 0] = np.float32(distance)
                trace[frame, pair_idx, 1] = np.float32(1.0 if active else 0.0)
                trace[frame, pair_idx, 2] = np.float32(risk_distance_m)
                trace[frame, pair_idx, 3] = np.float32(self.ai_sapiens_capsule_proxy_barrier_clearance_m)
                trace[frame, pair_idx, 4] = np.float32(pair_idx)
            summary["pairs"].append({
                "name": str(pair["name"]),
                "side": str(pair["side"]),
                "point_body": str(pair["point_body"]),
                "segment_a_body": str(pair["segment_a_body"]),
                "segment_b_body": str(pair["segment_b_body"]),
                "active_frames": int(active_count),
                "distance_p05_m": self._safe_percentile(distances, 5),
                "distance_min_m": float(np.min(distances)) if distances else float("nan"),
            })
        return active_masks, trace, summary

    def _compute_ai_sapiens_risk_window_objective_scales(self, targets):
        chain_count = len(self.ai_sapiens_risk_window_data)
        frames = int(targets.shape[0])
        trace = np.zeros((frames, max(1, chain_count), 13), dtype=np.float32)
        trace[:, :, 5:13] = np.float32(1.0)
        summary = {
            "enabled": bool(self.ai_sapiens_risk_window_forearm_priority_enabled),
            "torso_lock_enabled": bool(self.ai_sapiens_risk_window_torso_lock_enabled),
            "chain_count": int(chain_count),
            "capsule_threshold_m": float(self.ai_sapiens_risk_window_capsule_threshold_m),
            "source_elbow_active_deg": float(self.ai_sapiens_risk_window_source_elbow_active_deg),
            "near_max_margin_m": float(self.ai_sapiens_risk_window_near_max_margin_m),
            "max_hand_hip_horizontal_m": float(self.ai_sapiens_risk_window_max_hand_hip_horizontal_m),
            "min_hand_hip_vertical_drop_m": float(self.ai_sapiens_risk_window_min_hand_hip_vertical_drop_m),
            "active_ratio_cap": float(self.ai_sapiens_risk_window_active_ratio_cap),
            "smoothing_frames": int(self.ai_sapiens_risk_window_smoothing_frames),
            "preactivation_frames": int(self.ai_sapiens_risk_window_preactivation_frames),
            "postactivation_frames": int(self.ai_sapiens_risk_window_postactivation_frames),
            "activation_mode": str(self.ai_sapiens_risk_window_activation_mode),
            "arm_t_scale": float(self.ai_sapiens_risk_window_arm_t_scale),
            "forearm_r_scale": float(self.ai_sapiens_risk_window_forearm_r_scale),
            "hips_r_scale": float(self.ai_sapiens_risk_window_hips_r_scale),
            "absolute_weights_enabled": bool(self.ai_sapiens_risk_window_absolute_weights_enabled),
            "absolute_weights": {
                "arm_t": float(self.ai_sapiens_risk_window_arm_t_weight),
                "hand_t": float(self.ai_sapiens_risk_window_hand_t_weight),
                "forearm_t": float(self.ai_sapiens_risk_window_forearm_t_weight),
                "hand_r": float(self.ai_sapiens_risk_window_hand_r_weight),
                "forearm_r": float(self.ai_sapiens_risk_window_forearm_r_weight),
                "chest_t": float(self.ai_sapiens_risk_window_chest_t_weight),
                "chest_r": float(self.ai_sapiens_risk_window_chest_r_weight),
                "hips_t": float(self.ai_sapiens_risk_window_hips_t_weight),
                "hips_r": float(self.ai_sapiens_risk_window_hips_r_weight),
            },
            "chains": [],
            "skipped": False,
            "skip_reason": None,
        }
        if not self.ai_sapiens_risk_window_forearm_priority_enabled:
            summary["skipped"] = True
            summary["skip_reason"] = "disabled"
            return trace, summary
        if not self.ai_sapiens_risk_window_data:
            summary["skipped"] = True
            summary["skip_reason"] = "no_valid_arm_chains"
            return trace, summary

        raw = np.asarray(targets, dtype=np.float64)
        start_frame, should_skip, skip_reason = self._projection_start_and_skip(raw)
        if should_skip:
            summary["skipped"] = True
            summary["skip_reason"] = skip_reason
            return trace, summary

        target_name_to_idx = {name: idx for idx, name in enumerate(self.mapped_joints)}
        hips_idx = target_name_to_idx.get("Hips")
        left_leg_idx = target_name_to_idx.get("LeftLeg")
        right_leg_idx = target_name_to_idx.get("RightLeg")
        threshold = max(0.0, float(self.ai_sapiens_risk_window_capsule_threshold_m))
        source_active_deg = float(self.ai_sapiens_risk_window_source_elbow_active_deg)
        near_max_margin = max(0.0, float(self.ai_sapiens_risk_window_near_max_margin_m))
        max_hand_horizontal = float(self.ai_sapiens_risk_window_max_hand_hip_horizontal_m)
        min_hand_vertical_drop = max(0.0, float(self.ai_sapiens_risk_window_min_hand_hip_vertical_drop_m))
        smooth_frames = max(1, int(self.ai_sapiens_risk_window_smoothing_frames))
        cap = float(self.ai_sapiens_risk_window_active_ratio_cap)
        eps = 1.0e-8

        for chain_idx, chain in enumerate(self.ai_sapiens_risk_window_data):
            root_idx = int(chain["root_idx"])
            mid_idx = int(chain["mid_idx"])
            end_idx = int(chain["end_idx"])
            root_name = str(chain["root_name"])
            side = "left" if root_name.lower().startswith("left") else "right"
            hip_idx = left_leg_idx if side == "left" else right_leg_idx
            risk_strength = np.zeros((frames,), dtype=np.float64)
            raw_active = np.zeros((frames,), dtype=np.float64)
            risk_distances = np.full((frames,), np.nan, dtype=np.float64)
            source_angles = np.full((frames,), np.nan, dtype=np.float64)
            reach_ratios = np.full((frames,), np.nan, dtype=np.float64)
            hand_hip_horizontal = np.full((frames,), np.nan, dtype=np.float64)
            hand_hip_vertical_drop = np.full((frames,), np.nan, dtype=np.float64)
            max_reach = max(eps, float(chain["upper_len"]) + float(chain["lower_len"]))
            active_count = 0

            for frame in range(frames):
                root = raw[frame, root_idx, 0:3]
                mid = raw[frame, mid_idx, 0:3]
                end = raw[frame, end_idx, 0:3]
                v0 = root - mid
                v1 = end - mid
                n0 = float(np.linalg.norm(v0))
                n1 = float(np.linalg.norm(v1))
                if n0 <= eps or n1 <= eps:
                    source_angle = 180.0
                else:
                    c = float(np.clip(np.dot(v0, v1) / (n0 * n1), -1.0, 1.0))
                    source_angle = float(np.degrees(np.arccos(c)))
                source_angles[frame] = source_angle

                root_end_distance = float(np.linalg.norm(end - root))
                reach_ratio = root_end_distance / max_reach
                reach_ratios[frame] = reach_ratio

                risk_distance = float("inf")
                if hips_idx is not None and hip_idx is not None:
                    pelvis = raw[frame, hips_idx, 0:3]
                    hip = raw[frame, hip_idx, 0:3]
                    wrist_distance = self._point_to_segment_distance_np(end, pelvis, hip)
                    elbow_distance = self._point_to_segment_distance_np(mid, pelvis, hip)
                    risk_distance = min(wrist_distance, elbow_distance)
                    hand_delta = end - pelvis
                    hand_hip_horizontal[frame] = float(np.linalg.norm(hand_delta[0:2]))
                    hand_hip_vertical_drop[frame] = float(pelvis[2] - end[2])
                risk_distances[frame] = risk_distance

                source_bent = source_angle <= source_active_deg
                contact_risk = risk_distance <= threshold
                near_max = root_end_distance >= (max_reach - near_max_margin)
                compact_horizontal = True
                if math.isfinite(max_hand_horizontal):
                    compact_horizontal = hand_hip_horizontal[frame] <= max_hand_horizontal
                low_hand = hand_hip_vertical_drop[frame] >= min_hand_vertical_drop
                active = bool(
                    frame >= int(start_frame)
                    and source_bent
                    and contact_risk
                    and near_max
                    and compact_horizontal
                    and low_hand
                )
                if active:
                    raw_active[frame] = 1.0
                    active_count += 1
                    distance_score = (threshold - risk_distance) / max(threshold, eps)
                    reach_score = max(0.0, reach_ratio - 0.95) / 0.15
                    risk_strength[frame] = float(np.clip(max(distance_score, reach_score), 0.0, 1.0))

            pre_frames = max(0, int(self.ai_sapiens_risk_window_preactivation_frames))
            post_frames = max(0, int(self.ai_sapiens_risk_window_postactivation_frames))
            if pre_frames > 0 or post_frames > 0:
                original_active = raw_active.copy()
                original_strength = risk_strength.copy()
                for active_frame in np.flatnonzero(original_active > 0.5):
                    lo = max(int(start_frame), int(active_frame) - pre_frames)
                    hi = min(frames, int(active_frame) + post_frames + 1)
                    if hi <= lo:
                        continue
                    raw_active[lo:hi] = np.maximum(raw_active[lo:hi], original_active[active_frame])
                    risk_strength[lo:hi] = np.maximum(
                        risk_strength[lo:hi],
                        original_strength[active_frame],
                    )
                active_count = int(np.count_nonzero(raw_active > 0.5))

            activation_signal = risk_strength
            if self.ai_sapiens_risk_window_activation_mode in {"binary", "raw", "raw_active"}:
                activation_signal = raw_active

            if smooth_frames > 1:
                kernel = np.ones((smooth_frames,), dtype=np.float64) / float(smooth_frames)
                smoothed = np.convolve(activation_signal, kernel, mode="same")
            else:
                smoothed = activation_signal
            smoothed = np.clip(smoothed, 0.0, 1.0)

            removed_count = 0
            if 0.0 < cap < 1.0:
                frame_count_for_cap = max(0, frames - int(start_frame))
                if frame_count_for_cap > 0:
                    active_local = np.flatnonzero(smoothed[start_frame:] > 1e-6)
                    keep_count = int(np.ceil(float(frame_count_for_cap) * cap))
                    keep_count = max(1, min(int(active_local.size), keep_count))
                    if active_local.size > keep_count:
                        values = smoothed[start_frame:][active_local]
                        order = np.argsort(values)
                        keep_local = set(int(v) for v in active_local[order[-keep_count:]])
                        for local_idx in active_local:
                            if int(local_idx) not in keep_local:
                                smoothed[start_frame + int(local_idx)] = 0.0
                                removed_count += 1

            hand_t_scale = 1.0 + (float(self.ai_sapiens_risk_window_hand_t_scale) - 1.0) * smoothed
            forearm_t_scale = 1.0 + (float(self.ai_sapiens_risk_window_forearm_t_scale) - 1.0) * smoothed
            hand_r_scale = 1.0 + (float(self.ai_sapiens_risk_window_hand_r_scale) - 1.0) * smoothed
            arm_t_scale = 1.0 + (float(self.ai_sapiens_risk_window_arm_t_scale) - 1.0) * smoothed
            forearm_r_scale = 1.0 + (float(self.ai_sapiens_risk_window_forearm_r_scale) - 1.0) * smoothed
            chest_t_scale = np.ones((frames,), dtype=np.float64)
            hips_t_scale = np.ones((frames,), dtype=np.float64)
            hips_r_scale = np.ones((frames,), dtype=np.float64)
            if self.ai_sapiens_risk_window_torso_lock_enabled:
                chest_t_scale = chest_t_scale + (
                    (float(self.ai_sapiens_risk_window_chest_t_scale) - 1.0) * smoothed
                )
                hips_t_scale = hips_t_scale + (
                    (float(self.ai_sapiens_risk_window_hips_t_scale) - 1.0) * smoothed
                )
                hips_r_scale = hips_r_scale + (
                    (float(self.ai_sapiens_risk_window_hips_r_scale) - 1.0) * smoothed
                )

            trace[:, chain_idx, 0] = smoothed.astype(np.float32)
            trace[:, chain_idx, 1] = raw_active.astype(np.float32)
            trace[:, chain_idx, 2] = risk_distances.astype(np.float32)
            trace[:, chain_idx, 3] = source_angles.astype(np.float32)
            trace[:, chain_idx, 4] = reach_ratios.astype(np.float32)
            trace[:, chain_idx, 5] = hand_t_scale.astype(np.float32)
            trace[:, chain_idx, 6] = forearm_t_scale.astype(np.float32)
            trace[:, chain_idx, 7] = hand_r_scale.astype(np.float32)
            trace[:, chain_idx, 8] = chest_t_scale.astype(np.float32)
            trace[:, chain_idx, 9] = hips_t_scale.astype(np.float32)
            trace[:, chain_idx, 10] = arm_t_scale.astype(np.float32)
            trace[:, chain_idx, 11] = forearm_r_scale.astype(np.float32)
            trace[:, chain_idx, 12] = hips_r_scale.astype(np.float32)
            summary["chains"].append({
                "root": root_name,
                "mid": str(chain["mid_name"]),
                "end": str(chain["end_name"]),
                "raw_active_frames": int(active_count),
                "active_frames_after_smooth_cap": int(np.count_nonzero(smoothed > 1e-6)),
                "active_cap_removed_frames": int(removed_count),
                "risk_strength_p95": self._safe_percentile(smoothed, 95),
                "risk_distance_p05_m": self._safe_percentile(risk_distances[np.isfinite(risk_distances)], 5),
                "source_angle_p50_deg": self._safe_percentile(source_angles[np.isfinite(source_angles)], 50),
                "reach_ratio_p95": self._safe_percentile(reach_ratios[np.isfinite(reach_ratios)], 95),
                "hand_hip_horizontal_p95_m": self._safe_percentile(
                    hand_hip_horizontal[np.isfinite(hand_hip_horizontal)], 95
                ),
                "hand_hip_vertical_drop_p50_m": self._safe_percentile(
                    hand_hip_vertical_drop[np.isfinite(hand_hip_vertical_drop)], 50
                ),
                "arm_t_scale_p95": self._safe_percentile(arm_t_scale, 95),
                "hand_t_scale_p95": self._safe_percentile(hand_t_scale, 95),
                "forearm_t_scale_p95": self._safe_percentile(forearm_t_scale, 95),
                "hand_r_scale_p05": self._safe_percentile(hand_r_scale, 5),
                "forearm_r_scale_p05": self._safe_percentile(forearm_r_scale, 5),
                "chest_t_scale_p95": self._safe_percentile(chest_t_scale, 95),
                "hips_t_scale_p05": self._safe_percentile(hips_t_scale, 5),
                "hips_r_scale_p05": self._safe_percentile(hips_r_scale, 5),
            })

        return trace, summary

    def _build_target_mapping(self, model, skeleton, retargeter_config):
        mapped_joints = []
        mapped_joint_indices = []
        mapped_body_link_pos_data = []
        mapped_body_link_rot_data = []
        body_names = [newton_utils.get_name_from_label(label) for label in self.robot_builder.body_label]
        for joint, mapping_data in retargeter_config["ik_map"].items():
            mapped_joints.append(joint)
            mapped_joint_indices.append(skeleton.joint_index(joint))
            t_offset = mapping_data.get("t_offset", [0.0, 0.0, 0.0])
            if len(t_offset) != 3:
                raise ValueError(f"ik_map[{joint}].t_offset must have length 3, got {t_offset}")
            mapped_body_link_pos_data.append((
                body_names.index(mapping_data['t_body']),
                mapping_data['t_weight'],
                (float(t_offset[0]), float(t_offset[1]), float(t_offset[2])),
            ))
            mapped_body_link_rot_data.append((body_names.index(mapping_data['r_body']), mapping_data['r_weight']))

        return (
            mapped_joints,
            mapped_joint_indices,
            mapped_body_link_pos_data,
            mapped_body_link_rot_data)

    def _create_ik_objectives(self, num_envs, model, state):
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)

        # Gather default body position and rotation based on model state to initialize
        # position and rotation objectives
        num_body_link_pos = len(self.mapped_body_link_pos_data)
        num_body_link_rot = len(self.mapped_body_link_rot_data)
        pos_targets = np.zeros((num_envs, num_body_link_pos), dtype=wp.vec3)
        rot_targets = np.zeros((num_envs, num_body_link_rot), dtype=wp.quat)

        body_q = state.body_q.numpy()
        for env in range(num_envs):
            base = env * self.num_body_count
            for ee_idx, data in enumerate(self.mapped_body_link_pos_data):
                link_idx = data[0]
                pos_targets[env, ee_idx] = body_q[base + link_idx][0:3]

            for ee_idx, (link_idx, _) in enumerate(self.mapped_body_link_rot_data):
                rot_wp = wp.quat(body_q[base + link_idx][3:7])
                rot_targets[env, ee_idx] = wp.normalize(rot_wp)

        pos_num_ees = len(self.mapped_body_link_pos_data)
        rot_num_ees = len(self.mapped_body_link_rot_data)
        pos_target_arrays, rot_target_arrays = [], []
        for ee_idx in range(pos_num_ees):
            pos_wp = wp.array(pos_targets[:, ee_idx], dtype=wp.vec3)
            pos_target_arrays.append(pos_wp)

        for ee_idx in range(rot_num_ees):
            rot_wp = wp.array(rot_targets[:, ee_idx], dtype=wp.vec4)
            rot_target_arrays.append(rot_wp)

        position_objectives = []
        for i, data in enumerate(self.mapped_body_link_pos_data):
            link_idx = data[0]
            w = data[1]
            t_offset = data[2] if len(data) > 2 else (0.0, 0.0, 0.0)
            objective = ik.IKObjectivePosition(
                link_index=link_idx,
                link_offset=wp.vec3(*t_offset),
                target_positions=pos_target_arrays[i],
                weight=w)
            position_objectives.append(objective)

        rotation_objectives = []
        for i, (link_idx, w) in enumerate(self.mapped_body_link_rot_data):
            objective = ik.IKObjectiveRotation(
                link_index=link_idx,
                link_offset_rotation=wp.quat_identity(),
                target_rotations=rot_target_arrays[i],
                weight=w)
            rotation_objectives.append(objective)

        joint_limit_objective = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.ik_model.joint_limit_lower,
            joint_limit_upper=self.ik_model.joint_limit_upper,
            weight=self.joint_limit_weight)

        # Weight is set to desired value once initialization frames have been processed
        smooth_joint_limiter_objective = IKSmoothJointFilter(
            joint_limit_lower=self.ik_model.joint_limit_lower,
            joint_limit_upper=self.ik_model.joint_limit_upper,
            weight=0.0,
            coord_masks=self.smooth_joint_filter_coord_masks)

        neutral_reference_objective = IKTemporalJointReference(
            joint_coord_count=self.ik_model.joint_coord_count,
            weight=self.neutral_reference_weight,
            coord_masks=self.neutral_reference_coord_masks)

        temporal_yaw_twist_reference_objective = IKTemporalJointReference(
            joint_coord_count=self.ik_model.joint_coord_count,
            weight=self.ai_sapiens_temporal_yaw_twist_reference_weight,
            coord_masks=self.ai_sapiens_temporal_yaw_twist_reference_coord_masks)

        arm_ik_temporal_reference_objective = IKTemporalJointReference(
            joint_coord_count=self.ik_model.joint_coord_count,
            weight=self.ai_sapiens_arm_ik_temporal_reference_weight,
            coord_masks=self.ai_sapiens_arm_ik_temporal_reference_coord_masks)

        wrist_roll_nullspace_temporal_objective = IKTemporalJointReference(
            joint_coord_count=self.ik_model.joint_coord_count,
            weight=self.ai_sapiens_wrist_roll_temporal_weight_base,
            coord_masks=self.ai_sapiens_wrist_roll_nullspace_coord_masks)

        arm_branch_nullspace_temporal_objective = IKTemporalJointReference(
            joint_coord_count=self.ik_model.joint_coord_count,
            weight=self.ai_sapiens_arm_branch_temporal_weight_base,
            coord_masks=self.ai_sapiens_arm_branch_nullspace_coord_masks)

        elbow_branch_hint_objectives = []
        if self.ai_sapiens_elbow_branch_hint_objective_enabled:
            hint_count = len(self.ai_sapiens_elbow_branch_hint_data)
            hint_targets = np.zeros((num_envs, hint_count), dtype=wp.vec3)
            body_q = state.body_q.numpy()
            for hint_idx, chain in enumerate(self.ai_sapiens_elbow_branch_hint_data):
                mid_target_idx = int(chain["mid_idx"])
                link_idx = int(self.mapped_body_link_pos_data[mid_target_idx][0])
                for env in range(num_envs):
                    base = env * self.num_body_count
                    hint_targets[env, hint_idx] = body_q[base + link_idx][0:3]
            for hint_idx, chain in enumerate(self.ai_sapiens_elbow_branch_hint_data):
                mid_target_idx = int(chain["mid_idx"])
                link_idx = int(self.mapped_body_link_pos_data[mid_target_idx][0])
                target_array = wp.array(hint_targets[:, hint_idx], dtype=wp.vec3)
                elbow_branch_hint_objectives.append(
                    ik.IKObjectivePosition(
                        link_index=link_idx,
                        link_offset=wp.vec3(0.0, 0.0, 0.0),
                        target_positions=target_array,
                        weight=self.ai_sapiens_elbow_branch_hint_weight,
                    )
                )

        arm_bend_angle_objective = None
        if self.ai_sapiens_bilateral_arm_bend_objective_enabled:
            root_links = []
            mid_links = []
            end_links = []
            for chain in self.ai_sapiens_bilateral_arm_bend_data:
                root_links.append(int(self.mapped_body_link_pos_data[int(chain["root_idx"])][0]))
                mid_links.append(int(self.mapped_body_link_pos_data[int(chain["mid_idx"])][0]))
                end_links.append(int(self.mapped_body_link_pos_data[int(chain["end_idx"])][0]))
            if root_links:
                arm_bend_angle_objective = IKArmBendAngleObjective(
                    root_links,
                    mid_links,
                    end_links,
                    num_envs,
                    weight=self.ai_sapiens_arm_bend_weight,
                )

        limb_bend_angle_objective = None
        if self.ai_sapiens_limb_bend_angle_objective_enabled:
            root_links = []
            mid_links = []
            end_links = []
            for chain in self.ai_sapiens_limb_bend_angle_data:
                root_links.append(int(self.mapped_body_link_pos_data[int(chain["root_idx"])][0]))
                mid_links.append(int(self.mapped_body_link_pos_data[int(chain["mid_idx"])][0]))
                end_links.append(int(self.mapped_body_link_pos_data[int(chain["end_idx"])][0]))
            if root_links:
                limb_bend_angle_objective = IKArmBendAngleObjective(
                    root_links,
                    mid_links,
                    end_links,
                    num_envs,
                    weight=self.ai_sapiens_limb_bend_angle_weight,
                )

        limb_plane_normal_objective = None
        if self.ai_sapiens_limb_plane_normal_objective_enabled:
            root_links = []
            mid_links = []
            end_links = []
            for chain in self.ai_sapiens_limb_plane_normal_data:
                root_links.append(int(self.mapped_body_link_pos_data[int(chain["root_idx"])][0]))
                mid_links.append(int(self.mapped_body_link_pos_data[int(chain["mid_idx"])][0]))
                end_links.append(int(self.mapped_body_link_pos_data[int(chain["end_idx"])][0]))
            if root_links:
                limb_plane_normal_objective = IKLimbPlaneNormalObjective(
                    root_links,
                    mid_links,
                    end_links,
                    num_envs,
                    weight=self.ai_sapiens_limb_plane_normal_weight,
                )

        limb_midpoint_position_objective = []
        if self.ai_sapiens_limb_midpoint_position_objective_enabled:
            hint_count = len(self.ai_sapiens_limb_midpoint_position_data)
            hint_targets = np.zeros((num_envs, hint_count), dtype=wp.vec3)
            body_q = state.body_q.numpy()
            for hint_idx, chain in enumerate(self.ai_sapiens_limb_midpoint_position_data):
                mid_target_idx = int(chain["mid_idx"])
                link_idx = int(self.mapped_body_link_pos_data[mid_target_idx][0])
                for env in range(num_envs):
                    base = env * self.num_body_count
                    hint_targets[env, hint_idx] = body_q[base + link_idx][0:3]
            for hint_idx, chain in enumerate(self.ai_sapiens_limb_midpoint_position_data):
                mid_target_idx = int(chain["mid_idx"])
                link_idx = int(self.mapped_body_link_pos_data[mid_target_idx][0])
                target_array = wp.array(hint_targets[:, hint_idx], dtype=wp.vec3)
                limb_midpoint_position_objective.append(
                    ik.IKObjectivePosition(
                        link_index=link_idx,
                        link_offset=wp.vec3(0.0, 0.0, 0.0),
                        target_positions=target_array,
                        weight=self.ai_sapiens_limb_midpoint_position_weight,
                    )
                )

        torso_local_limb_midpoint_objective = []
        if self.ai_sapiens_torso_local_limb_midpoint_objective_enabled:
            hint_count = len(self.ai_sapiens_torso_local_limb_midpoint_data)
            hint_targets = np.zeros((num_envs, hint_count), dtype=wp.vec3)
            body_q = state.body_q.numpy()
            for hint_idx, chain in enumerate(self.ai_sapiens_torso_local_limb_midpoint_data):
                mid_target_idx = int(chain["mid_idx"])
                link_idx = int(self.mapped_body_link_pos_data[mid_target_idx][0])
                for env in range(num_envs):
                    base = env * self.num_body_count
                    hint_targets[env, hint_idx] = body_q[base + link_idx][0:3]
            for hint_idx, chain in enumerate(self.ai_sapiens_torso_local_limb_midpoint_data):
                mid_target_idx = int(chain["mid_idx"])
                link_idx = int(self.mapped_body_link_pos_data[mid_target_idx][0])
                target_array = wp.array(hint_targets[:, hint_idx], dtype=wp.vec3)
                torso_local_limb_midpoint_objective.append(
                    ik.IKObjectivePosition(
                        link_index=link_idx,
                        link_offset=wp.vec3(0.0, 0.0, 0.0),
                        target_positions=target_array,
                        weight=self.ai_sapiens_torso_local_limb_midpoint_weight,
                    )
                )

        soft_bend_wrist_reference_objective = None
        if self.ai_sapiens_soft_bend_wrist_reference_enabled:
            end_links = []
            for chain in self.ai_sapiens_bilateral_arm_bend_data:
                end_links.append(int(self.mapped_body_link_pos_data[int(chain["end_idx"])][0]))
            if end_links:
                soft_bend_wrist_reference_objective = IKMaskedBodyPositionObjective(
                    end_links,
                    num_envs,
                    weight=self.ai_sapiens_soft_bend_wrist_reference_weight,
                )

        contact_gated_elbow_hint_objective = None
        if self.ai_sapiens_contact_gated_elbow_midpoint_hint_enabled:
            mid_links = []
            for chain in self.ai_sapiens_bilateral_arm_bend_data:
                mid_links.append(int(self.mapped_body_link_pos_data[int(chain["mid_idx"])][0]))
            if mid_links:
                contact_gated_elbow_hint_objective = IKMaskedBodyPositionObjective(
                    mid_links,
                    num_envs,
                    weight=self.ai_sapiens_contact_gated_elbow_hint_weight,
                )

        contact_micro_wrist_reference_objective = None
        if self.ai_sapiens_contact_micro_wrist_reference_enabled:
            end_links = []
            for chain in self.ai_sapiens_bilateral_arm_bend_data:
                end_links.append(int(self.mapped_body_link_pos_data[int(chain["end_idx"])][0]))
            if end_links:
                contact_micro_wrist_reference_objective = IKMaskedBodyPositionObjective(
                    end_links,
                    num_envs,
                    weight=self.ai_sapiens_contact_micro_wrist_reference_weight,
                )

        capsule_proxy_barrier_objective = None
        if self.ai_sapiens_capsule_proxy_barrier_enabled and self.ai_sapiens_capsule_proxy_barrier_data:
            capsule_proxy_barrier_objective = IKPointSegmentDistanceBarrier(
                [int(pair["point_body_idx"]) for pair in self.ai_sapiens_capsule_proxy_barrier_data],
                [int(pair["segment_a_body_idx"]) for pair in self.ai_sapiens_capsule_proxy_barrier_data],
                [int(pair["segment_b_body_idx"]) for pair in self.ai_sapiens_capsule_proxy_barrier_data],
                num_envs,
                clearance_m=self.ai_sapiens_capsule_proxy_barrier_clearance_m,
                weight=self.ai_sapiens_capsule_proxy_barrier_weight,
            )

        return (
            position_objectives,
            rotation_objectives,
            joint_limit_objective,
            smooth_joint_limiter_objective,
            neutral_reference_objective,
            temporal_yaw_twist_reference_objective,
            arm_ik_temporal_reference_objective,
            wrist_roll_nullspace_temporal_objective,
            arm_branch_nullspace_temporal_objective,
            elbow_branch_hint_objectives,
            arm_bend_angle_objective,
            limb_bend_angle_objective,
            limb_plane_normal_objective,
            limb_midpoint_position_objective,
            torso_local_limb_midpoint_objective,
            soft_bend_wrist_reference_objective,
            contact_gated_elbow_hint_objective,
            contact_micro_wrist_reference_objective,
            capsule_proxy_barrier_objective,
        )
