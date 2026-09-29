# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations
import math

import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Dict, List, Tuple

if TYPE_CHECKING:
    from torch import Tensor
    from isaaclab.envs import VecEnvStepReturn
else:
    Tensor = torch.Tensor
    VecEnvStepReturn = tuple

from tqdm import tqdm

import carb
import isaaclab.sim as sim_utils
import omni.physics.tensors.impl.api as physx
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_conjugate, quat_mul, axis_angle_from_quat, saturate, quat_inv, wrap_to_pi, subtract_frame_transforms
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from dexx.tasks.hand_imitation.dataset.transform import aa_to_quat, aa_to_rotmat, rotmat_to_aa, rot6d_to_aa
from dexx.tasks.hand_imitation.envs.factory import DexHandFactory
from dexx.tasks.hand_imitation.dataset.factory import ManipDataFactory
from dexx.tasks.hand_imitation.dataset.oakink2_dataset_utils import oakink2_obj_mass
from dexx.utils.debug_draw import DebugDraw
from dexx import deploy_config as _dcfg
# Auto-register hands
import dexx.tasks.hand_imitation.envs.sharpa  # noqa: F401
from isaaclab.assets import RigidObjectCfg
import os

# BPS encoding
try:
    from bps_torch.bps import bps_torch
    BPS_AVAILABLE = True
except ImportError:
    BPS_AVAILABLE = False
    print("WARNING: bps_torch not available. BPS encoding will be disabled.")

if TYPE_CHECKING:
    from .franka_sharpa_env_cfg import FrankaSharpaEnvCfg
else:
    from .franka_sharpa_env_cfg import update_cfg_for_hand_side

ROBOT_HEIGHT = 0.00214874
# Yaw-rotation helpers (replay retarget-time aug_yaw_deg augmentation).
def yaw_rotmat_2x2_batch(yaw_deg: torch.Tensor) -> torch.Tensor:
    t = torch.deg2rad(yaw_deg)
    c = torch.cos(t)
    s = torch.sin(t)
    R = torch.zeros(yaw_deg.shape[0], 2, 2, device=yaw_deg.device, dtype=torch.float32)
    R[:, 0, 0] = c
    R[:, 0, 1] = -s
    R[:, 1, 0] = s
    R[:, 1, 1] = c
    return R

def yaw_rotmat_3x3_batch(yaw_deg: torch.Tensor) -> torch.Tensor:
    t = torch.deg2rad(yaw_deg)
    c = torch.cos(t)
    s = torch.sin(t)
    R = torch.zeros(yaw_deg.shape[0], 3, 3, device=yaw_deg.device, dtype=torch.float32)
    R[:, 0, 0] = c
    R[:, 0, 1] = -s
    R[:, 1, 0] = s
    R[:, 1, 1] = c
    R[:, 2, 2] = 1.0
    return R

class FrankaSharpaEnv(DirectRLEnv):
    cfg: "FrankaSharpaEnvCfg"

    def __init__(self, cfg: "FrankaSharpaEnvCfg", render_mode: str | None = None, **kwargs):
        # Update configuration based on hand_side before initializing
        # This ensures all hand_side-dependent parameters are correctly set
        update_cfg_for_hand_side(cfg, cfg.hand_side)

        # Auto-set action_space based on arm control mode
        n_hand = len(cfg.actuated_joint_names)  # 22
        self.freeze_arm = getattr(cfg, 'freeze_arm', False)
        if self.freeze_arm:
            cfg.action_space = n_hand  # hand only, arm stays at reset position
        elif getattr(cfg, 'use_pid_control', False) or getattr(cfg, 'use_osc_control', False):
            cfg.action_space = 9 + n_hand   # pos_error(3) + rot_error_6d(6) + hand
        elif getattr(cfg, 'use_joint_pos_control', False):
            cfg.action_space = 7 + n_hand   # arm_joint_pos(7) + hand
        elif getattr(cfg, 'use_joint_delta_control', False):
            cfg.action_space = 7 + n_hand   # arm_joint_delta(7) + hand
        else:
            cfg.action_space = 6 + n_hand   # force(3) + torque(3) + hand
        arm_mode = "frozen" if self.freeze_arm else \
                   "osc" if getattr(cfg, 'use_osc_control', False) else \
                   "pid" if getattr(cfg, 'use_pid_control', False) else \
                   "joint_pos" if getattr(cfg, 'use_joint_pos_control', False) else \
                   "joint_delta" if getattr(cfg, 'use_joint_delta_control', False) else "force"
        print(f"[INFO] Auto-set action_space={cfg.action_space} (arm control: {arm_mode})")

        # Reward tracking mode flags
        self._use_wrist_tracking: bool = getattr(cfg, 'use_wrist_tracking_reward', True)
        self._use_abs_hand_tracking: bool = getattr(cfg, 'use_absolute_hand_tracking_reward', True)
        self._use_rel_hand_tracking: bool = getattr(cfg, 'use_relative_hand_tracking_reward', False)
        print(f"[INFO] Tracking rewards: wrist={self._use_wrist_tracking}, "
              f"abs_hand={self._use_abs_hand_tracking}, rel_hand={self._use_rel_hand_tracking}")

        self.reset_height_lower = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.reset_height_upper = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)

        self.hand_side = cfg.hand_side
        max_episode_length = math.ceil(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation))
        self.data_indices = cfg.data_indices
        # Expand glob patterns or task-only indices for robotool_batch
        # e.g. "rt/blue_cup/*" or "rt/blue_cup" -> all experiments under blue_cup/
        _needs_expand = any(
            "*" in idx or (idx.startswith("rt/") and idx.count("/") == 1)
            for idx in self.data_indices
        )
        if _needs_expand:
            from dexx.tasks.hand_imitation.dataset.robotool_batch_dataset_dexhand import expand_rt_indices
            self.data_indices = expand_rt_indices(self.data_indices)
            print(f"Expanded data_indices to {len(self.data_indices)} entries")
        self.dexhand = DexHandFactory.create_hand("sharpa", self.hand_side)
        print("Is building data ...")
        dataset_list = list(set([ManipDataFactory.dataset_type(data_idx) for data_idx in self.data_indices]))


        self._table_surface_z = _dcfg.TABLE_SURFACE_Z
        mujoco2gym_transf = torch.eye(4, dtype=torch.float32, device=cfg.sim.device)

        m1 = aa_to_rotmat(torch.tensor([0, 0, -np.pi/2], dtype=torch.float32, device=cfg.sim.device))
        m2 = aa_to_rotmat(torch.tensor([np.pi/2, 0, 0], dtype=torch.float32, device=cfg.sim.device))

        mujoco2gym_transf[:3, :3] = m1 @ m2
        mujoco2gym_transf[:3, 3] = torch.tensor([0, 0, self._table_surface_z], device=cfg.sim.device)

        self.mujoco2gym_transf = torch.tensor(mujoco2gym_transf, device=cfg.sim.device, dtype=torch.float32)


        self.demo_dataset_dict = {}
        for dataset_type in dataset_list:
            dataset_kwargs = dict(
                manipdata_type=dataset_type,
                side=self.hand_side,
                device=cfg.sim.device,
                mujoco2gym_transf=self.mujoco2gym_transf,
                max_seq_len=max_episode_length,
                dexhand=self.dexhand,
                embodiment="sharpa",
            )
            retarget_root = getattr(cfg, "robotool_batch_retarget_root", "")
            if dataset_type == "robotool_batch" and retarget_root:
                dataset_kwargs["retarget_root"] = retarget_root
                print(f"[INFO] RobotoolBatch retarget_root override: {retarget_root}")
            self.demo_dataset_dict[dataset_type] = ManipDataFactory.create_data(**dataset_kwargs)
        # Expand robotool_batch indices to all retarget-augmentation entries.
        _has_rt = any(idx.startswith("rt/") or idx.startswith("rt_") for idx in self.data_indices)
        if _has_rt:
            from dexx.tasks.hand_imitation.dataset.robotool_batch_dataset_dexhand import expand_rt_indices

            # Normalize rt_ / rt/task style indices into base keys.
            normalized_indices = expand_rt_indices(self.data_indices)

            # Find the dataset instance for the actual dataset_type.
            rt_dataset = None
            for idx in normalized_indices:
                ds_type = ManipDataFactory.dataset_type(idx)
                if "robotool_batch" in ds_type:
                    rt_dataset = self.demo_dataset_dict.get(ds_type)
                    if rt_dataset is not None:
                        break

            print(f"[INFO] normalized rt indices: {len(normalized_indices)}")
            print(f"[INFO] rt_dataset found: {type(rt_dataset).__name__ if rt_dataset is not None else None}")

            import os as _os
            if rt_dataset is not None and not _os.environ.get("DISABLE_AUG_EXPAND"):
                expanded = rt_dataset.expand_with_all_aug(normalized_indices)

                print(f"[INFO] Expanded data_indices from {len(self.data_indices)} to {len(expanded)} aug entries")
                for i, idx in enumerate(expanded[:10]):
                    print(f"  [{i}] {idx}")
                if len(expanded) > 10:
                    print(f"  ... and {len(expanded)-10} more")

                self.data_indices = expanded
            else:
                if _os.environ.get("DISABLE_AUG_EXPAND"):
                    print(f"[INFO] DISABLE_AUG_EXPAND=1: using {len(normalized_indices)} base demos (no aug)")
                self.data_indices = normalized_indices

        self.data_keys = list(self.demo_dataset_dict.keys())
        self._env0_obj_urdf = self.demo_dataset_dict[self.data_keys[0]][self.data_indices[0]]["obj_urdf_path"]
        super().__init__(cfg, render_mode, **kwargs)

        
        # num_hand_dofs should be the number of actuated hand joints only (not including arm joints)
        # This will be set after actuated_dof_indices is determined
        self.num_hand_dofs = len(cfg.actuated_joint_names)  # Number of actuated hand joints
        # list of actuated joints (hand joints only, not arm joints)
        self.actuated_dof_indices = list()
        for joint_name in cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self.hand.joint_names.index(joint_name))
        self.actuated_dof_indices.sort()
        
        self._identify_arm_joints()
        self._build_data()

        # Adaptive initialization: rollout state buffer
        self._state_buffer = None
        if getattr(cfg, 'adaptive_init_enabled', False):
            from .adaptive_init import AdaptiveStateBuffer
            self._state_buffer = AdaptiveStateBuffer(cfg, self.num_envs, self.device)
            print(f"[INFO] AdaptiveStateBuffer enabled: size={self._state_buffer.buffer_size}, "
                  f"prob={self._state_buffer.prob}, warmup={self._state_buffer.warmup}")

        # Adaptive trajectory-fraction sampling (failure-bin bias).
        # Two buffers on `bin_count` × scalar:
        #   _adapt_bin_failed_count   — slow EMA, drives the sampler probabilities
        #   _adapt_bin_current_failed — accumulates within one reset call, then EMA'd
        # Smoothing kernel is precomputed (geometric, length K).
        self._adapt_enabled = bool(getattr(cfg, 'adaptive_sampling_enabled', False))
        if self._adapt_enabled:
            n_bins = int(getattr(cfg, 'adaptive_sampling_bins', 50))
            K = int(getattr(cfg, 'adaptive_sampling_kernel_size', 3))
            lam = float(getattr(cfg, 'adaptive_sampling_lambda', 0.8))
            self._adapt_n_bins = n_bins
            self._adapt_kernel_size = K
            self._adapt_bin_failed_count = torch.zeros(n_bins, dtype=torch.float, device=self.device)
            self._adapt_bin_current_failed = torch.zeros(n_bins, dtype=torch.float, device=self.device)
            kern = torch.tensor([lam ** i for i in range(K)], dtype=torch.float, device=self.device)
            self._adapt_kernel = (kern / kern.sum()).view(1, 1, -1)
            print(f"[INFO] Adaptive trajectory-fraction sampling enabled: "
                  f"bins={n_bins} kernel={K} λ={lam} "
                  f"uniform_ratio={cfg.adaptive_sampling_uniform_ratio} α={cfg.adaptive_sampling_alpha}")

        self._init_hand_data()
        self._init_arm_diff_ik_controller()
        self._init_osc_controller()

        # Arm gravity compensation: disable PhysX PD for arm, we'll apply effort manually
        # For OSC mode, arm stiffness/damping must also be zeroed (OSC outputs effort directly)
        self.use_osc_control = getattr(self.cfg, 'use_osc_control', False)
        self._arm_gravity_comp = getattr(self.cfg, 'arm_gravity_compensation', False)
        if self._arm_gravity_comp or self.use_osc_control:
            # Store the cfg gains for manual PD computation
            arm_actuator = self.hand.actuators["arm_joints"]
            self._arm_K = arm_actuator.stiffness.clone()
            self._arm_D = arm_actuator.damping.clone()
            if self._arm_K.ndim == 1:
                self._arm_K = self._arm_K.unsqueeze(0)
                self._arm_D = self._arm_D.unsqueeze(0)
            # Expand to per-env tensors so arm-PD Domain Randomization can
            # vary K/D per env at reset. Keep `_default` copies to scale against.
            if self._arm_K.shape[0] == 1:
                self._arm_K = self._arm_K.expand(self.num_envs, -1).contiguous()
                self._arm_D = self._arm_D.expand(self.num_envs, -1).contiguous()
            self._arm_K_default = self._arm_K.clone()
            self._arm_D_default = self._arm_D.clone()
            # Zero out PhysX PD drives for arm joints (we compute torque manually)
            # Must write to PhysX directly, not just Python tensor
            arm_actuator.stiffness[:] = 0.0
            arm_actuator.damping[:] = 0.0
            # Write zeros to PhysX simulation
            cur_stiffness = self.hand.root_physx_view.get_dof_stiffnesses()
            cur_damping = self.hand.root_physx_view.get_dof_dampings()
            arm_phys_ids = arm_actuator.joint_indices.cpu().tolist()
            for idx in arm_phys_ids:
                cur_stiffness[:, idx] = 0.0
                cur_damping[:, idx] = 0.0
            self.hand.root_physx_view.set_dof_stiffnesses(cur_stiffness, torch.arange(self.num_envs))
            self.hand.root_physx_view.set_dof_dampings(cur_damping, torch.arange(self.num_envs))
            self._arm_phys_ids = arm_phys_ids
            print(f"[INFO] Arm gravity compensation enabled. PhysX drives zeroed for arm joints.")
            print(f"[INFO] Manual PD: K={self._arm_K[0].cpu().numpy()}, D={self._arm_D[0].cpu().numpy()}")
            print(f"[INFO] arm_joint_indices (robot order): {self.arm_joint_indices}")
            print(f"[INFO] arm_phys_ids (actuator order):   {arm_phys_ids}")

        # Loop trajectory config
        self.loop_trajectory = getattr(self.cfg, 'loop_trajectory', False)

        # Action delay buffer for sim2real — with optional per-env DR.
        # If randomize_action_delay=True: each env's delay is sampled uniformly
        # in [action_delay_min, action_delay_max] at reset; buffer is sized
        # to the MAX delay so any env-specific delay can be read from it.
        # If randomize_action_delay=False: all envs use the same
        # `action_delay_steps`.
        self.action_delay_steps = getattr(self.cfg, 'action_delay_steps', 0)
        self._randomize_action_delay = bool(getattr(self.cfg, 'randomize_action_delay', False))
        self._action_delay_min = int(getattr(self.cfg, 'action_delay_min', 0))
        self._action_delay_max = int(getattr(self.cfg, 'action_delay_max', self.action_delay_steps))
        if self._randomize_action_delay:
            max_delay = max(self._action_delay_max, 1)
        else:
            max_delay = self.action_delay_steps
        if max_delay > 0:
            # Circular buffer of shape (max_delay+1, num_envs, action_dim).
            self._action_delay_buf = torch.zeros(
                (max_delay + 1, self.num_envs, self.cfg.action_space),
                dtype=torch.float, device=self.device,
            )
            self._action_delay_max_buf = max_delay  # capacity minus 1
            self._action_delay_head = 0             # newest slot
            # Per-env delay (samples to look back, in [0, max_delay]). Default
            # to `action_delay_steps` for all envs; DR overwrites in _reset_idx.
            default = min(max(self.action_delay_steps, 0), max_delay)
            self._env_action_delay = torch.full(
                (self.num_envs,), default, dtype=torch.long, device=self.device,
            )
            print(f"[INFO] Action delay enabled: max_buffer={max_delay}, "
                  f"default={default}, randomize={self._randomize_action_delay} "
                  f"(range=[{self._action_delay_min}, {self._action_delay_max}])")
        else:
            self._action_delay_buf = None
            self._env_action_delay = None

        # buffers for position targets
        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        
        # buffers for object
        self.object_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.object_pos_prev = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_rot_prev = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.last_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device,)

        # buffers for data
        # History buffer for proprioceptive observations (needed for ProprioAdapt)
        # ProprioAdaptTConv expects input_shape[0]//3 dimensions
        # For observation_space=583, this is 583//3 = 194
        self.proprio_hist_dim = self.cfg.observation_space // 3
        self.obs_buf_lag_history = torch.zeros((self.num_envs, max(80, self.cfg.prop_hist_len + 10), self.proprio_hist_dim), device=self.device, dtype=torch.float)
        self.at_reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.proprio_hist_buf = torch.zeros((self.num_envs, self.cfg.prop_hist_len, self.proprio_hist_dim), device=self.device, dtype=torch.float)
        self.priv_info_buf = torch.zeros((self.num_envs, self.cfg.priv_info_dim), device=self.device, dtype=torch.float)
        self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.reward_execute = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        
        # finger bodies
        self.finger_bodies = list()
        for body_name in self.cfg.fingertip_body_names:
            self.finger_bodies.append(self.hand.body_names.index(body_name))
        self.num_fingertips = len(self.finger_bodies)

        # joint limits (only for hand joints, not arm joints)
        joint_pos_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        # Extract limits only for actuated hand joints
        if joint_pos_limits.ndim == 3:
            # Shape: [num_envs, num_dofs, 2]
            self.hand_dof_lower_limits = joint_pos_limits[:, self.actuated_dof_indices, 0] * self.cfg.dof_limits_scale
            self.hand_dof_upper_limits = joint_pos_limits[:, self.actuated_dof_indices, 1] * self.cfg.dof_limits_scale
        else:
            # Shape: [num_dofs, 2]
            self.hand_dof_lower_limits = joint_pos_limits[self.actuated_dof_indices, 0] * self.cfg.dof_limits_scale
            self.hand_dof_upper_limits = joint_pos_limits[self.actuated_dof_indices, 1] * self.cfg.dof_limits_scale

        # Tighten hand joint limits to the real-hand reachable range (measured on
        # the Sharpa Wave hand) so sim
        # action space and reset sampling match the deployable range. Real limits
        # are in cfg (Sharpa) order; reorder to sorted `actuated_dof_indices` layout.
        if getattr(self.cfg, 'use_real_hand_limits', True) and self.num_hand_dofs == 22:
            from dexx.tasks.franka_sharpa.sim2real.real_hand_limits import (
                SHARPA_REAL_LOWER_NP, SHARPA_REAL_UPPER_NP,
            )
            real_lower_cfg = torch.as_tensor(SHARPA_REAL_LOWER_NP,
                                             dtype=self.hand_dof_lower_limits.dtype,
                                             device=self.device)
            real_upper_cfg = torch.as_tensor(SHARPA_REAL_UPPER_NP,
                                             dtype=self.hand_dof_upper_limits.dtype,
                                             device=self.device)
            # cfg[j] -> position in sorted actuated_dof_indices
            full2sorted = {idx: pos for pos, idx in enumerate(self.actuated_dof_indices)}
            cfg_full = [self.hand.joint_names.index(n) for n in self.cfg.actuated_joint_names]
            cfg_to_sorted = torch.tensor([full2sorted[f] for f in cfg_full],
                                         dtype=torch.long, device=self.device)
            real_lower_sorted = torch.empty_like(real_lower_cfg)
            real_upper_sorted = torch.empty_like(real_upper_cfg)
            real_lower_sorted[cfg_to_sorted] = real_lower_cfg
            real_upper_sorted[cfg_to_sorted] = real_upper_cfg

            self.hand_dof_lower_limits = torch.maximum(self.hand_dof_lower_limits, real_lower_sorted)
            self.hand_dof_upper_limits = torch.minimum(self.hand_dof_upper_limits, real_upper_sorted)
            print(f"[FrankaSharpaEnv] Applied real-hand limits to hand_dof_lower/upper "
                  f"(dof_limits_scale={self.cfg.dof_limits_scale}).")

        self.p_gain_default = self.hand.data.default_joint_stiffness[:, self.actuated_dof_indices].clone()
        self.d_gain_default = self.hand.data.default_joint_damping[:, self.actuated_dof_indices].clone()

        self.p_gain = self.p_gain_default.clone()
        self.d_gain = self.d_gain_default.clone()

        if self.cfg.torque_control:
            print("using torque control")
            self.hand.data.default_joint_stiffness = torch.zeros_like(self.p_gain_default, device=self.device)
            self.hand.data.default_joint_damping = torch.zeros_like(self.d_gain_default, device=self.device)
        
            for key, act in self.hand.actuators.items():
                act.stiffness = torch.zeros_like(act.stiffness, device=self.device)
                act.damping = torch.zeros_like(act.damping, device=self.device)

        # contact buffers
        self._contact_body_ids = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        self._contact_body_ids_disable = torch.tensor(self.cfg.disable_tactile_ids, dtype=torch.long)
        CONTACT_HISTORY_LEN = 3
        # last smoothed contact force
        self.last_contacts = torch.zeros((self.num_envs, len(self._contact_body_ids)), dtype=torch.float, device=self.device)
        self.tips_contact_history = torch.ones(self.num_envs, CONTACT_HISTORY_LEN, len(self._contact_body_ids), device=self.device).bool()
        # Eval-only per-env seq_idx range overrides. Set via
        # `env.set_eval_init_range(lo_per_env, hi_per_env)` to force resets
        # within [lo, hi] inclusive (per-env). Set to None to disable.
        self._eval_init_seq_idx_lo: torch.Tensor | None = None
        self._eval_init_seq_idx_hi: torch.Tensor | None = None
        self.elastomer_ids = [self.hand.body_names.index(body_name) for body_name in 
                              [f"{self.hand_side}_thumb_elastomer", 
                               f"{self.hand_side}_index_elastomer", 
                               f"{self.hand_side}_middle_elastomer",
                               f"{self.hand_side}_ring_elastomer", 
                               f"{self.hand_side}_pinky_elastomer"]]

        # align real
        # Update effort limit for hand joints actuator
        if "hand_joints" in self.hand.actuators:
            self.hand.actuators['hand_joints'].effort_limit *= self.cfg.current_coef
        elif "joints" in self.hand.actuators:
            # Fallback for backward compatibility
            self.hand.actuators['joints'].effort_limit *= self.cfg.current_coef

        # randomize
        if self.cfg.randomize_friction:
            rand_friction = torch.empty(self.num_envs).uniform_(self.cfg.randomize_friction_scale_lower, self.cfg.randomize_friction_scale_upper)
            rand_friction = rand_friction.reshape(self.num_envs, 1)
            rand_friction_object = rand_friction.clone() * self.cfg.object_base_friction
            self.set_friction(self.object, rand_friction_object, self.num_envs)
            # Elastomer collision-shape indices into the material array. These are
            # asset-dependent: the stock merged FR3+Wave articulation cooks to 34
            # PhysX shapes (8 arm + 26 hand) and the elastomers land on
            # [27,28,30,32,33] -- verified with tools/calibrate_elastomer_ids.py.
            # Re-calibrate after ANY asset change: ids that fall outside the actual
            # shape count are dropped by the filter below, which disables elastomer
            # friction DR silently.
            material_elastomer_ids = getattr(self.cfg, "material_elastomer_ids", None) or [27, 28, 30, 32, 33]
            # Size the friction buffer from the ACTUAL shape count (not a hardcoded 34).
            n_shapes = self.hand.root_physx_view.get_material_properties().shape[1]
            rand_friction_hand = rand_friction.clone().repeat(1, n_shapes) * self.cfg.metal_base_friction
            _elast_ids = [i for i in material_elastomer_ids if i < n_shapes]
            rand_friction_hand[:, _elast_ids] = rand_friction_hand[:, _elast_ids] / self.cfg.metal_base_friction * self.cfg.elastomer_base_friction
            self.set_friction(self.hand, rand_friction_hand, self.num_envs)
            self.priv_info_buf[:, 22] = rand_friction.squeeze()
        if self.cfg.randomize_com:
            rand_com = torch.empty([self.num_envs, 3]).uniform_(self.cfg.randomize_com_lower, self.cfg.randomize_com_upper)
            self.set_com(self.object, rand_com, self.num_envs)
            self.priv_info_buf[:, 24:27] = self.object.root_physx_view.get_coms().reshape(self.num_envs, -1)[:, :3]
        if self.cfg.randomize_mass:
            rand_mass = torch.empty(self.num_envs).uniform_(self.cfg.randomize_mass_lower, self.cfg.randomize_mass_upper)
            self.set_mass(self.object, rand_mass, self.num_envs)
            self.priv_info_buf[:, 23] = self.object.root_physx_view.get_masses().reshape(self.num_envs)

        # physics_sim_view
        self.physics_sim_view: physx.SimulationView = sim_utils.SimulationContext.instance().physics_sim_view
        
        # Gravity scheduler (curriculum)
        self.gravity_scheduler_enabled = self.cfg.gravity_scheduler_enabled
        
        # Virtual object force curriculum buffers
        if hasattr(self.cfg, 'virtual_object_force_enabled') and self.cfg.virtual_object_force_enabled:
            self.virtual_object_force_enabled = True
            self.virtual_object_force = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
            self.virtual_object_torque = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        else:
            self.virtual_object_force_enabled = False

        # Auto-disable debug_draw in headless mode
        if self.cfg.debug_draw and not self.sim.has_gui():
            self.cfg.debug_draw = False
        if self.cfg.debug_draw:
            self.debug_draw = DebugDraw()
        

    def _setup_scene(self):
        print("Setting up scene ... ")
        # add hand, in-hand object, and goal object
        self.hand = Articulation(self.cfg.robot_cfg)
        # # -------------------------------------
        # for multiple object
        self.object_urdf_path_list = []
        self.object_scale_list = []
        _missing_assets = []
        for data_idx in self.data_indices:
            dset = self.demo_dataset_dict[ManipDataFactory.dataset_type(data_idx)][data_idx]
            asset_path = dset["obj_urdf_path"]
            self.object_urdf_path_list.append(asset_path)
            self.object_scale_list.append(float(dset.get("obj_scale", 1.0)))
            # Sanity-check the asset path exists on disk now, so we fail with a
            # clear message tied to the offending data_idx rather than later
            # under a CUDA "_external_force_b out-of-bounds" assert (which is
            # what happens when MultiAssetSpawnerCfg silently fails to spawn a
            # body for some envs).
            if asset_path and not os.path.exists(asset_path):
                _missing_assets.append((data_idx, asset_path))
        if _missing_assets:
            msg_lines = ["Object asset paths missing on disk:"]
            for di, p in _missing_assets:
                msg_lines.append(f"  data_idx={di} -> {p}")
            raise FileNotFoundError("\n".join(msg_lines))
        # Body / collision / mass props applied to each spawned instance.
        # Defined once and forwarded into every inner per-asset cfg below so
        # rigid body API is attached even for the USD branch — without it,
        # USD-spawned prims under MultiAssetSpawnerCfg can come out without
        # rigid-body API, leaving num_instances < num_envs and tripping
        # `_external_force_b[env_ids] = 0` at the first reset.
        _shared_rigid_props = sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            disable_gravity=False,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0025,
            max_depenetration_velocity=1000.0,
        )
        _shared_collision_props = sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.002,
            rest_offset=0.0,
        )
        _shared_mass_props = sim_utils.MassPropertiesCfg(mass=0.05)

        # Helper: build per-asset spawn cfg, picking USD vs URDF by extension.
        # The dataset returns hand-authored tool.usd when present (preferred for
        # better collision approximation), otherwise the auto-generated .urdf.
        def _build_asset_cfg(asset_path, scale):
            ext = os.path.splitext(asset_path)[1].lower()
            if ext == ".usd" or ext == ".usda" or ext == ".usdc":
                return sim_utils.UsdFileCfg(
                    usd_path=asset_path,
                    scale=(scale, scale, scale),
                    rigid_props=_shared_rigid_props,
                    collision_props=_shared_collision_props,
                    mass_props=_shared_mass_props,
                )
            # Default: URDF
            return sim_utils.UrdfFileCfg(
                asset_path=asset_path,
                fix_base=False,
                joint_drive=None,
                scale=(scale, scale, scale),
                make_instanceable=False,
                rigid_props=_shared_rigid_props,
                collision_props=_shared_collision_props,
                mass_props=_shared_mass_props,
            )

        object_contact_filter_path = "/World/envs/env_.*/object"
        if len(self.object_urdf_path_list) > 1:
            object_spawn_prim_path = "/World/envs/env_.*/object"
            multiasset_spawn_cfg = sim_utils.MultiAssetSpawnerCfg(
                assets_cfg=[
                    _build_asset_cfg(asset_path, s)
                    for asset_path, s in zip(self.object_urdf_path_list, self.object_scale_list)
                ],
                random_choice=False,
                rigid_props=_shared_rigid_props,
                collision_props=_shared_collision_props,
                mass_props=_shared_mass_props,
            )
            object_init_state = RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, 1.0),
                rot=(1.0, 0.0, 0.0, 0.0),
            )

            if getattr(self.cfg, "bind_multiasset_object_root_link", False):
                # Robotool URDFs spawn a wrapper prim at /object and the actual
                # rigid body at /object/<root_link>. RigidObject must bind to
                # the rigid body prim.
                multiasset_spawn_cfg.func(
                    object_spawn_prim_path,
                    multiasset_spawn_cfg,
                    translation=object_init_state.pos,
                    orientation=object_init_state.rot,
                )
                root_link_name = getattr(self.cfg, "multiasset_object_root_link_name", "base")
                object_rigid_prim_path = f"{object_spawn_prim_path}/{root_link_name}"
                object_cfg = RigidObjectCfg(
                    prim_path=object_rigid_prim_path,
                    spawn=None,
                    init_state=object_init_state,
                )
                object_contact_filter_path = object_rigid_prim_path
            else:
                object_cfg = RigidObjectCfg(
                    prim_path=object_spawn_prim_path,
                    spawn=multiasset_spawn_cfg,
                    init_state=object_init_state,
                )

            self.object = RigidObject(object_cfg)

        else:
            # Single object: if dataset provided a USD, replace cfg.object_cfg.spawn
            # with a UsdFileCfg (cfg's default spawn type is UrdfFileCfg). Otherwise
            # keep the cfg URDF spawn and just point asset_path at the dataset's URDF.
            single_path = self._env0_obj_urdf
            ext = os.path.splitext(single_path)[1].lower()
            scale_val = self.object_scale_list[0] if self.object_scale_list else 1.0

            if ext in (".usd", ".usda", ".usdc"):
                # Build a fresh USD spawn cfg, preserving rigid/collision/mass
                # props from the original cfg.object_cfg if available.
                orig_spawn = self.cfg.object_cfg.spawn
                self.cfg.object_cfg.spawn = sim_utils.UsdFileCfg(
                    usd_path=single_path,
                    scale=(scale_val, scale_val, scale_val),
                    rigid_props=getattr(orig_spawn, "rigid_props", None),
                    collision_props=getattr(orig_spawn, "collision_props", None),
                    mass_props=getattr(orig_spawn, "mass_props", None),
                )
                print(f"[INFO] Object asset = USD: {single_path}")
            else:
                self.cfg.object_cfg.spawn.asset_path = single_path
                if scale_val != 1.0:
                    self.cfg.object_cfg.spawn.scale = (scale_val, scale_val, scale_val)
                print(f"[INFO] Object asset = URDF: {single_path}")
            if scale_val != 1.0:
                print(f"[INFO] Object scale from retarget data: {scale_val}")
            self.object = RigidObject(self.cfg.object_cfg)

        # -------- Aux object (optional, static / kinematic) --------
        # Mixed aux / no-aux batches are supported: env where the demo has no
        # aux gets a placeholder spawned (= reuses the FIRST real aux asset,
        # placed underground at z=-10) so the MultiAssetSpawner sees a uniform
        # prim hierarchy across all envs. The placeholder is invisible during
        # training (it lives below the table and never moves).
        self.aux_object = None
        self.has_aux = False
        self.aux_urdf_path_list = []
        self.aux_scale_list = []
        self.aux_present_mask_list = []   # per-env bool: True = this env has a real aux

        aux_flags = []
        for data_idx in self.data_indices:
            dset = self.demo_dataset_dict[ManipDataFactory.dataset_type(data_idx)][data_idx]
            aux_flags.append(bool(dset.get("has_aux", False)))

        self.has_aux = any(aux_flags)
        self.aux_present_mask_list = list(aux_flags)

        if self.has_aux:
            # Find a placeholder aux asset (first real one) for non-aux envs.
            placeholder_path = None
            for data_idx, flag in zip(self.data_indices, aux_flags):
                if not flag:
                    continue
                dset = self.demo_dataset_dict[ManipDataFactory.dataset_type(data_idx)][data_idx]
                p = dset.get("aux_obj_urdf_path", "")
                if p and os.path.exists(p):
                    placeholder_path = p
                    break
            if placeholder_path is None:
                raise RuntimeError("has_aux=True but no usable aux asset found in batch")

            n_real = sum(aux_flags)
            n_placeholder = len(aux_flags) - n_real
            print(f"[INFO] Aux mixed batch: {n_real}/{len(aux_flags)} real aux, "
                  f"{n_placeholder} placeholders (using {os.path.basename(placeholder_path)} "
                  f"scaled tiny, hidden underground)")

            for data_idx, flag in zip(self.data_indices, aux_flags):
                dset = self.demo_dataset_dict[ManipDataFactory.dataset_type(data_idx)][data_idx]
                if flag:
                    p = dset.get("aux_obj_urdf_path", "")
                    if not p or not os.path.exists(p):
                        raise FileNotFoundError(
                            f"aux asset missing for data_idx={data_idx}: {p!r}"
                        )
                    self.aux_urdf_path_list.append(p)
                    self.aux_scale_list.append(float(dset.get("aux_obj_scale", 1.0)))
                else:
                    # Placeholder: same asset shape (so MultiAssetSpawner views
                    # are uniform), tiny scale so it's effectively a speck.
                    self.aux_urdf_path_list.append(placeholder_path)
                    self.aux_scale_list.append(0.01)

            _aux_shared_rigid = sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,     # ← aux is static; PhysX won't simulate dynamics
                disable_gravity=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                max_depenetration_velocity=1000.0,
            )
            _aux_shared_collision = sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.002,
                rest_offset=0.0,
            )
            _aux_shared_mass = sim_utils.MassPropertiesCfg(mass=0.5)  # arbitrary; kinematic ignores it

            def _build_aux_asset_cfg(asset_path, scale):
                ext = os.path.splitext(asset_path)[1].lower()
                if ext in (".usd", ".usda", ".usdc"):
                    return sim_utils.UsdFileCfg(
                        usd_path=asset_path,
                        scale=(scale, scale, scale),
                        rigid_props=_aux_shared_rigid,
                        collision_props=_aux_shared_collision,
                        mass_props=_aux_shared_mass,
                    )
                return sim_utils.UrdfFileCfg(
                    asset_path=asset_path,
                    fix_base=False,
                    joint_drive=None,
                    scale=(scale, scale, scale),
                    make_instanceable=False,
                    rigid_props=_aux_shared_rigid,
                    collision_props=_aux_shared_collision,
                    mass_props=_aux_shared_mass,
                )

            aux_cfg = RigidObjectCfg(
                prim_path="/World/envs/env_.*/aux_object",
                spawn=sim_utils.MultiAssetSpawnerCfg(
                    assets_cfg=[_build_aux_asset_cfg(p, s)
                                 for p, s in zip(self.aux_urdf_path_list, self.aux_scale_list)],
                    random_choice=False,
                    rigid_props=_aux_shared_rigid,
                    collision_props=_aux_shared_collision,
                    mass_props=_aux_shared_mass,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 1.0), rot=(1.0, 0.0, 0.0, 0.0),
                ),
            )
            self.aux_object = RigidObject(aux_cfg)
            print(f"[INFO] Aux object enabled ({len(self.aux_urdf_path_list)} assets, "
                  f"kinematic=True).")

        self.table = RigidObject(self.cfg.table_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate (no need to filter for this environment)
        # self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions()

        # add articulation to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["table"] = self.table
        self.scene.rigid_objects["object"] = self.object
        if self.aux_object is not None:
            self.scene.rigid_objects["aux_object"] = self.aux_object
            # Per-env mask: True = this env has a real aux (write real pose at reset),
            # False = placeholder (force underground at z=-10).
            # data_indices cycles through self.data_indices; map each env to its slot.
            n_idx = len(self.data_indices)
            self._aux_present_mask = torch.tensor(
                [self.aux_present_mask_list[i % n_idx] for i in range(self.num_envs)],
                dtype=torch.bool, device=self.device,
            )

        if getattr(self.cfg, "bind_multiasset_object_root_link", False):
            for contact_sensor_cfg in self.cfg.contact_sensor:
                contact_sensor_cfg.filter_prim_paths_expr = [object_contact_filter_path]

        # contact sensors
        self._contact_sensor = []
        for id in range(len(self.cfg.contact_sensor)):
            self._contact_sensor.append(ContactSensor(self.cfg.contact_sensor[id]))
            self.scene.sensors[f"contact_sensor_{id}"] = self._contact_sensor[id]


        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        actions = saturate(actions, torch.tensor(-self.cfg.clip_actions), torch.tensor(self.cfg.clip_actions))

        # Apply per-env action delay via a circular buffer.
        #   buffer shape:   (max_delay+1, num_envs, action_dim)
        #   head (writer):  points at slot holding the NEWEST action
        #   read index:     (head - env_delay) mod (max_delay+1) per env
        # Delay=0 → read the slot we're about to overwrite (the previous step's
        # newest action — effectively no delay once we write). This is handled
        # by writing AFTER reading, so delay=0 means current action is executed.
        if self._action_delay_buf is not None:
            max_buf = self._action_delay_max_buf  # capacity-1
            cap = max_buf + 1
            # write newest at head
            self._action_delay_buf[self._action_delay_head] = actions
            # compute per-env read slot
            env_delay = self._env_action_delay if self._env_action_delay is not None else None
            if env_delay is None:
                delayed_actions = self._action_delay_buf[
                    (self._action_delay_head - max_buf) % cap
                ].clone()
            else:
                read_idx = (self._action_delay_head - env_delay) % cap   # [num_envs]
                env_idx = torch.arange(self.num_envs, device=self.device)
                delayed_actions = self._action_delay_buf[read_idx, env_idx].clone()
            # advance head
            self._action_delay_head = (self._action_delay_head + 1) % cap
            actions = delayed_actions

        self.actions = actions.clone()

        # Action structure depends on control mode:
        # freeze_arm:              [hand(22)] = 22
        # use_pid_control:         [pos_error(3), rot_error_6d(6), hand(22)] = 31
        # use_osc_control:         [pos_error(3), rot_error_6d(6), hand(22)] = 31
        # use_joint_pos_control:   [arm_joint_pos(7), hand(22)] = 29
        # use_joint_delta_control: [arm_joint_delta(7), hand(22)] = 29
        # force/torque:            [force(3), torque(3), hand(22)] = 28

        if self.freeze_arm:
            root_control_dim = 0
        elif self.use_joint_delta_control:
            root_control_dim = 7
        elif self.use_pid_control or self.use_osc_control:
            root_control_dim = 9
        elif self.use_joint_pos_control:
            root_control_dim = 7
        else:
            root_control_dim = 6

        # Compute and store wrist action magnitude
        if root_control_dim > 0:
            wrist_actions = actions[:, :3]
            wrist_action_magnitude = torch.norm(wrist_actions, dim=-1)
        else:
            wrist_action_magnitude = torch.zeros(self.num_envs, device=self.device)
        self.wrist_action_magnitude_buf = wrist_action_magnitude.clone()

        # Process joint actions
        dof_actions = actions[:, root_control_dim:root_control_dim + self.num_hand_dofs]
        dof_actions = torch.clamp(dof_actions, -1, 1)
        
        # Scale actions to joint limits
        self.cur_targets = scale(
            dof_actions,
            self.hand_dof_lower_limits,
            self.hand_dof_upper_limits,
        )
        # Apply moving average smoothing
        act_moving_average = self.actions_moving_average
        self.cur_targets = (
            act_moving_average * self.cur_targets
            + (1.0 - act_moving_average) * self.prev_targets
        )
        self.cur_targets = saturate(
            self.cur_targets,
            self.hand_dof_lower_limits,
            self.hand_dof_upper_limits,
        )
        self.actions = actions
        # Process arm control (skip if arm is frozen — arm_joint_pos_des stays at reset value)
        if self.freeze_arm:
            pass  # arm_joint_pos_des unchanged from reset
        elif self.use_joint_delta_control:
            # Joint delta control: action[-1..1] * delta_scale → add to current joint pos
            arm_delta_actions = actions[:, :root_control_dim]
            arm_delta_actions = torch.clamp(arm_delta_actions, -1, 1)

            self._refresh_lab()
            arm_joint_pos_des = self.arm_joint_pos + arm_delta_actions * self.cfg.joint_delta_scale

            # Smooth — use arm-specific alpha (default 0.15, more aggressive LP than
            # hand's 0.4) to approximate real Franka impedance ~50ms bandwidth and
            # damp sim2real arm shake. Falls back to global alpha if cfg missing.
            _arm_alpha = getattr(self.cfg, 'arm_actions_moving_average', act_moving_average)
            arm_joint_pos_des = (
                _arm_alpha * arm_joint_pos_des
                + (1.0 - _arm_alpha) * self.arm_joint_pos_des_prev
            )
            # Clamp to joint limits
            arm_joint_pos_des = saturate(
                arm_joint_pos_des,
                self.arm_joint_lower_limits,
                self.arm_joint_upper_limits,
            )
            self.arm_joint_pos_des_prev = arm_joint_pos_des.clone()
            self.arm_joint_pos_des = arm_joint_pos_des.clone()
        elif self.use_pid_control or self.use_osc_control:

            ###### using velocity control ######
            # translation_vel = actions[:, :3]  # [num_envs, 3]
            # rotation_vel = actions[:, 3:9]  # [num_envs, 6]
            # rotation_vel_aa = rot6d_to_aa(rotation_vel)  # [num_envs, 3]
            # dt = self.physics_dt * 5
            # position_error = translation_vel * dt
            # rotation_error_6d = rotation_vel * dt
            # rotation_error_aa = rot6d_to_aa(rotation_error_6d)
            cur_idx = self._get_demo_idx()
            ###################################################################
            ###### using position control ###### Get wrist pose error from actions
            position_error = actions[:, :3] * self.translation_scale   # [num_envs, 3], scaled from [-1,1] to meters
            rotation_error_6d = actions[:, 3:9] * self.orientation_scale   # [num_envs, 6], scaled
            rotation_error_aa = rot6d_to_aa(rotation_error_6d)  # [num_envs, 3]
            
            # # Get current wrist pose
            self._refresh_lab()

            current_wrist_pos = self.base_pos  # [num_envs, 3]
            current_wrist_quat = self.base_quat  # [num_envs, 4]
            
            
            # Compute target wrist pose from error
            # Target position = current position + position_error
            target_wrist_pos = current_wrist_pos + position_error
            
            # Target rotation: convert error to quaternion and multiply with current quaternion
            rotation_error_quat = aa_to_quat(rotation_error_aa)  # [num_envs, 4]
            target_wrist_quat = quat_mul(rotation_error_quat, current_wrist_quat)  # [num_envs, 4]

            # ####### Canchoose:
            # Apply moving average smoothing to wrist target (similar to hand joints)
            self.cur_wrist_target_pos = (
                act_moving_average * target_wrist_pos
                + (1.0 - act_moving_average) * self.prev_wrist_target_pos
            )
            # For quaternion smoothing, we need to use SLERP or normalize after interpolation
            # Using simple interpolation and normalization
            self.cur_wrist_target_quat = (
                act_moving_average * target_wrist_quat
                + (1.0 - act_moving_average) * self.prev_wrist_target_quat
            )
            self.cur_wrist_target_pos = target_wrist_pos
            self.cur_wrist_target_quat = target_wrist_quat
            # Normalize quaternion
            quat_norm = torch.norm(self.cur_wrist_target_quat, dim=-1, keepdim=True)
            self.cur_wrist_target_quat = self.cur_wrist_target_quat / (quat_norm + 1e-8)
            
            # Update previous targets for next iteration
            self.prev_wrist_target_pos = self.cur_wrist_target_pos.clone()
            self.prev_wrist_target_quat = self.cur_wrist_target_quat.clone()
            
            # Convert target wrist pose from world frame (minus env_origins) to root frame
            # IK controller.compute() receives ee in root frame, so command must also be in root frame
            root_pose_w = self.hand.data.root_pose_w
            target_pos_w = self.cur_wrist_target_pos + self.scene.env_origins  # restore to world frame
            target_pos_b, target_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7],
                target_pos_w, self.cur_wrist_target_quat,
            )
            ik_commands = torch.cat([target_pos_b, target_quat_b], dim=-1)  # [num_envs, 7] in root frame

            # Debug: verify coordinate frame alignment (print every 500 steps for env 0)
            if not hasattr(self, '_ik_debug_counter'):
                self._ik_debug_counter = 0
            self._ik_debug_counter += 1
            if self._ik_debug_counter % 500 == 1:
                ee_pose_w = self.hand.data.body_pose_w[:, self.arm_robot_entity_cfg.body_ids[0]]
                ee_pos_b_check, _ = subtract_frame_transforms(
                    root_pose_w[0, 0:3], root_pose_w[0, 3:7], ee_pose_w[0, 0:3], ee_pose_w[0, 3:7]
                )
                ik_pos_error = target_pos_b[0] - ee_pos_b_check

                # Get demo target for comparison
                demo_idx = self._get_demo_idx()
                demo_wrist_pos = self.demo_data["wrist_pos"][0, demo_idx[0]]
                demo_wrist_rot = self.demo_data["wrist_rot"][0, demo_idx[0]]

                print(f"[IK DEBUG] step={self._ik_debug_counter}, demo_frame={demo_idx[0].item()}")
                print(f"  root_pos_w[0]:         {root_pose_w[0, :3].cpu().numpy()}")
                print(f"  env_origins[0]:         {self.scene.env_origins[0].cpu().numpy()}")
                print(f"  arm_base_pos (cfg):     {self.cfg.arm_base_pos}")
                print(f"  --- Demo target ---")
                print(f"  demo_wrist_pos:         {demo_wrist_pos.cpu().numpy()}")
                print(f"  --- Current state ---")
                print(f"  cur_wrist_pos (w-eo):   {current_wrist_pos[0].cpu().detach().numpy()}")
                print(f"  cur_wrist_pos world Z:  {(current_wrist_pos[0, 2] + self.scene.env_origins[0, 2]).item():.4f}")
                print(f"  --- Policy action ---")
                print(f"  action pos_error:       {position_error[0].cpu().detach().numpy()}")
                print(f"  action rot_error (aa):  {rotation_error_aa[0].cpu().detach().numpy()}")
                print(f"  --- IK target ---")
                print(f"  target_pos (w-eo):      {self.cur_wrist_target_pos[0].cpu().detach().numpy()}")
                print(f"  target_pos_b (root):    {target_pos_b[0].cpu().detach().numpy()}")
                print(f"  ee_pos_b (root):        {ee_pos_b_check.cpu().numpy()}")
                print(f"  IK pos_error:           {ik_pos_error.cpu().detach().numpy()}")
                print(f"  --- Sanity check ---")
                print(f"  demo-current gap:       {(demo_wrist_pos - current_wrist_pos[0]).cpu().detach().numpy()}")
            ########################################################

            # Type 2: data wrist pose + delta action 
            
            # Use advanced indexing: [torch.arange(num_envs), cur_idx] to select different indices for each environment
            # target_wrist_pos = self.demo_data["opt_wrist_pos"][torch.arange(self.num_envs), cur_idx] + actions[:, :3] * self.translation_scale  # [num_envs, 3]
            # # Convert 6D rotation action to axis-angle, then add to demo wrist rotation
            # rotation_action_6d = actions[:, 3:9] * self.orientation_scale  # [num_envs, 6]
            # rotation_action_aa = rot6d_to_aa(rotation_action_6d)  # [num_envs, 3]
            # target_wrist_rot_aa = self.demo_data["opt_wrist_rot"][torch.arange(self.num_envs), cur_idx] + rotation_action_aa  # [num_envs, 3]
            # target_wrist_quat = aa_to_quat(target_wrist_rot_aa)  # [num_envs, 4]


            # # Type 4: only reference demo data
            # target_wrist_pos = self.demo_data["opt_wrist_pos"][torch.arange(self.num_envs), cur_idx] # [num_envs, 3]
            # # Convert 6D rotation action to axis-angle, then add to demo wrist rotation
            # rotation_action_6d = actions[:, 3:9] * self.orientation_scale  # [num_envs, 6]
            # rotation_action_aa = rot6d_to_aa(rotation_action_6d)  # [num_envs, 3]
            # target_wrist_rot_aa = self.demo_data["opt_wrist_rot"][torch.arange(self.num_envs), cur_idx] # [num_envs, 3]
            # target_wrist_quat = aa_to_quat(target_wrist_rot_aa)  # [num_envs, 4]
            
            
            
            # # Prepare IK command: [pos(3), quat(4)]
            # ik_commands = torch.cat([target_wrist_pos, target_wrist_quat], dim=-1)  # [num_envs, 7]
            
            # Set command for arm controller (Diff IK or OSC)
            if self.use_osc_control:
                # OSC expects [pos(3), quat(4)] + optional Kp, same as ik_commands
                osc_command = torch.zeros(self.num_envs, self._osc.action_dim, device=self.device)
                osc_command[:, :7] = ik_commands
                self._osc.set_command(command=osc_command)
            else:
                self.arm_diff_ik_controller.set_command(ik_commands)
            
            # # Get current quantities from simulation for IK computation
            # jacobian = self.hand.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.arm_robot_entity_cfg.joint_ids]
            # ee_pose_w = self.hand.data.body_pose_w[:, self.arm_robot_entity_cfg.body_ids[0]]
            # root_pose_w = self.hand.data.root_pose_w
            # joint_pos = self.hand.data.joint_pos[:, self.arm_robot_entity_cfg.joint_ids]
            # # joint_pos = self.arm_joint_pos_des.clone()
            
            # # Compute frame in root frame
            # ee_pos_b, ee_quat_b = subtract_frame_transforms(
            #     root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            # )
            
            # # Compute target joint positions using Differential IK controller
            # arm_joint_pos_des = self.arm_diff_ik_controller.compute(
            #     ee_pos_b, ee_quat_b, jacobian, joint_pos
            # )
            
            # self.arm_joint_pos_des = arm_joint_pos_des.clone()
        elif self.use_joint_pos_control:
            # not working well
            arm_dof_actions = actions[:, :root_control_dim]
            arm_dof_actions = torch.clamp(arm_dof_actions, -1, 1)
        
            # Scale actions to joint limits
            arm_joint_pos_des = scale(
                arm_dof_actions,
                self.arm_joint_lower_limits,
                self.arm_joint_upper_limits,
            )
            # smooth — use arm-specific alpha if set (more aggressive LP for sim2real arm-shake)
            _arm_alpha = getattr(self.cfg, 'arm_actions_moving_average', act_moving_average)
            arm_joint_pos_des_smooth = (
                _arm_alpha * arm_joint_pos_des
                + (1.0 - _arm_alpha) * self.arm_joint_pos_des_prev
            )
            arm_joint_pos_des_smooth = saturate(
                arm_joint_pos_des_smooth,
                self.arm_joint_lower_limits,
                self.arm_joint_upper_limits,
            )
            self.arm_joint_pos_des_prev = arm_joint_pos_des_smooth.clone()
            self.arm_joint_pos_des = arm_joint_pos_des_smooth.clone()
        else:
            # Direct force/torque control mode
            translation_scale = self.translation_scale
            orientation_scale = self.orientation_scale
            
            self.apply_forces = (
                act_moving_average * (actions[:, 0:3] * self.physics_dt * translation_scale * 500)
                + (1.0 - act_moving_average) * self.apply_forces
            )
            self.apply_torque = (
                act_moving_average * (actions[:, 3:6] * self.physics_dt * orientation_scale * 200)
                + (1.0 - act_moving_average) * self.apply_torque
            )
            # Apply forces and torques to wrist
            self.hand.set_external_force_and_torque(
                forces=self.apply_forces.unsqueeze(1),
                torques=self.apply_torque.unsqueeze(1),
                body_ids=self.wrist_body_idx,
            )
        
        # Apply virtual object force curriculum
        if self.virtual_object_force_enabled and hasattr(self, 'object') and self.object is not None:
            self._refresh_lab()  # Refresh to get current object states
            
            # Compute current gains based on curriculum
            if hasattr(self.cfg, 'virtual_object_force_decay_method') and self.cfg.virtual_object_force_decay_method != "None":
                last_step = self.common_step_counter
                kp_initial = getattr(self.cfg, 'virtual_object_kp_initial', 100.0)
                kp_final = getattr(self.cfg, 'virtual_object_kp_final', 0.0)
                kd_initial = getattr(self.cfg, 'virtual_object_kd_initial', 10.0)
                kd_final = getattr(self.cfg, 'virtual_object_kd_final', 0.0)
                decay_steps = getattr(self.cfg, 'virtual_object_force_decay_steps', 10000)
                force_scale = getattr(self.cfg, 'virtual_force_scale', 0.5)
                
                if self.cfg.virtual_object_force_decay_method == "linear":
                    progress = min(last_step / decay_steps, 1.0)
                    current_kp = kp_initial + (kp_final - kp_initial) * progress
                    current_kd = kd_initial + (kd_final - kd_initial) * progress
                elif self.cfg.virtual_object_force_decay_method == "exp":
                    progress = min(last_step / decay_steps, 1.0)
                    if kp_initial > 0:
                        current_kp = kp_initial * ((kp_final / kp_initial) ** progress) if kp_final > 0 else kp_initial * (1 - progress)
                    else:
                        current_kp = kp_final
                    if kd_initial > 0:
                        current_kd = kd_initial * ((kd_final / kd_initial) ** progress) if kd_final > 0 else kd_initial * (1 - progress)
                    else:
                        current_kd = kd_final
                elif self.cfg.virtual_object_force_decay_method == "cos":
                    progress = min(last_step / decay_steps, 1.0)
                    cos_progress = (1 - math.cos(progress * math.pi)) / 2
                    current_kp = kp_initial + (kp_final - kp_initial) * cos_progress
                    current_kd = kd_initial + (kd_final - kd_initial) * cos_progress
                else:
                    current_kp = kp_final
                    current_kd = kd_final
            else:
                current_kp = getattr(self.cfg, 'virtual_object_kp_initial', 100.0)
                current_kd = getattr(self.cfg, 'virtual_object_kd_initial', 10.0)
            
            # Get target object states from demo data
            cur_idx = self._get_demo_idx()
            target_obj_transf = self.demo_data["obj_trajectory"][torch.arange(self.num_envs), cur_idx]
            target_obj_pos = target_obj_transf[:, :3, 3]  # [num_envs, 3]
            
            target_obj_quat = rotmat_to_quat(target_obj_transf[:, :3, :3])  # [num_envs, 4]
            
            # Get current object states
            current_obj_pos = self.object.data.root_pos_w - self.scene.env_origins  # [num_envs, 3]
            current_obj_quat = self.object.data.root_quat_w  # [num_envs, 4]
            current_obj_vel = self.object.data.root_lin_vel_w  # [num_envs, 3]
            current_obj_ang_vel = self.object.data.root_ang_vel_w  # [num_envs, 3]
            
            # Compute position error and force: Ft = kp(ˆp_o_t - p_o_t) - kd*v_o_t
            pos_error = target_obj_pos - current_obj_pos  # [num_envs, 3]
            virtual_force = current_kp * pos_error - current_kd * current_obj_vel  # [num_envs, 3]
            virtual_force = virtual_force * force_scale

            # Compute rotation error and torque: Tt = kp*(ˆθ_o_t ⊖ θ_o_t) - kd*ω_o_t
            # Rotation difference: quat_mul(target, quat_conjugate(current))
            rot_error_quat = quat_mul(target_obj_quat, quat_conjugate(current_obj_quat))  # [num_envs, 4]
            # Convert quaternion error to axis-angle for torque computation
            rot_error_aa = axis_angle_from_quat(rot_error_quat)  # [num_envs, 3]
            virtual_torque = current_kp * rot_error_aa - current_kd * current_obj_ang_vel  # [num_envs, 3]
            virtual_torque = virtual_torque * force_scale

            # Store for logging
            self.virtual_object_force = virtual_force
            self.virtual_object_torque = virtual_torque
            self.virtual_object_kp = current_kp
            self.virtual_object_kd = current_kd
            
            # Apply virtual forces and torques to object
            self.object.set_external_force_and_torque(
                forces=virtual_force.unsqueeze(1),  # [num_envs, 1, 3]
                torques=virtual_torque.unsqueeze(1),  # [num_envs, 1, 3]
            )

        # compute action rate
        self.action_rate = self.actions - self.last_actions
        self.last_actions[:] = self.actions

        # # ==== DEBUG wrist force and torque ====
        # print(f"DEBUG force torque command:{self.apply_forces.mean().item()},{self.apply_torque.mean().item()}")


    def _apply_action(self) -> None:
        self._refresh_lab()
        # Set joint position targets for all joints (arm + hand)
        # self.cur_targets contains only hand joint targets (shape: [num_envs, num_hand_dofs])
        # We need to set targets for all joints
        all_joint_targets = torch.zeros((self.num_envs, self.hand.num_joints), device=self.device)
        # Set hand joint targets
        # all_joint_targets[:, self.hand_joint_indices] = self.cur_targets
        all_joint_targets[:, self.actuated_dof_indices] = self.cur_targets
        
        # # Set arm joint targets from Differential IK controller
        # if self.use_pid_control and hasattr(self, 'arm_joint_pos_des'):
        #     # Use desired joint positions from Differential IK controller
        #     all_joint_targets[:, self.arm_joint_indices] = self.arm_joint_pos_des
        # elif hasattr(self, 'arm_joint_indices') and len(self.arm_joint_indices) > 0:
        #     # Fallback: keep arm joints at current positions
        #     if hasattr(self, 'arm_joint_pos'):
        #         all_joint_targets[:, self.arm_joint_indices] = self.arm_joint_pos
        if self.use_osc_control:
            # OSC: compute joint efforts directly (includes gravity comp internally)
            from isaaclab.utils.math import matrix_from_quat, quat_inv, quat_apply_inverse

            arm_joint_ids = self.arm_robot_entity_cfg.joint_ids
            ee_body_id = self.arm_robot_entity_cfg.body_ids[0]

            # Jacobian in world frame → root frame
            jacobian_w = self.hand.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, arm_joint_ids]
            jacobian_b = jacobian_w.clone()
            root_rot_matrix = matrix_from_quat(quat_inv(self.hand.data.root_quat_w))
            jacobian_b[:, :3, :] = torch.bmm(root_rot_matrix, jacobian_b[:, :3, :])
            jacobian_b[:, 3:, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 3:, :])

            # Mass matrix and gravity
            mass_matrix = self.hand.root_physx_view.get_generalized_mass_matrices()[:, arm_joint_ids, :][:, :, arm_joint_ids]
            gravity = self.hand.root_physx_view.get_gravity_compensation_forces()
            if not isinstance(gravity, torch.Tensor):
                gravity = torch.tensor(gravity, device=self.device, dtype=torch.float32)
            gravity = gravity.reshape(self.num_envs, -1)[:, arm_joint_ids]

            # EE pose in root frame
            root_pose_w = self.hand.data.root_pose_w
            ee_pose_w = self.hand.data.body_pose_w[:, ee_body_id]
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, :3], root_pose_w[:, 3:7], ee_pose_w[:, :3], ee_pose_w[:, 3:7]
            )
            ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)

            # EE velocity in root frame
            ee_vel_w = self.hand.data.body_vel_w[:, ee_body_id, :]
            root_vel_w = self.hand.data.root_vel_w
            relative_vel_w = ee_vel_w - root_vel_w
            ee_lin_vel_b = quat_apply_inverse(self.hand.data.root_quat_w, relative_vel_w[:, :3])
            ee_ang_vel_b = quat_apply_inverse(self.hand.data.root_quat_w, relative_vel_w[:, 3:])
            ee_vel_b = torch.cat([ee_lin_vel_b, ee_ang_vel_b], dim=-1)

            joint_pos = self.hand.data.joint_pos[:, arm_joint_ids]
            joint_vel = self.hand.data.joint_vel[:, arm_joint_ids]

            # Nullspace target: current demo joint positions
            cur_idx = self._get_demo_idx()
            ns_target = self.demo_data["opt_arm_joint_pos"][torch.arange(self.num_envs), cur_idx]

            arm_efforts = self._osc.compute(
                jacobian_b=jacobian_b,
                current_ee_pose_b=ee_pose_b,
                current_ee_vel_b=ee_vel_b,
                mass_matrix=mass_matrix,
                gravity=gravity,
                current_joint_pos=joint_pos,
                current_joint_vel=joint_vel,
                nullspace_joint_pos_target=ns_target,
            )

            # Apply: arm via effort, hand via position target
            all_joint_effort = torch.zeros((self.num_envs, self.hand.num_joints), device=self.device)
            all_joint_effort[:, self.arm_joint_indices] = arm_efforts
            self.hand.set_joint_effort_target(all_joint_effort)
            self.hand.set_joint_position_target(all_joint_targets)

            # Store arm_joint_pos_des for obs/reward consistency
            self.arm_joint_pos_des = joint_pos.clone()

        elif self.use_pid_control:
            jacobian = self.hand.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.arm_robot_entity_cfg.joint_ids]
            ee_pose_w = self.hand.data.body_pose_w[:, self.arm_robot_entity_cfg.body_ids[0]]
            root_pose_w = self.hand.data.root_pose_w
            joint_pos = self.hand.data.joint_pos[:, self.arm_robot_entity_cfg.joint_ids]

            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )

            arm_joint_pos_des = self.arm_diff_ik_controller.compute(
                ee_pos_b, ee_quat_b, jacobian, joint_pos
            )
            self.arm_joint_pos_des = arm_joint_pos_des.clone()

            # Debug
            if hasattr(self, '_ik_debug_counter') and self._ik_debug_counter % 500 == 1:
                ik_cmd = self.arm_diff_ik_controller._command
                ik_pos_err = ik_cmd[0, :3] - ee_pos_b[0]
                print(f"[IK _apply_action DEBUG]")
                print(f"  ee_pos_b[0] (root):    {ee_pos_b[0].cpu().numpy()}")
                print(f"  IK command[0] (root):  {ik_cmd[0, :3].cpu().detach().numpy()}")
                print(f"  IK pos_error:          {ik_pos_err.cpu().detach().numpy()}")
                print(f"  arm_joint_pos_des[0]:   {arm_joint_pos_des[0].cpu().detach().numpy()}")

            all_joint_targets[:, self.arm_joint_indices] = self.arm_joint_pos_des

            # Gravity comp (manual PD + gravity + coriolis)
            if self._arm_gravity_comp:
                gravity_forces = self.hand.root_physx_view.get_gravity_compensation_forces()
                coriolis_forces = self.hand.root_physx_view.get_coriolis_and_centrifugal_compensation_forces()
                if not isinstance(gravity_forces, torch.Tensor):
                    gravity_forces = torch.tensor(gravity_forces, device=self.device, dtype=torch.float32)
                if not isinstance(coriolis_forces, torch.Tensor):
                    coriolis_forces = torch.tensor(coriolis_forces, device=self.device, dtype=torch.float32)
                gravity_forces = gravity_forces.reshape(self.num_envs, -1)
                coriolis_forces = coriolis_forces.reshape(self.num_envs, -1)
                arm_phys_ids = self.hand.actuators["arm_joints"].joint_indices
                arm_gravity = gravity_forces[:, arm_phys_ids]
                arm_coriolis = coriolis_forces[:, arm_phys_ids]
                arm_pos = self.hand.data.joint_pos[:, self.arm_joint_indices]
                arm_vel = self.hand.data.joint_vel[:, self.arm_joint_indices]
                tau_pd = self._arm_K * (self.arm_joint_pos_des - arm_pos) + self._arm_D * (-arm_vel)
                arm_effort = tau_pd + arm_gravity + arm_coriolis
                all_joint_effort = torch.zeros((self.num_envs, self.hand.num_joints), device=self.device)
                all_joint_effort[:, self.arm_joint_indices] = arm_effort
                self.hand.set_joint_effort_target(all_joint_effort)
                self.hand.set_joint_position_target(all_joint_targets)
            else:
                self.hand.set_joint_position_target(all_joint_targets)

        else:
            # Joint delta / joint pos / force-torque modes
            all_joint_targets[:, self.arm_joint_indices] = self.arm_joint_pos_des

            if self._arm_gravity_comp:
                gravity_forces = self.hand.root_physx_view.get_gravity_compensation_forces()
                coriolis_forces = self.hand.root_physx_view.get_coriolis_and_centrifugal_compensation_forces()
                if not isinstance(gravity_forces, torch.Tensor):
                    gravity_forces = torch.tensor(gravity_forces, device=self.device, dtype=torch.float32)
                if not isinstance(coriolis_forces, torch.Tensor):
                    coriolis_forces = torch.tensor(coriolis_forces, device=self.device, dtype=torch.float32)
                gravity_forces = gravity_forces.reshape(self.num_envs, -1)
                coriolis_forces = coriolis_forces.reshape(self.num_envs, -1)
                arm_phys_ids = self.hand.actuators["arm_joints"].joint_indices
                arm_gravity = gravity_forces[:, arm_phys_ids]
                arm_coriolis = coriolis_forces[:, arm_phys_ids]
                arm_pos = self.hand.data.joint_pos[:, self.arm_joint_indices]
                arm_vel = self.hand.data.joint_vel[:, self.arm_joint_indices]
                tau_pd = self._arm_K * (self.arm_joint_pos_des - arm_pos) + self._arm_D * (-arm_vel)
                arm_effort = tau_pd + arm_gravity + arm_coriolis
                all_joint_effort = torch.zeros((self.num_envs, self.hand.num_joints), device=self.device)
                all_joint_effort[:, self.arm_joint_indices] = arm_effort
                self.hand.set_joint_effort_target(all_joint_effort)
                self.hand.set_joint_position_target(all_joint_targets)
            else:
                self.hand.set_joint_position_target(all_joint_targets)

        self.prev_targets = self.cur_targets

    def _get_observations(self) -> dict:
        self._refresh_lab()
        obs = self.compute_observations()
        observations = {
            "policy": obs,
            "priv_info": self.priv_info_buf,
            "proprio_hist": self.proprio_hist_buf,
        }
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """Compute keypoint tracking reward."""
        target_state = {}
        max_length = torch.clamp(self.demo_data["seq_len"], 0, self.max_episode_length).float()
        cur_idx = self._get_demo_idx()

        # Get target states from demo data
        target_state["wrist_pos"] = self.demo_data["target_wrist_pos"][torch.arange(self.num_envs), cur_idx]
        cur_wrist_rot = self.demo_data["target_wrist_rot"][torch.arange(self.num_envs), cur_idx]
        target_state["wrist_quat"] = aa_to_quat(cur_wrist_rot)
        
        target_state["wrist_vel"] = self.demo_data["target_wrist_velocity"][torch.arange(self.num_envs), cur_idx]
        target_state["wrist_ang_vel"] = self.demo_data["target_wrist_angular_velocity"][torch.arange(self.num_envs), cur_idx]
        
        cur_joints_pos = self.demo_data["target_joints_pos"][torch.arange(self.num_envs), cur_idx]
        target_state["joints_pos"] = cur_joints_pos.reshape(self.num_envs, -1, 3)
        target_state["joints_vel"] = self.demo_data["target_joints_velocity"][torch.arange(self.num_envs), cur_idx].reshape(self.num_envs, -1, 3)
        
        # Object states (if object exists)
        cur_obj_transf = self.demo_data["obj_trajectory"][torch.arange(self.num_envs), cur_idx]
        target_obj_pos = cur_obj_transf[:, :3, 3]


        target_state["manip_obj_pos"] = target_obj_pos
        target_state["manip_obj_quat"] = rotmat_to_quat(cur_obj_transf[:, :3, :3])
        target_state["manip_obj_vel"] = self.demo_data["obj_velocity"][torch.arange(self.num_envs), cur_idx]
        target_state["manip_obj_ang_vel"] = self.demo_data["obj_angular_velocity"][torch.arange(self.num_envs), cur_idx]
        target_state["tips_distance"] = self.demo_data["tips_distance"][torch.arange(self.num_envs), cur_idx]

        # ---- Final-frame target (for success / approach reward) ----
        # Per-env final frame index (seq_len-1, clamped to >=0 for empty seqs).
        seq_len_clamped = torch.clamp(self.demo_data["seq_len"], 0, self.max_episode_length)
        final_idx = torch.clamp(seq_len_clamped - 1, min=0).long()  # [N]
        env_idx = torch.arange(self.num_envs, device=self.device)
        obj_traj = self.demo_data["obj_trajectory"]  # [N, T_max, 4, 4]

        final_obj_transf = obj_traj[env_idx, final_idx]                # [N, 4, 4]
        target_state["final_obj_pos"] = final_obj_transf[:, :3, 3]      # [N, 3]
        target_state["final_obj_quat"] = rotmat_to_quat(final_obj_transf[:, :3, :3])  # [N, 4]

        # Last K frames' target positions (K from cfg, default 5).
        # last_K_idx[i, k] = final_idx[i] - (K - 1 - k) clamped to >=0.
        K = int(getattr(self.cfg, "success_reward_window", 5))
        arange_K = torch.arange(K, device=self.device)
        last_K_idx = (final_idx[:, None] - (K - 1 - arange_K[None, :])).clamp(min=0)  # [N, K]
        last_K_transf = obj_traj[env_idx[:, None], last_K_idx]          # [N, K, 4, 4]
        target_state["final_K_obj_pos"] = last_K_transf[..., :3, 3]    # [N, K, 3]

        # target_state["tip_force"] = self.last_contacts
        target_state["tip_force"] = torch.stack(
            [
                # (N, B, 3) -> (N, 3)
                self._contact_sensor[body_id].data.net_forces_w.sum(dim=1)
                for body_id in self._contact_body_ids
            ],
            dim=1,
        )

        self.tips_contact_history = torch.concat(
            [
                self.tips_contact_history[:, 1:],
                (torch.norm(target_state["tip_force"], dim=-1) > 0)[:, None],
            ],
            dim=1,
        )
        target_state["tip_contact_state"] = self.tips_contact_history

        # Compute power from joint torques and velocities
        power = torch.abs(torch.multiply(self.hand_dof_torque, self.hand_dof_vel)).sum(dim=-1)
        target_state["power"] = power
        
        # Compute wrist power (force * velocity + torque * angular_velocity)
        wrist_lin_vel = self.hand.data.body_lin_vel_w[:, self.wrist_body_idx]
        wrist_ang_vel = self.hand.data.body_ang_vel_w[:, self.wrist_body_idx]
        wrist_power = torch.abs(
            torch.sum(self.apply_forces * wrist_lin_vel, dim=-1)
        ) + torch.abs(
            torch.sum(self.apply_torque * wrist_ang_vel, dim=-1)
        )
        target_state["wrist_power"] = wrist_power
        
        # Compute scale factor for tightening (curriculum learning)
        if hasattr(self.cfg, 'tighten_method') and self.cfg.tighten_method != "None":
            last_step = self.common_step_counter
            if self.cfg.tighten_method == "const":
                scale_factor = self.cfg.tighten_factor
            elif self.cfg.tighten_method == "linear_decay":
                scale_factor = 1 - (1 - self.cfg.tighten_factor) / self.cfg.tighten_steps * min(last_step, self.cfg.tighten_steps)
            elif self.cfg.tighten_method == "exp_decay":
                scale_factor = (math.e * 2) ** (-1 * last_step / self.cfg.tighten_steps) * (1 - self.cfg.tighten_factor) + self.cfg.tighten_factor
            elif self.cfg.tighten_method == "cos":
                scale_factor = self.cfg.tighten_factor + abs(
                    -1 * (1 - self.cfg.tighten_factor) * math.cos(last_step / self.cfg.tighten_steps * math.pi)
                ) * (2 ** (-1 * last_step / self.cfg.tighten_steps))
            else:
                scale_factor = 1.0
        else:
            scale_factor = 1.0

        # Update gravity based on scheduler
        current_gravity = None
        gravity_progress = None
        if self.cfg.gravity_scheduler_enabled:
            if hasattr(self.cfg, 'gravity_scheduler_method') and self.cfg.gravity_scheduler_method != "None":
                last_step = self.common_step_counter
                gravity_initial = getattr(self.cfg, 'gravity_initial', 0.1)
                gravity_final = getattr(self.cfg, 'gravity_final', 9.81)
                gravity_steps = getattr(self.cfg, 'gravity_scheduler_steps', 10000)
                
                if self.cfg.gravity_scheduler_method == "linear":
                    # Linear interpolation
                    progress = min(last_step / gravity_steps, 1.0)
                    current_gravity = gravity_initial + (gravity_final - gravity_initial) * progress
                    gravity_progress = progress
                elif self.cfg.gravity_scheduler_method == "exp":
                    # Exponential interpolation
                    progress = min(last_step / gravity_steps, 1.0)
                    # Use exponential curve: g = g_initial * (g_final / g_initial) ^ progress
                    current_gravity = gravity_initial * ((gravity_final / gravity_initial) ** progress)
                    gravity_progress = progress
                elif self.cfg.gravity_scheduler_method == "cos":
                    # Cosine interpolation (smooth start and end)
                    progress = min(last_step / gravity_steps, 1.0)
                    # Cosine interpolation: smooth transition
                    cos_progress = (1 - math.cos(progress * math.pi)) / 2
                    current_gravity = gravity_initial + (gravity_final - gravity_initial) * cos_progress
                    gravity_progress = progress
                else:
                    current_gravity = gravity_final
                    gravity_progress = 1.0
                
                # Set gravity (negative for z-axis)
                new_gravity = carb.Float3(0.0, 0.0, -current_gravity)
                self.physics_sim_view.set_gravity(new_gravity)
        
        # Get current states
        current_states = self._get_current_states()
        
        # Compute reward using imitation reward function
        # Convert max_length to float tensor for JIT compatibility
        max_length_tensor = max_length.float()
        _eval_no_terminate = bool(getattr(self.cfg, 'eval_no_terminate', False))
        if not hasattr(self, "_eval_no_terminate_logged"):
            print(f"[DEBUG _get_rewards] eval_no_terminate = {_eval_no_terminate}")
            self._eval_no_terminate_logged = True
        self.reward_execute[:], self.reset_buf[:], self.success_buf[:], self.failure_buf[:], reward_dict = compute_imitation_reward(
            self.reset_buf,
            self.progress_buf,
            self.running_progress_buf,
            self.actions,
            current_states,
            target_state,
            max_length_tensor,
            scale_factor,
            self.dexhand_weight_idx,   # by-name resolved (asserted equal to `self.dexhand.weight_idx` at init)
            self._use_wrist_tracking,
            self._use_abs_hand_tracking,
            self._use_rel_hand_tracking,
            getattr(self.cfg, 'arm_action_rate_penalty', 0.1),
            getattr(self.cfg, 'no_slip_weight', 0.0),
            getattr(self.cfg, 'approach_shaping_v2', False),
            getattr(self.cfg, 'success_pos_weight', 0.0),
            getattr(self.cfg, 'success_rot_weight', 0.0),
            getattr(self.cfg, 'success_approach_weight', 0.0),
            float(getattr(self.cfg, 'success_alpha_pos', 30.0)),
            float(getattr(self.cfg, 'success_alpha_rot', 3.0)),
            bool(getattr(self.cfg, 'success_reward_ramp', False)),
            int(getattr(self.cfg, 'success_reward_window', 5)),
            float(getattr(self.cfg, 'premature_contact_dist_threshold', 0.005)),
            int(getattr(self.cfg, 'premature_contact_progress_threshold', 50)),
            bool(getattr(self.cfg, 'premature_contact_enabled', True)),
            bool(getattr(self.cfg, 'eval_no_terminate', False)),
        )
        # Eval-mode mask applied OUTSIDE the @torch.jit.script function:
        # JIT caches the compiled function and silently ignores newly-added
        # bool params in some setups. Apply the mask here so it always takes
        # effect. When True: zero reset_buf and failure_buf so the only way
        # episodes end is via env's `time_out = episode_length_buf >= max_length`
        # in _get_dones. Also force success_buf to 1 at time-out boundary
        # (progress >= max_length - 3) to keep the success metric meaningful.
        if _eval_no_terminate:
            self.reset_buf[:] = 0   # don't terminate, let time_out handle it

        self.total_rew_buf += self.reward_execute
        self.reward_dict = reward_dict
        
        # Update extras for logging
        for key, value in reward_dict.items():
            self.extras[key] = value.mean() if isinstance(value, torch.Tensor) else value
        self.extras['total_reward'] =  self.reward_execute.mean()
        # Per-ENV vectors. `success_buf` is 1 only on the step an episode ends and
        # is cleared in `_reset_idx`, so a mean over all envs at every step is a
        # near-zero number that is NOT the episode success rate. Consumers must
        # select the envs that just terminated — see `AverageScalarMeter` usage
        # in algo/ppo/ppo.py and the caveat in docs/EVAL.md.
        self.extras['succeeded_per_env'] = self.success_buf.float()
        # Conservative training proxy: also requires survival, and PPO counts
        # bad inits as zero. This is not eval.py's strict3; see docs/TRAINING.md.
        if 'succ/strict' in reward_dict:
            self.extras['succeeded_strict_per_env'] = reward_dict['succ/strict']
        self.extras['failed_per_env'] = self.failure_buf.float()
        # Scalars kept for backward compatibility with existing log keys. Do not
        # read these as episode rates.
        self.extras['succeeded'] = self.success_buf.float().mean()
        self.extras['failed_execute'] = self.failure_buf.float().mean()
        
        # Record curriculum information
        self.extras['curriculum_scale_factor'] = scale_factor
        if current_gravity is not None:
            self.extras['curriculum_gravity'] = current_gravity
        if gravity_progress is not None:
            self.extras['curriculum_gravity_progress'] = gravity_progress
        # Always record current gravity value from simulation
        gravity_vec = self.physics_sim_view.get_gravity()
        self.extras['gravity_z'] = -gravity_vec[2]  # Store as positive value

        # Init curriculum status
        if getattr(self.cfg, 'init_curriculum_enabled', False):
            progress = min(self.common_step_counter / self.cfg.init_curriculum_steps, 1.0)
            earliest_frac = self.cfg.init_curriculum_start + \
                (self.cfg.init_curriculum_end - self.cfg.init_curriculum_start) * progress
            self.extras['curriculum_init_earliest_frac'] = max(earliest_frac, self.cfg.init_curriculum_end)

        # Adaptive init: capture high-reward states into buffer
        if self._state_buffer is not None:
            self._state_buffer.maybe_capture(self, self.reward_execute)

        return self.reward_execute
    
    def _get_current_states(self) -> dict:
        """Get current hand states for reward computation."""
        self._refresh_lab()
        base_state = self.base_state
        # base_state[:, :3] = self.base_pos
        # Get joint positions and velocities
        q = self.hand_dof_pos
        dq = self.hand_dof_vel
        
        # joint state means corresponding mano joints position
        joints_state = self.hand.data.body_state_w[:, :, :10]
        joints_state[:, :, :3] = self.hand.data.body_state_w[:, :, :3] - self.scene.env_origins[:, None, :]

        hand_joints_state = joints_state[:, self.hand_body_indices, :]

        # action rate (total)
        action_rate_l1 = torch.sum(self.action_rate, dim=-1)
        action_rate_l2 = torch.sum(self.action_rate ** 2, dim=-1)

        # arm action rate (first root_control_dim dims of action)
        root_control_dim = 0 if self.freeze_arm else (9 if self.use_pid_control else (7 if self.use_joint_pos_control else 6))
        arm_action_rate_l2 = torch.sum(self.action_rate[:, :root_control_dim] ** 2, dim=-1)

        dt = self.physics_dt if hasattr(self, 'physics_dt') else 1.0 / 60.0  # Default to 60Hz if not available
        self.arm_joint_acc = (self.arm_joint_vel - self.prev_arm_joint_vel) / (dt + 1e-8)
        self.prev_arm_joint_vel = self.arm_joint_vel.clone()
        self.wrist_lin_acc = (self.base_lin_vel - self.prev_wrist_lin_vel) / (dt + 1e-8)
        self.wrist_ang_acc = (self.base_ang_vel - self.prev_wrist_ang_vel) / (dt + 1e-8)
        self.prev_wrist_lin_vel = self.base_lin_vel.clone()
        self.prev_wrist_ang_vel = self.base_ang_vel.clone()
        
        
        # Get arm body positions (for collision detection with table)
        arm_body_positions = None
        if hasattr(self, 'arm_body_indices') and len(self.arm_body_indices) > 0:
            # Get arm body positions in world frame, then subtract env_origins to get relative positions
            arm_body_pos_w = self.hand.data.body_pos_w[:, self.arm_body_indices]  # [num_envs, num_arm_bodies, 3]
            arm_body_positions = arm_body_pos_w - self.scene.env_origins.unsqueeze(1)  # [num_envs, num_arm_bodies, 3]
        else:
            # Fallback: use empty tensor if arm bodies not identified
            arm_body_positions = torch.zeros((self.num_envs, 0, 3), device=self.device)
        
        # Get wrist height (Z coordinate of wrist position)
        wrist_height = self.base_pos[:, 2]  # [num_envs]
        
        states = {
            "q": q,
            "cos_q": torch.cos(q),
            "sin_q": torch.sin(q),
            "dq": dq,
            "base_state": base_state,
            "joints_state": joints_state,
            "hand_joints_state": hand_joints_state,
            "action_rate_l1": action_rate_l1,
            "action_rate_l2": action_rate_l2,
            "arm_action_rate_l2": arm_action_rate_l2,
            # Arm joint velocity and acceleration for penalty
            "arm_joint_vel": self.arm_joint_vel,
            "arm_joint_acc": self.arm_joint_acc,
            # Wrist velocity and acceleration for penalty
            "wrist_lin_vel": self.base_lin_vel,
            "wrist_ang_vel": self.base_ang_vel,
            "wrist_lin_acc": self.wrist_lin_acc,
            "wrist_ang_acc": self.wrist_ang_acc,
            # Arm body positions for collision detection
            "arm_body_positions": arm_body_positions[:,1:],
            # Wrist height for reward
            "wrist_height": wrist_height,
        }
        
        # Add object states if object exists
        if hasattr(self, 'object') and self.object is not None:
            states["manip_obj_pos"] = self.object.data.root_pos_w - self.scene.env_origins
            states["manip_obj_quat"] = self.object.data.root_quat_w
            states["manip_obj_vel"] = self.object.data.root_lin_vel_w
            states["manip_obj_ang_vel"] = self.object.data.root_ang_vel_w
        else:
            # Placeholder values when object doesn't exist
            states["manip_obj_pos"] = torch.zeros((self.num_envs, 3), device=self.device)
            states["manip_obj_quat"] = torch.zeros((self.num_envs, 4), device=self.device)
            states["manip_obj_vel"] = torch.zeros((self.num_envs, 3), device=self.device)
            states["manip_obj_ang_vel"] = torch.zeros((self.num_envs, 3), device=self.device)
        
        return states

    def set_eval_init_range(self, lo_per_env: torch.Tensor, hi_per_env: torch.Tensor) -> None:
        """Force resets to sample seq_idx uniformly in [lo[i], hi[i]] per env.
        Overrides curriculum / adaptive / fixed paths. Set both to None to
        clear and restore default sampling.
        """
        if lo_per_env is None or hi_per_env is None:
            self._eval_init_seq_idx_lo = None
            self._eval_init_seq_idx_hi = None
            return
        assert lo_per_env.shape == hi_per_env.shape == (self.num_envs,), (
            f"lo/hi must be (num_envs,)={self.num_envs}; got {lo_per_env.shape} / {hi_per_env.shape}"
        )
        self._eval_init_seq_idx_lo = lo_per_env.long().to(self.device)
        self._eval_init_seq_idx_hi = hi_per_env.long().to(self.device)

    def _get_demo_idx(self, progress: torch.Tensor | None = None) -> torch.Tensor:
        """Get demo data index from progress_buf.

        - loop_trajectory=True : modulo wrap.
        - loop_trajectory=False: clamp to seq_len-1. Required because
          `progress_buf` is set to seq_idx (≥0) on reset and incremented every
          step, while the episode terminates on `episode_length_buf >= seq_len-1`
          (steps-since-reset). With non-zero seq_idx (init_curriculum or random
          init), progress_buf overshoots seq_len-1 long before the time_out
          fires, so direct demo_data indexing would go OOB.
        """
        if progress is None:
            progress = self.progress_buf
        seq_len = self.demo_data["seq_len"]  # per-env sequence lengths
        if self.loop_trajectory:
            return progress % seq_len
        return torch.minimum(progress, seq_len - 1)

    # ------------------------------------------------------------------
    # Adaptive trajectory-fraction sampling
    # ------------------------------------------------------------------

    def _adaptive_sample_seq_idx(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Bin the trajectory into N fractional bins, sample reset frames from a
        smoothed failure-biased distribution, and return per-env seq_idx.

        Pipeline (extends whole_body_tracking.commands.MotionCommand._adaptive_sampling
        with warmup + optional curriculum window so early-training reward
        doesn't tank when a few bad bins immediately collapse the pmf):

          0. WARMUP. For the first `warmup_steps` env steps, force pmf uniform
             across the curriculum window (or full range if curriculum off).
             This matches the baseline reset distribution while bin_failed_count
             is still noisy.

          1. CURRICULUM MASK (optional). If init_curriculum is also enabled
             AND `compose_with_curriculum=True`, zero out bins below the
             current `earliest_frac` so adaptive only samples within the
             curriculum's expanding window. This preserves the easy-first
             warmup behavior even with adaptive on.

          2. FAILURE ACCUMULATION. For each resetting env that *failed*
             (failure_buf=1 — actual failure, not time-out), bump the bin
             matching its progress_buf at the moment of reset.

          3. SAMPLING PMF.
                p_raw      = bin_failed_count + uniform_floor   (per bin)
                p_masked   = p_raw * curriculum_mask             (optional)
                p_smoothed = conv1d(p_masked, geometric kernel)  (non-causal)
                p          = p_smoothed / sum(p_smoothed)
             During warmup, replaces p with uniform over allowed bins.

          4. multinomial(p, len(env_ids)) + uniform jitter inside bin →
             per-env seq_idx (respects per-env seq_len).

          5. EMA update of slow stat: bin_failed_count = α·current + (1-α)·prev.

          6. Log entropy / top-1 prob / top-1 bin / warmup flag.
        """
        n_bins = self._adapt_n_bins
        K = self._adapt_kernel_size

        # ---- 0. Curriculum window (only used as a bin mask, not for sampling math)
        cmask = None  # 1.0 = bin allowed, 0.0 = bin masked out
        if (getattr(self.cfg, 'adaptive_sampling_compose_with_curriculum', True)
                and getattr(self.cfg, 'init_curriculum_enabled', False)):
            progress = min(self.common_step_counter / self.cfg.init_curriculum_steps, 1.0)
            method = self.cfg.init_curriculum_method
            start_f = self.cfg.init_curriculum_start
            end_f = self.cfg.init_curriculum_end
            if method == "linear":
                earliest_frac = start_f + (end_f - start_f) * progress
            elif method == "exp":
                earliest_frac = start_f * ((end_f + 1e-6) / (start_f + 1e-6)) ** progress
            elif method == "cos":
                cos_progress = (1 - math.cos(progress * math.pi)) / 2
                earliest_frac = start_f + (end_f - start_f) * cos_progress
            else:
                earliest_frac = start_f
            earliest_frac = max(earliest_frac, end_f)
            # Bin index where earliest_frac lives; mask bins strictly below it.
            min_bin = int(min(max(earliest_frac, 0.0), 0.999) * n_bins)
            cmask = torch.zeros(n_bins, dtype=torch.float, device=self.device)
            cmask[min_bin:] = 1.0

        # ---- WARMUP check.
        warmup_steps = int(getattr(self.cfg, 'adaptive_sampling_warmup_steps', 0))
        in_warmup = self.common_step_counter < warmup_steps

        # ---- 2. Failure accumulation (still done during warmup so EMA is warm
        # by the time we exit warmup — but the warmup pmf below ignores it).
        failed_mask = self.failure_buf[env_ids].bool() if hasattr(self, 'failure_buf') else None
        if failed_mask is not None and failed_mask.any():
            seq_lens = self.demo_data["seq_len"][env_ids].clamp(min=1).float()
            progress_frac = self.progress_buf[env_ids].float() / seq_lens
            progress_frac = progress_frac.clamp(0.0, 1.0 - 1e-6)
            bin_idx_at_fail = (progress_frac * n_bins).long().clamp(0, n_bins - 1)
            fail_bins = bin_idx_at_fail[failed_mask]
            bc = torch.bincount(fail_bins, minlength=n_bins).float()
            self._adapt_bin_current_failed = self._adapt_bin_current_failed + bc

        # ---- 3. Build pmf.
        if in_warmup:
            # Uniform over allowed (curriculum-masked) bins.
            base = cmask.clone() if cmask is not None else torch.ones(n_bins, device=self.device)
            if base.sum() <= 0:
                base = torch.ones(n_bins, device=self.device)
            p_smoothed = base / base.sum()
        else:
            uniform_floor = float(self.cfg.adaptive_sampling_uniform_ratio) / float(n_bins)
            p = self._adapt_bin_failed_count + uniform_floor
            if cmask is not None:
                p = p * cmask
                # If curriculum window is empty (shouldn't happen but defensive):
                if p.sum() <= 0:
                    p = cmask.clone()
            p_padded = torch.nn.functional.pad(
                p.view(1, 1, -1),
                (0, K - 1),         # non-causal: pad on the right
                mode='replicate',
            )
            p_smoothed = torch.nn.functional.conv1d(p_padded, self._adapt_kernel).view(-1)
            # Re-apply curriculum mask AFTER smoothing too — replicate-pad +
            # conv1d would otherwise leak mass to bins outside the window.
            if cmask is not None:
                p_smoothed = p_smoothed * cmask
            denom = p_smoothed.sum().clamp(min=1e-12)
            p_smoothed = p_smoothed / denom

        # ---- 4. Sample bins.
        sampled_bins = torch.multinomial(p_smoothed, len(env_ids), replacement=True)
        jitter = torch.rand(len(env_ids), device=self.device)

        seq_lens_e = self.demo_data["seq_len"][env_ids].float()
        seq_idx = ((sampled_bins.float() + jitter) / float(n_bins) * (seq_lens_e - 1)).long()
        seq_idx = seq_idx.clamp(min=torch.zeros_like(seq_idx),
                                max=(seq_lens_e.long() - 1).clamp(min=0))

        # ---- 5. EMA update.
        alpha = float(self.cfg.adaptive_sampling_alpha)
        self._adapt_bin_failed_count = (alpha * self._adapt_bin_current_failed
                                        + (1.0 - alpha) * self._adapt_bin_failed_count)
        self._adapt_bin_current_failed = torch.zeros_like(self._adapt_bin_current_failed)

        # ---- 6. Logging.
        ent = -(p_smoothed * (p_smoothed + 1e-12).log()).sum()
        ent_norm = ent / math.log(max(n_bins, 2))
        pmax, imax = p_smoothed.max(dim=0)
        self.extras["adaptive_sampling/entropy_norm"] = ent_norm.detach()
        self.extras["adaptive_sampling/top1_prob"] = pmax.detach()
        self.extras["adaptive_sampling/top1_bin"] = imax.float().detach() / float(n_bins)
        self.extras["adaptive_sampling/failed_count_total"] = self._adapt_bin_failed_count.sum().detach()
        self.extras["adaptive_sampling/in_warmup"] = float(in_warmup)
        if cmask is not None:
            self.extras["adaptive_sampling/curriculum_min_bin_frac"] = float(
                cmask.argmax().item() / max(n_bins, 1)
            )

        return seq_idx

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._refresh_lab()

        if self.loop_trajectory:
            # When looping, trajectory end is not a termination condition.
            # Only reset on max_episode_length or explicit terminate (reset_buf).
            time_out = self.episode_length_buf >= (self.max_episode_length - 1)
            terminated = self.reset_buf.bool() & (~time_out)
        else:
            max_length = torch.clamp(self.demo_data["seq_len"] - 1, 0, self.max_episode_length - 1).float()
            time_out = self.episode_length_buf >= max_length
            terminated = self.reset_buf.bool() & (~time_out)

        self.extras["time_out"] = time_out.float().mean()
        self.extras["terminated"] = terminated.float().mean()

        return terminated, time_out

    def _rand_pd_scales(self, lower, upper, num_envs, n_dofs):
        rand_scale_s = torch.distributions.Uniform(lower, 1).sample((num_envs, n_dofs)).to(self.device)
        rand_scale_l = torch.distributions.Uniform(1, upper).sample((num_envs, n_dofs)).to(self.device)
        mask_choice = torch.rand((num_envs, n_dofs), device=self.device) > 0.5
        rand_scale = torch.where(mask_choice, rand_scale_s, rand_scale_l)
        return rand_scale

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        # resets articulation and rigid body attributes
        super()._reset_idx(env_ids)
        
        # pd randomize
        if self.cfg.randomize_pd_gains:
            assert self.cfg.randomize_p_gain_scale_lower <= 1, "pd gain scale: lower must be <= 1 and upper must be >= 1"
            assert self.cfg.randomize_p_gain_scale_upper >= 1, "pd gain scale: lower must be <= 1 and upper must be >= 1"
            assert self.cfg.randomize_d_gain_scale_lower <= 1, "pd gain scale: lower must be <= 1 and upper must be >= 1"
            assert self.cfg.randomize_d_gain_scale_upper >= 1, "pd gain scale: lower must be <= 1 and upper must be >= 1"
            rand_scale = self._rand_pd_scales(self.cfg.randomize_p_gain_scale_lower, self.cfg.randomize_p_gain_scale_upper, len(env_ids), self.num_hand_dofs)
            self.p_gain[env_ids] = self.p_gain_default[env_ids] * rand_scale
            rand_scale = self._rand_pd_scales(self.cfg.randomize_d_gain_scale_lower, self.cfg.randomize_d_gain_scale_upper, len(env_ids), self.num_hand_dofs)
            self.d_gain[env_ids] = self.d_gain_default[env_ids] * rand_scale

        # Arm PD DR — scales arm K / D per env at reset. Needs manual PD path
        # (arm_gravity_compensation=True or use_osc_control=True), because arm
        # stiffness/damping are applied manually via tau_pd, not via PhysX drives.
        # Skipped automatically in deploy envs (which set _is_deploy_env=True).
        if (getattr(self.cfg, 'randomize_arm_pd_gains', False)
                and hasattr(self, '_arm_K_default')
                and not getattr(self, '_is_deploy_env', False)):
            n_arm = self._arm_K.shape[-1]
            rand_k = self._rand_pd_scales(
                self.cfg.randomize_arm_p_scale_lower,
                self.cfg.randomize_arm_p_scale_upper,
                len(env_ids), n_arm,
            )
            rand_d = self._rand_pd_scales(
                self.cfg.randomize_arm_d_scale_lower,
                self.cfg.randomize_arm_d_scale_upper,
                len(env_ids), n_arm,
            )
            self._arm_K[env_ids] = self._arm_K_default[env_ids] * rand_k
            self._arm_D[env_ids] = self._arm_D_default[env_ids] * rand_d

        # Action-delay DR — per-env delay in [action_delay_min, action_delay_max].
        # Skipped in deploy envs; real hardware has its own fixed latency already.
        if (self._randomize_action_delay
                and self._env_action_delay is not None
                and not getattr(self, '_is_deploy_env', False)):
            lo = self._action_delay_min
            hi = self._action_delay_max
            self._env_action_delay[env_ids] = torch.randint(
                lo, hi + 1, (len(env_ids),), device=self.device, dtype=torch.long,
            )

        # reset data buffers
        self.last_contacts[env_ids] = 0
        if hasattr(self, "last_contacts_vec_w"):
            self.last_contacts_vec_w[env_ids] = 0
        self.proprio_hist_buf[env_ids] = 0
        self.at_reset_buf[env_ids] = 1

        # maniptrans style
        # Adaptive trajectory-fraction sampling — replaces curriculum / uniform
        # random when enabled. Done first so we don't pay cost of the other paths.
        # The helper ALSO records the resetting envs' progress bins (for failures
        # only) and runs the EMA update; on first call after init,
        # bin_failed_count is all zero so the smoothed pmf is uniform.
        # ---- Eval-mode override: per-env seq_idx ranges (highest priority) ----
        # `self._eval_init_seq_idx_lo/hi` are int64 tensors of shape (num_envs,).
        # When set (via env.set_eval_init_range(lo, hi)), the env samples
        # seq_idx uniformly per env in [lo[i], hi[i]] each reset. Overrides
        # random_state_init / adaptive / curriculum / fixed paths.
        if (hasattr(self, "_eval_init_seq_idx_lo") and self._eval_init_seq_idx_lo is not None):
            lo = self._eval_init_seq_idx_lo[env_ids]
            hi = self._eval_init_seq_idx_hi[env_ids]
            rng = (hi - lo + 1).clamp(min=1).float()
            seq_idx = (lo.float() + torch.floor(rng * torch.rand_like(rng))).long()
            # safety clamp
            seq_lens = self.demo_data["seq_len"][env_ids].long()
            seq_idx = torch.minimum(seq_idx, (seq_lens - 1).clamp(min=0))
            seq_idx = torch.maximum(seq_idx, torch.zeros_like(seq_idx))
        elif self.random_state_init and self._adapt_enabled:
            seq_idx = self._adaptive_sample_seq_idx(env_ids)
        elif self.random_state_init:
            if getattr(self.cfg, 'init_curriculum_enabled', False):
                # Reverse curriculum: start from near-grasp, gradually expand to full trajectory
                progress = min(self.common_step_counter / self.cfg.init_curriculum_steps, 1.0)
                if self.cfg.init_curriculum_method == "linear":
                    earliest_frac = self.cfg.init_curriculum_start + \
                        (self.cfg.init_curriculum_end - self.cfg.init_curriculum_start) * progress
                elif self.cfg.init_curriculum_method == "exp":
                    earliest_frac = self.cfg.init_curriculum_start * \
                        ((self.cfg.init_curriculum_end + 1e-6) / (self.cfg.init_curriculum_start + 1e-6)) ** progress
                elif self.cfg.init_curriculum_method == "cos":
                    cos_progress = (1 - math.cos(progress * math.pi)) / 2
                    earliest_frac = self.cfg.init_curriculum_start + \
                        (self.cfg.init_curriculum_end - self.cfg.init_curriculum_start) * cos_progress
                else:
                    earliest_frac = self.cfg.init_curriculum_start
                earliest_frac = max(earliest_frac, self.cfg.init_curriculum_end)
                # Sample from [earliest_frac * seq_len, 0.98 * seq_len)
                seq_len_f = self.demo_data["seq_len"][env_ids].float()
                low = (seq_len_f * earliest_frac).long()
                high = (seq_len_f * 0.98).long()
                rand_range = (high - low).clamp(min=1).float()
                seq_idx = low + torch.floor(rand_range * torch.rand_like(rand_range)).long()
            else:
                seq_idx = torch.floor(
                        self.demo_data["seq_len"][env_ids]
                        * 0.98
                        * torch.rand_like(self.demo_data["seq_len"][env_ids].float())
                    ).long()
        else:
            fixed_fr = getattr(self.cfg, "fixed_reset_demo_frame", None)
            if fixed_fr is not None:
                fr = int(fixed_fr)
                seq_lens = self.demo_data["seq_len"][env_ids].long()
                max_idx = (seq_lens - 1).clamp(min=0)
                seq_idx = torch.full_like(max_idx, fr, dtype=torch.long)
                seq_idx = torch.minimum(seq_idx, max_idx)
            else:
                seq_idx = torch.zeros_like(self.demo_data["seq_len"][env_ids].long())

        # Combine arm and hand joint positions for full robot
        all_joint_pos = torch.zeros(len(env_ids), self.hand.num_joints, device=self.device)
        all_joint_vel = torch.zeros(len(env_ids), self.hand.num_joints, device=self.device)
        

        dof_pos = self.demo_data["opt_dof_pos"][env_ids, seq_idx]
        # change the real robot joint order to the isaaclab dof order
        all_joint_pos[:, self.hand_joint_indices] = dof_pos
        all_joint_pos[:, 7:] = torch.clamp(
            all_joint_pos[:, 7:],
            self.hand_dof_lower_limits[env_ids],
            self.hand_dof_upper_limits[env_ids],
        )
        dof_vel = self.demo_data["opt_dof_velocity"][env_ids, seq_idx]
        all_joint_vel[:, self.hand_joint_indices] = dof_vel

        # Get arm joint positions from retargeted data
        franka_joint_pos = self.demo_data["opt_arm_joint_pos"][env_ids, seq_idx]  # [num_reset, 7]
        franka_joint_pos = torch.clamp(
            franka_joint_pos,
            self.arm_joint_lower_limits[env_ids] if self.arm_joint_lower_limits.ndim > 1 else self.arm_joint_lower_limits,
            self.arm_joint_upper_limits[env_ids] if self.arm_joint_upper_limits.ndim > 1 else self.arm_joint_upper_limits,
        )
        
        franka_joint_vel = torch.zeros_like(franka_joint_pos)

        arm_base_pos = torch.tensor(
            self.cfg.arm_base_pos,
            device=self.device,
            dtype=torch.float32
        ).unsqueeze(0).repeat(len(env_ids), 1)  # [num_reset, 3]
        arm_base_rot = torch.tensor(
            self.cfg.arm_base_rot,
            device=self.device,
            dtype=torch.float32
        ).unsqueeze(0).repeat(len(env_ids), 1)  # [num_reset, 4]
        
        
        # Set arm joint positions and velocities
        all_joint_pos[:, self.arm_joint_indices] = franka_joint_pos
        all_joint_vel[:, self.arm_joint_indices] = franka_joint_vel
        
        all_joint_vel = torch.clamp(
            all_joint_vel,
            -self.hand.data.joint_vel_limits[env_ids],
            self.hand.data.joint_vel_limits[env_ids],
        )

        # Adaptive init: override selected envs with buffered high-reward states
        self._adaptive_init_obj_state = None
        if self._state_buffer is not None:
            seq_idx, buf_joint_pos, buf_joint_vel, buf_obj_state, use_buf = \
                self._state_buffer.sample_init(self, env_ids, seq_idx)
            if use_buf.any():
                buf_ids = use_buf.nonzero(as_tuple=False).squeeze(-1)
                all_joint_pos[buf_ids] = buf_joint_pos[buf_ids]
                all_joint_vel[buf_ids] = buf_joint_vel[buf_ids]
                franka_joint_pos[buf_ids] = all_joint_pos[buf_ids][:, self.arm_joint_indices]
                self._adaptive_init_obj_state = (use_buf, buf_obj_state)

        base_state = torch.zeros(len(env_ids), 13, device=self.device)
        base_state[:, :3] = arm_base_pos
        base_state[:, 3:7] = arm_base_rot
        base_state[:, 7:] = 0  # Zero velocities for base (fixed base)

        # Set robot root state (arm base position)
        base_default_state = self.hand.data.default_root_state.clone()[env_ids]
        base_default_state[:, :] = base_state
        base_default_state[:, :3] += self.scene.env_origins[env_ids]

        # write target and state into sim (all joints: arm + hand)
        self.hand.write_joint_state_to_sim(all_joint_pos, all_joint_vel, env_ids=env_ids)
        
        self.hand.set_joint_position_target(all_joint_pos, env_ids=env_ids)

        self.hand.write_root_state_to_sim(base_default_state, env_ids)

        self.prev_targets[env_ids] = all_joint_pos[:, 7:]
        self.cur_targets[env_ids] = all_joint_pos[:, 7:]

        # Always reset arm targets to init pose (needed for all control modes)
        self.arm_joint_pos_des_prev[env_ids] = franka_joint_pos.clone()
        self.arm_joint_pos_des[env_ids] = franka_joint_pos.clone()

        # Initialize Differential IK controller with wrist pose (similar to diff_ik.py)
        if self.use_pid_control:
            # After setting joint positions, get current wrist pose in root frame for IK controller initialization
            # Refresh to get current states after reset (joint positions are now set)
            self._refresh_lab()
            
            # Get current quantities from simulation
            jacobian = self.hand.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.arm_robot_entity_cfg.joint_ids]
            ee_pose_w = self.hand.data.body_pose_w[:, self.arm_robot_entity_cfg.body_ids[0]]
            root_pose_w = self.hand.data.root_pose_w
            
            # Compute frame in root frame (wrist pose relative to arm base)
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            
            # Prepare IK command from current wrist pose: [pos(3), quat(4)]
            ik_commands = torch.cat([ee_pos_b, ee_quat_b], dim=-1)  # [num_envs, 7]

            # Reset and set command for Differential IK controller
            self.arm_diff_ik_controller.reset(env_ids)
            self.arm_diff_ik_controller.set_command(ik_commands)
            
            self.arm_joint_pos_des_prev[env_ids] = franka_joint_pos.clone()
            self.arm_joint_pos_des[env_ids] = franka_joint_pos.clone()
            
            # Reset wrist target buffers to current wrist pose (in world frame)
            # Get current wrist pose in world frame
            current_wrist_pos = self.base_pos[env_ids]  # [len(env_ids), 3]
            current_wrist_quat = self.base_quat[env_ids]  # [len(env_ids), 4]
            
            # Initialize both prev and cur to current pose
            self.prev_wrist_target_pos[env_ids] = current_wrist_pos.clone()
            self.prev_wrist_target_quat[env_ids] = current_wrist_quat.clone()
            self.cur_wrist_target_pos[env_ids] = current_wrist_pos.clone()
            self.cur_wrist_target_quat[env_ids] = current_wrist_quat.clone()
        
        # Reset object state if object exists
        if hasattr(self, 'object') and self.object is not None:
            obj_pos_init = self.demo_data["obj_trajectory"][env_ids, seq_idx, :3, 3]
            obj_rot_init = self.demo_data["obj_trajectory"][env_ids, seq_idx, :3, :3]
            obj_rot_init = rotmat_to_quat(obj_rot_init)
            
            obj_vel = self.demo_data["obj_velocity"][env_ids, seq_idx]
            obj_ang_vel = self.demo_data["obj_angular_velocity"][env_ids, seq_idx]

            obj_default_state = self.object.data.default_root_state.clone()[env_ids]
            obj_default_state[:, :3] = obj_pos_init + self.scene.env_origins[env_ids]
            obj_default_state[:, 3:7] = obj_rot_init
            obj_default_state[:, 7:10] = obj_vel
            obj_default_state[:, 10:13] = obj_ang_vel

            # Adaptive init: override object state for buffer envs
            if self._adaptive_init_obj_state is not None:
                use_buf, buf_obj_state = self._adaptive_init_obj_state
                buf_ids = use_buf.nonzero(as_tuple=False).squeeze(-1)
                # buf_obj_state[:3] is already world-frame (rebased to target env by sample_init)
                obj_default_state[buf_ids, :3] = buf_obj_state[buf_ids, :3]
                obj_default_state[buf_ids, 3:13] = buf_obj_state[buf_ids, 3:13]

            # Eval-time object position perturbation (sim2real robustness study).
            # When env._eval_perturb_obj_xy = δ is set (scripts/eval.py
            # --perturb_obj_xy), inject a uniform(-δ,+δ) xy offset into obj_pos
            # at reset, to test dependence on the object being at the demo position.
            # cfg.randomize_obj_xy is the TRAINING-time knob; the instance
            # attribute is the eval-time override. Take whichever is larger so
            # an eval can widen a trained policy's displacement but never
            # silently narrow it below what training used.
            _xy_perturb = max(
                float(getattr(self.cfg, "randomize_obj_xy", 0.0) or 0.0),
                float(getattr(self, "_eval_perturb_obj_xy", 0.0) or 0.0),
            )
            if _xy_perturb > 0:
                perturb = torch.empty(len(env_ids), 2, device=self.device).uniform_(
                    -_xy_perturb, _xy_perturb
                )
                obj_default_state[:, :2] += perturb

            self.object.write_root_state_to_sim(obj_default_state, env_ids)

        # Reset aux object (static / kinematic) — write its per-env pose every reset.
        # For non-aux envs (placeholder), stash the body underground at z=-10
        # so it doesn't interfere with the scene.
        if getattr(self, "aux_object", None) is not None and self.has_aux:
            aux_default_state = self.aux_object.data.default_root_state.clone()[env_ids]
            aux_default_state[:, :3] = (
                self.demo_data["aux_obj_pos"][env_ids]
                + self.scene.env_origins[env_ids]
            )
            aux_default_state[:, 3:7] = self.demo_data["aux_obj_quat"][env_ids]
            aux_default_state[:, 7:13] = 0.0   # zero lin/ang vel (kinematic)

            # For placeholder envs (this demo has no aux), force underground.
            if hasattr(self, "_aux_present_mask"):
                env_mask = self._aux_present_mask[env_ids]   # [n_resets]
                if (~env_mask).any():
                    # Override placeholders: env_origin + (0, 0, -10)
                    no_aux_idx = (~env_mask).nonzero(as_tuple=False).squeeze(-1)
                    aux_default_state[no_aux_idx, 0:2] = self.scene.env_origins[env_ids][no_aux_idx, 0:2]
                    aux_default_state[no_aux_idx, 2] = self.scene.env_origins[env_ids][no_aux_idx, 2] - 10.0
                    aux_default_state[no_aux_idx, 3:7] = torch.tensor(
                        [1.0, 0.0, 0.0, 0.0], device=self.device, dtype=aux_default_state.dtype,
                    )

            self.aux_object.write_root_state_to_sim(aux_default_state, env_ids)


        # ---- Init-curriculum debug: print first 3 resets so we can verify
        # that seq_idx sampling, demo frame-0 arm pose, and downstream writes
        # behave the same regardless of init_curriculum_enabled. Only the first
        # 3 calls trigger to avoid log spam.
        if not hasattr(self, '_debug_reset_count'):
            self._debug_reset_count = 0
        if self._debug_reset_count < 3 and len(env_ids) > 0:
            curr_on = bool(getattr(self.cfg, 'init_curriculum_enabled', False))
            sl = self.demo_data["seq_len"][env_ids]
            print(f"\n[reset_debug #{self._debug_reset_count}] curriculum={curr_on}  "
                  f"random_state_init={self.random_state_init}  num_resets={len(env_ids)}")
            print(f"  seq_len     min/max/mean = {sl.min().item()}/{sl.max().item()}/{sl.float().mean().item():.1f}")
            print(f"  seq_idx     min/max/mean = {seq_idx.min().item()}/{seq_idx.max().item()}/{seq_idx.float().mean().item():.1f}")
            print(f"  seq_idx/seq_len mean    = {(seq_idx.float() / sl.float().clamp(min=1)).mean().item():.3f}")
            # Sanity: any zeros / NaN in retargeted opt_arm_joint_pos for the sampled frames?
            sampled_arm = self.demo_data["opt_arm_joint_pos"][env_ids, seq_idx]
            print(f"  sampled opt_arm_joint_pos shape = {tuple(sampled_arm.shape)}")
            print(f"  sampled arm pose mean/std       = {sampled_arm.mean().item():+.4f} / {sampled_arm.std().item():.4f}")
            print(f"  sampled arm pose abs max        = {sampled_arm.abs().max().item():.4f}")
            zero_envs = (sampled_arm.abs().sum(dim=-1) < 1e-6).sum().item()
            nan_envs  = torch.isnan(sampled_arm).any(dim=-1).sum().item()
            print(f"  envs with all-zero arm pose     = {zero_envs}/{len(env_ids)}  (suggests retarget fallback)")
            print(f"  envs with NaN arm pose          = {nan_envs}/{len(env_ids)}")
            # Env 0 specifically: seq_idx, sampled arm pose, current arm pose post-write
            e0_idx = 0
            print(f"  env0: seq_idx={seq_idx[e0_idx].item()}  "
                  f"sampled_pose={sampled_arm[e0_idx].cpu().numpy()}")
            self._debug_reset_count += 1

        self.progress_buf[env_ids] = seq_idx
        self.running_progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.success_buf[env_ids] = 0
        self.failure_buf[env_ids] = 0
        self.error_buf[env_ids] = 0
        self.total_rew_buf[env_ids] = 0
        self.apply_forces[env_ids] = 0
        self.apply_torque[env_ids] = 0
        self.last_actions[env_ids] = 0.0
        if self._action_delay_buf is not None:
            self._action_delay_buf[:, env_ids] = 0.0
        self.tips_contact_history[env_ids] = 0
        self.wrist_action_magnitude_buf[env_ids] = 0.0
        if self.virtual_object_force_enabled:
            self.virtual_object_force[env_ids] = 0
            self.virtual_object_torque[env_ids] = 0
        if self.use_pid_control:
            self.prev_pos_error[env_ids] = 0
            self.prev_rot_error[env_ids] = 0
            self.pos_error_integral[env_ids] = 0
            self.rot_error_integral[env_ids] = 0
            # Reset Differential IK controller
            if hasattr(self, 'arm_diff_ik_controller'):
                self.arm_diff_ik_controller.reset(env_ids)
        # Reset tips contact history
        self.tips_contact_history[env_ids] = torch.ones_like(self.tips_contact_history[env_ids]).bool()
        
        # Reset velocity buffers for acceleration computation
        # After reset, refresh to get current velocities, then set previous velocities to current
        self._refresh_lab()
        self.prev_arm_joint_vel[env_ids] = self.arm_joint_vel[env_ids].clone()
        self.prev_wrist_lin_vel[env_ids] = self.base_lin_vel[env_ids].clone()
        self.prev_wrist_ang_vel[env_ids] = self.base_ang_vel[env_ids].clone()


    def _refresh_lab(self):
        # data for hand
        self.fingertip_pos = self.hand.data.body_pos_w[:, self.finger_bodies]
        self.fingertip_rot = self.hand.data.body_quat_w[:, self.finger_bodies]
        self.fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(self.num_envs, self.num_fingertips, 3)
        self.fingertip_velocities = self.hand.data.body_vel_w[:, self.finger_bodies]

        # Get hand joint positions and velocities (only actuated hand joints, not arm joints)
        # self.hand.data.joint_pos contains all joints (arm + hand)
        # We need to extract only the hand (actuated) joints
        self.hand_dof_pos = self.hand.data.joint_pos[:, self.actuated_dof_indices]
        self.hand_dof_vel = self.hand.data.joint_vel[:, self.actuated_dof_indices]
        self.hand_dof_torque = self.hand.data.applied_torque[:, self.actuated_dof_indices]

        self.real_hand_dof_pos = self.hand.data.joint_pos[:, self.hand_joint_indices]
        self.real_hand_dof_vel = self.hand.data.joint_vel[:, self.hand_joint_indices]
        self.real_hand_dof_torque = self.hand.data.applied_torque[:, self.hand_joint_indices]
        
        self.arm_joint_pos = self.hand.data.joint_pos[:, self.arm_joint_indices]
        self.arm_joint_vel = self.hand.data.joint_vel[:, self.arm_joint_indices]
        
        

        # data for object
        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w
        self.object_velocities = self.object.data.root_vel_w
        self.object_linvel = self.object.data.root_lin_vel_w
        self.object_angvel = self.object.data.root_ang_vel_w

        # hand base state
        # self.base_state = self.hand.data.root_state_w
        # self.base_pos = self.base_state[:, :3] - self.scene.env_origins
        # self.base_quat = self.base_state[:, 3:7]
        # self.base_lin_vel = self.base_state[:, 7:10]
        # self.base_ang_vel = self.base_state[:, 10:13]

        # Use the IK end-effector body for wrist state (must match IK controller's EE frame)
        ee_body_idx = self.arm_robot_entity_cfg.body_ids[0]
        self.base_pos = self.hand.data.body_pos_w[:, ee_body_idx] - self.scene.env_origins
        self.base_quat = self.hand.data.body_quat_w[:, ee_body_idx]
        self.base_lin_vel = self.hand.data.body_lin_vel_w[:, ee_body_idx]
        self.base_ang_vel = self.hand.data.body_ang_vel_w[:, ee_body_idx]
        # self.base_pos = self.hand.data.body_pos_w[:, self.wrist_body_idx] - self.scene.env_origins
        # self.base_quat = self.hand.data.body_quat_w[:, self.wrist_body_idx]
        # self.base_lin_vel = self.hand.data.body_lin_vel_w[:, self.wrist_body_idx]
        # self.base_ang_vel = self.hand.data.body_ang_vel_w[:, self.wrist_body_idx]
        # Observation noise injection for sim2real robustness (only during training, not deploy)
        if not getattr(self, '_is_deploy_env', False):
            joint_noise_std = getattr(self.cfg, 'obs_joint_pos_noise', 0.0)
            wrist_pos_noise_std = getattr(self.cfg, 'obs_wrist_pos_noise', 0.0)
            wrist_rot_noise_std = getattr(self.cfg, 'obs_wrist_rot_noise', 0.0)

            if joint_noise_std > 0:
                self.hand_dof_pos = self.hand_dof_pos + joint_noise_std * torch.randn_like(self.hand_dof_pos)
            if wrist_pos_noise_std > 0:
                self.base_pos = self.base_pos + wrist_pos_noise_std * torch.randn_like(self.base_pos)
            if wrist_rot_noise_std > 0:
                # Small rotation noise via axis-angle perturbation
                rot_noise_aa = wrist_rot_noise_std * torch.randn(self.num_envs, 3, device=self.device)
                rot_noise_quat = aa_to_quat(rot_noise_aa)
                self.base_quat = quat_mul(rot_noise_quat, self.base_quat)
                self.base_quat = self.base_quat / torch.norm(self.base_quat, dim=-1, keepdim=True)

        self.base_state = torch.cat([self.base_pos, self.base_quat, self.base_lin_vel, self.base_ang_vel], dim=-1)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        # Save success/failure counts BEFORE super().step() which calls _reset_idx and clears them
        # (success_buf/failure_buf are set in _get_rewards, then cleared in _reset_idx)
        # We can't read them here either because _get_rewards hasn't run yet.
        # Instead, we rely on extras['succeeded'] and extras['failed_execute'] set in _get_rewards.

        # Call parent step method - it returns (obs_dict, rew, terminated, truncated, extras)
        obs_dict, rew, terminated, truncated, extras = super().step(action)

        terminated[:] = self.at_reset_buf

        # Overwrite terminated/time_out extras with actual values
        # (_get_dones sets these from previous step's reset_buf which is stale;
        #  the real reset comes from compute_imitation_reward in _get_rewards)
        extras["terminated"] = self.at_reset_buf.float().mean()
        extras["time_out"] = truncated.float().mean()

        # Add total_rewards to extras
        extras["total_rewards"] = self.total_rew_buf
        extras["total_steps"] = self.progress_buf

        self.progress_buf += 1
        self.running_progress_buf += 1

        # success_rate/failure_rate: use values from _get_rewards (set before _reset_idx clears buffers)
        extras["success_rate"] = extras.get("succeeded", torch.tensor(0.0)).item() if isinstance(extras.get("succeeded", 0.0), torch.Tensor) else extras.get("succeeded", 0.0)
        extras["failure_rate"] = extras.get("failed_execute", torch.tensor(0.0)).item() if isinstance(extras.get("failed_execute", 0.0), torch.Tensor) else extras.get("failed_execute", 0.0)

        # Add average wrist action magnitude to extras
        extras["wrist_action_magnitude"] = self.wrist_action_magnitude_buf.mean().item()

        # Adaptive init buffer stats
        if self._state_buffer is not None:
            for k, v in self._state_buffer.stats().items():
                extras[f"adaptive_init/{k}"] = v

        at_reset_env_ids = self.at_reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self.at_reset_buf[at_reset_env_ids] = 0

        return obs_dict, rew, terminated, truncated, extras


    def compute_observations(self):
        """Compute observations including proprioception and target states for keypoint tracking."""
        self._refresh_lab()
        # Proprioception observations (joint states, base state)
        obs_values = []
        
        # Joint positions (cos/sin encoding)
        q = self.hand_dof_pos
        obs_values.append(q)
        obs_values.append(torch.cos(q))
        obs_values.append(torch.sin(q))
        
        
        # Ignore base position, only use orientation and velocities
        base_obs = torch.cat([torch.zeros_like(self.base_pos), self.base_quat, self.base_lin_vel, self.base_ang_vel], dim=-1)
        obs_values.append(base_obs)
        
        proprioception_obs = torch.cat(obs_values, dim=-1)
        
        # Target observations (future states)
        obs_future_length = self.obs_future_length
        seq_len = self.demo_data["seq_len"]
        if self.loop_trajectory:
            future_indices = torch.stack([(self._get_demo_idx() + 1 + t) % seq_len for t in range(obs_future_length)], dim=-1)
        else:
            # Clamp each future step independently — adding t without re-clamping
            # can exceed seq_len-1 for envs whose sequence equals the buffer max
            # length, triggering CUDA index out-of-bounds.
            future_indices = torch.stack(
                [torch.clamp(self.progress_buf + 1 + t, torch.zeros_like(seq_len), seq_len - 1)
                 for t in range(obs_future_length)], dim=-1
            )  # [B, K]
        nE, nT = self.demo_data["target_wrist_pos"].shape[:2]
        nF = obs_future_length
        
        def indicing(data, idx):
            """Index data with future indices."""
            assert data.shape[0] == nE and data.shape[1] == nT
            remaining_shape = data.shape[2:]
            expanded_idx = idx
            for _ in remaining_shape:
                expanded_idx = expanded_idx.unsqueeze(-1)
            expanded_idx = expanded_idx.expand(-1, -1, *remaining_shape)
            return torch.gather(data, 1, expanded_idx)
        
        # Get target wrist states
        target_wrist_pos = indicing(self.demo_data["target_wrist_pos"], future_indices)  # [B, K, 3]
        cur_wrist_pos = self.base_pos  # [B, 3]
        delta_wrist_pos = (target_wrist_pos - cur_wrist_pos[:, None]).reshape(nE, -1)
        
        target_wrist_vel = indicing(self.demo_data["target_wrist_velocity"], future_indices)
        cur_wrist_vel = self.base_lin_vel
        wrist_vel = target_wrist_vel.reshape(nE, -1)
        delta_wrist_vel = (target_wrist_vel - cur_wrist_vel[:, None]).reshape(nE, -1)
        
        target_wrist_rot_raw = indicing(self.demo_data["target_wrist_rot"], future_indices)
        # Ensure target_wrist_rot has shape [nE, nF, 3]
        # If it has extra dimensions (e.g., [nE, nF, K, 3]), take the first element
        if target_wrist_rot_raw.ndim > 3:
            target_wrist_rot_raw = target_wrist_rot_raw[:, :, 0, :]
        target_wrist_rot = target_wrist_rot_raw
        target_wrist_quat = aa_to_quat(target_wrist_rot.reshape(nE * nF, -1))  # Convert to (w,x,y,z)
        delta_wrist_quat = quat_mul(
            self.base_quat[:, None].repeat(1, nF, 1).reshape(nE * nF, -1),
            quat_conjugate(target_wrist_quat),
        ).reshape(nE, -1)
        wrist_quat = target_wrist_quat.reshape(nE, -1)
        
        target_wrist_ang_vel_raw = indicing(self.demo_data["target_wrist_angular_velocity"], future_indices)
        # Ensure target_wrist_ang_vel has shape [nE, nF, 3]
        # If it has extra dimensions (e.g., [nE, nF, K, 3]), take the first element
        if target_wrist_ang_vel_raw.ndim > 3:
            target_wrist_ang_vel_raw = target_wrist_ang_vel_raw[:, :, 0, :]
        target_wrist_ang_vel = target_wrist_ang_vel_raw
        cur_wrist_ang_vel = self.base_ang_vel
        wrist_ang_vel = target_wrist_ang_vel.reshape(nE, -1)
        delta_wrist_ang_vel = (target_wrist_ang_vel - cur_wrist_ang_vel[:, None]).reshape(nE, -1)
        
        # Get target joint states
        target_joints_pos = indicing(self.demo_data["target_joints_pos"], future_indices).reshape(nE, nF, -1, 3)

        cur_joint_pos = self.hand.data.body_pos_w[:, self.hand_body_indices[1:]] - self.scene.env_origins.unsqueeze(1)
        # cur_joint_pos = torch.stack(cur_joint_pos, dim=1)  # [B, n_joints-1, 3]
        delta_joints_pos = (target_joints_pos - cur_joint_pos[:, None]).reshape(self.num_envs, -1) # check this
        
        target_joints_vel = indicing(self.demo_data["target_joints_velocity"], future_indices).reshape(nE, nF, -1, 3)
        cur_joint_vel = self.hand.data.body_lin_vel_w[:, self.hand_body_indices[1:]]
        joints_vel = target_joints_vel.reshape(self.num_envs, -1)
        delta_joints_vel = (target_joints_vel - cur_joint_vel[:, None]).reshape(self.num_envs, -1)
        
        # Get target object states (if object exists)
        target_obs_list = [
            delta_wrist_pos,
            wrist_vel,
            delta_wrist_vel,
            wrist_quat,
            delta_wrist_quat,
            wrist_ang_vel,
            delta_wrist_ang_vel,
            delta_joints_pos,
            joints_vel,
            delta_joints_vel,
        ]

        
        
        if hasattr(self, 'object') and self.object is not None:
            target_obj_transf = indicing(self.demo_data["obj_trajectory"], future_indices)
            target_obj_transf = target_obj_transf.reshape(nE * nF, 4, 4)
            
            target_obj_pos = target_obj_transf[:, :3, 3].reshape(nE, nF, -1)  # [nE, nF, 3]
            
            # Object position delta
            cur_obj_pos = self.object.data.root_pos_w - self.scene.env_origins
            delta_manip_obj_pos = (
                target_obj_pos - cur_obj_pos[:, None]
            ).reshape(nE, -1)
            target_obs_list.append(delta_manip_obj_pos)
            
            # Object velocity
            target_obj_vel = indicing(self.demo_data["obj_velocity"], future_indices)
            cur_obj_vel = self.object.data.root_lin_vel_w
            manip_obj_vel = target_obj_vel.reshape(nE, -1)
            delta_manip_obj_vel = (target_obj_vel - cur_obj_vel[:, None]).reshape(nE, -1)
            target_obs_list.append(manip_obj_vel)
            target_obs_list.append(delta_manip_obj_vel)
            
            # Object quaternion
            target_obj_quat = rotmat_to_quat(target_obj_transf[:, :3, :3])
            cur_obj_quat = self.object.data.root_quat_w
            delta_manip_obj_quat = quat_mul(
                cur_obj_quat[:, None].repeat(1, nF, 1).reshape(nE * nF, -1),
                quat_conjugate(target_obj_quat),
            ).reshape(nE, -1)
            manip_obj_quat = target_obj_quat.reshape(nE, -1)
            target_obs_list.append(manip_obj_quat)
            target_obs_list.append(delta_manip_obj_quat)
            
            # Object angular velocity
            target_obj_ang_vel = indicing(self.demo_data["obj_angular_velocity"], future_indices)
            cur_obj_ang_vel = self.object.data.root_ang_vel_w
            manip_obj_ang_vel = target_obj_ang_vel.reshape(nE, -1)
            delta_manip_obj_ang_vel = (target_obj_ang_vel - cur_obj_ang_vel[:, None]).reshape(nE, -1)
            target_obs_list.append(manip_obj_ang_vel)
            target_obs_list.append(delta_manip_obj_ang_vel)
            
            # Object to joints distance
            obj_to_joints = torch.norm(
                cur_obj_pos[:, None] - cur_joint_pos, dim=-1
            ).reshape(self.num_envs, -1)
            target_obs_list.append(obj_to_joints)
            
            # Tips distance
            gt_tips_distance = indicing(self.demo_data["tips_distance"], future_indices).reshape(nE, -1)
            target_obs_list.append(gt_tips_distance)

        
        # Add BPS features if available
        if self.obj_bps is not None:
            target_obs_list.append(self.obj_bps)

        # Add contact
        net_contact_forces_history = torch.cat([self._contact_sensor[id].data.net_forces_w_history[:, :, 0, :].unsqueeze(2) for id in self._contact_body_ids], dim=2)
        norm_contact_forces_history = torch.norm(net_contact_forces_history, dim=-1)
        smooth_contact_forces = norm_contact_forces_history[:, 0, :] * self.cfg.contact_smooth + norm_contact_forces_history[:, 1, :] * (1 - self.cfg.contact_smooth)
        latency_samples = torch.rand_like(self.last_contacts)
        latency = torch.where(latency_samples < self.cfg.contact_latency, 1.0, 0.0)
        self.last_contacts = self.last_contacts * latency + smooth_contact_forces * (1 - latency)
        sensed_contacts = self.last_contacts.clone().reshape(nE, -1)

        # Apply configuration flags (shared with force env)
        if getattr(self.cfg, 'binary_contact', False):
            threshold = getattr(self.cfg, 'contact_threshold', 0.2)
            sensed_contacts = torch.where(sensed_contacts > threshold,
                                          torch.ones_like(sensed_contacts),
                                          torch.zeros_like(sensed_contacts))
        if getattr(self.cfg, 'enable_contact_force', True) is False:
            sensed_contacts[:] = 0.0
        if getattr(self.cfg, 'enable_tactile', True) is False:
            sensed_contacts[:] = 0.0

        target_obs_list.append(sensed_contacts)

        # Combine target observations
        target_obs = torch.cat(target_obs_list, dim=-1)

        # Combine proprioception and target observations
        obs_buf = torch.cat([proprioception_obs, target_obs], dim=-1)

        # Update observation history buffer for ProprioAdapt
        # Store first observation_space//3 dimensions (ProprioAdaptTConv expects this)
        # This includes proprioception_obs (76 dims) + part of target_obs
        obs_part_for_hist = obs_buf[:, :self.proprio_hist_dim]  # [num_envs, 194]
        prev_obs_buf = self.obs_buf_lag_history[:, 1:].clone()
        cur_obs_buf = obs_part_for_hist.unsqueeze(1)  # [num_envs, 1, 194]
        self.obs_buf_lag_history[:] = torch.cat([prev_obs_buf, cur_obs_buf], dim=1)
        
        # Extract proprioceptive history for ProprioAdapt
        # Take the last prop_hist_len steps
        if self.cfg.prop_hist_len > 0:
            self.proprio_hist_buf[:] = self.obs_buf_lag_history[:, -self.cfg.prop_hist_len:].clone()
        
        # Handle reset: refill history buffer for reset environments
        at_reset_env_ids = self.at_reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(at_reset_env_ids) > 0:
            # Fill history with current observation for reset environments
            reset_obs = obs_part_for_hist[at_reset_env_ids]  # [num_reset, 194]
            self.obs_buf_lag_history[at_reset_env_ids] = reset_obs.unsqueeze(1).repeat(1, self.obs_buf_lag_history.shape[1], 1)
            self.proprio_hist_buf[at_reset_env_ids] = reset_obs.unsqueeze(1).repeat(1, self.cfg.prop_hist_len, 1)

        # privileged info
        dq = self.hand_dof_vel
        self.priv_info_buf[:,:22] = dq
        priv_obs_values = [self.object_pos, self.object_rot, self.object_velocities]

        self.priv_info_buf[:, 27:40] = torch.cat(priv_obs_values, dim=-1)
        # TODO tip forces not enabled now
        
        return obs_buf
    
    def set_friction(self, assets, values, num_envs):
        """
        Supports:
        - batched asset (hand)
        - list of single-env assets (objects)
        """

        # List case (a different object per env).
        if isinstance(assets, (list, tuple)):

            for env_idx, asset in enumerate(assets):

                # Friction for this env.
                value = values[env_idx]

                materials = asset.root_physx_view.get_material_properties()

                # value shape may be (1,) or (1, 1).
                if value.ndim > 1:
                    value = value.squeeze()

                materials[..., 0] = value
                materials[..., 1] = value

                # Single-env asset: only env 0 can be written.
                env_ids = torch.tensor([0], device="cpu")

                asset.root_physx_view.set_material_properties(materials, env_ids)

            return

        # =========================
        # Batched case (hand).
        # =========================

        materials = assets.root_physx_view.get_material_properties()

        materials[..., 0] = values
        materials[..., 1] = values

        env_ids = torch.arange(num_envs, device="cpu")

        assets.root_physx_view.set_material_properties(materials, env_ids)

    def set_com(self, asset, value, num_envs):
        coms = asset.root_physx_view.get_coms().clone()
        coms[:, :3] += value
        env_ids = torch.arange(num_envs, device="cpu")
        asset.root_physx_view.set_coms(coms, env_ids)

    def set_mass(self, asset, value, num_envs):
        env_ids = torch.arange(num_envs, device="cpu")
        asset.root_physx_view.set_masses(value, env_ids)


    def _init_hand_data(self):
        
        # Keypoint tracking buffers
        self.progress_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.running_progress_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.success_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.failure_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.error_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        self.total_rew_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.reward_dict = {}

        # self.dexhand_handles = {} # dexhand_handles[body_name] = body_index
        # for name in self.hand.body_names:
        #     ids, _ = self.hand.find_bodies(name)
        #     self.dexhand_handles[name] = ids[0]

        wrist_body_name = self.dexhand.to_dex("wrist")[0]
        self.wrist_body_idx = self.hand.body_names.index(wrist_body_name)
        
        
        self.apply_forces = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.apply_torque = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        
        # PID control buffers (for wrist control)
        self.use_pid_control = getattr(self.cfg, 'use_pid_control', False)
        if self.use_pid_control:
            self.prev_pos_error = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.prev_rot_error = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.pos_error_integral = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.rot_error_integral = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.use_joint_pos_control = getattr(self.cfg, 'use_joint_pos_control', False)
        self.use_joint_delta_control = getattr(self.cfg, 'use_joint_delta_control', False)

        # Validate: only one arm control mode can be active
        arm_modes = sum([self.use_pid_control, self.use_joint_pos_control, self.use_joint_delta_control])
        assert arm_modes <= 1, (
            f"Only one arm control mode can be active, got: "
            f"use_pid_control={self.use_pid_control}, "
            f"use_joint_pos_control={self.use_joint_pos_control}, "
            f"use_joint_delta_control={self.use_joint_delta_control}"
        )
        active_mode = "pid/IK" if self.use_pid_control else "joint_pos" if self.use_joint_pos_control else "joint_delta" if self.use_joint_delta_control else "force/torque"
        print(f"[INFO] Arm control mode: {active_mode}")

        self.random_state_init = self.cfg.random_state_init
        self.use_quat_rot = self.cfg.use_quat_rot
        self.actions_moving_average = self.cfg.actions_moving_average
        self.obs_future_length = self.cfg.obs_future_length
        self.translation_scale = self.cfg.translation_scale
        self.orientation_scale = self.cfg.orientation_scale

        # Initialize default DOF pose
        default_pose = torch.ones(self.num_hand_dofs, device=self.device) * np.pi / 36
        self.dexhand_default_dof_pos = default_pose

        # Store DOF speed limits for reset
        # use the hand ids
        # self._dexhand_dof_speed_limits = self.hand.data.joint_vel_limits[:, self.hand_joint_indices].clone()
        self._dexhand_dof_speed_limits = self.hand.data.joint_vel_limits[:, self.actuated_dof_indices].clone()

        self.arm_joint_pos_des_prev = torch.zeros((self.num_envs, len(self.arm_joint_indices)), device=self.device)

        self.arm_joint_pos_des = torch.zeros((self.num_envs, len(self.arm_joint_indices)), device=self.device)
        
        # Buffers for wrist target smoothing (similar to hand joint targets)
        self.prev_wrist_target_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.prev_wrist_target_quat = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.cur_wrist_target_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.cur_wrist_target_quat = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        
        # Buffers for velocity/acceleration penalty (to reduce jitter)
        self.prev_arm_joint_vel = torch.zeros((self.num_envs, len(self.arm_joint_indices)), dtype=torch.float, device=self.device)
        self.prev_wrist_lin_vel = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.prev_wrist_ang_vel = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        
        # Buffer for wrist action magnitude tracking
        self.wrist_action_magnitude_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
    
    def _build_data(self):
        def segment_data(k):
            idx = self.data_indices[k % len(self.data_indices)]
            return self.demo_dataset_dict[ManipDataFactory.dataset_type(idx)][idx]
        
        # Print the assignment.
        print(f"[INFO] Building data for {self.num_envs} envs from {len(self.data_indices)} data indices:")
        for i in range(min(self.num_envs, 20)):
            idx = self.data_indices[i % len(self.data_indices)]
            print(f"  env {i:4d} -> {idx}")
        if self.num_envs > 20:
            print(f"  ... (cycling through {len(self.data_indices)} indices total)")
        

        
        raw = [segment_data(i) for i in tqdm(range(self.num_envs))]

        seq_lens = torch.tensor([len(d["obj_trajectory"]) for d in raw], device=self.device)
        max_len = int(seq_lens.max())
        assert max_len <= self.max_episode_length, f"max_len ({max_len}) should be <= max_episode_length ({self.max_episode_length})"
        packed = {"seq_len": seq_lens}

        def fill(xs):
            out = []
            for x in xs:
                if len(x) < max_len:
                    x = torch.cat([x, x[-1:].repeat(max_len - len(x), *[1 for _ in x.shape[1:]])], dim=0)
                out.append(x)
            return torch.stack(out)

        
        for k in raw[0].keys():
            if "alt" in k:
                continue
            if k in ["mano_joints", "mano_joints_velocity"]:
                mj = []
                for d in raw:
                    curr_joints = torch.concat(
                        [d[k][self.dexhand.to_hand(j)[0]]
                        for j in self.hand_body_names
                        if self.dexhand.to_hand(j)[0] != "wrist"],
                        dim=-1
                    )
                    mj.append(curr_joints)
                packed[k] = fill(mj)
            elif isinstance(raw[0][k], torch.Tensor):
                if k == "obj_verts":
                    packed[k] = torch.stack([(d[k] if k in d else raw[0][k]) for d in raw])
                elif k == "xy_offset":
                    packed[k] = torch.stack([(d[k] if k in d else raw[0][k]) for d in raw])
                elif k == "z_offset":
                    packed[k] = torch.stack([(d[k] if k in d else raw[0][k]) for d in raw])
                elif k == "aug_yaw_deg":
                    packed[k] = torch.stack([(d[k] if k in d else raw[0][k]) for d in raw]).float()   # [num_envs]
                elif k in ("aux_obj_pos", "aux_obj_quat"):
                    # static per-env (single 3- or 4-vec, not a trajectory)
                    packed[k] = torch.stack([(d[k] if k in d else raw[0][k]) for d in raw])
                else:
                    packed[k] = fill([(d[k] if k in d else raw[0][k]) for d in raw])
            else:
                packed[k] = [(d[k] if k in d else raw[0][k]) for d in raw]

        def to_dev(x):
            if isinstance(x, torch.Tensor): return x.to(self.device)
            if isinstance(x, list): return [to_dev(xx) for xx in x]
            if isinstance(x, dict): return {k: to_dev(v) for k,v in x.items()}
            return x

        self.demo_data = to_dev(packed)

        self._obj_urdf_list = self.demo_data['obj_urdf_path']
        # === Replay aug_yaw_deg exactly like retargeting, BEFORE the xy/z offset ===
        if "aug_yaw_deg" in self.demo_data:
            yaw_deg = self.demo_data["aug_yaw_deg"].float().to(self.device)   # [N]
        else:
            yaw_deg = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        if torch.any(torch.abs(yaw_deg) > 1e-6):
            R2 = yaw_rotmat_2x2_batch(yaw_deg)   # [N,2,2]
            R3 = yaw_rotmat_3x3_batch(yaw_deg)   # [N,3,3]

            # retarget is rotating around arm_center_xy = [arm_base_pos_x, arm_base_pos_y]
            center_xy = torch.tensor(
                [self.cfg.arm_base_pos[0], self.cfg.arm_base_pos[1]],
                device=self.device,
                dtype=torch.float32,
            ).unsqueeze(0).repeat(self.num_envs, 1)   # [N,2]

            # 1) wrist_pos: [N,T,3]
            wrist_pos = self.demo_data["wrist_pos"]
            wrist_xy = wrist_pos[:, :, :2] - center_xy[:, None, :]
            wrist_xy = torch.einsum("nij,ntj->nti", R2, wrist_xy) + center_xy[:, None, :]
            wrist_pos[:, :, :2] = wrist_xy
            self.demo_data["wrist_pos"] = wrist_pos

            # 2) wrist_rot: axis-angle [N,T,3]
            wrist_rot = self.demo_data["wrist_rot"]
            N, T, _ = wrist_rot.shape
            wrist_rot_mat = aa_to_rotmat(wrist_rot.reshape(-1, 3)).reshape(N, T, 3, 3)
            wrist_rot_mat = torch.einsum("nij,ntjk->ntik", R3, wrist_rot_mat)
            self.demo_data["wrist_rot"] = rotmat_to_aa(wrist_rot_mat.reshape(-1, 3, 3)).reshape(N, T, 3)

            # 3) obj_trajectory: [N,T,4,4]
            obj_traj = self.demo_data["obj_trajectory"]
            obj_xy = obj_traj[:, :, :2, 3] - center_xy[:, None, :]
            obj_xy = torch.einsum("nij,ntj->nti", R2, obj_xy) + center_xy[:, None, :]
            obj_traj[:, :, :2, 3] = obj_xy
            obj_traj[:, :, :3, :3] = torch.einsum("nij,ntjk->ntik", R3, obj_traj[:, :, :3, :3])
            self.demo_data["obj_trajectory"] = obj_traj

            # 3b) aux_obj_pos/quat: same in-plane yaw rotation around center_xy
            if self.has_aux and "aux_obj_pos" in self.demo_data:
                aux_pos = self.demo_data["aux_obj_pos"]    # [N, 3]
                aux_xy = aux_pos[:, :2] - center_xy
                aux_xy = torch.einsum("nij,nj->ni", R2, aux_xy) + center_xy
                aux_pos[:, :2] = aux_xy
                self.demo_data["aux_obj_pos"] = aux_pos

                # Rotate quat by yaw around z: q_new = q_yaw * q_old (wxyz)
                aux_quat = self.demo_data["aux_obj_quat"]  # [N, 4] wxyz
                yaw_half = (yaw_deg * np.pi / 180.0 / 2.0).to(aux_quat.dtype)
                q_yaw_w = torch.cos(yaw_half)
                q_yaw_z = torch.sin(yaw_half)
                # q_yaw = (w, 0, 0, z) ; q_old = (w0, x0, y0, z0)
                w0, x0, y0, z0 = aux_quat[:, 0], aux_quat[:, 1], aux_quat[:, 2], aux_quat[:, 3]
                qw = q_yaw_w * w0 - q_yaw_z * z0
                qx = q_yaw_w * x0 - q_yaw_z * y0
                qy = q_yaw_w * y0 + q_yaw_z * x0
                qz = q_yaw_w * z0 + q_yaw_z * w0
                self.demo_data["aux_obj_quat"] = torch.stack([qw, qx, qy, qz], dim=-1)

            # 4) mano_joints: [N,T,J*3]
            num_hand_bodies = len(self.hand_body_names) - 1
            joints_reshaped = self.demo_data["mano_joints"].reshape(self.num_envs, max_len, num_hand_bodies, 3)
            mj_xy = joints_reshaped[:, :, :, :2] - center_xy[:, None, None, :]
            mj_xy = torch.einsum("nij,ntkj->ntki", R2, mj_xy) + center_xy[:, None, None, :]
            joints_reshaped[:, :, :, :2] = mj_xy
            self.demo_data["mano_joints"] = joints_reshaped.reshape(self.num_envs, max_len, num_hand_bodies * 3)

            print(f"[INFO] Applied aug_yaw_deg in env, env0 yaw={yaw_deg[0].item():.2f} deg")
            # 5) wrist_velocity: [N,T,3]  linear velocity: rotate xy only
            if "wrist_velocity" in self.demo_data:
                wrist_vel = self.demo_data["wrist_velocity"]
                wrist_vel_xy = torch.einsum("nij,ntj->nti", R2, wrist_vel[:, :, :2])
                wrist_vel[:, :, :2] = wrist_vel_xy
                self.demo_data["wrist_velocity"] = wrist_vel

            # 6) wrist_angular_velocity: [N,T,3]  angular velocity: rotate in 3D
            if "wrist_angular_velocity" in self.demo_data:
                wrist_ang_vel = self.demo_data["wrist_angular_velocity"]
                wrist_ang_vel = torch.einsum("nij,ntj->nti", R3, wrist_ang_vel)
                self.demo_data["wrist_angular_velocity"] = wrist_ang_vel

            # 7) obj_velocity: [N,T,3]  linear velocity: rotate xy only
            if "obj_velocity" in self.demo_data:
                obj_vel = self.demo_data["obj_velocity"]
                obj_vel_xy = torch.einsum("nij,ntj->nti", R2, obj_vel[:, :, :2])
                obj_vel[:, :, :2] = obj_vel_xy
                self.demo_data["obj_velocity"] = obj_vel

            # 8) obj_angular_velocity: [N,T,3]  angular velocity: rotate in 3D
            if "obj_angular_velocity" in self.demo_data:
                obj_ang_vel = self.demo_data["obj_angular_velocity"]
                obj_ang_vel = torch.einsum("nij,ntj->nti", R3, obj_ang_vel)
                self.demo_data["obj_angular_velocity"] = obj_ang_vel

            # 9) mano_joints_velocity: [N,T,J*3]
            if "mano_joints_velocity" in self.demo_data:
                joints_vel = self.demo_data["mano_joints_velocity"].reshape(self.num_envs, max_len, num_hand_bodies, 3)
                joints_vel = torch.einsum("nij,ntkj->ntki", R3, joints_vel)
                self.demo_data["mano_joints_velocity"] = joints_vel.reshape(self.num_envs, max_len, num_hand_bodies * 3)
        self.xy_offset = self.demo_data["xy_offset"]  # [num_envs, 2] - XY offset only
        # === end aug_yaw_deg replay ===
        
        
        # Apply xy_offset to object trajectory (only XY translation)
        # obj_trajectory shape: [num_envs, T, 4, 4]
        obj_traj = self.demo_data["obj_trajectory"]
        obj_traj[:, :, :2, 3] += self.xy_offset.unsqueeze(1)  # Apply to XY coordinates only
        # Apply xy_offset and z_offset to mano_joints
        num_hand_bodies = len(self.hand_body_names) - 1  # Excludes wrist
        joints_reshaped = self.demo_data["mano_joints"].view(self.num_envs, max_len, num_hand_bodies, 3)  # (N, F, J, 3)
        joints_reshaped[:, :, :, :2] += self.xy_offset[:, None, None, :2]  # Apply to XY coordinates only
        
        # Get z_offset from demo_data, default to 0.1 if not present
        if "z_offset" in self.demo_data:
            self.z_offset = self.demo_data["z_offset"]
            # z_offset may be stored as [N] (scalar per env) or [N,1] depending on the
            # dataset variant. Normalize to a [N,1] column so the broadcasts below align
            # on T (not on num_envs). reshape works for both shapes.
            _zcol = self.z_offset.reshape(self.num_envs, 1)  # [N,1]
            obj_traj[:, :, 2, 3] += _zcol  # Apply Z offset (broadcasts to [num_envs, T])
            self.demo_data["wrist_pos"][:, :, 2] += _zcol  # Apply Z offset (broadcasts to [num_envs, T])
            joints_reshaped[:, :, :, 2] += _zcol[:, None]  # Apply Z offset (broadcasts to [num_envs, T, J])
            print(f"Applied z_offset to wrist_pos and mano_joints: {self.z_offset[0].item()}")

        self.demo_data["obj_trajectory"] = obj_traj
        self.demo_data["wrist_pos"][:, :, :2] += self.xy_offset.unsqueeze(1)  # Apply to XY coordinates only
        self.demo_data["mano_joints"] = joints_reshaped.view(self.num_envs, max_len, num_hand_bodies * 3)
        print(f"Applied xy_offset to wrist_pos and mano_joints: {self.xy_offset[0].cpu().numpy()}")

        # Apply xy_offset (and z_offset if present) to aux_obj_pos
        if self.has_aux and "aux_obj_pos" in self.demo_data:
            aux_pos = self.demo_data["aux_obj_pos"]   # [N, 3]
            aux_pos[:, :2] += self.xy_offset
            if "z_offset" in self.demo_data:
                # z_offset can be shape [N] (scalar per env) OR [N, 1] (1-elem
                # vector per env) depending on dataset variant. Squeeze trailing
                # singleton so it broadcasts cleanly with aux_pos[:, 2] = [N].
                zo = self.demo_data["z_offset"]
                while zo.dim() > 1:
                    zo = zo.squeeze(-1)
                aux_pos[:, 2] += zo
            self.demo_data["aux_obj_pos"] = aux_pos
            print(f"Applied xy_offset to aux_obj_pos: env0 aux_pos = {aux_pos[0].cpu().numpy()}")

        # Reference source: pick MANO vs retargeted opt_* based on cfg.
        # "auto" — prefer opt_* if present, fall back to MANO
        # "retarget" — strict, raise if opt_* missing
        # "mano" — force MANO even if opt_* present
        _ref_src = str(getattr(self.cfg, "reference_source", "auto")).lower()
        if _ref_src not in ("auto", "retarget", "mano"):
            print(f"[WARN] reference_source={_ref_src!r} unknown; using 'auto'")
            _ref_src = "auto"

        def _pick_ref(opt_key: str, fallback_key: str):
            if _ref_src == "mano":
                return self.demo_data[fallback_key], "mano"
            if _ref_src == "retarget":
                if opt_key not in self.demo_data:
                    raise KeyError(
                        f"reference_source='retarget' but {opt_key!r} not in demo_data; "
                        f"either retarget the data or switch cfg.reference_source to 'auto'/'mano'"
                    )
                return self.demo_data[opt_key], "retarget"
            # auto: prefer opt_* if present
            if opt_key in self.demo_data:
                return self.demo_data[opt_key], "retarget"
            return self.demo_data[fallback_key], "mano"

        self.demo_data["target_wrist_pos"], _wrist_src_pos = _pick_ref("opt_wrist_pos", "wrist_pos")
        self.demo_data["target_wrist_rot"], _wrist_src_rot = _pick_ref("opt_wrist_rot", "wrist_rot")
        self.demo_data["target_wrist_velocity"], _ = _pick_ref(
            "opt_wrist_velocity", "wrist_velocity"
        )
        self.demo_data["target_wrist_angular_velocity"], _ = _pick_ref(
            "opt_wrist_angular_velocity", "wrist_angular_velocity"
        )
        print(f"[INFO] Reference source for wrist: cfg='{_ref_src}', "
              f"resolved pos={_wrist_src_pos}, rot={_wrist_src_rot}")

        target_body_names = [
            b for b in self.hand_body_names
            if self.dexhand.to_hand(b)[0] != "wrist"
        ]
        source_body_names = self.demo_data.get("opt_joints_body_names", None)
        if (
            isinstance(source_body_names, list)
            and len(source_body_names) > 0
            and isinstance(source_body_names[0], list)
        ):
            source_body_names = source_body_names[0]

        def _target_joints_from_retarget(opt_key, fallback_key):
            target = self.demo_data[fallback_key]
            used_retarget = False
            # Honor cfg.reference_source:
            #   "mano"     → always use fallback (MANO joints), skip retarget
            #   "retarget" → strict, raise if opt_key missing
            #   "auto"     → prefer opt_key if present
            if _ref_src == "mano":
                return target, used_retarget
            if opt_key not in self.demo_data:
                if _ref_src == "retarget":
                    raise KeyError(
                        f"reference_source='retarget' but {opt_key!r} not in demo_data"
                    )
                return target, used_retarget

            opt = self.demo_data[opt_key]
            if opt.ndim == 3 and opt.shape[-1] % 3 == 0:
                opt = opt.reshape(self.num_envs, max_len, -1, 3)
            elif opt.ndim != 4:
                print(f"[WARN] {opt_key} unsupported ndim={opt.ndim}")
                return target, used_retarget

            if opt.shape[-1] != 3:
                print(f"[WARN] {opt_key} shape mismatch: {tuple(opt.shape)}")
                return target, used_retarget

            aligned = None
            align_mode = None
            if (
                isinstance(source_body_names, list)
                and all(name in source_body_names for name in target_body_names)
            ):
                src_idx = torch.tensor(
                    [source_body_names.index(name) for name in target_body_names],
                    dtype=torch.long,
                    device=self.device,
                )
                aligned = opt.index_select(2, src_idx)
                align_mode = "body_names"
            elif opt.shape[2] == num_hand_bodies + 1:
                aligned = opt[:, :, 1:, :]
                align_mode = "drop_wrist"
            elif opt.shape[2] == num_hand_bodies:
                aligned = opt
                align_mode = "direct"

            if aligned is None:
                print(
                    f"[WARN] {opt_key} shape mismatch: {tuple(opt.shape)} "
                    f"vs target bodies {num_hand_bodies}"
                )
                return target, used_retarget

            target = aligned.reshape(self.num_envs, max_len, num_hand_bodies * 3)
            used_retarget = True
            if opt_key == "opt_joints_pos":
                print(f"[RetargetRef] {opt_key} aligned by {align_mode}")
            return target, used_retarget

        self.demo_data["target_joints_pos"], used_retarget_joints = _target_joints_from_retarget(
            "opt_joints_pos", "mano_joints"
        )
        self.demo_data["target_joints_velocity"], _ = _target_joints_from_retarget(
            "opt_joints_velocity", "mano_joints_velocity"
        )
        print(
            "[INFO] Reward/target keypoints use "
            f"{'retargeted opt_joints_pos' if used_retarget_joints else 'MANO mano_joints'}"
        )
        
        # Apply obj_scale to obj_verts if retarget provided a non-1.0 scale
        if "obj_scale" in self.demo_data:
            scales = self.demo_data["obj_scale"]  # list of floats (one per env)
            if isinstance(scales, list):
                for i, s in enumerate(scales):
                    if isinstance(s, (int, float)) and s != 1.0:
                        self.demo_data["obj_verts"][i] = self.demo_data["obj_verts"][i] * s
                        print(f"[INFO] Scaled obj_verts[{i}] by {s}")
            elif isinstance(scales, (int, float)) and scales != 1.0:
                self.demo_data["obj_verts"] = self.demo_data["obj_verts"] * scales
                print(f"[INFO] Scaled all obj_verts by {scales}")

        # Initialize BPS encoding if available
        if BPS_AVAILABLE:
            self.bps_feat_type = "dists"
            self.bps_layer = bps_torch(
                bps_type="grid_sphere", n_bps_points=128, radius=0.2, randomize=False, device=self.device
            )
            obj_verts = self.demo_data["obj_verts"]
            self.obj_bps = self.bps_layer.encode(obj_verts, feature_type=self.bps_feat_type)[self.bps_feat_type]
        else:
            self.bps_layer = None
            self.obj_bps = None
            print("WARNING: BPS encoding disabled. Observations will not include BPS features.")
        
        # Object management (will be initialized in _setup_scene or _create_objects)
        self.manip_obj_mass = None
        self.manip_obj_com = None
        self.objs_assets = {}  # Cache for object assets (not used in IsaacLab, kept for compatibility)
        
        # Initialize object mass and COM if needed
        if "obj_id" in self.demo_data:
            self.manip_obj_mass = []
            self.manip_obj_com = []
            for i in range(self.num_envs):
                obj_id = self.demo_data["obj_id"][i] if isinstance(self.demo_data["obj_id"], list) else self.demo_data["obj_id"][i]
                if obj_id in oakink2_obj_mass:
                    self.manip_obj_mass.append(oakink2_obj_mass[obj_id])
                else:
                    self.manip_obj_mass.append(0.05)  # Default mass
                self.manip_obj_com.append(torch.zeros(3, device=self.device))  # Default COM
            self.manip_obj_mass = torch.tensor(self.manip_obj_mass, device=self.device)
            self.manip_obj_com = torch.stack(self.manip_obj_com, dim=0).to(self.device)
        self.urdf_path = self.demo_data["obj_urdf_path"][0]
    
    def _identify_arm_joints(self):
        """Identify Franka arm joint indices."""
        all_joint_names = self.hand.data.joint_names
        
        # Arm joints (FR3)
        self.arm_joint_names = [
            "fr3_joint1",
            "fr3_joint2",
            "fr3_joint3",
            "fr3_joint4",
            "fr3_joint5",
            "fr3_joint6",
            "fr3_joint7",
        ]
        self.arm_joint_indices = []
        for joint_name in self.arm_joint_names:
            if joint_name in all_joint_names:
                self.arm_joint_indices.append(all_joint_names.index(joint_name))
            else:
                # Try case-insensitive match
                found = False
                for i, robot_joint_name in enumerate(all_joint_names):
                    if joint_name.lower() == robot_joint_name.lower():
                        self.arm_joint_indices.append(i)
                        found = True
                        break
                if not found:
                    print(f"Warning: Arm joint {joint_name} not found in robot joints")
        
        print(f"Arm joint indices: {self.arm_joint_indices}")
        print(f"Arm joint names: {[all_joint_names[i] for i in self.arm_joint_indices]}")
        
        # Get arm joint limits
        joint_pos_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        if joint_pos_limits.ndim == 3:
            all_joint_lower_limits = joint_pos_limits[0, :, 0]
            all_joint_upper_limits = joint_pos_limits[0, :, 1]
        else:
            all_joint_lower_limits = joint_pos_limits[:, 0]
            all_joint_upper_limits = joint_pos_limits[:, 1]
        
        if len(self.arm_joint_indices) > 0:
            self.arm_joint_lower_limits = all_joint_lower_limits[self.arm_joint_indices]
            self.arm_joint_upper_limits = all_joint_upper_limits[self.arm_joint_indices]
        else:
            # Fallback limits
            self.arm_joint_lower_limits = torch.tensor([-2.9, -1.8, -2.9, -3.1, -2.9, 0.4, -3.1], device=self.device)
            self.arm_joint_upper_limits = torch.tensor([2.9, 1.8, 2.9, -0.1, 2.9, 4.6, 3.1], device=self.device)
        
        # End effector link name (wrist link)
        self.arm_end_effector_link = f"{self.hand_side}_hand_C_MC"  # Wrist link name in combined URDF

        self.hand_body_names = []
        for body_name in self.hand.body_names:
            if f"{self.hand_side}_" in body_name:
                self.hand_body_names.append(body_name)
        print(f"Hand body names: {self.hand_body_names}")

        # ====== weight_idx: body-name resolver + sanity assert ======
        # Build `self.dexhand_weight_idx` by looking up `weight_idx_body_names`
        # against the actual PhysX body order. This is robust to URDF / USD
        # parse-order changes and to left/right hand variants. The hardcoded
        # int indices in `sharpa.py:weight_idx` serve as a cross-check: if the
        # resolved indices ever diverge from the hardcoded ones a hard assert
        # fires so we catch the discrepancy at startup instead of in training.
        _wb = self.hand_body_names
        self.dexhand_weight_idx: dict = {}
        _by_name = getattr(self.dexhand, "weight_idx_body_names", None)
        if _by_name is None:
            print("[WEIGHT_IDX] WARN: dexhand has no `weight_idx_body_names`; "
                  "falling back to hardcoded int indices.")
            self.dexhand_weight_idx = {k: list(v) for k, v in self.dexhand.weight_idx.items()}
        else:
            _missing = []
            for _key, _names in _by_name.items():
                _idx_list = []
                for _n in _names:
                    _full = f"{self.hand_side}_{_n}"
                    if _full in _wb:
                        _idx_list.append(_wb.index(_full))
                    else:
                        _missing.append((_key, _full))
                self.dexhand_weight_idx[_key] = _idx_list
            if _missing:
                raise RuntimeError(
                    f"[WEIGHT_IDX] resolver failed: bodies not found in "
                    f"hand_body_names: {_missing}.\n"
                    f"hand_body_names = {_wb}"
                )

            # Sanity: resolved indices should match the hardcoded weight_idx
            # exactly (as sets — order within a list is irrelevant for the
            # reward terms that use `.mean(dim=-1)`).
            _legacy = self.dexhand.weight_idx
            _mismatches = []
            for _key in self.dexhand_weight_idx.keys():
                _r = sorted(self.dexhand_weight_idx[_key])
                _l = sorted(_legacy.get(_key, []))
                if _r != _l:
                    _mismatches.append((_key, _r, _l))
            if _mismatches:
                print("[WEIGHT_IDX] resolved hand_body_names order:")
                for i, n in enumerate(_wb):
                    print(f"  [{i:2d}] {n}")
                print("[WEIGHT_IDX] resolved (by-name):")
                for k, v in self.dexhand_weight_idx.items():
                    print(f"  {k}: {sorted(v)}  → {[_wb[i] for i in v]}")
                print("[WEIGHT_IDX] hardcoded (sharpa.py:weight_idx):")
                for k, v in _legacy.items():
                    print(f"  {k}: {sorted(v)}")
                raise AssertionError(
                    f"[WEIGHT_IDX] resolved vs hardcoded disagree on: "
                    f"{[(k, r, l) for k, r, l in _mismatches]}. "
                    f"URDF / USD body order probably changed — update the "
                    f"hardcoded `weight_idx` in sharpa.py, OR (preferred) "
                    f"remove it now that the by-name resolver is trustworthy."
                )
            print(f"[WEIGHT_IDX] by-name resolver matches hardcoded indices "
                  f"for all {len(self.dexhand_weight_idx)} keys ✓")
        # ====== END weight_idx ======

        self.hand_joint_indices = list()
        for joint_name in self.cfg.actuated_joint_names:
            self.hand_joint_indices.append(self.hand.joint_names.index(joint_name))
        self.hand_body_indices = [self.hand.body_names.index(body_name) for body_name in self.hand_body_names]
        
        # Identify arm body indices (FR3 arm bodies, excluding hand bodies)
        self.arm_body_indices = []
        for i, body_name in enumerate(self.hand.body_names):
            # Arm bodies typically contain "fr3" or are part of the arm structure
            # Exclude hand bodies (which contain hand_side prefix) and root/base
            if ("fr3" in body_name.lower() or "link" in body_name.lower()) and f"{self.hand_side}_" not in body_name:
                # Also exclude the end effector link (wrist) as it's the boundary
                if self.arm_end_effector_link not in body_name:
                    self.arm_body_indices.append(i)
        print(f"Arm body indices: {self.arm_body_indices}")
        print(f"Arm body names: {[self.hand.body_names[i] for i in self.arm_body_indices]}")
        
    
    def _init_arm_diff_ik_controller(self):
        """Initialize Differential IK controller for arm control."""
        # Create SceneEntityCfg for arm joints and end effector
        # Use arm joints only for IK, end effector is wrist link
        self.arm_robot_entity_cfg = SceneEntityCfg(
            "robot", 
            joint_names=["fr3_joint.*"], 
            body_names=[self.arm_end_effector_link]
        )
        # Resolve the scene entities
        self.arm_robot_entity_cfg.resolve(self.scene)
        
        # Obtain the frame index of the end-effector
        # For a fixed base robot, the frame index is one less than the body index
        if self.hand.is_fixed_base:
            self.ee_jacobi_idx = self.arm_robot_entity_cfg.body_ids[0] - 1
        else:
            self.ee_jacobi_idx = self.arm_robot_entity_cfg.body_ids[0]
        
        # Create Differential IK controller
        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", 
            use_relative_mode=False, 
            ik_method="dls"
        )
        self.arm_diff_ik_controller = DifferentialIKController(
            diff_ik_cfg, 
            num_envs=self.num_envs, 
            device=self.device
        )
        
        print(f"[INFO]: Initialized Differential IK controller for arm")
        print(f"  Arm joint indices: {self.arm_joint_indices}")
        print(f"  End effector body: {self.arm_end_effector_link}")
        print(f"  End effector body ID: {self.arm_robot_entity_cfg.body_ids[0]}")
        print(f"  End effector Jacobian index: {self.ee_jacobi_idx}")

    def _init_osc_controller(self):
        """Initialize Operational Space Controller for arm control."""
        if not getattr(self.cfg, 'use_osc_control', False):
            self._osc = None
            return

        from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg

        osc_cfg = OperationalSpaceControllerCfg(
            target_types=["pose_abs"],
            impedance_mode="fixed",
            inertial_dynamics_decoupling=True,
            partial_inertial_dynamics_decoupling=getattr(self.cfg, 'osc_partial_decoupling', True),
            gravity_compensation=True,
            motion_stiffness_task=(
                self.cfg.osc_kp_xyz, self.cfg.osc_kp_xyz, self.cfg.osc_kp_xyz,
                self.cfg.osc_kp_rot, self.cfg.osc_kp_rot, self.cfg.osc_kp_rot,
            ),
            motion_damping_ratio_task=self.cfg.osc_damping_ratio,
            nullspace_control="position",
            nullspace_stiffness=getattr(self.cfg, 'osc_nullspace_stiffness', 10.0),
        )
        self._osc = OperationalSpaceController(osc_cfg, num_envs=self.num_envs, device=self.device)
        print(f"[INFO]: Initialized OSC controller for arm")
        print(f"  Kp xyz={self.cfg.osc_kp_xyz}, Kp rot={self.cfg.osc_kp_rot}, "
              f"damping={self.cfg.osc_damping_ratio}, partial_decoupling={self.cfg.osc_partial_decoupling}")

@torch.jit.script
def quat_to_angle_axis(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes axis-angle representation from quaternion q.
    q must be normalized. Returns (angle, axis).
    """
    min_theta = 1e-5
    qx, qy, qz, qw = 1, 2, 3, 0  # IsaacLab uses (w, x, y, z)
    
    sin_theta = torch.sqrt(torch.clamp(1 - q[..., qw] * q[..., qw], min=0.0))
    angle = 2 * torch.acos(torch.clamp(q[..., qw], -1.0, 1.0))
    angle = wrap_to_pi(angle)  # Normalize angle to [-pi, pi]
    sin_theta_expand = sin_theta.clamp(min=min_theta).unsqueeze(-1)
    axis = q[..., qx:qz+1] / sin_theta_expand
    
    mask = torch.abs(sin_theta) > min_theta
    default_axis = torch.zeros_like(axis)
    default_axis[..., -1] = 1
    
    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)
    return angle, axis

# Training strict-proxy threshold, in metres. The 3 cm distance matches eval
# strict3, but episode filtering differs (see docs/TRAINING.md). The eval CLI's
# --success_dist controls closest-approach success, not post-hoc strict3.
# Baked into the TorchScript function below at script time.
STRICT_SUCCESS_DIST: float = 0.03

# Table top height (m), from dexx.deploy_config. Baked into the TorchScript
# reward below as a parameter default, like STRICT_SUCCESS_DIST.
TABLE_SURFACE_Z: float = float(_dcfg.TABLE_SURFACE_Z)


@torch.jit.script
def compute_imitation_reward(
    reset_buf: torch.Tensor,
    progress_buf: torch.Tensor,
    running_progress_buf: torch.Tensor,
    actions: torch.Tensor,
    states: Dict[str, torch.Tensor],
    target_states: Dict[str, torch.Tensor],
    max_length: torch.Tensor,
    scale_factor: float,
    dexhand_weight_idx: Dict[str, List[int]],
    use_wrist_tracking: bool = True,
    use_abs_hand_tracking: bool = True,
    use_rel_hand_tracking: bool = False,
    arm_action_rate_penalty: float = 0.1,
    no_slip_weight: float = 0.0,
    approach_shaping_v2: bool = False,
    success_pos_weight: float = 0.0,
    success_rot_weight: float = 0.0,
    success_approach_weight: float = 0.0,
    success_alpha_pos: float = 30.0,
    success_alpha_rot: float = 3.0,
    success_reward_ramp: bool = False,
    success_reward_window: int = 5,
    premature_contact_dist_threshold: float = 0.005,
    premature_contact_progress_threshold: int = 50,
    premature_contact_enabled: bool = True,
    eval_no_terminate: bool = False,
    # Passed as parameters, not read from the module: TorchScript cannot close
    # over a global float. The defaults are evaluated by Python at definition
    # time, so the scripted body still sees one number with one definition.
    strict_success_dist: float = STRICT_SUCCESS_DIST,
    table_surface_z: float = TABLE_SURFACE_Z,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:

    # end effector pose reward
    current_eef_pos = states["base_state"][:, :3]
    current_eef_quat = states["base_state"][:, 3:7]

    target_eef_pos = target_states["wrist_pos"]
    target_eef_quat = target_states["wrist_quat"]
    diff_eef_pos = target_eef_pos - current_eef_pos
    diff_eef_pos_dist = torch.norm(diff_eef_pos, dim=-1)
    # print(f"DEBUG: target eef pos: {target_eef_pos}, current eef pos: {current_eef_pos}, diff_eef_pos_dist: {diff_eef_pos_dist}")

    current_eef_vel = states["base_state"][:, 7:10]
    current_eef_ang_vel = states["base_state"][:, 10:13]
    target_eef_vel = target_states["wrist_vel"]
    target_eef_ang_vel = target_states["wrist_ang_vel"]

    diff_eef_vel = target_eef_vel - current_eef_vel
    diff_eef_ang_vel = target_eef_ang_vel - current_eef_ang_vel

    joints_pos = states["hand_joints_state"][:, 1:, :3]
    target_joints_pos = target_states["joints_pos"]
    diff_joints_pos = target_joints_pos - joints_pos
    diff_joints_pos_dist = torch.norm(diff_joints_pos, dim=-1)

    # assign different weights to different joints
    diff_thumb_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["thumb_tip"]]].mean(dim=-1)
    diff_index_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["index_tip"]]].mean(dim=-1)
    diff_middle_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["middle_tip"]]].mean(dim=-1)
    diff_ring_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["ring_tip"]]].mean(dim=-1)
    diff_pinky_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["pinky_tip"]]].mean(dim=-1)
    diff_level_1_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["level_1_joints"]]].mean(dim=-1)
    diff_level_2_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["level_2_joints"]]].mean(dim=-1)

    joints_vel = states["hand_joints_state"][:, 1:, 7:10]
    target_joints_vel = target_states["joints_vel"]
    diff_joints_vel = target_joints_vel - joints_vel

    # ---- Relative (wrist-frame) hand tracking ----
    # Transform hand body positions into wrist local frame to decouple hand shape from wrist error.
    # relative_pos = R_wrist^{-1} @ (body_pos - wrist_pos)
    num_envs = current_eef_pos.shape[0]
    device = current_eef_pos.device
    zeros_like_dist = torch.zeros(num_envs, device=device)

    if use_rel_hand_tracking:
        # Current hand bodies in current wrist frame
        cur_wrist_q_inv = quat_conjugate(current_eef_quat)  # [N, 4]
        cur_bodies_centered = joints_pos - current_eef_pos.unsqueeze(1)  # [N, J, 3]
        # Rotate each body vector: for each env, rotate J vectors by the same quat
        # Using quaternion rotation: v' = v + 2*w*(q_xyz x v) + 2*(q_xyz x (q_xyz x v))
        nJ = cur_bodies_centered.shape[1]
        q_flat = cur_wrist_q_inv.unsqueeze(1).expand(num_envs, nJ, 4).reshape(num_envs * nJ, 4)
        v_flat = cur_bodies_centered.reshape(num_envs * nJ, 3)
        # quat_rotate inline: q * (0,v) * q_inv
        q_w = q_flat[:, 0:1]
        q_vec = q_flat[:, 1:4]
        t = 2.0 * torch.cross(q_vec, v_flat, dim=-1)
        cur_rel_flat = v_flat + q_w * t + torch.cross(q_vec, t, dim=-1)
        cur_rel = cur_rel_flat.reshape(num_envs, nJ, 3)

        # Target hand bodies in target wrist frame
        tgt_wrist_q_inv = quat_conjugate(target_eef_quat)
        tgt_bodies_centered = target_joints_pos - target_eef_pos.unsqueeze(1)
        q_flat_t = tgt_wrist_q_inv.unsqueeze(1).expand(num_envs, nJ, 4).reshape(num_envs * nJ, 4)
        v_flat_t = tgt_bodies_centered.reshape(num_envs * nJ, 3)
        q_w_t = q_flat_t[:, 0:1]
        q_vec_t = q_flat_t[:, 1:4]
        t_t = 2.0 * torch.cross(q_vec_t, v_flat_t, dim=-1)
        tgt_rel_flat = v_flat_t + q_w_t * t_t + torch.cross(q_vec_t, t_t, dim=-1)
        tgt_rel = tgt_rel_flat.reshape(num_envs, nJ, 3)

        diff_rel = tgt_rel - cur_rel
        diff_rel_dist = torch.norm(diff_rel, dim=-1)  # [N, J]

        rel_thumb_tip_dist = diff_rel_dist[:, [k - 1 for k in dexhand_weight_idx["thumb_tip"]]].mean(dim=-1)
        rel_index_tip_dist = diff_rel_dist[:, [k - 1 for k in dexhand_weight_idx["index_tip"]]].mean(dim=-1)
        rel_middle_tip_dist = diff_rel_dist[:, [k - 1 for k in dexhand_weight_idx["middle_tip"]]].mean(dim=-1)
        rel_ring_tip_dist = diff_rel_dist[:, [k - 1 for k in dexhand_weight_idx["ring_tip"]]].mean(dim=-1)
        rel_pinky_tip_dist = diff_rel_dist[:, [k - 1 for k in dexhand_weight_idx["pinky_tip"]]].mean(dim=-1)
        rel_level_1_dist = diff_rel_dist[:, [k - 1 for k in dexhand_weight_idx["level_1_joints"]]].mean(dim=-1)
        rel_level_2_dist = diff_rel_dist[:, [k - 1 for k in dexhand_weight_idx["level_2_joints"]]].mean(dim=-1)
    else:
        rel_thumb_tip_dist = zeros_like_dist
        rel_index_tip_dist = zeros_like_dist
        rel_middle_tip_dist = zeros_like_dist
        rel_ring_tip_dist = zeros_like_dist
        rel_pinky_tip_dist = zeros_like_dist
        rel_level_1_dist = zeros_like_dist
        rel_level_2_dist = zeros_like_dist

    # ---- Compute rewards ----
    # Relative hand tracking rewards
    reward_rel_thumb_tip = torch.exp(-100 * rel_thumb_tip_dist)
    reward_rel_index_tip = torch.exp(-90 * rel_index_tip_dist)
    reward_rel_middle_tip = torch.exp(-80 * rel_middle_tip_dist)
    reward_rel_pinky_tip = torch.exp(-60 * rel_pinky_tip_dist)
    reward_rel_ring_tip = torch.exp(-60 * rel_ring_tip_dist)
    reward_rel_level_1 = torch.exp(-50 * rel_level_1_dist)
    reward_rel_level_2 = torch.exp(-40 * rel_level_2_dist)

    # Multi-scale EEF position reward: sharp (near) + medium + wide (far)
    # At dist=0.12m: exp(-40*0.12)=0.008, exp(-20*0.12)=0.091, exp(-5*0.12)=0.549
    reward_eef_pos = (torch.exp(-40 * diff_eef_pos_dist)
                      + 0.25 * torch.exp(-20 * diff_eef_pos_dist)
                      + 0.5 * torch.exp(-5 * diff_eef_pos_dist))  # wide component: strong signal at 5-20cm
    # Absolute hand tracking rewards
    reward_thumb_tip_pos = torch.exp(-100 * diff_thumb_tip_pos_dist)
    reward_index_tip_pos = torch.exp(-90 * diff_index_tip_pos_dist)
    reward_middle_tip_pos = torch.exp(-80 * diff_middle_tip_pos_dist)
    reward_pinky_tip_pos = torch.exp(-60 * diff_pinky_tip_pos_dist)
    reward_ring_tip_pos = torch.exp(-60 * diff_ring_tip_pos_dist)
    reward_level_1_pos = torch.exp(-50 * diff_level_1_pos_dist)
    reward_level_2_pos = torch.exp(-40 * diff_level_2_pos_dist)

    reward_eef_vel = torch.exp(-1 * diff_eef_vel.abs().mean(dim=-1))
    reward_eef_ang_vel = torch.exp(-1 * diff_eef_ang_vel.abs().mean(dim=-1))
    reward_joints_vel = torch.exp(-1 * diff_joints_vel.abs().mean(dim=-1).mean(-1))

    current_dof_vel = states["dq"]

    diff_eef_rot = quat_mul(target_eef_quat, quat_conjugate(current_eef_quat))
    diff_eef_rot_angle = quat_to_angle_axis(diff_eef_rot)[0]
    reward_eef_rot = torch.exp(-1 * (diff_eef_rot_angle).abs())

    reward_power = torch.exp(-0.01 * target_states["power"])
    reward_wrist_power = torch.exp(-2 * target_states["wrist_power"])
    reward_action_rate_l1 = torch.exp(-0.05 * states["action_rate_l1"])
    reward_action_rate_l2 = torch.exp(-0.05 * states["action_rate_l2"]) # -0.01

    penal_action_rate_l2 = states["action_rate_l2"]

    # Arm action rate (separate from total)
    reward_arm_action_rate_l2 = torch.exp(-0.1 * states["arm_action_rate_l2"])
    penal_arm_action_rate_l2 = states["arm_action_rate_l2"]

    
    # Object pose reward
    current_obj_pos = states["manip_obj_pos"]
    current_obj_quat = states["manip_obj_quat"]
    target_obj_pos = target_states["manip_obj_pos"]
    target_obj_quat = target_states["manip_obj_quat"]
    diff_obj_pos = target_obj_pos - current_obj_pos
    diff_obj_pos_dist = torch.norm(diff_obj_pos, dim=-1)
    reward_obj_pos = torch.exp(-80 * diff_obj_pos_dist) # this is small TODO
    # print(f"debug: current obj pos: {current_obj_pos}, target obj pos: {target_obj_pos}")s
    
    diff_obj_rot = quat_mul(target_obj_quat, quat_conjugate(current_obj_quat))
    diff_obj_rot_angle = quat_to_angle_axis(diff_obj_rot)[0]
    # exp(-4·|Δrot|) penalizes ≥15° rotation errors strongly: the reward is
    # ≈0.35 at 15° and ≈0.12 at 30°.
    reward_obj_rot = torch.exp(-4 * (diff_obj_rot_angle).abs())
    
    current_obj_vel = states["manip_obj_vel"]
    target_obj_vel = target_states["manip_obj_vel"]
    diff_obj_vel = target_obj_vel - current_obj_vel
    reward_obj_vel = torch.exp(-1 * diff_obj_vel.abs().mean(dim=-1))
    
    current_obj_ang_vel = states["manip_obj_ang_vel"]
    target_obj_ang_vel = target_states["manip_obj_ang_vel"]
    diff_obj_ang_vel = target_obj_ang_vel - current_obj_ang_vel
    reward_obj_ang_vel = torch.exp(-1 * diff_obj_ang_vel.abs().mean(dim=-1))
    
    # Finger tip force reward
    finger_tip_force = target_states["tip_force"] # last contact force in sim
    finger_tip_distance = target_states["tips_distance"]
    contact_range = [0.005, 0.015] # TODO check value [0.02, 0.03]
    finger_tip_weight = torch.clamp(
        (contact_range[1] - finger_tip_distance) / (contact_range[1] - contact_range[0]), 0, 1
    )
    # finger_tip_force_masked = finger_tip_force * finger_tip_weight[:, None]
    finger_tip_force_masked = finger_tip_force * finger_tip_weight[:, :, None]
    
    reward_finger_tip_force = torch.exp(-1 * (1 / (torch.norm(finger_tip_force_masked, dim=-1).sum(-1) + 1e-5)))

    # Approach reward: guide hand toward object when far away (fills reward gap before contact)
    min_tip_distance = finger_tip_distance.min(dim=-1).values  # closest fingertip to object
    if approach_shaping_v2:
        # Dense rational shaping with non-vanishing gradient at all distances.
        # 1/(1+5d): d=0→1.0, d=0.1→0.67, d=0.5→0.29, d=1.0→0.17.
        # Always active (no mask): policy gets useful gradient even when 50cm+ away.
        reward_approach = 1.0 / (1.0 + 5.0 * min_tip_distance)
    else:
        approach_mask = (min_tip_distance > contact_range[1]).float()  # only when far (> 0.015m)
        reward_approach = torch.exp(-5.0 * min_tip_distance) * approach_mask

    # ---- No-slip reward (contact stability) ----
    # When fingertips are in contact with the object, penalize relative velocity
    # between fingertip and object surface. Targets `fail/obj_pos_drift` directly:
    # slip ≡ object moving inside the grasp ≡ trajectory goes off-demo ≡ fail.
    # Per-finger contact weight ramps in over [contact_range[1], contact_range[0]]
    # so the term is smooth (matches finger_tip_force masking).
    # joints_vel: shape [B, J-1, 3] (lin vel for hand body 1+).
    # Each tip's lin vel = joints_vel at the tip body index from dexhand_weight_idx.
    thumb_tip_vel  = joints_vel[:, [k - 1 for k in dexhand_weight_idx["thumb_tip"]]].mean(dim=1)
    index_tip_vel  = joints_vel[:, [k - 1 for k in dexhand_weight_idx["index_tip"]]].mean(dim=1)
    middle_tip_vel = joints_vel[:, [k - 1 for k in dexhand_weight_idx["middle_tip"]]].mean(dim=1)
    ring_tip_vel   = joints_vel[:, [k - 1 for k in dexhand_weight_idx["ring_tip"]]].mean(dim=1)
    pinky_tip_vel  = joints_vel[:, [k - 1 for k in dexhand_weight_idx["pinky_tip"]]].mean(dim=1)
    fingertip_vels = torch.stack(
        [thumb_tip_vel, index_tip_vel, middle_tip_vel, ring_tip_vel, pinky_tip_vel], dim=1
    )  # [B, 5, 3]
    # Broadcast [B, 1, 3] against [B, 5, 3] (avoid .expand(-1, ...) which is JIT-fussy).
    diff_vel = fingertip_vels - states["manip_obj_vel"].unsqueeze(1)
    slip_per_finger = torch.norm(diff_vel, dim=-1)  # [B, 5]
    # Use the same contact ramp that finger_tip_force uses, so the term only
    # activates for fingers that are actually (or near-actually) in contact.
    slip_weighted = slip_per_finger * finger_tip_weight  # [B, 5]
    slip_total = slip_weighted.sum(dim=-1)               # [B]
    reward_no_slip = torch.exp(-2.0 * slip_total)

    # Arm joint velocity and acceleration penalties (to reduce jitter)
    # Penalize high arm joint velocities
    arm_joint_vel_penalty = torch.norm(states["arm_joint_vel"], dim=-1).mean(dim=-1)
    # Penalize high arm joint accelerations (velocity changes)
    arm_joint_acc_penalty = torch.norm(states["arm_joint_acc"], dim=-1).mean(dim=-1)
    
    # # Wrist velocity and acceleration penalties (to reduce jitter)
    # # Penalize high wrist linear velocities
    wrist_lin_vel_penalty = torch.norm(states["wrist_lin_vel"], dim=-1)
    # Penalize high wrist angular velocities
    wrist_ang_vel_penalty = torch.norm(states["wrist_ang_vel"], dim=-1)
    # # Penalize high wrist linear accelerations (velocity changes)
    wrist_lin_acc_penalty = torch.norm(states["wrist_lin_acc"], dim=-1)
    # Penalize high wrist angular accelerations (velocity changes)
    wrist_ang_acc_penalty = torch.norm(states["wrist_ang_acc"], dim=-1)
    
    # Convert penalties to rewards (negative exponential)
    reward_arm_joint_vel = torch.exp(-0.1 * arm_joint_vel_penalty)
    reward_arm_joint_acc = torch.exp(-0.5 * arm_joint_acc_penalty)
    reward_wrist_lin_vel = torch.exp(-1 * wrist_lin_vel_penalty)
    reward_wrist_ang_vel = torch.exp(-0.5 * wrist_ang_vel_penalty)
    reward_wrist_lin_acc = torch.exp(-0.01 * wrist_lin_acc_penalty)
    reward_wrist_ang_acc = torch.exp(-0.01 * wrist_ang_acc_penalty)
    # print(f"DEBUG: wrist_lin_acc_penalty: {wrist_lin_acc_penalty}, reward_wrist_lin_acc: {reward_wrist_lin_acc}, wrist_ang_acc_penalty: {wrist_ang_acc_penalty}, reward_wrist_ang_acc: {reward_wrist_ang_acc}")
    # print(f"DEBUG: wrist_lin_vel_penalty: {wrist_lin_vel_penalty}, reward_wrist_lin_vel: {reward_wrist_lin_vel}, wrist_ang_vel_penalty: {wrist_ang_vel_penalty}, reward_wrist_ang_vel: {reward_wrist_ang_vel}")
    survival_reward_scale = 1.0  # Scale factor for survival reward
    survival_reward = survival_reward_scale * (running_progress_buf.float() / max_length.clamp(min=1.0))
    
    # Arm collision detection with table (table_surface_z defaults to
    # dexx.deploy_config.TABLE_SURFACE_Z).
    # Hardcoded collision threshold: 5cm
    arm_collision_threshold = 0.05
    
    # Check if any arm body height is below table surface + threshold
    num_envs = states["wrist_height"].shape[0]
    device = states["wrist_height"].device
    arm_collision_penalty = torch.zeros(num_envs, device=device, dtype=torch.float32)
    arm_collision_failed = torch.zeros(num_envs, device=device, dtype=torch.bool)
    
    arm_body_positions = states["arm_body_positions"]  # [num_envs, num_arm_bodies, 3]
    num_arm_bodies = arm_body_positions.shape[1]
    # Get minimum Z coordinate (height) of all arm bodies for each environment
    # If no arm bodies, skip collision check (penalty remains zero)
    if num_arm_bodies > 0:
        arm_body_heights = arm_body_positions[:, :, 2]  # [num_envs, num_arm_bodies]
        min_arm_body_height = arm_body_heights.min(dim=1)[0]  # [num_envs] - minimum height per env
        
        # Check if minimum arm body height is below table surface + threshold
        table_threshold_z = table_surface_z + arm_collision_threshold
        arm_below_table = min_arm_body_height < table_threshold_z
        
        # Penalty: negative reward proportional to how far below the threshold
        # Use exponential penalty: more negative the further below
        height_diff = table_threshold_z - min_arm_body_height  # Positive when below threshold
        arm_collision_penalty = torch.where(
            arm_below_table,
            -10.0 * torch.exp(10.0 * height_diff),  # Exponential penalty
            torch.zeros_like(arm_collision_penalty)
        )
        
        # Terminate if arm body is below threshold
        arm_collision_failed = arm_below_table
    
    # Wrist height reward: encourage higher wrist positions
    # Reward increases with wrist height (exponential reward)
    wrist_height_reward = torch.exp(1.0 * states["wrist_height"])  # [num_envs]
    # Normalize to reasonable range (e.g., reward for height around 0.3-0.5m)
    wrist_height_reward = wrist_height_reward / torch.exp(torch.tensor(2.0 * 0.4, device=wrist_height_reward.device))

    error_buf = (
        (torch.norm(current_eef_vel, dim=-1) > 10)
        | (torch.norm(current_eef_ang_vel, dim=-1) > 20)
        | (torch.norm(joints_vel, dim=-1).mean(-1) > 10)
        | (torch.abs(current_dof_vel).mean(-1) > 20)
    ) 
    # Add object velocity checks
    current_obj_vel = states["manip_obj_vel"]
    current_obj_ang_vel = states["manip_obj_ang_vel"]
    error_buf = error_buf | (
        (torch.norm(current_obj_vel, dim=-1) > 10)
        | (torch.norm(current_obj_ang_vel, dim=-1) > 20)
    )

    # ---- Failure conditions: use abs or rel distances depending on flags ----
    # Pick whichever tracking mode is active for failure thresholds.
    # When both are active, use the minimum (more lenient) of the two distances.
    if use_abs_hand_tracking and use_rel_hand_tracking:
        fail_thumb = torch.minimum(diff_thumb_tip_pos_dist, rel_thumb_tip_dist)
        fail_index = torch.minimum(diff_index_tip_pos_dist, rel_index_tip_dist)
        fail_middle = torch.minimum(diff_middle_tip_pos_dist, rel_middle_tip_dist)
        fail_pinky = torch.minimum(diff_pinky_tip_pos_dist, rel_pinky_tip_dist)
        fail_ring = torch.minimum(diff_ring_tip_pos_dist, rel_ring_tip_dist)
        fail_l1 = torch.minimum(diff_level_1_pos_dist, rel_level_1_dist)
        fail_l2 = torch.minimum(diff_level_2_pos_dist, rel_level_2_dist)
    elif use_rel_hand_tracking:
        fail_thumb = rel_thumb_tip_dist
        fail_index = rel_index_tip_dist
        fail_middle = rel_middle_tip_dist
        fail_pinky = rel_pinky_tip_dist
        fail_ring = rel_ring_tip_dist
        fail_l1 = rel_level_1_dist
        fail_l2 = rel_level_2_dist
    else:
        fail_thumb = diff_thumb_tip_pos_dist
        fail_index = diff_index_tip_pos_dist
        fail_middle = diff_middle_tip_pos_dist
        fail_pinky = diff_pinky_tip_pos_dist
        fail_ring = diff_ring_tip_pos_dist
        fail_l1 = diff_level_1_pos_dist
        fail_l2 = diff_level_2_pos_dist

    # Per-cause failure masks (logged to reward_dict for wandb breakdown).
    # A single env may trigger multiple causes — sum of fail/* fractions
    # can exceed fail/any. Use this to see WHICH threshold is killing eps.
    fail_hand_tracking = (
        (
            (fail_thumb > 0.4 / 0.6 * scale_factor)
            | (fail_index > 0.45 / 0.6 * scale_factor)
            | (fail_middle > 0.5 / 0.6 * scale_factor)
            | (fail_pinky > 0.6 / 0.6 * scale_factor)
            | (fail_ring > 0.6 / 0.6 * scale_factor)
            | (fail_l1 > 0.7 / 0.6 * scale_factor)
            | (fail_l2 > 0.8 / 0.6 * scale_factor)
        )
        & (running_progress_buf >= 20)
    )

    diff_obj_pos_dist = torch.norm(target_states["manip_obj_pos"] - states["manip_obj_pos"], dim=-1)
    diff_obj_rot_angle = quat_to_angle_axis(
        quat_mul(target_states["manip_obj_quat"], quat_conjugate(states["manip_obj_quat"]))
    )[0]
    # Trajectory-tracking failure thresholds:
    #   pos: |obj − demo_current| > 8cm
    #   rot: |Δrot|                > ~123° (30/0.243 * scale^3)
    fail_obj_pos = (diff_obj_pos_dist > 0.03 / 0.243 * scale_factor**3) & (running_progress_buf >= 20)
    fail_obj_rot = (
        (diff_obj_rot_angle.abs() / 3.141592653589793 * 180 > 30 / 0.243 * scale_factor**3)
        & (running_progress_buf >= 30)
    )
    # if premature_contact_enabled:
    #     fail_premature_contact = (
    #         torch.any((finger_tip_distance < premature_contact_dist_threshold) & ~(target_states["tip_contact_state"].any(1)), dim=-1)
    #         & (running_progress_buf >= premature_contact_progress_threshold)
    #     )
    # else:
    #    
    # 
    fail_premature_contact = torch.zeros_like(running_progress_buf, dtype=torch.bool)

    failed_execute = (
        fail_hand_tracking
        | error_buf
        | fail_obj_pos
        | fail_obj_rot
        | fail_premature_contact
        | arm_collision_failed
    )

    # Eval infra: when set, suppress ALL terminations so policy rolls to
    # episode_length_buf >= max_length. Per-cause fail/* metrics are still
    # logged in reward_dict for diagnostic breakdown. Use for offline eval +
    # debug only — DO NOT enable during training.
    if eval_no_terminate:
        failed_execute = torch.zeros_like(failed_execute)

    # ---- Final-frame "success" rewards ----------------------------------
    # `final_obj_pos / final_obj_quat / final_K_obj_pos` come from
    # `_get_rewards`, computed once per step from `obj_trajectory[:, seq_len-1]`
    # (and the last K frames). Two reward terms:
    #   reward_final_pos      — exp(-α·||cur − final||)            full credit at endpoint
    #   reward_final_rot      — exp(-α·|Δrot|)                     orientation at endpoint
    #   reward_final_approach — exp(-α·min_k ||cur − last_K[k]||)  partial credit near endpoint
    # Weights come from cfg.success_*_weight.
    # Optional time-ramp: if success_reward_ramp=True, scale by ramp that's
    # 0 before max_length-K and 1 at max_length. This prevents the policy
    # from "skipping ahead" to dump the object early.
    has_final = (
        "final_obj_pos" in target_states
        and "final_obj_quat" in target_states
        and "final_K_obj_pos" in target_states
    )
    # We compute the reward terms whenever final-frame targets are available
    # so the diag/ metrics show up in wandb even when weights are still 0.
    if has_final:
        dist_to_final = torch.norm(
            target_states["final_obj_pos"] - states["manip_obj_pos"], dim=-1
        )  # [B]
        reward_final_pos = torch.exp(-success_alpha_pos * dist_to_final)

        diff_final_rot = quat_mul(
            target_states["final_obj_quat"], quat_conjugate(states["manip_obj_quat"])
        )
        final_rot_angle = quat_to_angle_axis(diff_final_rot)[0]
        reward_final_rot = torch.exp(-success_alpha_rot * final_rot_angle.abs())

        # Min distance to any of last K target positions
        # final_K_obj_pos: [B, K, 3]; cur_obj_pos: [B, 3]
        cur_obj_pos = states["manip_obj_pos"]
        dist_to_last_K = torch.norm(
            target_states["final_K_obj_pos"] - cur_obj_pos[:, None], dim=-1
        )  # [B, K]
        min_dist_to_last_K = dist_to_last_K.min(dim=-1).values  # [B]
        reward_final_approach = torch.exp(-success_alpha_pos * min_dist_to_last_K)

        if success_reward_ramp:
            # Ramp from 0 at (max_length - K) to 1 at max_length.
            ramp = (
                (progress_buf.float() - (max_length - float(success_reward_window))) / float(success_reward_window)
            ).clamp(0.0, 1.0)
            reward_final_pos = reward_final_pos * ramp
            reward_final_rot = reward_final_rot * ramp
            reward_final_approach = reward_final_approach * ramp
    else:
        reward_final_pos = torch.zeros_like(progress_buf, dtype=torch.float)
        reward_final_rot = torch.zeros_like(progress_buf, dtype=torch.float)
        reward_final_approach = torch.zeros_like(progress_buf, dtype=torch.float)
        dist_to_final = torch.zeros_like(progress_buf, dtype=torch.float)
        final_rot_angle = torch.zeros_like(progress_buf, dtype=torch.float)
        min_dist_to_last_K = torch.zeros_like(progress_buf, dtype=torch.float)
    # ---------------------------------------------------------------------

    # ---- Compose reward_execute with gated tracking terms ----
    # Wrist tracking (gated by use_wrist_tracking)
    wrist_w = 1.0 if use_wrist_tracking else 0.0
    # Absolute hand tracking (gated by use_abs_hand_tracking)
    abs_w = 2.0 if use_abs_hand_tracking else 0.0
    # Relative hand tracking (gated by use_rel_hand_tracking)
    rel_w = 1.0 if use_rel_hand_tracking else 0.0

    reward_execute = (
        # Wrist tracking
        wrist_w * 4.0 * reward_eef_pos
        + wrist_w * 2.0 * reward_eef_rot
        + wrist_w * 0.1 * reward_eef_vel
        + wrist_w * 0.05 * reward_eef_ang_vel
        # Absolute hand body tracking
        + abs_w * 0.9 * reward_thumb_tip_pos
        + abs_w * 0.8 * reward_index_tip_pos
        + abs_w * 0.75 * reward_middle_tip_pos
        + abs_w * 0.6 * reward_pinky_tip_pos
        + abs_w * 0.6 * reward_ring_tip_pos
        # Mid/base finger joints are weighted so the policy reproduces the
        # demo's finger SHAPE inside the grasp, not just the tip contact points.
        + abs_w * 0.7 * reward_level_1_pos
        + abs_w * 0.5 * reward_level_2_pos
        # Relative (wrist-frame) hand body tracking
        + rel_w * 0.9 * reward_rel_thumb_tip
        + rel_w * 0.8 * reward_rel_index_tip
        + rel_w * 0.75 * reward_rel_middle_tip
        + rel_w * 0.6 * reward_rel_pinky_tip
        + rel_w * 0.6 * reward_rel_ring_tip
        + rel_w * 0.7 * reward_rel_level_1
        + rel_w * 0.5 * reward_rel_level_2
        # Object tracking (always on)
        + 8.0 * reward_obj_pos
        # obj_rot weight 6.0 so rotation tracking carries weight comparable to obj_pos for rotate/spin demos.
        + 6.0 * reward_obj_rot
        + 0.1 * reward_joints_vel
        + 0.1 * reward_obj_vel
        # obj_ang_vel weight 0.4 so the policy learns to MATCH demo angular velocity, not just final orientation.
        + 0.4 * reward_obj_ang_vel
        + 3.0 * reward_finger_tip_force
        + 2.0 * reward_approach
        + no_slip_weight * reward_no_slip
        + 0.1 * reward_action_rate_l2
        - arm_action_rate_penalty * penal_arm_action_rate_l2
        + arm_collision_penalty
        # Final-frame "success" shaping (weights from cfg.success_*_weight).
        + success_pos_weight * reward_final_pos
        + success_rot_weight * reward_final_rot
        + success_approach_weight * reward_final_approach
    )

    succeeded = (
        (progress_buf + 1 + 3).float() >= max_length
    ) & ~failed_execute  # reached the end of the trajectory, +3 for max future 3 steps
    reset_buf = torch.where(
        succeeded | failed_execute,
        torch.ones_like(reset_buf),
        reset_buf,
    )
    # ---- Conservative training strict proxy --------------------------------
    # `succeeded` above only means "reached the end of the trajectory without a
    # failure termination" — survival, not task success. strict additionally
    # requires the object to finish near its demo endpoint, with no object-
    # position drift. Bad inits score zero and remain in PPO's denominator.
    # Eval strict3 excludes bad inits and does not require `succeeded`, so this
    # training diagnostic must not be reported as the evaluation protocol.
    if has_final:
        succeeded_strict = (
            succeeded
            & (dist_to_final < strict_success_dist)
            & ~fail_obj_pos
            & (running_progress_buf > 5)
        )
        strict_available = torch.ones_like(succeeded.float())
    else:
        # No final-frame target: strict is not computable. Report zero rather
        # than silently falling back to `succeeded` — a flat zero line is a
        # visible anomaly, a loose number wearing a strict label is not.
        succeeded_strict = torch.zeros_like(succeeded)
        strict_available = torch.zeros_like(succeeded.float())

    reward_dict = {
        "reward_eef_pos": reward_eef_pos,
        "reward_eef_rot": reward_eef_rot,
        "reward_eef_vel": reward_eef_vel,
        "reward_eef_ang_vel": reward_eef_ang_vel,
        "reward_joints_vel": reward_joints_vel,
        # Absolute hand tracking aggregate
        "reward_joints_pos": (
            reward_thumb_tip_pos
            + reward_index_tip_pos
            + reward_middle_tip_pos
            + reward_pinky_tip_pos
            + reward_ring_tip_pos
            + reward_level_1_pos
            + reward_level_2_pos
        ),
        "reward_level_1_pos": reward_level_1_pos,
        "reward_level_2_pos": reward_level_2_pos,
        "reward_thumb_tip_pos": reward_thumb_tip_pos,
        "reward_index_tip_pos": reward_index_tip_pos,
        "reward_middle_tip_pos": reward_middle_tip_pos,
        "reward_pinky_tip_pos": reward_pinky_tip_pos,
        "reward_ring_tip_pos": reward_ring_tip_pos,
        # Relative hand tracking aggregate
        "reward_rel_joints_pos": (
            reward_rel_thumb_tip
            + reward_rel_index_tip
            + reward_rel_middle_tip
            + reward_rel_pinky_tip
            + reward_rel_ring_tip
            + reward_rel_level_1
            + reward_rel_level_2
        ),
        "reward_rel_thumb_tip": reward_rel_thumb_tip,
        "reward_rel_index_tip": reward_rel_index_tip,
        "reward_rel_middle_tip": reward_rel_middle_tip,
        "reward_rel_pinky_tip": reward_rel_pinky_tip,
        "reward_rel_ring_tip": reward_rel_ring_tip,
        "reward_rel_level_1": reward_rel_level_1,
        "reward_rel_level_2": reward_rel_level_2,
        "reward_power": reward_power,
        "reward_wrist_power": reward_wrist_power,
        "reward_obj_pos": reward_obj_pos,
        "reward_obj_rot": reward_obj_rot,
        "reward_obj_vel": reward_obj_vel,
        "reward_obj_ang_vel": reward_obj_ang_vel,
        "reward_finger_tip_force": reward_finger_tip_force,
        "reward_action_rate_l1": reward_action_rate_l1,
        "reward_action_rate_l2": reward_action_rate_l2,
        "penal_action_rate_l2": -0.1 * penal_action_rate_l2,
        "reward_arm_action_rate_l2": reward_arm_action_rate_l2,
        "penal_arm_action_rate_l2": -arm_action_rate_penalty * penal_arm_action_rate_l2,
        "reward_arm_joint_vel": reward_arm_joint_vel,
        "reward_arm_joint_acc": reward_arm_joint_acc,
        "reward_wrist_lin_vel": reward_wrist_lin_vel,
        "reward_wrist_ang_vel": reward_wrist_ang_vel,
        "reward_wrist_lin_acc": reward_wrist_lin_acc,
        "reward_wrist_ang_acc": reward_wrist_ang_acc,
        "survival_reward": survival_reward,
        "arm_collision_penalty": arm_collision_penalty,
        "reward_wrist_height": wrist_height_reward,
        "reward_approach": reward_approach,
        "reward_no_slip": reward_no_slip,
        # Final-frame "success" rewards (raw, before weight; weight applied in
        # reward_execute composition above)
        "reward_final_pos": reward_final_pos,
        "reward_final_rot": reward_final_rot,
        "reward_final_approach": reward_final_approach,
        "diag/final_pos_dist": dist_to_final,
        "diag/final_rot_angle": final_rot_angle.abs(),
        "diag/final_K_min_dist": min_dist_to_last_K,
        "diag/slip_total": slip_total,
        # ---- Diagnostics (raw distances, not transformed by exp) ----
        "diag/eef_pos_dist": diff_eef_pos_dist,
        "diag/eef_rot_angle": diff_eef_rot_angle,
        "diag/obj_pos_dist": diff_obj_pos_dist,
        "diag/obj_rot_angle": diff_obj_rot_angle.abs(),
        "diag/min_tip_distance": min_tip_distance,
        "diag/rel_thumb_tip_dist": rel_thumb_tip_dist,
        "diag/rel_index_tip_dist": rel_index_tip_dist,
        "diag/arm_action_norm": torch.norm(actions[:, :7], dim=-1) if actions.shape[-1] >= 7 else torch.zeros(actions.shape[0], device=actions.device),
        "diag/episode_length": running_progress_buf.float(),
        # ---- Termination cause breakdown (each is per-env 0/1 indicator) ----
        # Mean over batch in extras gives the fraction-of-envs hitting each cause
        # this step. A reset that's caused by multiple conditions counts in all
        # of them (so sum of fail/* > fail/any when overlaps exist). The
        # `_progress_at_fail` is `running_progress_buf` masked by failure so
        # mean tells you "where in the trajectory failures concentrate".
        "fail/any": failed_execute.float(),
        "fail/hand_tracking": fail_hand_tracking.float(),
        "fail/obj_pos_drift": fail_obj_pos.float(),
        "fail/obj_rot_drift": fail_obj_rot.float(),
        "fail/premature_contact": fail_premature_contact.float(),
        "fail/arm_below_table": arm_collision_failed.float(),
        "fail/error_buf_velocity_explosion": error_buf.float(),
        "fail/progress_at_fail": (running_progress_buf.float() * failed_execute.float()),
        "succ/any": succeeded.float(),
        "succ/strict": succeeded_strict.float(),
        "succ/strict_available": strict_available,
    }

    return reward_execute, reset_buf, succeeded, failed_execute, reward_dict

    

@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower

@torch.jit.script
def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector(s) v about the rotation described by quaternion(s) q.

    Args:
        q: Quaternion(s) in (w, x, y, z). Shape (..., 4).
        v: Vector(s). Shape (..., 3).

    Returns:
        Rotated vector(s). Shape (..., 3).
    """
    # make v into pure quaternion (0, v)
    zeros = torch.zeros_like(v[..., :1])
    v_as_quat = torch.cat([zeros, v], dim=-1)  # (..., 4)
    # rotate: q * v * q^-1
    v_rot = quat_mul(quat_mul(q, v_as_quat), quat_inv(q))
    return v_rot[..., 1:]  # drop scalar part


@torch.jit.script
def transform_between_frames(p_A: torch.Tensor, q_A: torch.Tensor,
                             q_B: torch.Tensor) -> torch.Tensor:
    """Transform a point from frame A to frame B (rotation only).

    Args:
        p_A: Point(s) in frame A, shape (..., 3).
        q_A: Quaternion of frame A in world, shape (..., 4).
        q_B: Quaternion of frame B in world, shape (..., 4).

    Returns:
        Point(s) in frame B, shape (..., 3).
    """
    # p in world frame
    p_world = quat_rotate(q_A, p_A)
    # p in B frame
    p_B = quat_rotate(quat_inv(q_B), p_world)
    return p_B

@torch.jit.script
def rotmat_to_quat(rotmat: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to quaternion (w, x, y, z).
    
    Args:
        rotmat: Rotation matrix, shape (..., 3, 3)
    
    Returns:
        Quaternion in (w, x, y, z) format, shape (..., 4)
    """
    # Extract matrix elements
    m00, m01, m02 = rotmat[..., 0, 0], rotmat[..., 0, 1], rotmat[..., 0, 2]
    m10, m11, m12 = rotmat[..., 1, 0], rotmat[..., 1, 1], rotmat[..., 1, 2]
    m20, m21, m22 = rotmat[..., 2, 0], rotmat[..., 2, 1], rotmat[..., 2, 2]
    
    # Compute quaternion components
    trace = m00 + m11 + m22
    
    cond = trace > 0
    s = torch.sqrt(trace + 1.0) * 2  # s = 4 * qw
    qw = 0.25 * s
    qx = (m21 - m12) / s
    qy = (m02 - m20) / s
    qz = (m10 - m01) / s
    
    cond2 = (m00 > m11) & (m00 > m22)
    s2 = torch.sqrt(1.0 + m00 - m11 - m22) * 2  # s = 4 * qx
    qw2 = (m21 - m12) / s2
    qx2 = 0.25 * s2
    qy2 = (m01 + m10) / s2
    qz2 = (m02 + m20) / s2
    
    cond3 = m11 > m22
    s3 = torch.sqrt(1.0 + m11 - m00 - m22) * 2  # s = 4 * qy
    qw3 = (m02 - m20) / s3
    qx3 = (m01 + m10) / s3
    qy3 = 0.25 * s3
    qz3 = (m12 + m21) / s3
    
    s4 = torch.sqrt(1.0 + m22 - m00 - m11) * 2  # s = 4 * qz
    qw4 = (m10 - m01) / s4
    qx4 = (m02 + m20) / s4
    qy4 = (m12 + m21) / s4
    qz4 = 0.25 * s4
    
    # Select based on conditions
    qw = torch.where(cond, qw, torch.where(cond2, qw2, torch.where(cond3, qw3, qw4)))
    qx = torch.where(cond, qx, torch.where(cond2, qx2, torch.where(cond3, qx3, qx4)))
    qy = torch.where(cond, qy, torch.where(cond2, qy2, torch.where(cond3, qy3, qy4)))
    qz = torch.where(cond, qz, torch.where(cond2, qz2, torch.where(cond3, qz3, qz4)))
    
    return torch.stack([qw, qx, qy, qz], dim=-1)
