# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
import carb
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .franka_sharpa_env_cfg import FrankaSharpaEnvCfg

from .franka_sharpa_env import FrankaSharpaEnv, rotmat_to_quat, transform_between_frames
from dexx.tasks.hand_imitation.dataset.transform import aa_to_quat
from isaaclab.utils.math import quat_conjugate, quat_mul


class FrankaSharpaForceEnv(FrankaSharpaEnv):
    """FrankaSharpaEnv with smooth contact forces for fingertip force calculation.
    
    This environment inherits from FrankaSharpaEnv but uses smooth_contact_forces
    instead of raw net_forces_w for 
    fingertip force calculation in reward computation.
    """
    cfg: "FrankaSharpaEnvCfg"
    def __init__(self, cfg: "FrankaSharpaEnvCfg", render_mode: str | None = None, **kwargs):
        # cfg.observation_space = 598
        cfg.observation_space = 543
        super().__init__(cfg, render_mode, **kwargs)

        # Dynamic obs dim reduction: asymmetric_ac removes tips+BPS as privileged;
        # enable_bps=False removes BPS even under non-asymmetric path.
        _asymmetric_ac = getattr(self.cfg, 'asymmetric_ac', False)
        _enable_bps = getattr(self.cfg, 'enable_bps', True)

        priv_obs_removed = 0
        if _asymmetric_ac:
            # tips_distance: 5 dims × obs_future_length
            if hasattr(self, 'object') and self.object is not None:
                priv_obs_removed += 5 * self.obs_future_length
            # BPS: always removed under asymmetric (treated as privileged)
            if self.obj_bps is not None:
                priv_obs_removed += self.obj_bps.shape[-1]
        else:
            # Non-asymmetric: BPS included by default; subtract if explicitly disabled
            if not _enable_bps and self.obj_bps is not None:
                priv_obs_removed += self.obj_bps.shape[-1]

        if priv_obs_removed > 0:
            new_obs_dim = self.cfg.observation_space - priv_obs_removed
            self.cfg.observation_space = new_obs_dim
            self.num_obs = new_obs_dim
            from isaaclab.envs.utils.spaces import spec_to_gym_space
            import gymnasium as gym
            self.single_observation_space["policy"] = spec_to_gym_space(new_obs_dim)
            self.observation_space = gym.vector.utils.batch_space(
                self.single_observation_space["policy"], self.num_envs
            )
            # Rebuild proprio_hist_dim and history buffers for ProprioAdapt compatibility
            self.proprio_hist_dim = new_obs_dim // 3
            self.obs_buf_lag_history = torch.zeros(
                (self.num_envs, max(80, self.cfg.prop_hist_len + 10), self.proprio_hist_dim),
                device=self.device, dtype=torch.float,
            )
            self.proprio_hist_buf = torch.zeros(
                (self.num_envs, self.cfg.prop_hist_len, self.proprio_hist_dim),
                device=self.device, dtype=torch.float,
            )
            print(f"[ForceEnv] Actor obs reduced: {new_obs_dim + priv_obs_removed} -> {new_obs_dim} "
                  f"(asymmetric_ac={_asymmetric_ac}, enable_bps={_enable_bps}, "
                  f"removed={priv_obs_removed}d, proprio_hist_dim={self.proprio_hist_dim})")

        # Set num_actions attribute required by wrapper
        self.num_actions = self.cfg.action_space
        print(f"[ForceEnv] arm_gravity_compensation={getattr(self.cfg, 'arm_gravity_compensation', 'NOT SET')}, "
              f"_arm_gravity_comp={getattr(self, '_arm_gravity_comp', 'NOT SET')}")

    def _get_rewards(self) -> torch.Tensor:
        """Compute keypoint tracking reward with smooth contact forces."""
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
        # Mirrors the same block in FrankaSharpaEnv._get_rewards. This subclass
        # overrides _get_rewards entirely, so we have to populate the keys here
        # too — otherwise compute_imitation_reward sees missing keys and falls
        # back to all-zero final rewards/diagnostics.
        seq_len_clamped = torch.clamp(self.demo_data["seq_len"], 0, self.max_episode_length)
        final_idx = torch.clamp(seq_len_clamped - 1, min=0).long()  # [N]
        env_idx = torch.arange(self.num_envs, device=self.device)
        obj_traj = self.demo_data["obj_trajectory"]  # [N, T_max, 4, 4]

        final_obj_transf = obj_traj[env_idx, final_idx]                 # [N, 4, 4]
        target_state["final_obj_pos"] = final_obj_transf[:, :3, 3]       # [N, 3]
        target_state["final_obj_quat"] = rotmat_to_quat(final_obj_transf[:, :3, :3])  # [N, 4]

        K = int(getattr(self.cfg, "success_reward_window", 5))
        arange_K = torch.arange(K, device=self.device)
        last_K_idx = (final_idx[:, None] - (K - 1 - arange_K[None, :])).clamp(min=0)  # [N, K]
        last_K_transf = obj_traj[env_idx[:, None], last_K_idx]           # [N, K, 4, 4]
        target_state["final_K_obj_pos"] = last_K_transf[..., :3, 3]     # [N, K, 3]

        # Compute smooth contact forces
        # Get contact force history
        net_contact_forces_history = torch.cat([
            self._contact_sensor[id].data.net_forces_w_history[:, :, 0, :].unsqueeze(2) 
            for id in self._contact_body_ids
        ], dim=2)
        
        # Compute norm of contact forces
        norm_contact_forces_history = torch.norm(net_contact_forces_history, dim=-1)
        
        # Apply smoothing: smooth_contact_forces = alpha * current + (1-alpha) * previous
        smooth_contact_forces = (
            norm_contact_forces_history[:, 0, :] * self.cfg.contact_smooth + 
            norm_contact_forces_history[:, 1, :] * (1 - self.cfg.contact_smooth)
        )
        
        # Disable forces for specified contact body IDs
        if hasattr(self, '_contact_body_ids_disable') and len(self._contact_body_ids_disable) > 0:
            smooth_contact_forces[:, self._contact_body_ids_disable] = 0.0
        
        # Convert smooth_contact_forces (scalar) to tip_force format (vector)
        # We need to get the direction from the current net_forces_w
        current_net_forces = torch.stack([
            self._contact_sensor[body_id].data.net_forces_w.sum(dim=1)
            for body_id in self._contact_body_ids
        ], dim=1)  # [num_envs, num_contact_bodies, 3]
        
        # Normalize to get direction, then scale by smooth_contact_forces magnitude
        current_net_forces_norm = torch.norm(current_net_forces, dim=-1, keepdim=True)  # [num_envs, num_contact_bodies, 1]
        
        # Handle zero forces: if norm is zero, use zero vector; otherwise normalize
        zero_mask = current_net_forces_norm.squeeze(-1) < 1e-8  # [num_envs, num_contact_bodies]
        current_net_forces_norm = torch.clamp(current_net_forces_norm, min=1e-8)
        current_net_forces_dir = current_net_forces / current_net_forces_norm  # [num_envs, num_contact_bodies, 3]
        
        # Set direction to zero where force norm is zero
        current_net_forces_dir[zero_mask] = 0.0
        
        # Scale by smooth_contact_forces magnitude
        smooth_contact_forces_expanded = smooth_contact_forces.unsqueeze(-1)  # [num_envs, num_contact_bodies, 1]
        target_state["tip_force"] = current_net_forces_dir * smooth_contact_forces_expanded  # [num_envs, num_contact_bodies, 3]

        # Update contact history
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
                    progress = min(last_step / gravity_steps, 1.0)
                    current_gravity = gravity_initial + (gravity_final - gravity_initial) * progress
                    gravity_progress = progress
                elif self.cfg.gravity_scheduler_method == "exp":
                    progress = min(last_step / gravity_steps, 1.0)
                    current_gravity = gravity_initial * ((gravity_final / gravity_initial) ** progress)
                    gravity_progress = progress
                elif self.cfg.gravity_scheduler_method == "cos":
                    progress = min(last_step / gravity_steps, 1.0)
                    cos_progress = (1 - math.cos(progress * math.pi)) / 2
                    current_gravity = gravity_initial + (gravity_final - gravity_initial) * cos_progress
                    gravity_progress = progress
                else:
                    current_gravity = gravity_final
                    gravity_progress = 1.0
                
                new_gravity = carb.Float3(0.0, 0.0, -current_gravity)
                self.physics_sim_view.set_gravity(new_gravity)
        
        # Get current states
        current_states = self._get_current_states()
        
        # Compute reward using imitation reward function
        max_length_tensor = max_length.float()
        from .franka_sharpa_env import compute_imitation_reward
        _eval_no_terminate = bool(getattr(self.cfg, 'eval_no_terminate', False))
        self.reward_execute[:], self.reset_buf[:], self.success_buf[:], self.failure_buf[:], reward_dict = compute_imitation_reward(
            self.reset_buf,
            self.progress_buf,
            self.running_progress_buf,
            self.actions,
            current_states,
            target_state,
            max_length_tensor,
            scale_factor,
            self.dexhand_weight_idx,   # by-name resolved (same content as the positional weight_idx)
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
            _eval_no_terminate,
        )

        # Eval-mode mask: prevent early termination but DON'T fake success.
        # `success_buf` from compute_imitation_reward = (progress+4>=max_length)
        # & ~failed_execute. With failure_buf intact, once any failure has
        # fired this episode, success_buf stays False even at episode end.
        # That gives the strict metric "reached end of demo without any
        # failure firing along the way".
        if _eval_no_terminate:
            self.reset_buf[:] = 0   # don't terminate → episode runs to time_out

        # DEBUG: print fail/* sums every 100 steps
        if not hasattr(self, "_dbg_count"):
            self._dbg_count = 0
        self._dbg_count += 1
        if self._dbg_count in (1, 100, 200):
            for k in ("fail/obj_pos_drift", "fail/obj_rot_drift", "fail/error_buf_velocity_explosion", "fail/premature_contact"):
                if k in reward_dict:
                    v = reward_dict[k]
                    print(f"[FORCE_ENV DEBUG step={self._dbg_count}] {k}: shape={tuple(v.shape)}, sum={int((v > 0.5).sum().item())}, mean={float(v.mean().item()):.4f}")

        self.total_rew_buf += self.reward_execute
        self.reward_dict = reward_dict

        # Update extras for logging
        for key, value in reward_dict.items():
            self.extras[key] = value.mean() if isinstance(value, torch.Tensor) else value
        self.extras['total_reward'] = self.reward_execute.mean()
        # Per-ENV vectors. `success_buf` is 1 only on the step an episode ends and
        # is cleared in `_reset_idx`, so a mean over all envs at every step is a
        # near-zero number that is NOT the episode success rate. Consumers select
        # the envs that just terminated — see algo/ppo/ppo.py.
        self.extras['succeeded_per_env'] = self.success_buf.float()
        # Conservative training proxy: also requires survival, and PPO counts
        # bad inits as zero. This is not eval.py's strict3; see docs/TRAINING.md.
        if 'succ/strict' in reward_dict:
            self.extras['succeeded_strict_per_env'] = reward_dict['succ/strict']
        self.extras['failed_per_env'] = self.failure_buf.float()
        # Scalars kept for the existing log keys. Do not read as episode rates.
        self.extras['succeeded'] = self.success_buf.float().mean()
        self.extras['failed_execute'] = self.failure_buf.float().mean()

        # ---- Anti-shake debug diagnostics (env 0 + batch stats) ----
        # These show up in wandb so you can see per-step:
        #   - how jittery the policy's arm action is (commanded jitter)
        #   - whether the arm is actually following or saturating
        #   - which joint is the worst offender
        with torch.no_grad():
            _root = 7 if getattr(self.cfg, 'use_joint_delta_control', False) else (
                9 if getattr(self.cfg, 'use_pid_control', False) else (
                7 if getattr(self.cfg, 'use_joint_pos_control', False) else 6))
            arm_act = self.actions[:, :_root]
            arm_act_diff = self.action_rate[:, :_root]  # actions[t] - actions[t-1]

            # action jitter (high = policy chattering)
            self.extras['shake/arm_act_diff_l2_mean'] = arm_act_diff.norm(dim=-1).mean()
            self.extras['shake/arm_act_diff_l2_max']  = arm_act_diff.norm(dim=-1).max()
            # action saturation
            self.extras['shake/arm_act_abs_max_mean'] = arm_act.abs().max(dim=-1)[0].mean()
            # cross-env action std (high = policy uncertain in similar states)
            self.extras['shake/arm_act_std_across_envs'] = arm_act.std(dim=0).mean()

            # arm tracking error (cur vs commanded des)
            if hasattr(self, 'arm_joint_pos_des'):
                track_err = (self.arm_joint_pos - self.arm_joint_pos_des).abs()
                self.extras['shake/arm_track_err_mean'] = track_err.mean()
                self.extras['shake/arm_track_err_max']  = track_err.max()

            # arm physical motion magnitudes
            self.extras['shake/arm_vel_norm_mean'] = self.arm_joint_vel.norm(dim=-1).mean()
            if hasattr(self, 'arm_joint_acc'):
                self.extras['shake/arm_acc_norm_mean'] = self.arm_joint_acc.norm(dim=-1).mean()

            # Per-joint action diff (find the offending joint)
            for jj in range(_root):
                self.extras[f'shake/arm_act_diff_j{jj}'] = arm_act_diff[:, jj].abs().mean()
        
        # Record curriculum information
        self.extras['curriculum_scale_factor'] = scale_factor
        if current_gravity is not None:
            self.extras['curriculum_gravity'] = current_gravity
        if gravity_progress is not None:
            self.extras['curriculum_gravity_progress'] = gravity_progress
        # Always record current gravity value from simulation
        gravity_vec = self.physics_sim_view.get_gravity()
        self.extras['gravity_z'] = -gravity_vec[2]

        return self.reward_execute

    def compute_observations(self):
        """Compute observations including proprioception, target states, and contact positions."""
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
        if self.loop_trajectory:
            seq_len = self.demo_data["seq_len"]
            cur_idx = (self._get_demo_idx() + 1) % seq_len
            future_indices = torch.stack([(self._get_demo_idx() + 1 + t) % seq_len for t in range(obs_future_length)], dim=-1)
        else:
            cur_idx = self.progress_buf + 1
            cur_idx = torch.clamp(cur_idx, torch.zeros_like(self.demo_data["seq_len"]), self.demo_data["seq_len"] - 1)
            future_indices = torch.stack([cur_idx + t for t in range(obs_future_length)], dim=-1)  # [B, K]
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
        if target_wrist_ang_vel_raw.ndim > 3:
            target_wrist_ang_vel_raw = target_wrist_ang_vel_raw[:, :, 0, :]
        target_wrist_ang_vel = target_wrist_ang_vel_raw
        cur_wrist_ang_vel = self.base_ang_vel
        wrist_ang_vel = target_wrist_ang_vel.reshape(nE, -1)
        delta_wrist_ang_vel = (target_wrist_ang_vel - cur_wrist_ang_vel[:, None]).reshape(nE, -1)

        # Get target joint states
        target_joints_pos = indicing(self.demo_data["target_joints_pos"], future_indices).reshape(nE, nF, -1, 3)

        cur_joint_pos = self.hand.data.body_pos_w[:, self.hand_body_indices[1:]] - self.scene.env_origins.unsqueeze(1)
        delta_joints_pos = (target_joints_pos - cur_joint_pos[:, None]).reshape(self.num_envs, -1)

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
        
        _asymmetric_ac = getattr(self.cfg, 'asymmetric_ac', False)
        if hasattr(self, 'object') and self.object is not None:
            # target_obj_transf = indicing(self.demo_data["obj_trajectory"], future_indices)
            # target_obj_transf = target_obj_transf.reshape(nE * nF, 4, 4)

            # target_obj_pos = target_obj_transf[:, :3, 3].reshape(nE, nF, -1)  # [nE, nF, 3]

            # # Object position delta
            # cur_obj_pos = self.object.data.root_pos_w - self.scene.env_origins
            # print(f"cur_obj_pos: {cur_obj_pos}")
            # delta_manip_obj_pos = (
            #     target_obj_pos - cur_obj_pos[:, None]
            # ).reshape(nE, -1)
            # target_obs_list.append(delta_manip_obj_pos)

            # # Object velocity
            # target_obj_vel = indicing(self.demo_data["obj_velocity"], future_indices)
            # cur_obj_vel = self.object.data.root_lin_vel_w
            # manip_obj_vel = target_obj_vel.reshape(nE, -1)
            # delta_manip_obj_vel = (target_obj_vel - cur_obj_vel[:, None]).reshape(nE, -1)
            # target_obs_list.append(manip_obj_vel)
            # target_obs_list.append(delta_manip_obj_vel)

            # # Object quaternion
            # target_obj_quat = rotmat_to_quat(target_obj_transf[:, :3, :3])
            # cur_obj_quat = self.object.data.root_quat_w
            # delta_manip_obj_quat = quat_mul(
            #     cur_obj_quat[:, None].repeat(1, nF, 1).reshape(nE * nF, -1),
            #     quat_conjugate(target_obj_quat),
            # ).reshape(nE, -1)
            # manip_obj_quat = target_obj_quat.reshape(nE, -1)
            # target_obs_list.append(manip_obj_quat)
            # target_obs_list.append(delta_manip_obj_quat)

            # # Object angular velocity
            # target_obj_ang_vel = indicing(self.demo_data["obj_angular_velocity"], future_indices)
            # cur_obj_ang_vel = self.object.data.root_ang_vel_w
            # manip_obj_ang_vel = target_obj_ang_vel.reshape(nE, -1)
            # delta_manip_obj_ang_vel = (target_obj_ang_vel - cur_obj_ang_vel[:, None]).reshape(nE, -1)
            # target_obs_list.append(manip_obj_ang_vel)
            # target_obs_list.append(delta_manip_obj_ang_vel)

            # # Object to joints distance
            # obj_to_joints = torch.norm(
            #     cur_obj_pos[:, None] - cur_joint_pos, dim=-1
            # ).reshape(self.num_envs, -1)
            # target_obs_list.append(obj_to_joints)

            # Tips distance (privileged: requires GT object pose, skip for asymmetric AC actor)
            if not _asymmetric_ac:
                gt_tips_distance = indicing(self.demo_data["tips_distance"], future_indices).reshape(nE, -1)
                target_obs_list.append(gt_tips_distance)

        # Add BPS features if available.
        # Gated by cfg.enable_bps (default True) — when False, BPS is omitted even
        # in non-asymmetric path (used for ablations / removing shape prior).
        # Under asymmetric_ac=True, BPS is always removed from actor (privileged).
        _include_bps = (self.obj_bps is not None
                        and not _asymmetric_ac
                        and getattr(self.cfg, 'enable_bps', True))
        if _include_bps:
            target_obs_list.append(self.obj_bps)

        # Add contact forces (with smoothing)
        net_contact_forces_history = torch.cat([self._contact_sensor[id].data.net_forces_w_history[:, :, 0, :].unsqueeze(2) for id in self._contact_body_ids], dim=2)
        norm_contact_forces_history = torch.norm(net_contact_forces_history, dim=-1)
        smooth_contact_forces = norm_contact_forces_history[:, 0, :] * self.cfg.contact_smooth + norm_contact_forces_history[:, 1, :] * (1 - self.cfg.contact_smooth)
        smooth_contact_forces[:, self._contact_body_ids_disable] = 0.0

        latency_samples = torch.rand_like(self.last_contacts)
        latency = torch.where(latency_samples < self.cfg.contact_latency, 1.0, 0.0)
        self.last_contacts = self.last_contacts * latency + smooth_contact_forces * (1 - latency)
        sensed_contacts = self.last_contacts.clone().reshape(nE, -1)

        # ALSO maintain a 3D world-frame copy for downstream consumers that want
        # the full vector (e.g. PointCloud env with pc_tactile_use_vec3=True).
        # Same smoothing + latency as the scalar path, applied per-axis.
        smooth_vec_w = (
            net_contact_forces_history[:, 0] * self.cfg.contact_smooth
            + net_contact_forces_history[:, 1] * (1 - self.cfg.contact_smooth)
        )  # (N, F, 3) world
        smooth_vec_w[:, self._contact_body_ids_disable, :] = 0.0
        if not hasattr(self, "last_contacts_vec_w"):
            self.last_contacts_vec_w = torch.zeros_like(smooth_vec_w)
        latency_vec = latency.unsqueeze(-1)  # (N, F, 1)
        self.last_contacts_vec_w = self.last_contacts_vec_w * latency_vec + smooth_vec_w * (1 - latency_vec)

        # Apply binary_contact in training (not just deploy)
        if getattr(self.cfg, 'binary_contact', False):
            threshold = getattr(self.cfg, 'contact_threshold', 0.2)
            sensed_contacts = torch.where(sensed_contacts > threshold,
                                          torch.ones_like(sensed_contacts),
                                          torch.zeros_like(sensed_contacts))

        # Add contact positions
        # Get tactile frame pose (elastomer body poses)
        tactile_frame_pose = self.hand.data.body_link_state_w[:, self.elastomer_ids, :7]
        tactile_frame_pos = tactile_frame_pose[..., :3]
        tactile_frame_quat = tactile_frame_pose[..., 3:7]
        world_quat = torch.zeros_like(tactile_frame_quat)
        world_quat[..., 0] = 1.0
        
        # Get contact positions from sensors
        contact_pos = torch.cat([
            self._contact_sensor[id].data.contact_pos_w[:, 0, 0, :].unsqueeze(1) 
            for id in self._contact_body_ids
        ], dim=1)
        contact_pos = torch.nan_to_num(contact_pos, nan=0.0)
        
        # Determine contact mask based on sensed contacts
        not_contact_mask = sensed_contacts < 1.0e-6
        not_contact_mask[:, self._contact_body_ids_disable] = True
        contact_mask = ~not_contact_mask
        
        # Transform contact positions to tactile frame
        contact_pos[contact_mask, :] = transform_between_frames(
            contact_pos[contact_mask, :] - tactile_frame_pos[contact_mask, :], 
            world_quat[contact_mask, :], 
            tactile_frame_quat[contact_mask, :]
        )
        contact_pos[not_contact_mask, :] = 0.0
        contact_pos = contact_pos.reshape(self.num_envs, -1)
        
        # Apply configuration flags
        if hasattr(self.cfg, 'enable_contact_pos') and not self.cfg.enable_contact_pos:
            contact_pos[:] = 0.0

        if hasattr(self.cfg, 'enable_contact_force') and not self.cfg.enable_contact_force:
            sensed_contacts[:] = 0.0

        if hasattr(self.cfg, 'enable_tactile') and not self.cfg.enable_tactile:
            contact_pos[:] = 0.0
            sensed_contacts[:] = 0.0

        # Contact domain randomization for sim2real (only during training)
        if not getattr(self, '_is_deploy_env', False):
            # 1) Force noise: multiplicative ±20%
            force_noise_std = getattr(self.cfg, 'contact_force_noise', 0.0)
            if force_noise_std > 0:
                force_noise = 1.0 + force_noise_std * torch.randn_like(sensed_contacts)
                sensed_contacts = sensed_contacts * force_noise
                sensed_contacts = torch.clamp(sensed_contacts, min=0.0)

            # 2) Position noise: additive ±3mm
            pos_noise_std = getattr(self.cfg, 'contact_pos_noise', 0.0)
            if pos_noise_std > 0:
                # Only add noise where contact exists
                pos_noise = pos_noise_std * torch.randn_like(contact_pos)
                contact_mask_flat = (sensed_contacts.repeat_interleave(3, dim=-1) > 1e-6)
                contact_pos = contact_pos + pos_noise * contact_mask_flat.float()

            # 3) Dropout: randomly zero out entire finger's tactile signal
            dropout_prob = getattr(self.cfg, 'contact_dropout_prob', 0.0)
            if dropout_prob > 0:
                dropout_mask = (torch.rand(nE, 5, device=sensed_contacts.device) > dropout_prob).float()
                sensed_contacts = sensed_contacts * dropout_mask
                # Also zero contact_pos for dropped fingers
                dropout_mask_pos = dropout_mask.repeat_interleave(3, dim=-1)
                contact_pos = contact_pos * dropout_mask_pos

        # Keep the tactile tail synchronized after flags / noise / dropout:
        # [..., sensed_contacts(5), contact_pos(15)].
        target_obs_list.append(sensed_contacts)
        target_obs_list.append(contact_pos)
        
        # Combine target observations
        target_obs = torch.cat(target_obs_list, dim=-1)
        
        # Combine proprioception and target observations
        obs_buf = torch.cat([proprioception_obs, target_obs], dim=-1)

        # Update observation history buffer for ProprioAdapt
        obs_part_for_hist = obs_buf[:, :self.proprio_hist_dim]  # [num_envs, 194]
        prev_obs_buf = self.obs_buf_lag_history[:, 1:].clone()
        cur_obs_buf = obs_part_for_hist.unsqueeze(1)  # [num_envs, 1, 194]
        self.obs_buf_lag_history[:] = torch.cat([prev_obs_buf, cur_obs_buf], dim=1)
        
        # Extract proprioceptive history for ProprioAdapt
        if self.cfg.prop_hist_len > 0:
            self.proprio_hist_buf[:] = self.obs_buf_lag_history[:, -self.cfg.prop_hist_len:].clone()
        
        # Handle reset: refill history buffer for reset environments
        at_reset_env_ids = self.at_reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(at_reset_env_ids) > 0:
            reset_obs = obs_part_for_hist[at_reset_env_ids]  # [num_reset, 194]
            self.obs_buf_lag_history[at_reset_env_ids] = reset_obs.unsqueeze(1).repeat(1, self.obs_buf_lag_history.shape[1], 1)
            self.proprio_hist_buf[at_reset_env_ids] = reset_obs.unsqueeze(1).repeat(1, self.cfg.prop_hist_len, 1)

        # privileged info
        dq = self.hand_dof_vel
        self.priv_info_buf[:,:22] = dq
        priv_obs_values = [self.object_pos, self.object_rot, self.object_velocities]
        self.priv_info_buf[:, 27:40] = torch.cat(priv_obs_values, dim=-1)

        # One-time observation-dimension check.
        if not hasattr(self, '_obs_dim_checked'):
            actual_dim = obs_buf.shape[-1]
            expected_dim = self.cfg.observation_space
            print(f"[DEBUG] Observation dimension check:")
            print(f"  Actual computed: {actual_dim}")
            print(f"  Expected (cfg): {expected_dim}")
            print(f"  Difference: {actual_dim - expected_dim}")
            self._obs_dim_checked = True
        
        return obs_buf
