# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Critic-horizon variant of FrankaSharpaForceEnv.

Actor obs (550d in both asymmetric and non-asymmetric paths) adds the
deployable signals
    + target_obj_pos[next]        (3)
    + target_obj_quat[next]       (4)
    + tips_distance[next]         (5)   ← privileged in the parent under asymmetric_ac
    + obj_bps (object shape)      (128) ← privileged in the parent under asymmetric_ac
All four are available at deploy time (demo pkl / static mesh encoding).

Critic priv_info (148d, for K=5) packs, in addition to the existing 40d
(dq, friction, mass, com, current object state):
    + target_obj_pos over K        (3K)
    + target_obj_quat over K       (4K)
    + target_obj_vel over K        (3K)
    + target_obj_ang_vel over K    (3K)
    + tips_distance over K         (5K)
    + delta_obj current frame      (13)   pos 3 + quat 4 + vel 3 + ang_vel 3
    + obj_to_fingertips            (5)

Kept completely separate from `FrankaSharpaForceEnv`.
"""
from __future__ import annotations

import torch
import gymnasium as gym

from isaaclab.envs.utils.spaces import spec_to_gym_space
from isaaclab.utils.math import quat_conjugate, quat_mul

from dexx.tasks.hand_imitation.dataset.transform import aa_to_quat

from .franka_sharpa_env import rotmat_to_quat, transform_between_frames
from .franka_sharpa_force_env import FrankaSharpaForceEnv
from .franka_sharpa_critic_horizon_cfg import FrankaSharpaCriticHorizonCfg


class FrankaSharpaForceCriticHorizonEnv(FrankaSharpaForceEnv):
    cfg: FrankaSharpaCriticHorizonCfg

    def __init__(self, cfg: FrankaSharpaCriticHorizonCfg, render_mode: str | None = None, **kwargs):
        # Let the parent (ForceEnv) init run to full completion. Parent will
        # force cfg.observation_space=543 and then (if asymmetric_ac) reduce
        # it to 410 — we undo + extend that here.
        super().__init__(cfg, render_mode, **kwargs)

        _asymmetric_ac = getattr(self.cfg, 'asymmetric_ac', False)
        _enable_bps = getattr(self.cfg, 'enable_bps', True)
        bps_dim = self.obj_bps.shape[-1] if (self.obj_bps is not None and _enable_bps) else 0

        # Additions to actor obs vs parent (parent obs-dim already reflects enable_bps):
        #   +7   for target_obj_pos (3) + target_obj_quat (4)
        #   +5   for tips_distance  (asymmetric path only)
        #   +bps_dim for obj_bps    (asymmetric path only, if enabled)
        #
        # Expected obs_dim matrix (for obj_bps=128):
        #   asymmetric=T, enable_bps=T: parent=410, add 12+128 = 550
        #   asymmetric=T, enable_bps=F: parent=410, add 12+0   = 422
        #   asymmetric=F, enable_bps=T: parent=543, add 7      = 550
        #   asymmetric=F, enable_bps=F: parent=415, add 7      = 422
        if _asymmetric_ac:
            new_obs_dim = 410 + 12 + bps_dim
        else:
            # parent already reduced for enable_bps=False → use its current value
            new_obs_dim = self.cfg.observation_space + 7

        # Sanity-check vs cfg class attribute to catch drift early.
        # cfg.observation_space was set to 550 in the cfg class but has been
        # mutated by parent __init__; we authoritatively reset it here.
        self.cfg.observation_space = new_obs_dim
        self.num_obs = new_obs_dim
        self.single_observation_space["policy"] = spec_to_gym_space(new_obs_dim)
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space["policy"], self.num_envs
        )

        # Rebuild history buffers to match the new actor-obs dim.
        self.proprio_hist_dim = new_obs_dim // 3
        self.obs_buf_lag_history = torch.zeros(
            (self.num_envs, max(80, self.cfg.prop_hist_len + 10), self.proprio_hist_dim),
            device=self.device, dtype=torch.float,
        )
        self.proprio_hist_buf = torch.zeros(
            (self.num_envs, self.cfg.prop_hist_len, self.proprio_hist_dim),
            device=self.device, dtype=torch.float,
        )

        # priv_info_buf was already sized by base env init using cfg.priv_info_dim
        # (148 for K=5). No need to re-allocate here.
        K = int(self.cfg.critic_future_length)
        expected_priv = 40 + 18 * K + 18
        assert self.priv_info_buf.shape[-1] == self.cfg.priv_info_dim == expected_priv, (
            f"priv_info_dim mismatch: buf={self.priv_info_buf.shape[-1]}, "
            f"cfg={self.cfg.priv_info_dim}, expected={expected_priv} (K={K})"
        )

        print(
            f"[CriticHorizonEnv] asymmetric_ac={_asymmetric_ac}, "
            f"obs_dim={new_obs_dim}, priv_info_dim={self.cfg.priv_info_dim}, K={K}, "
            f"actor_masks(extra={getattr(self.cfg, 'actor_mask_extra_geometry', False)}, "
            f"obj={getattr(self.cfg, 'actor_mask_target_obj_pose', False)}, "
            f"tips={getattr(self.cfg, 'actor_mask_tips_distance', False)}, "
            f"bps={getattr(self.cfg, 'actor_mask_bps', False)}, "
            f"hand={getattr(self.cfg, 'actor_mask_target_hand', False)})"
        )

    # ------------------------------------------------------------------
    # compute_observations: forked from FrankaSharpaForceEnv with 3 changes
    #   (A) append target_obj_pos/quat (next frame) to target_obs_list
    #   (B) always append tips_distance (next frame) to target_obs_list
    #       (do not treat as privileged in this variant)
    #   (C) populate extended priv_info_buf with K-frame future + delta + obj_to_tips
    # ------------------------------------------------------------------
    def compute_observations(self):
        self._refresh_lab()

        # ---- Proprioception (unchanged) ----
        q = self.hand_dof_pos
        proprioception_obs = torch.cat([
            q,
            torch.cos(q),
            torch.sin(q),
            torch.cat([
                torch.zeros_like(self.base_pos),
                self.base_quat,
                self.base_lin_vel,
                self.base_ang_vel,
            ], dim=-1),
        ], dim=-1)

        # ---- Future indices (actor uses obs_future_length, critic uses K) ----
        obs_future_length = self.obs_future_length
        seq_len = self.demo_data["seq_len"]
        if self.loop_trajectory:
            actor_future_indices = torch.stack(
                [(self._get_demo_idx() + 1 + t) % seq_len for t in range(obs_future_length)], dim=-1
            )
            K = int(self.cfg.critic_future_length)
            critic_future_indices = torch.stack(
                [(self._get_demo_idx() + 1 + t) % seq_len for t in range(K)], dim=-1
            )
        else:
            actor_future_indices = torch.stack(
                [torch.clamp(self.progress_buf + 1 + t, torch.zeros_like(seq_len), seq_len - 1)
                 for t in range(obs_future_length)], dim=-1
            )
            K = int(self.cfg.critic_future_length)
            # Clamp each future step to valid range.
            critic_future_indices = torch.stack(
                [torch.clamp(self.progress_buf + 1 + t, torch.zeros_like(seq_len), seq_len - 1)
                 for t in range(K)], dim=-1
            )
        nE, nT = self.demo_data["target_wrist_pos"].shape[:2]
        nF = obs_future_length

        def indicing(data, idx):
            assert data.shape[0] == nE and data.shape[1] == nT
            remaining_shape = data.shape[2:]
            expanded_idx = idx
            for _ in remaining_shape:
                expanded_idx = expanded_idx.unsqueeze(-1)
            expanded_idx = expanded_idx.expand(-1, -1, *remaining_shape)
            return torch.gather(data, 1, expanded_idx)

        # ---- Wrist target: use resolved reference source (retarget/MANO) ----
        target_wrist_pos = indicing(self.demo_data["target_wrist_pos"], actor_future_indices)
        cur_wrist_pos = self.base_pos
        delta_wrist_pos = (target_wrist_pos - cur_wrist_pos[:, None]).reshape(nE, -1)

        target_wrist_vel = indicing(self.demo_data["target_wrist_velocity"], actor_future_indices)
        cur_wrist_vel = self.base_lin_vel
        wrist_vel = target_wrist_vel.reshape(nE, -1)
        delta_wrist_vel = (target_wrist_vel - cur_wrist_vel[:, None]).reshape(nE, -1)

        target_wrist_rot_raw = indicing(self.demo_data["target_wrist_rot"], actor_future_indices)
        if target_wrist_rot_raw.ndim > 3:
            target_wrist_rot_raw = target_wrist_rot_raw[:, :, 0, :]
        target_wrist_quat = aa_to_quat(target_wrist_rot_raw.reshape(nE * nF, -1))
        delta_wrist_quat = quat_mul(
            self.base_quat[:, None].repeat(1, nF, 1).reshape(nE * nF, -1),
            quat_conjugate(target_wrist_quat),
        ).reshape(nE, -1)
        wrist_quat = target_wrist_quat.reshape(nE, -1)

        target_wrist_ang_vel_raw = indicing(self.demo_data["target_wrist_angular_velocity"], actor_future_indices)
        if target_wrist_ang_vel_raw.ndim > 3:
            target_wrist_ang_vel_raw = target_wrist_ang_vel_raw[:, :, 0, :]
        wrist_ang_vel = target_wrist_ang_vel_raw.reshape(nE, -1)
        delta_wrist_ang_vel = (target_wrist_ang_vel_raw - self.base_ang_vel[:, None]).reshape(nE, -1)

        # ---- Hand joints target: use resolved reference source (retarget/MANO) ----
        target_joints_pos = indicing(self.demo_data["target_joints_pos"], actor_future_indices).reshape(nE, nF, -1, 3)
        cur_joint_pos = self.hand.data.body_pos_w[:, self.hand_body_indices[1:]] - self.scene.env_origins.unsqueeze(1)
        delta_joints_pos = (target_joints_pos - cur_joint_pos[:, None]).reshape(self.num_envs, -1)

        target_joints_vel = indicing(self.demo_data["target_joints_velocity"], actor_future_indices).reshape(nE, nF, -1, 3)
        cur_joint_vel = self.hand.data.body_lin_vel_w[:, self.hand_body_indices[1:]]
        joints_vel = target_joints_vel.reshape(self.num_envs, -1)
        delta_joints_vel = (target_joints_vel - cur_joint_vel[:, None]).reshape(self.num_envs, -1)

        if getattr(self.cfg, "actor_mask_target_hand", False):
            delta_joints_pos = torch.zeros_like(delta_joints_pos)
            joints_vel = torch.zeros_like(joints_vel)
            delta_joints_vel = torch.zeros_like(delta_joints_vel)

        target_obs_list = [
            delta_wrist_pos, wrist_vel, delta_wrist_vel,
            wrist_quat, delta_wrist_quat, wrist_ang_vel, delta_wrist_ang_vel,
            delta_joints_pos, joints_vel, delta_joints_vel,
        ]
        # Named slot map over the assembled actor obs. Recorded from the tensors
        # actually concatenated, never from arithmetic on a doc comment, so a
        # change to obs_future_length or to any block's width cannot silently
        # shift a consumer's indices. Consumed by the student's
        # `--student_drop_slots`, which names the channels it drops.
        _slots = [("proprioception", int(proprioception_obs.shape[-1])),
                  ("ref_tracking", int(sum(t.shape[-1] for t in target_obs_list)))]

        _asymmetric_ac = getattr(self.cfg, 'asymmetric_ac', False)
        has_object = hasattr(self, 'object') and self.object is not None

        # ---- (A) target_obj_pos / target_obj_quat next-frame — deployable ----
        if has_object:
            target_obj_transf_actor = indicing(self.demo_data["obj_trajectory"], actor_future_indices)  # [B,K,4,4]
            target_obj_pos_actor = target_obj_transf_actor[:, :, :3, 3].reshape(nE, -1)  # [B, K*3]
            target_obj_quat_actor = rotmat_to_quat(
                target_obj_transf_actor.reshape(-1, 4, 4)[:, :3, :3]
            ).reshape(nE, -1)  # [B, K*4]
            mask_extra_geom = getattr(self.cfg, 'actor_mask_extra_geometry', False)
            if mask_extra_geom or getattr(self.cfg, 'actor_mask_target_obj_pose', False):
                target_obj_pos_actor = torch.zeros_like(target_obj_pos_actor)
                target_obj_quat_actor = torch.zeros_like(target_obj_quat_actor)
            target_obs_list.append(target_obj_pos_actor)
            target_obs_list.append(target_obj_quat_actor)
            _slots.append(("target_obj_pose",
                           int(target_obj_pos_actor.shape[-1] + target_obj_quat_actor.shape[-1])))

            # ---- (B) tips_distance as deployable actor obs (demo-precomputed) ----
            gt_tips_distance = indicing(self.demo_data["tips_distance"], actor_future_indices).reshape(nE, -1)
            if mask_extra_geom or getattr(self.cfg, 'actor_mask_tips_distance', False):
                gt_tips_distance = torch.zeros_like(gt_tips_distance)
            target_obs_list.append(gt_tips_distance)
            _slots.append(("tips_distance", int(gt_tips_distance.shape[-1])))

        # ---- BPS appended to actor obs in both asymmetric and non-asymmetric paths,
        # gated by cfg.enable_bps (default True). Mesh-based static encoding computed
        # in _build_data from obj_verts, available at deploy via the same code path.
        if self.obj_bps is not None and getattr(self.cfg, 'enable_bps', True):
            obj_bps_actor = self.obj_bps
            if getattr(self.cfg, 'actor_mask_extra_geometry', False) or getattr(self.cfg, 'actor_mask_bps', False):
                obj_bps_actor = torch.zeros_like(obj_bps_actor)
            target_obs_list.append(obj_bps_actor)
            _slots.append(("obj_bps", int(obj_bps_actor.shape[-1])))

        # ---- Contact forces (unchanged from parent) ----
        net_contact_forces_history = torch.cat([
            self._contact_sensor[id].data.net_forces_w_history[:, :, 0, :].unsqueeze(2)
            for id in self._contact_body_ids
        ], dim=2)
        norm_contact_forces_history = torch.norm(net_contact_forces_history, dim=-1)
        smooth_contact_forces = (
            norm_contact_forces_history[:, 0, :] * self.cfg.contact_smooth
            + norm_contact_forces_history[:, 1, :] * (1 - self.cfg.contact_smooth)
        )
        smooth_contact_forces[:, self._contact_body_ids_disable] = 0.0

        latency_samples = torch.rand_like(self.last_contacts)
        latency = torch.where(latency_samples < self.cfg.contact_latency, 1.0, 0.0)
        self.last_contacts = self.last_contacts * latency + smooth_contact_forces * (1 - latency)
        sensed_contacts = self.last_contacts.clone().reshape(nE, -1)

        # 3D world-frame copy for vec3 PointCloud consumer.
        smooth_vec_w = (
            net_contact_forces_history[:, 0] * self.cfg.contact_smooth
            + net_contact_forces_history[:, 1] * (1 - self.cfg.contact_smooth)
        )
        smooth_vec_w[:, self._contact_body_ids_disable, :] = 0.0
        if not hasattr(self, "last_contacts_vec_w"):
            self.last_contacts_vec_w = torch.zeros_like(smooth_vec_w)
        latency_vec = latency.unsqueeze(-1)
        self.last_contacts_vec_w = self.last_contacts_vec_w * latency_vec + smooth_vec_w * (1 - latency_vec)

        if getattr(self.cfg, 'binary_contact', False):
            threshold = getattr(self.cfg, 'contact_threshold', 0.2)
            sensed_contacts = torch.where(sensed_contacts > threshold,
                                          torch.ones_like(sensed_contacts),
                                          torch.zeros_like(sensed_contacts))

        # ---- Contact positions in tactile frame (unchanged from parent) ----
        tactile_frame_pose = self.hand.data.body_link_state_w[:, self.elastomer_ids, :7]
        tactile_frame_pos = tactile_frame_pose[..., :3]
        tactile_frame_quat = tactile_frame_pose[..., 3:7]
        world_quat = torch.zeros_like(tactile_frame_quat)
        world_quat[..., 0] = 1.0

        contact_pos = torch.cat([
            self._contact_sensor[id].data.contact_pos_w[:, 0, 0, :].unsqueeze(1)
            for id in self._contact_body_ids
        ], dim=1)
        contact_pos = torch.nan_to_num(contact_pos, nan=0.0)

        not_contact_mask = sensed_contacts < 1.0e-6
        not_contact_mask[:, self._contact_body_ids_disable] = True
        contact_mask = ~not_contact_mask

        contact_pos[contact_mask, :] = transform_between_frames(
            contact_pos[contact_mask, :] - tactile_frame_pos[contact_mask, :],
            world_quat[contact_mask, :],
            tactile_frame_quat[contact_mask, :],
        )
        contact_pos[not_contact_mask, :] = 0.0
        contact_pos = contact_pos.reshape(self.num_envs, -1)

        if hasattr(self.cfg, 'enable_contact_pos') and not self.cfg.enable_contact_pos:
            contact_pos[:] = 0.0
        if hasattr(self.cfg, 'enable_contact_force') and not self.cfg.enable_contact_force:
            sensed_contacts[:] = 0.0
        if hasattr(self.cfg, 'enable_tactile') and not self.cfg.enable_tactile:
            contact_pos[:] = 0.0
            sensed_contacts[:] = 0.0

        # Tactile domain randomization (unchanged)
        if not getattr(self, '_is_deploy_env', False):
            force_noise_std = getattr(self.cfg, 'contact_force_noise', 0.0)
            if force_noise_std > 0:
                force_noise = 1.0 + force_noise_std * torch.randn_like(sensed_contacts)
                sensed_contacts = sensed_contacts * force_noise
                sensed_contacts = torch.clamp(sensed_contacts, min=0.0)

            pos_noise_std = getattr(self.cfg, 'contact_pos_noise', 0.0)
            if pos_noise_std > 0:
                pos_noise = pos_noise_std * torch.randn_like(contact_pos)
                contact_mask_flat = (sensed_contacts.repeat_interleave(3, dim=-1) > 1e-6)
                contact_pos = contact_pos + pos_noise * contact_mask_flat.float()

            dropout_prob = getattr(self.cfg, 'contact_dropout_prob', 0.0)
            if dropout_prob > 0:
                dropout_mask = (torch.rand(nE, 5, device=sensed_contacts.device) > dropout_prob).float()
                sensed_contacts = sensed_contacts * dropout_mask
                dropout_mask_pos = dropout_mask.repeat_interleave(3, dim=-1)
                contact_pos = contact_pos * dropout_mask_pos

        # Keep the tactile tail synchronized after flags / noise / dropout:
        # [..., sensed_contacts(5), contact_pos(15)].
        target_obs_list.append(sensed_contacts)
        target_obs_list.append(contact_pos)
        _slots.append(("tactile", int(sensed_contacts.shape[-1] + contact_pos.shape[-1])))

        # ---- Assemble final obs_buf ----
        target_obs = torch.cat(target_obs_list, dim=-1)
        obs_buf = torch.cat([proprioception_obs, target_obs], dim=-1)

        if getattr(self, "actor_obs_slots", None) is None:
            slots, off = {}, 0
            for _name, _w in _slots:
                slots[_name] = (off, off + _w)
                off += _w
            # The head block is `proprioception_obs`, which is cat'd BEFORE
            # target_obs — the recorded order already reflects that.
            if off != int(obs_buf.shape[-1]):
                raise RuntimeError(
                    f"actor obs slot map covers {off} dims but obs_buf has "
                    f"{int(obs_buf.shape[-1])}; a block was appended without "
                    f"being recorded in `_slots`.")
            self.actor_obs_slots = slots
            print(f"[obs-slots] {self.__class__.__name__} actor obs "
                  f"{int(obs_buf.shape[-1])}d: "
                  + "  ".join(f"{k}[{a}:{b}]" for k, (a, b) in slots.items()), flush=True)

        # History for ProprioAdapt (unchanged structure)
        obs_part_for_hist = obs_buf[:, :self.proprio_hist_dim]
        prev_obs_buf = self.obs_buf_lag_history[:, 1:].clone()
        cur_obs_buf = obs_part_for_hist.unsqueeze(1)
        self.obs_buf_lag_history[:] = torch.cat([prev_obs_buf, cur_obs_buf], dim=1)
        if self.cfg.prop_hist_len > 0:
            self.proprio_hist_buf[:] = self.obs_buf_lag_history[:, -self.cfg.prop_hist_len:].clone()

        at_reset_env_ids = self.at_reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(at_reset_env_ids) > 0:
            reset_obs = obs_part_for_hist[at_reset_env_ids]
            self.obs_buf_lag_history[at_reset_env_ids] = reset_obs.unsqueeze(1).repeat(
                1, self.obs_buf_lag_history.shape[1], 1)
            self.proprio_hist_buf[at_reset_env_ids] = reset_obs.unsqueeze(1).repeat(
                1, self.cfg.prop_hist_len, 1)

        # ---- (C) Critic-only priv_info: existing 40d + K-frame future + delta + obj_to_tips ----
        # Reset to zero to guarantee no stale values survive across reset boundaries
        # or across disabled-flag toggles.
        self.priv_info_buf.zero_()
        self.priv_info_buf[:, :22] = self.hand_dof_vel
        # Slots 22/23/24:27 are populated elsewhere (friction, mass, com) during reset;
        # they remain zero here, matching parent behavior when no randomization applies.
        if has_object:
            self.priv_info_buf[:, 27:30] = self.object_pos
            self.priv_info_buf[:, 30:34] = self.object_rot
            self.priv_info_buf[:, 34:40] = self.object_velocities

        if has_object:
            off = 40
            K = int(self.cfg.critic_future_length)

            if self.cfg.critic_enable_target_obj_future:
                target_obj_transf_K = indicing(self.demo_data["obj_trajectory"], critic_future_indices)
                target_obj_pos_K = target_obj_transf_K[:, :, :3, 3].reshape(nE, -1)             # K*3
                target_obj_quat_K = rotmat_to_quat(
                    target_obj_transf_K.reshape(-1, 4, 4)[:, :3, :3]
                ).reshape(nE, -1)                                                                # K*4
                target_obj_vel_K = indicing(self.demo_data["obj_velocity"], critic_future_indices).reshape(nE, -1)   # K*3
                target_obj_angv_K = indicing(self.demo_data["obj_angular_velocity"], critic_future_indices).reshape(nE, -1)  # K*3
                self.priv_info_buf[:, off:off + 3 * K] = target_obj_pos_K;   off += 3 * K
                self.priv_info_buf[:, off:off + 4 * K] = target_obj_quat_K;  off += 4 * K
                self.priv_info_buf[:, off:off + 3 * K] = target_obj_vel_K;   off += 3 * K
                self.priv_info_buf[:, off:off + 3 * K] = target_obj_angv_K;  off += 3 * K
            else:
                off += 13 * K  # reserve slots even if disabled

            if self.cfg.critic_enable_tips_distance_future:
                tips_dist_K = indicing(self.demo_data["tips_distance"], critic_future_indices).reshape(nE, -1)  # K*5
                self.priv_info_buf[:, off:off + 5 * K] = tips_dist_K
            off += 5 * K

            if self.cfg.critic_enable_delta_obj_current:
                # First critic frame target vs current object state
                target_obj_transf_0 = target_obj_transf_K[:, 0] if self.cfg.critic_enable_target_obj_future \
                    else indicing(self.demo_data["obj_trajectory"], critic_future_indices)[:, 0]
                target_obj_pos_0 = target_obj_transf_0[:, :3, 3]                                 # 3
                target_obj_quat_0 = rotmat_to_quat(target_obj_transf_0[:, :3, :3])               # 4
                target_obj_vel_0 = indicing(self.demo_data["obj_velocity"], critic_future_indices)[:, 0]
                target_obj_angv_0 = indicing(self.demo_data["obj_angular_velocity"], critic_future_indices)[:, 0]

                delta_obj_pos = target_obj_pos_0 - self.object_pos
                delta_obj_quat = quat_mul(self.object_rot, quat_conjugate(target_obj_quat_0))
                delta_obj_vel = target_obj_vel_0 - self.object_velocities[:, :3]
                delta_obj_angv = target_obj_angv_0 - self.object_velocities[:, 3:]
                self.priv_info_buf[:, off:off + 3] = delta_obj_pos;   sub = 3
                self.priv_info_buf[:, off + sub:off + sub + 4] = delta_obj_quat; sub += 4
                self.priv_info_buf[:, off + sub:off + sub + 3] = delta_obj_vel;  sub += 3
                self.priv_info_buf[:, off + sub:off + sub + 3] = delta_obj_angv
            off += 13

            if self.cfg.critic_enable_obj_to_tips_current:
                # Distance from current object position to 5 fingertips.
                # self.fingertip_pos (nE, 5, 3) is set in _refresh_lab (already env-local).
                obj_to_tips = torch.norm(self.object_pos[:, None] - self.fingertip_pos, dim=-1)  # (nE, 5)
                self.priv_info_buf[:, off:off + 5] = obj_to_tips
            off += 5

        # One-shot obs dim sanity print
        if not hasattr(self, '_obs_dim_checked_ch'):
            print(
                f"[CriticHorizonEnv] obs_buf={obs_buf.shape[-1]} (cfg={self.cfg.observation_space}), "
                f"priv_info={self.priv_info_buf.shape[-1]} (cfg={self.cfg.priv_info_dim})"
            )
            self._obs_dim_checked_ch = True

        return obs_buf
