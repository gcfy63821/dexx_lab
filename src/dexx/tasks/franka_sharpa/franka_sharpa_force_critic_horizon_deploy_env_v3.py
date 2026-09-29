# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""V3 deploy env: replaces sim.step FK with pytorch_kinematics CPU FK.

Why V3:
  A PhysX sim.step costs ~80ms per env step on real hardware — fixed GPU
  launch / sync overhead that does NOT scale down with num_envs=1 /
  decimation=1 / no-render. V2 needs sim.step only to refresh
  `self.hand.data.body_pos_w`, which the fingertip-tracking obs
  (delta_joints_pos / delta_joints_vel) reads.

What V3 does:
  Builds a pytorch_kinematics chain from the same URDF used in retargeting
  and runs FK on CPU each env step (~1-3 ms for a 29-joint chain at batch=1).
  No sim.step / scene.update / scene.write_data_to_sim. Everything else is
  inherited unchanged from FrankaSharpaForceCriticHorizonDeployEnv (which
  itself extends V2): HandSDKWorker, Polymetis arm client, e-stop,
  emergency hold, debug recording, two-stage reset ramp via FR3 home pose,
  manual reset via 'r' key, etc.

Coordinate frame note:
  pk FK returns body poses in the URDF root frame (fr3_link0). Sim's
  body_pos_w is in the global world frame. The two differ by a constant
  rigid offset (Franka base mount). We capture that offset once on the
  first FK call by comparing pk output against the still-valid sim
  body_pos_w from spawn-time init, then add it on every subsequent call.
  The result is a body_pos_w-equivalent tensor that drops straight into
  the existing obs computation (`body_pos - env_origin`).

Velocities:
  body_lin_vel_w is obtained by finite-difference of consecutive FK pos
  outputs. Same rule for the rest of the deploy env's wallclock dt
  handling — uses control_freq nominal dt at first frame, then real
  elapsed dt clamped against pathological stalls.
"""
from __future__ import annotations

import os
import time
import numpy as np
import torch
import cv2

import pytorch_kinematics as pk

from isaaclab.utils.math import quat_conjugate, quat_mul

from dexx.tasks.hand_imitation.dataset.transform import aa_to_quat

from .franka_sharpa_env import rotmat_to_quat
from .franka_sharpa_force_critic_horizon_deploy_env import (
    FrankaSharpaForceCriticHorizonDeployEnv,
)
from .franka_sharpa_critic_horizon_cfg import FrankaSharpaCriticHorizonCfg

# Staleness limit for the tactile stream, separate from deploy_max_sensor_age_s
# (0.25 s, sized for the 200 Hz arm/hand state streams). Tactile frames are
# polled non-blocking once per env step, and a channel legitimately returns
# None on many polls (the Sharpa stream runs on its own ~130 ms cycle, not in
# lockstep with the 30 Hz loop), so a per-channel gap of a few hundred ms is
# normal. 1.0 s without a single good frame on some channel means that
# channel is dead and its cached force is no longer an observation.
TACTILE_MAX_AGE_S = 1.0


class FrankaSharpaForceCriticHorizonDeployEnvV3(FrankaSharpaForceCriticHorizonDeployEnv):
    """Critic-Horizon deploy env, V3 — pytorch_kinematics FK in lieu of sim.step."""

    cfg: FrankaSharpaCriticHorizonCfg

    def __init__(self, cfg: FrankaSharpaCriticHorizonCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Cache for non-blocking tactile fetch (per-channel last good frame).
        # See get_tactile_info below — V2's blocking poll would dominate
        # the step time.
        self._tactile_force_cache = torch.zeros(5, dtype=torch.float32, device=self.device)
        self._tactile_pos_cache = torch.zeros((5, 3), dtype=torch.float32, device=self.device)
        # Per-finger 3D force vector cache (elastomer-local frame from Sharpa F6).
        # Needed for PointCloud deploy with `cfg.pc_tactile_use_vec3=True` —
        # the scalar `_tactile_force_cache` above drops direction. Updated in
        # `get_tactile_info` alongside the scalar magnitude.
        self._tactile_force_vec3_cache = torch.zeros((5, 3), dtype=torch.float32, device=self.device)
        # Output after flip / disable / scale / EMA — exposed for downstream
        # (PC deploy env reads `self.last_contacts_vec3` shape (1, 5, 3)).
        self.last_contacts_vec3 = torch.zeros((1, 5, 3), dtype=torch.float32, device=self.device)
        self._prev_tactile_force_vec3 = torch.zeros((1, 5, 3), dtype=torch.float32, device=self.device)
        # time.monotonic() of each channel's last good frame, and the start of
        # the window the staleness check measures from (set lazily on the first
        # check of a rollout, cleared on reset). See _sensor_ages.
        self._tactile_last_good_t = [float("-inf")] * 5
        self._tactile_age_t0 = None

        self._build_pk_chain()
        self._build_pk_joint_mapping()
        self._build_pk_body_mapping()

        # FK output buffers (filled each step). Shape mirrors what
        # compute_observations expects of body_pos_w[:, hand_body_indices].
        n_bodies = len(self.hand_body_names)
        self._fk_body_pos_minus_env = torch.zeros(
            (1, n_bodies, 3), device=self.device, dtype=torch.float32
        )
        self._fk_body_lin_vel = torch.zeros_like(self._fk_body_pos_minus_env)
        self._fk_prev_pos_chain = None  # for finite-diff vel
        self._fk_to_world_offset = None  # filled lazily on 1st FK call

        self.get_logger().info(
            f'[V3] pk chain ready: {len(self.pk_joint_names)} joints, '
            f'{n_bodies} hand bodies, dtype=fp32 device=cpu'
        )

    # -----------------------------------------------------------------
    # pk chain setup
    # -----------------------------------------------------------------

    def _build_pk_chain(self):
        side = self.cfg.hand_side
        urdf_filename = (
            "fr3_with_right_sharpa_wave.urdf" if side == "right"
            else "fr3_with_left_sharpa_wave.urdf"
        )
        # Repo root is the cwd convention for this project.
        urdf_path = os.path.join(
            os.getcwd(), "assets", "generated", urdf_filename
        )
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f'[V3] pk chain URDF not found: {urdf_path}')

        # Read as bytes: the generated URDF carries an XML encoding
        # declaration, which lxml rejects when handed a str.
        with open(urdf_path, 'rb') as f:
            urdf_string = f.read()

        chain = pk.build_chain_from_urdf(urdf_string)
        # Keep on CPU. pk FK on a single 29-joint chain at batch=1 is
        # CPU-faster than GPU because per-call CUDA launch overhead is
        # large relative to the actual compute. Only ~5 finger chains × 4
        # links each — trivially small for CPU.
        chain = chain.to(dtype=torch.float32, device='cpu')
        self.pk_chain = chain
        self.pk_joint_names = chain.get_joint_parameter_names()

    def _build_pk_joint_mapping(self):
        # Position in input vector → index into pk's joint name list.
        # We splat arm + hand joint values into the right chain slots
        # each step. Note: arm_joint_names is an INSTANCE attribute set by
        # FrankaSharpaEnv._identify_arm_joints (not a cfg field), and
        # actuated_joint_names IS on cfg.
        self._pk_arm_chain_idx = []
        for jn in self.arm_joint_names:
            if jn not in self.pk_joint_names:
                raise RuntimeError(f'[V3] arm joint {jn} missing from pk chain joints')
            self._pk_arm_chain_idx.append(self.pk_joint_names.index(jn))
        self._pk_hand_chain_idx = []
        for jn in self.cfg.actuated_joint_names:
            if jn not in self.pk_joint_names:
                raise RuntimeError(f'[V3] hand joint {jn} missing from pk chain joints')
            self._pk_hand_chain_idx.append(self.pk_joint_names.index(jn))
        self._pk_arm_chain_idx_t = torch.tensor(self._pk_arm_chain_idx, dtype=torch.long)
        self._pk_hand_chain_idx_t = torch.tensor(self._pk_hand_chain_idx, dtype=torch.long)

    def _build_pk_body_mapping(self):
        # Probe the chain by running FK once to learn its output frame names.
        # All hand_body_names should match.
        zero_q = torch.zeros((1, len(self.pk_joint_names)), dtype=torch.float32)
        ret = self.pk_chain.forward_kinematics(zero_q)
        frame_names = list(ret.keys())
        missing = [n for n in self.hand_body_names if n not in frame_names]
        if missing:
            self.get_logger().warn(
                f'[V3] {len(missing)} hand body names missing from URDF chain; '
                f'will be zero-filled in FK output: {missing}'
            )
        self._pk_hand_body_present = [n in frame_names for n in self.hand_body_names]

    # -----------------------------------------------------------------
    # FK
    # -----------------------------------------------------------------

    def _read_real_arm_hand_joints_cpu(self):
        """Pull latest arm + hand joint readings, return CPU fp32 tensors.
        Arm comes from the Polymetis client; hand from the SDK worker cache."""
        arm_pos = self.arm_client.arm_joint_positions.detach().cpu().to(dtype=torch.float32)
        if arm_pos.shape[0] < 7:
            arm_pos = torch.cat([arm_pos, torch.zeros(7 - arm_pos.shape[0], dtype=torch.float32)])

        hand_angles = getattr(self, '_cached_hand_angles_np', None)
        if hand_angles is None and self._hand_io is not None:
            hand_angles = self._hand_io.cached_angles()
            if hand_angles is not None:
                self._cached_hand_angles_np = hand_angles
        if hand_angles is None:
            # No cached reading yet: read synchronously. A failure propagates —
            # FK on made-up joint angles would feed the policy a fake hand.
            hand_angles = np.array(self.real_hand.get_states().angles, dtype=np.float32)
            self._cached_hand_angles_np = hand_angles
        hand_pos = torch.from_numpy(np.asarray(hand_angles, dtype=np.float32))
        return arm_pos, hand_pos

    def _compute_fk(self):
        """Run pk FK on current real-arm + real-hand readings, populate
        self._fk_body_pos_minus_env / self._fk_body_lin_vel."""
        arm_pos_cpu, hand_pos_cpu = self._read_real_arm_hand_joints_cpu()

        full_q = torch.zeros((1, len(self.pk_joint_names)), dtype=torch.float32)
        full_q[0, self._pk_arm_chain_idx_t] = arm_pos_cpu
        full_q[0, self._pk_hand_chain_idx_t] = hand_pos_cpu

        fk = self.pk_chain.forward_kinematics(full_q)

        n_bodies = len(self.hand_body_names)
        pos_chain = torch.zeros((1, n_bodies, 3), dtype=torch.float32)
        for i, name in enumerate(self.hand_body_names):
            if not self._pk_hand_body_present[i]:
                continue
            T = fk[name].get_matrix()  # (1, 4, 4) in fr3_link0 frame
            pos_chain[0, i, :] = T[0, :3, 3]

        # Lazily learn the chain → world offset on the first call. At this
        # point the sim still has spawn-init body_pos_w which is valid; we
        # use its hand-body slice to anchor pk to world. After this we never
        # touch sim FK again.
        if self._fk_to_world_offset is None:
            try:
                sim_pos_w_minus_env = (
                    self.hand.data.body_pos_w[0, self.hand_body_indices]
                    - self.scene.env_origins[0]
                ).detach().cpu()
                offset = sim_pos_w_minus_env - pos_chain[0]
                # If sim wasn't spawned at fr3_link0 = env_origin, offset is
                # roughly constant across all bodies; small per-body variation
                # comes from numerical noise. Average it for a robust estimate.
                self._fk_to_world_offset = offset.mean(dim=0)
                self.get_logger().info(
                    f'[V3] FK→world offset captured (mean across bodies): '
                    f'{self._fk_to_world_offset.tolist()}'
                )
            except Exception as e:
                # No fallback: a zero offset would shift every fingertip
                # observation by the base mount and the policy would act on it.
                raise RuntimeError(
                    f'[V3] failed to capture the FK->world offset: {e}') from e

        pos_world_minus_env_cpu = pos_chain + self._fk_to_world_offset[None, None, :]

        # Finite-diff velocity in chain frame (offset is constant so it
        # cancels out when subtracting consecutive frames).
        if self._fk_prev_pos_chain is None:
            vel_cpu = torch.zeros_like(pos_chain)
        else:
            # Same dt logic as _refresh_lab: use real elapsed time when
            # available, fallback to nominal control period.
            now_t = time.perf_counter()
            prev_t = getattr(self, '_fk_prev_t', None)
            nominal_dt = 1.0 / self.cfg.control_freq
            dt = nominal_dt if prev_t is None else max(nominal_dt * 0.5, now_t - prev_t)
            self._fk_prev_t = now_t
            vel_cpu = (pos_chain - self._fk_prev_pos_chain) / dt
        self._fk_prev_pos_chain = pos_chain.clone()
        if not hasattr(self, '_fk_prev_t'):
            self._fk_prev_t = time.perf_counter()

        # Upload to sim device (single host→device copy; small tensor).
        self._fk_body_pos_minus_env = pos_world_minus_env_cpu.to(self.device)
        self._fk_body_lin_vel = vel_cpu.to(self.device)

    # -----------------------------------------------------------------
    # compute_observations: clone of parent's, two lines (cur_joint_pos,
    # cur_joint_vel) swapped to read from FK buffers instead of sim.
    # -----------------------------------------------------------------

    def compute_observations(self):
        self._refresh_lab()
        self._compute_fk()

        # ---- Proprioception (unchanged from parent deploy) ----
        q = self.hand_dof_pos
        proprioception_obs = torch.cat([
            q, torch.cos(q), torch.sin(q),
            torch.cat([
                torch.zeros_like(self.base_pos), self.base_quat,
                self.base_lin_vel, self.base_ang_vel,
            ], dim=-1),
        ], dim=-1)

        # ---- Future indices ----
        obs_future_length = self.obs_future_length
        if self.loop_trajectory:
            seq_len = self.demo_data["seq_len"]
            future_indices = torch.stack(
                [(self._get_demo_idx() + 1 + t) % seq_len for t in range(obs_future_length)], dim=-1
            )
        else:
            # Clamp every future step, as the training env does.
            seq_len = self.demo_data["seq_len"]
            future_indices = torch.stack(
                [torch.clamp(self.progress_buf + 1 + t, torch.zeros_like(seq_len), seq_len - 1)
                 for t in range(obs_future_length)], dim=-1
            )
        # References are the retargeted robot targets (target_*), exactly what
        # the training env reads — not the raw MANO human-hand keys.
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

        # ---- Wrist target (unchanged) ----
        target_wrist_pos = indicing(self.demo_data["target_wrist_pos"], future_indices)
        delta_wrist_pos = (target_wrist_pos - self.base_pos[:, None]).reshape(nE, -1)

        target_wrist_vel = indicing(self.demo_data["target_wrist_velocity"], future_indices)
        wrist_vel = target_wrist_vel.reshape(nE, -1)
        delta_wrist_vel = (target_wrist_vel - self.base_lin_vel[:, None]).reshape(nE, -1)

        target_wrist_rot_raw = indicing(self.demo_data["target_wrist_rot"], future_indices)
        if target_wrist_rot_raw.ndim > 3:
            target_wrist_rot_raw = target_wrist_rot_raw[:, :, 0, :]
        target_wrist_quat = aa_to_quat(target_wrist_rot_raw.reshape(nE * nF, -1))
        delta_wrist_quat = quat_mul(
            self.base_quat[:, None].repeat(1, nF, 1).reshape(nE * nF, -1),
            quat_conjugate(target_wrist_quat),
        ).reshape(nE, -1)
        wrist_quat = target_wrist_quat.reshape(nE, -1)

        target_wrist_ang_vel_raw = indicing(self.demo_data["target_wrist_angular_velocity"], future_indices)
        if target_wrist_ang_vel_raw.ndim > 3:
            target_wrist_ang_vel_raw = target_wrist_ang_vel_raw[:, :, 0, :]
        wrist_ang_vel = target_wrist_ang_vel_raw.reshape(nE, -1)
        delta_wrist_ang_vel = (target_wrist_ang_vel_raw - self.base_ang_vel[:, None]).reshape(nE, -1)

        # ---- Joints target — FK buffers replace sim body_pos_w / body_lin_vel_w ----
        target_joints_pos = indicing(self.demo_data["target_joints_pos"], future_indices).reshape(nE, nF, -1, 3)
        # The sim env reads:
        #   cur_joint_pos = self.hand.data.body_pos_w[:, self.hand_body_indices[1:]] - self.scene.env_origins.unsqueeze(1)
        # V3 reads from pk FK output. self._fk_body_pos_minus_env is shape (1, n_hand_bodies, 3)
        # and matches "body_pos_w - env_origin" semantics exactly.
        cur_joint_pos = self._fk_body_pos_minus_env[:, 1:]
        delta_joints_pos = (target_joints_pos - cur_joint_pos[:, None]).reshape(self.num_envs, -1)

        target_joints_vel = indicing(self.demo_data["target_joints_velocity"], future_indices).reshape(nE, nF, -1, 3)
        # The sim env reads:
        #   cur_joint_vel = self.hand.data.body_lin_vel_w[:, self.hand_body_indices[1:]]
        # V3 reads from FK finite-diff vel.
        cur_joint_vel = self._fk_body_lin_vel[:, 1:]
        joints_vel = target_joints_vel.reshape(self.num_envs, -1)
        delta_joints_vel = (target_joints_vel - cur_joint_vel[:, None]).reshape(self.num_envs, -1)

        target_obs_list = [
            delta_wrist_pos, wrist_vel, delta_wrist_vel,
            wrist_quat, delta_wrist_quat, wrist_ang_vel, delta_wrist_ang_vel,
            delta_joints_pos, joints_vel, delta_joints_vel,
        ]

        _asymmetric_ac = getattr(self.cfg, 'asymmetric_ac', False)
        has_object = hasattr(self, 'object') and self.object is not None

        if has_object:
            target_obj_transf = indicing(self.demo_data["obj_trajectory"], future_indices)
            target_obs_list.append(target_obj_transf[:, :, :3, 3].reshape(nE, -1))
            target_obs_list.append(
                rotmat_to_quat(target_obj_transf.reshape(-1, 4, 4)[:, :3, :3]).reshape(nE, -1)
            )
            gt_tips_distance = indicing(self.demo_data["tips_distance"], future_indices).reshape(nE, -1)
            target_obs_list.append(gt_tips_distance)

        if self.obj_bps is not None and getattr(self.cfg, 'enable_bps', True):
            target_obs_list.append(self.obj_bps)

        sensed_contacts = self.last_contacts.clone().reshape(nE, -1)
        target_obs_list.append(sensed_contacts)
        contact_pos = (
            self._last_tactile_pos.clone().reshape(nE, -1)
            if hasattr(self, '_last_tactile_pos')
            else torch.zeros(nE, 15, device=self.device)
        )
        target_obs_list.append(contact_pos)

        target_obs = torch.cat(target_obs_list, dim=-1)
        obs_buf = torch.cat([proprioception_obs, target_obs], dim=-1)

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

        self.priv_info_buf[:, :22] = self.hand_dof_vel
        if has_object:
            self.priv_info_buf[:, 27:30] = self.object_pos
            self.priv_info_buf[:, 30:34] = self.object_rot
            self.priv_info_buf[:, 34:40] = self.object_velocities

        if not hasattr(self, '_obs_dim_checked_v3'):
            actual_dim = obs_buf.shape[-1]
            expected_dim = self.cfg.observation_space
            self.get_logger().info(
                f"[V3 obs] dim={actual_dim} (expected={expected_dim}, "
                f"diff={actual_dim - expected_dim}), priv_info={self.priv_info_buf.shape[-1]}"
            )
            self._obs_dim_checked_v3 = True

        return obs_buf

    # -----------------------------------------------------------------
    # Deploy-only arm target EMA to suppress real-encoder-noise chatter.
    # -----------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """After parent computes arm_joint_pos_des, run a low-pass EMA on
        it to kill the step-to-step chatter that arises because the
        integration `current + delta` reads a noisy real-arm encoder
        each step. Sim has perfect `arm_joint_pos` so training never
        exposed this — at deploy the encoder noise feeds straight into
        the target trajectory, amplifying the policy's action sign-flip
        rate, visible as arm shake.

        EMA on the FINAL target is the conservative choice: it doesn't
        change the integration semantics the policy was trained with, it
        just low-passes the published target before sending to the arm
        impedance controller. With alpha=0.4 at 30Hz the cutoff is
        ~5-7Hz; the 15Hz Nyquist chatter is removed.
        """
        super()._pre_physics_step(actions)

        if not hasattr(self, '_arm_des_ema') or self._arm_des_ema is None:
            self._arm_des_ema = self.arm_joint_pos_des.clone()

        # cfg-overridable so a future user can tune without code edits.
        alpha = float(getattr(self.cfg, 'deploy_arm_des_ema_alpha', 0.4))
        self._arm_des_ema = alpha * self.arm_joint_pos_des + (1.0 - alpha) * self._arm_des_ema
        self.arm_joint_pos_des = self._arm_des_ema.clone()

        # Keep arm_joint_pos_des_prev (used by parent next step's MA
        # computation) consistent with what we actually sent — otherwise
        # parent's MA fights against our EMA.
        if hasattr(self, 'arm_joint_pos_des_prev'):
            self.arm_joint_pos_des_prev = self._arm_des_ema.clone()

    def _reset_idx(self, env_ids):
        # Drop the EMA cache so the first post-reset target isn't pulled
        # backward toward the pre-reset rollout's stale value.
        super()._reset_idx(env_ids)
        self._arm_des_ema = None
        # The next rollout's tactile staleness is measured from its own start,
        # not from frames missed while waiting at the reset prompts.
        self._tactile_age_t0 = None

    # -----------------------------------------------------------------
    # Sensor staleness (extends V2: adds the tactile stream).
    # -----------------------------------------------------------------

    def _tactile_age(self) -> float:
        """Max over the 5 channels of seconds since its last good frame,
        counted from the first staleness check of the current rollout."""
        now = time.monotonic()
        if self._tactile_age_t0 is None:
            self._tactile_age_t0 = now
        return max(now - max(t, self._tactile_age_t0) for t in self._tactile_last_good_t)

    def _sensor_ages(self) -> dict:
        ages = super()._sensor_ages()
        # Tactile is only fetched every env step during a rollout (the hand
        # worker is enabled then); reset does not poll it. With
        # enable_tactile=False every tactile reading is zeroed anyway.
        hand_io = getattr(self, "_hand_io", None)
        if (hand_io is not None and hand_io.is_enabled()
                and getattr(self.cfg, "enable_tactile", True)):
            ages["tactile"] = self._tactile_age()
        return ages

    def _stale_sensors(self) -> dict:
        limit = float(getattr(self.cfg, "deploy_max_sensor_age_s", 0.25))
        return {k: a for k, a in self._sensor_ages().items()
                if a > (TACTILE_MAX_AGE_S if k == "tactile" else limit)}

    # -----------------------------------------------------------------
    # Non-blocking tactile fetch (override of V2's busy-loop polling).
    # -----------------------------------------------------------------

    def get_tactile_info(self):
        """Drop-in replacement for V2.get_tactile_info that does NOT block
        until every channel has a fresh frame. V2's loop re-polls channels
        with `timeout=0.1s` per RPC until all 5 have returned a frame; Sharpa
        tactile streams asynchronously on a ~130ms cycle, so some channels
        never push during one polling pass and that loop dominates the step
        time (~80ms / step).

        Strategy here:
          - Try-fetch each of the 5 channels with timeout=0 (truly non-blocking).
          - If a channel returns a frame, recompute its force / contact-pos
            and update the cache.
          - If a channel returns None, KEEP the previous cached value (this
            is what real systems do — old reading until new arrives).
          - Apply the same post-processing pipeline as V2 (config gates,
            flip, mask, scale, threshold, EMA smoothing).
        """
        force_raw = self._tactile_force_cache.clone()
        force_vec3_raw = self._tactile_force_vec3_cache.clone()
        contact_pos_raw = self._tactile_pos_cache.clone()

        for ch in range(5):
            try:
                ret = self.real_hand.fetch_tactile_frame(ch, timeout=0.0)
            except Exception:
                ret = None
            if ret is None:
                continue
            try:
                deform_data = ret["content"].get("DEFORM")
                f6_data = torch.tensor(ret["content"].get("F6"))
                # Scalar magnitude (used by proprio-force obs).
                force_raw[ch] = torch.norm(f6_data[:3])
                # 3D force vector in elastomer-local frame (for PC vec3 path).
                force_vec3_raw[ch] = f6_data[:3].to(self.device)
                self._tactile_last_good_t[ch] = time.monotonic()
                if deform_data is not None:
                    deform = deform_data.reshape(240, 240).astype(np.uint8)
                    _, binary = cv2.threshold(deform, 30, 255, cv2.THRESH_BINARY)
                    center = self._largest_component_centroid(binary.astype(np.uint8))
                    if center[0] is not None and center[1] is not None:
                        if self.tac_uv_map is None:
                            continue
                        center_pos_ch = self.tac_uv_map[ch][int(center[0]), int(center[1])]
                        contact_pos_raw[ch] = torch.tensor(center_pos_ch[:3]) / 1000.0
            except Exception:
                # Bad frame; keep cached.
                pass

        # Persist to cache.
        self._tactile_force_cache = force_raw.clone()
        self._tactile_force_vec3_cache = force_vec3_raw.clone()
        self._tactile_pos_cache = contact_pos_raw.clone()

        # Same post-processing as V2.
        force = force_raw.clone()
        force_vec3 = force_vec3_raw.clone()        # (5, 3)
        contact_pos = contact_pos_raw.clone()
        if not self.cfg.enable_contact_pos:
            contact_pos[:] = 0.0
        if not self.cfg.enable_tactile:
            force[:] = 0.0
            force_vec3[:] = 0.0
            contact_pos[:] = 0.0

        # ---- scalar path (proprio-force obs path; unchanged) ----
        force = torch.flip(force, dims=[0])
        force[self.cfg.disable_tactile_ids] = 0.0
        force = force.reshape(1, -1)
        force *= getattr(self.cfg, 'force_scale', 1.0)
        force[force < self.cfg.contact_threshold] = 0.0
        if self.cfg.binary_contact:
            force = torch.where(force > self.cfg.contact_threshold, 1.0, 0.0)
        force = self.cfg.contact_smooth * force + (1 - self.cfg.contact_smooth) * self._prev_tactile_force
        self._prev_tactile_force = force.clone()

        # ---- vec3 path (for PC tactile_pc when pc_tactile_use_vec3=True) ----
        # Apply same flip + disable + scale + EMA, per-component. NO binary
        # thresholding (vec3 + binary is undefined; sim env also disallows it).
        # Each finger's xyz is kept as a coherent vector — we DO NOT
        # contact-threshold each component independently (would distort direction);
        # instead we zero the WHOLE vector when its magnitude is below threshold.
        force_vec3 = torch.flip(force_vec3, dims=[0])           # (5, 3)
        force_vec3[self.cfg.disable_tactile_ids, :] = 0.0
        force_vec3 = force_vec3.unsqueeze(0)                    # (1, 5, 3)
        force_vec3 = force_vec3 * getattr(self.cfg, 'force_scale', 1.0)
        # Per-finger magnitude → zero whole vector if below threshold
        vec3_mag = torch.linalg.norm(force_vec3, dim=-1)        # (1, 5)
        thr = float(getattr(self.cfg, 'contact_threshold', 0.2))
        zero_mask = (vec3_mag < thr).unsqueeze(-1)              # (1, 5, 1)
        force_vec3 = torch.where(zero_mask, torch.zeros_like(force_vec3), force_vec3)
        # EMA smoothing (per component)
        a = self.cfg.contact_smooth
        force_vec3 = a * force_vec3 + (1 - a) * self._prev_tactile_force_vec3
        self._prev_tactile_force_vec3 = force_vec3.clone()
        self.last_contacts_vec3 = force_vec3.clone()            # exposed for PC env

        contact_pos = torch.flip(contact_pos, dims=[0])
        contact_pos[self.cfg.disable_tactile_ids, :] = 0.0
        contact_pos = contact_pos.reshape(1, -1)

        return force, contact_pos

    # -----------------------------------------------------------------
    # step: same as V2 but no sim.step / scene.update — pk FK runs inside
    # compute_observations.
    # -----------------------------------------------------------------

    def step(self, action: torch.Tensor):
        period = 1.0 / self.cfg.control_freq
        if not hasattr(self, '_step_next_deadline'):
            self._step_next_deadline = time.perf_counter()

        t0 = time.perf_counter()
        action = action.to(self.device)
        self._latest_policy_action = action[0].detach().cpu().numpy().copy()

        # 1. Action -> targets
        self._pre_physics_step(action)
        t1 = time.perf_counter()

        # 2. Send to real hardware (worker non-blocking) + Polymetis
        self._apply_action()
        t2 = time.perf_counter()

        # 3. Episode bookkeeping (advance progress AFTER reward, see step 6)
        self.episode_length_buf += 1

        # 4. Observation — runs _refresh_lab + _compute_fk + builds obs.
        # NO sim.step, NO scene.update, NO scene.write_data_to_sim.
        obs_dict = self._get_observations()
        t3 = time.perf_counter()

        # 5. Reward / dones
        reward = self._get_rewards()
        terminated, truncated = self._get_dones()

        # 6. Advance progress AFTER reward/done compute, matching the order
        # in FrankaSharpaEnv.step. Without this the policy gets
        # stuck on a single demo frame for the whole rollout.
        self.progress_buf += 1
        if hasattr(self, 'running_progress_buf'):
            self.running_progress_buf += 1

        # 7. Auto-reset
        self.reset_buf = terminated | truncated
        if self.reset_buf.any():
            reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            if reset_env_ids.numel() > 0:
                self._reset_idx(reset_env_ids)
                # Reset blew away FK history — drop the prev-pos cache so
                # the first post-reset velocity is taken as zero (correct,
                # since arm/hand were physically held during reset ramp).
                self._fk_prev_pos_chain = None
                if hasattr(self, '_fk_prev_t'):
                    delattr(self, '_fk_prev_t')
                obs_dict = self._get_observations()

        extras = {}
        t4 = time.perf_counter()

        # 8. Debug record + rate limit
        self._record_debug_step(obs_dict)
        if self.quit_requested and self._debug_record_active and len(self._debug_records) > 0:
            self._flush_debug_recording(reason="quit")

        self._step_next_deadline += period
        sleep_for = self._step_next_deadline - time.perf_counter()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            self._step_next_deadline = time.perf_counter()
        t5 = time.perf_counter()

        # 9. Per-stage profiler. obs+done bucket includes pk FK; if it's
        # still > ~10ms, profile inside _compute_fk for breakdown.
        if not hasattr(self, '_step_prof_v3'):
            self._step_prof_v3 = {k: 0.0 for k in
                                  ('pre', 'apply', 'obs_done_reset', 'rec_sleep', 'cycle')}
            self._step_prof_v3['n'] = 0
            self._step_prof_v3['last_log'] = time.perf_counter()
        self._step_prof_v3['pre']            += t1 - t0
        self._step_prof_v3['apply']          += t2 - t1
        self._step_prof_v3['obs_done_reset'] += t4 - t2
        self._step_prof_v3['rec_sleep']      += t5 - t4
        self._step_prof_v3['cycle']          += t5 - t0
        self._step_prof_v3['n'] += 1
        if time.perf_counter() - self._step_prof_v3['last_log'] >= 1.0:
            n = max(self._step_prof_v3['n'], 1)
            avg_cycle = self._step_prof_v3['cycle'] / n
            self.get_logger().info(
                f'[STEP-PROF V3] {n} steps in 1s | '
                f'cycle={avg_cycle*1000:.1f}ms ({1.0/avg_cycle:.1f}Hz) | '
                f'pre={self._step_prof_v3["pre"]/n*1000:.1f} '
                f'apply={self._step_prof_v3["apply"]/n*1000:.1f} '
                f'obs+done={self._step_prof_v3["obs_done_reset"]/n*1000:.1f} '
                f'rec+sleep={self._step_prof_v3["rec_sleep"]/n*1000:.1f} ms '
                f'(target {period*1000:.1f}ms)'
            )
            for k in ('pre', 'apply', 'obs_done_reset', 'rec_sleep', 'cycle'):
                self._step_prof_v3[k] = 0.0
            self._step_prof_v3['n'] = 0
            self._step_prof_v3['last_log'] = time.perf_counter()

        return obs_dict, reward, terminated, truncated, extras
