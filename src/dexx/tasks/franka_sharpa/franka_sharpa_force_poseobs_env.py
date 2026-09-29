# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""franka-sharpa-force-poseobs: critic-horizon env + observed object pose.

What this env adds vs `FrankaSharpaForceCriticHorizonEnv`:
  + 7 actor obs dims at the TAIL: obj_pos (3) + obj_quat wxyz (4)
    in world / env-local frame (same as `self.object_pos` / `self.object_rot`).
  + A pose-estimator-like noise model on top of that ground-truth pose:
      - per-step Gaussian (pos + rot)
      - per-episode constant bias (re-sampled at reset)
      - latency FIFO + dropout hold-last-good

The pose is privileged information for the teacher. There is no deploy
counterpart: deployable students drop these 7 dims (`obj_pose_tail`).

Design choices (rationale):
  - World / env-local frame, not wrist-relative, like `self.object_pos`.
  - Appended at TAIL, NOT inserted into proprio_hist:
      * Keeps `proprio_hist_dim` / `obs_buf_lag_history` unchanged.
      * ProprioAdapt student is intentionally proprio-only; obj pose obs sits
        outside its history slice so the student's input shape doesn't change.
  - Critic priv_info already carries GT object pose at slots 27:34, so asymmetric
    PPO learns the value baseline from clean data while the actor sees the noisy
    obs.
"""
from __future__ import annotations

import torch
import gymnasium as gym

from isaaclab.envs.utils.spaces import spec_to_gym_space
from isaaclab.utils.math import quat_from_angle_axis, quat_mul

from .franka_sharpa_force_critic_horizon_env import FrankaSharpaForceCriticHorizonEnv
from .franka_sharpa_force_poseobs_cfg import FrankaSharpaPoseObsCfg


def _rotvec_to_quat(rv: torch.Tensor) -> torch.Tensor:
    """(N, 3) axis-angle rotvec → (N, 4) wxyz quat. Handles zero-length."""
    angle = torch.linalg.norm(rv, dim=-1)                       # (N,)
    safe_norm = angle.unsqueeze(-1).clamp_min(1e-8)
    axis = rv / safe_norm                                       # (N, 3)
    return quat_from_angle_axis(angle, axis)


class FrankaSharpaForcePoseObsEnv(FrankaSharpaForceCriticHorizonEnv):
    """Critic-horizon env with observed object pose appended to actor obs."""

    cfg: FrankaSharpaPoseObsCfg

    def __init__(self, cfg: FrankaSharpaPoseObsCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Parent settled cfg.observation_space at the critic-horizon value
        # (550 in default config). Extend by 7 for obj_pose obs (pos 3 + quat 4).
        parent_obs_dim = self.cfg.observation_space
        new_obs_dim = parent_obs_dim + 7

        self.cfg.observation_space = new_obs_dim
        self.num_obs = new_obs_dim
        self.single_observation_space["policy"] = spec_to_gym_space(new_obs_dim)
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space["policy"], self.num_envs
        )

        # NOTE: deliberately do NOT touch self.proprio_hist_dim /
        # obs_buf_lag_history / proprio_hist_buf. The 7 obj_pose obs dims sit
        # OUTSIDE the proprio-history slice (i.e. ProprioAdapt student doesn't
        # see them in its history). This matches the asymmetric design where
        # priv_info carries GT pose for the critic and the actor consumes the
        # observation directly without compressing it into a history.

        # ---- Noise / bias / latency state buffers ----
        L = max(1, int(self.cfg.obj_pose_latency_steps) + 1)
        self._obj_pose_latency_buf = torch.zeros(
            (self.num_envs, L, 7), device=self.device, dtype=torch.float
        )
        self._obj_pose_last_good = torch.zeros(
            (self.num_envs, 7), device=self.device, dtype=torch.float
        )
        # Per-episode constant bias: pos (3) + rotvec (3). Resampled at reset.
        self._obj_pose_bias_pos = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float
        )
        self._obj_pose_bias_rotvec = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float
        )
        self._obj_pose_init_done = False  # latency buf needs a warm fill

        print(
            f"[PoseObsEnv] obs_dim={new_obs_dim} (parent={parent_obs_dim}+7), "
            f"noise: pos={self.cfg.obj_pose_pos_noise}, rot={self.cfg.obj_pose_rot_noise}, "
            f"const_bias: pos={self.cfg.obj_pose_const_bias_pos}, rot={self.cfg.obj_pose_const_bias_rot}, "
            f"latency={self.cfg.obj_pose_latency_steps}, dropout={self.cfg.obj_pose_dropout}, "
            f"enabled={self.cfg.enable_obj_pose_noise}"
        )

    # ------------------------------------------------------------------
    # Per-episode constant bias resampling (called from compute_observations
    # using at_reset_buf — same trigger the parent uses for hist resets).
    # ------------------------------------------------------------------
    def _resample_obj_pose_bias(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        n = env_ids.numel()
        pb = self.cfg.obj_pose_const_bias_pos
        rb = self.cfg.obj_pose_const_bias_rot

        if pb > 0.0:
            self._obj_pose_bias_pos[env_ids] = (
                torch.rand((n, 3), device=self.device) * 2.0 - 1.0
            ) * pb
        else:
            self._obj_pose_bias_pos[env_ids] = 0.0

        if rb > 0.0:
            self._obj_pose_bias_rotvec[env_ids] = (
                torch.randn((n, 3), device=self.device) * rb
            )
        else:
            self._obj_pose_bias_rotvec[env_ids] = 0.0

        # Clear latency buffer + last-good for these envs so the FIFO is filled
        # with the post-reset (current) pose on the next compute step.
        self._obj_pose_latency_buf[env_ids] = 0.0
        self._obj_pose_last_good[env_ids] = 0.0

    # ------------------------------------------------------------------
    # Build the 7-d obj_pose obs, applying noise/bias/latency.
    # ------------------------------------------------------------------
    def _compute_obj_pose_obs(self) -> torch.Tensor:
        # Ground truth (env-local pos, world-frame quat — same as priv_info).
        gt_pos = self.object_pos                  # (nE, 3)
        gt_quat = self.object_rot                 # (nE, 4) wxyz

        if not self.cfg.enable_obj_pose_noise:
            return torch.cat([gt_pos, gt_quat], dim=-1)

        # ---- Per-step Gaussian noise ----
        if self.cfg.obj_pose_pos_noise > 0:
            gt_pos = gt_pos + self.cfg.obj_pose_pos_noise * torch.randn_like(gt_pos)

        # ---- Per-episode constant pos bias ----
        gt_pos = gt_pos + self._obj_pose_bias_pos

        # Rotation noise + bias: build delta quat = const_bias * per_step_gauss,
        # then compose on the RIGHT (object-local). Doing both in one composed
        # quat keeps it well-defined for any angle (no small-angle assumption).
        delta_rv = self._obj_pose_bias_rotvec
        if self.cfg.obj_pose_rot_noise > 0:
            delta_rv = delta_rv + self.cfg.obj_pose_rot_noise * torch.randn_like(delta_rv)
        if torch.any(delta_rv != 0):
            delta_quat = _rotvec_to_quat(delta_rv)
            quat = quat_mul(gt_quat, delta_quat)
        else:
            quat = gt_quat

        # ---- Latency: FIFO push current, output oldest ----
        cur = torch.cat([gt_pos, quat], dim=-1).unsqueeze(1)  # (nE, 1, 7)
        if not self._obj_pose_init_done:
            # First step after construction: fill entire buffer with current
            self._obj_pose_latency_buf[:] = cur.expand_as(self._obj_pose_latency_buf)
            self._obj_pose_last_good[:] = cur.squeeze(1)
            self._obj_pose_init_done = True

        self._obj_pose_latency_buf = torch.cat(
            [self._obj_pose_latency_buf[:, 1:], cur], dim=1
        )
        delayed = self._obj_pose_latency_buf[:, 0]   # (nE, 7)

        # ---- Dropout: with prob p hold last good ----
        if self.cfg.obj_pose_dropout > 0:
            keep = (
                torch.rand((self.num_envs,), device=self.device) >= self.cfg.obj_pose_dropout
            ).float().unsqueeze(-1)
            out = keep * delayed + (1.0 - keep) * self._obj_pose_last_good
        else:
            out = delayed

        self._obj_pose_last_good = out
        return out

    # ------------------------------------------------------------------
    # Override compute_observations: call parent, then append 7 obs at tail.
    # ------------------------------------------------------------------
    def compute_observations(self):
        # Parent runs full obs assembly + writes obs_buf to obs_buf_lag_history
        # using ITS slicing (proprio_hist_dim from critic-horizon parent). We
        # don't touch that slice — obj_pose obs is appended AFTER it.
        obs_buf = super().compute_observations()

        # Resample per-episode constant bias for envs that just reset. The
        # at_reset_buf is maintained by base env; same hook the parent uses
        # for hist reset at the bottom of its compute_observations.
        at_reset_env_ids = self.at_reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if at_reset_env_ids.numel() > 0:
            self._resample_obj_pose_bias(at_reset_env_ids)

        # Append obj_pose obs.
        obj_pose_obs = self._compute_obj_pose_obs()    # (nE, 7)
        full_obs = torch.cat([obs_buf, obj_pose_obs], dim=-1)

        # Extend the parent's named slot map with this tail, so a student can
        # drop it by name rather than by a hardcoded `-7`. This is the channel
        # that at deploy is demo replay (open-loop) whenever no pose publisher
        # is running, so naming it explicitly matters.
        slots = getattr(self, "actor_obs_slots", None)
        if slots is not None and "obj_pose_tail" not in slots:
            start = max(b for _, b in slots.values())
            slots["obj_pose_tail"] = (start, start + int(obj_pose_obs.shape[-1]))
            print(f"[obs-slots] + obj_pose_tail[{start}:{slots['obj_pose_tail'][1]}]"
                  f"  -> total {int(full_obs.shape[-1])}d", flush=True)

        if not hasattr(self, "_poseobs_dim_checked"):
            print(
                f"[PoseObsEnv obs] dim={full_obs.shape[-1]} "
                f"(parent_obs={obs_buf.shape[-1]} + poseobs={obj_pose_obs.shape[-1]}), "
                f"cfg.observation_space={self.cfg.observation_space}"
            )
            assert full_obs.shape[-1] == self.cfg.observation_space, (
                f"obs dim mismatch: got {full_obs.shape[-1]} vs cfg "
                f"{self.cfg.observation_space}"
            )
            self._poseobs_dim_checked = True

        return full_obs
