"""Cfg for franka-sharpa-force-poseobs.

Extends the critic-horizon cfg by:
  - +7 to actor obs (obj_pos 3 + obj_quat 4 in world / env-local frame)
  - pose-estimator-like noise as domain randomization

Pose obs is in world (env-local) frame, NOT wrist-relative — the same frame as
`self.object_pos` / `self.object_rot`. It is privileged: the teacher sees it,
deployable students drop the slot (obj_pose_tail).

The noise model has THREE components:
  1. Per-step Gaussian (translates to high-frequency tracking jitter):
       pos ~ N(0, obj_pose_pos_noise),    rot ~ axis-angle N(0, obj_pose_rot_noise)
  2. Per-episode constant bias (models a constant ~5 cm pose-estimator offset):
       pos ~ U(-obj_pose_const_bias_pos, +obj_pose_const_bias_pos) per axis,
       rot ~ axis-angle N(0, obj_pose_const_bias_rot)
     Re-sampled at every env reset; held constant within an episode.
  3. Latency + dropout: FIFO buffer + occasional hold-last-good.

Set all noise/bias/latency to 0 to debug with ground-truth pose obs.
"""

from isaaclab.utils import configclass

from .franka_sharpa_critic_horizon_cfg import FrankaSharpaCriticHorizonCfg


@configclass
class FrankaSharpaPoseObsCfg(FrankaSharpaCriticHorizonCfg):
    # ---- Obs space: critic-horizon base (550) + 7 (obj pose obs) = 557 ----
    # Note: critic_horizon parent's __init__ mutates cfg.observation_space, then
    # the poseobs env adds another +7 on top at the end of __init__. The value
    # below is informational only; runtime authority is the env init.
    observation_space = 557

    # ---- Per-step (high-frequency) pose noise ----
    obj_pose_pos_noise: float = 0.008   # m, 1σ  (~5-10mm tracking jitter)
    obj_pose_rot_noise: float = 0.06    # rad, 1σ  (~3.5° tracking jitter)

    # ---- Per-episode constant bias (systematic offset, ~5cm) ----
    obj_pose_const_bias_pos: float = 0.05    # m, half-range U(-0.05, 0.05) per axis
    obj_pose_const_bias_rot: float = 0.05    # rad, 1σ axis-angle (~3°)

    # ---- Latency + dropout ----
    obj_pose_latency_steps: int = 2     # @30Hz → ~67ms
    obj_pose_dropout: float = 0.02      # per-step chance of holding last good

    # ---- Toggle for eval / debugging ----
    # When False, env publishes ground-truth (env-local) pose with zero noise.
    # Useful for sanity-checking that the policy actually uses the obs.
    enable_obj_pose_noise: bool = True
