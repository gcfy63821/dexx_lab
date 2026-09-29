"""Cfg subclass for the critic-horizon variant of franka-sharpa-force.

Isolated from the base `FrankaSharpaEnvCfg` so the base cfg's defaults stay
unchanged for other subclasses.

Differences vs base cfg:
 - `observation_space` starts from a new base that includes `target_obj_pos`/`quat`
   and `tips_distance` as deployable actor obs (computed at env init).
 - `priv_info_dim` enlarged to hold K future frames of object target + tips_distance
   + 1 frame of delta-vs-current-object + 5 obj-to-fingertip distances.
 - New fields controlling which critic-only signals are written.
"""

from isaaclab.utils import configclass

from dexx.robot_constants import ARM_TUNED_KP, ARM_TUNED_KD, ARM_KD_POLYMETIS_IT2  # noqa: F401 (re-exported)

from .franka_sharpa_env_cfg import FrankaSharpaEnvCfg

# The arm gains live in dexx.robot_constants; re-exported here for callers that import them from this module.
__all__ = ["FrankaSharpaCriticHorizonCfg", "ARM_TUNED_KP", "ARM_TUNED_KD", "ARM_KD_POLYMETIS_IT2"]


@configclass
class FrankaSharpaCriticHorizonCfg(FrankaSharpaEnvCfg):
    # Critic-only future horizon for object/tips_distance targets.
    critic_future_length: int = 5

    # Per-component enable flags (all default on).
    critic_enable_target_obj_future: bool = True    # K × (pos 3 + quat 4 + vel 3 + ang_vel 3)
    critic_enable_tips_distance_future: bool = True  # K × 5
    critic_enable_delta_obj_current: bool = True     # 1 × 13  (pos 3 + quat 4 + vel 3 + ang_vel 3)
    critic_enable_obj_to_tips_current: bool = True   # 5 (fingertip distances to object)

    # Actor-visible masks. These keep observation dimensionality fixed
    # and zero selected demo/object/trajectory hints for ablations.
    actor_mask_extra_geometry: bool = False  # masks target object pose + tips_distance + BPS
    actor_mask_target_obj_pose: bool = False # masks actor target_obj_pos/quat only
    actor_mask_tips_distance: bool = False   # masks actor demo/retarget tips_distance only
    actor_mask_bps: bool = False             # masks actor object BPS only
    actor_mask_target_hand: bool = False     # masks actor future hand keypoint targets only

    # This variant is designed to be trained with asymmetric AC by default.
    asymmetric_ac: bool = True

    # Observation-space: actor-visible obs. When enable_bps=True (default),
    # both asymmetric and non-asymmetric paths converge to 550.
    # Matrix (bps=128):
    #   asymmetric=T, enable_bps=T: parent=410, +12 tips+target_obj +128 bps = 550
    #   asymmetric=T, enable_bps=F: parent=410, +12 tips+target_obj          = 422
    #   asymmetric=F, enable_bps=T: parent=543, +7 target_obj                = 550
    #   asymmetric=F, enable_bps=F: parent=415, +7 target_obj                = 422
    observation_space = 550

    # priv_info_dim = 40 (existing slots) + 18*K + 18 (delta + obj_to_tips)
    #              = 40 + 18*5 + 18 = 148   (for K=5 default)
    priv_info_dim = 148

    # ---- Reset / curriculum: easy-first ----
    # adaptive_sampling stays off by default for stability; opt in via --env_cfg.
    random_state_init: bool = True
    init_curriculum_enabled: bool = True
    init_curriculum_method: str = "linear"
    init_curriculum_start: float = 0.3
    init_curriculum_end: float = 0.0
    init_curriculum_steps: int = 15000
    adaptive_sampling_enabled: bool = False

    # ---- Anti-shake training knobs ----
    # Lower moving-average → smoother arm command (sacrifices a bit of responsiveness).
    # 0.4 filters high-frequency policy jitter.
    actions_moving_average: float = 0.4

    # Coefficient on `penal_arm_action_rate_l2` reward term; higher forces the
    # policy to output smoother arm actions.
    arm_action_rate_penalty: float = 0.15

    # Per-joint tuned arm PD (dexx.robot_constants.ARM_TUNED_KP/KD). Set in
    # __post_init__ so we don't have to redefine the whole robot_cfg.
    use_per_joint_tuned_arm_gains: bool = True

    # ---- Reward shaping: no-slip + approach v2 ----
    # `no_slip_weight`: penalize fingertip-vs-object relative velocity when in
    # contact. Targets `fail/obj_pos_drift`. Set 0 to disable.
    no_slip_weight: float = 1.5

    # `approach_shaping_v2`: replace exp(-5d) approach reward (which has near-
    # zero gradient when far) with 1/(1+5d) — dense gradient at all distances.
    # Helps policy learn to approach instead of avoiding contact.
    approach_shaping_v2: bool = True

    def __post_init__(self):
        super().__post_init__() if hasattr(super(), "__post_init__") else None
        if self.use_per_joint_tuned_arm_gains:
            self.robot_cfg.actuators["arm_joints"].stiffness = dict(ARM_TUNED_KP)
            self.robot_cfg.actuators["arm_joints"].damping = dict(ARM_TUNED_KD)
