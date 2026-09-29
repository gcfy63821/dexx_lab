# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import os
from dexx import deploy_config as _dcfg  # single source of truth for arm_base/table
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.actuators.actuator_cfg import IdealPDActuatorCfg, ImplicitActuatorCfg

from dexx.robot_constants import (
    ARM_ARMATURE, ARM_FRICTION, ARM_TUNED_KD, ARM_TUNED_KP, hand_gain_dicts as _hand_gain_dicts,
)
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from dexx.tasks.hand_imitation.dataset.oakink2_dataset_utils import oakink2_obj_scale, oakink2_obj_mass

def get_workspace_root():
    """Get the workspace root directory (the dexx_release repo root).

    This function navigates up from the current file's directory to find
    the repo root (dexx_release). The file is located at:
    dexx_release/src/dexx/tasks/franka_sharpa/franka_sharpa_env_cfg.py
    so the repo root is 4 levels up (franka_sharpa -> tasks -> dexx -> src -> dexx_release).
    """
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate up 4 levels: franka_sharpa -> tasks -> dexx -> src -> dexx_release
    workspace_root = os.path.join(current_file_dir, "..", "..", "..", "..")
    return os.path.normpath(workspace_root)


def franka_sharpa_urdf(side: str = "right") -> str:
    """Merged FR3 + Sharpa Wave articulation URDF for one hand side.

    Produced by ``scripts/build_merged_urdf.py`` from the two vendored public
    models (``assets/franka_fr3`` + ``assets/sharpa_wave``) and committed, so
    training needs no build step. Isaac Lab converts it to USD on first spawn.
    """
    path = os.path.join(get_workspace_root(), "assets", "generated",
                        f"fr3_with_{side}_sharpa_wave.urdf")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"merged URDF not found: {path}\n"
            f"Generate it with:  python scripts/build_merged_urdf.py --side {side}"
        )
    return path


# Self-collision pairs that MUST be filtered out, as undirected link-name suffix
# pairs. The palm sits directly against the proximal phalanges, and each of these
# pairs is in contact at rest; with self-collision on and no filter the fingers
# cannot close against the palm at all.
#
# This is not cosmetic. URDF cannot express collision filtering and Isaac Lab's
# UrdfConverterCfg has no field for it, so they are applied to the converted USD
# here. Without them the failures are silent (no NaN, no warning): the hand
# simply never closes and the object is never moved.
SELF_COLLISION_FILTER_PAIRS = (
    ("hand_C_MC", "index_PP"),
    ("hand_C_MC", "middle_PP"),
    ("hand_C_MC", "ring_PP"),
    ("hand_C_MC", "pinky_PP"),
    ("hand_C_MC", "thumb_MC"),
    ("pinky_MC", "pinky_PP"),
    ("thumb_MC", "thumb_PP"),
)


def _apply_self_collision_filters(usd_path: str, side: str) -> int:
    """Author `physics:filteredPairs` on the converted robot USD. Idempotent."""
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    root = stage.GetDefaultPrim()
    n = 0
    for a_suf, b_suf in SELF_COLLISION_FILTER_PAIRS:
        a = stage.GetPrimAtPath(root.GetPath().AppendChild(f"{side}_{a_suf}"))
        b = stage.GetPrimAtPath(root.GetPath().AppendChild(f"{side}_{b_suf}"))
        if not (a and a.IsValid() and b and b.IsValid()):
            raise RuntimeError(
                f"self-collision filter pair not found in {usd_path}: "
                f"{side}_{a_suf} <-> {side}_{b_suf}"
            )
        # Author both directions.
        for src, dst in ((a, b), (b, a)):
            UsdPhysics.FilteredPairsAPI.Apply(src)
            rel = src.GetRelationship("physics:filteredPairs")
            targets = list(rel.GetTargets())
            if dst.GetPath() not in targets:
                rel.AddTarget(dst.GetPath())
                n += 1
    if n:
        stage.GetRootLayer().Save()
    return n


def urdf_source_hash(side: str = "right") -> str:
    """Content hash of the merged URDF, used to detect a stale shipped USD."""
    import hashlib

    with open(franka_sharpa_urdf(side), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def shipped_robot_usd(side: str = "right") -> str | None:
    """Path to the committed, pre-built robot USD for `side`, or None if absent.

    Warns (loudly, but does not fail) when the asset was built from a different
    URDF than the one currently in the tree -- a stale asset is exactly the kind
    of mismatch that produces a silently wrong simulation rather than an error.
    """
    d = os.path.join(get_workspace_root(), "assets", "robot", f"fr3_with_{side}_sharpa_wave")
    usd = os.path.join(d, f"fr3_with_{side}_sharpa_wave.usd")
    if not os.path.isfile(usd):
        return None

    stamp = os.path.join(d, ".source_hash")
    try:
        with open(stamp) as f:
            built_from = f.read().strip()
    except OSError:
        built_from = None
    if built_from != urdf_source_hash(side):
        print(
            f"[cfg] WARNING: {usd} was built from a different "
            f"{os.path.basename(franka_sharpa_urdf(side))} than the one on disk. "
            f"Re-run: python scripts/build_robot_usd.py --side {side}",
            flush=True,
        )
    return usd


def franka_sharpa_robot_usd(side: str = "right") -> str:
    """Resolve the robot USD to spawn: the committed asset, else convert on the fly.

    The committed asset already carries the self-collision filter pairs (which
    URDF cannot express and `UrdfConverterCfg` has no field for). The fallback
    conversion path reproduces them, so a checkout without the built asset still
    works -- it is just slower on first use and depends on the local Isaac Lab
    version producing the same output.
    """
    shipped = shipped_robot_usd(side)
    if shipped is not None:
        return shipped

    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    print(f"[cfg] no built robot USD for '{side}'; converting the URDF "
          f"(run scripts/build_robot_usd.py to avoid this)", flush=True)
    converter = UrdfConverter(
        UrdfConverterCfg(
            asset_path=franka_sharpa_urdf(side),
            usd_dir=usd_cache_dir(side),
            fix_base=True,
            # MUST stay False: merging would delete the `*_elastomer` (tactile
            # sensor prims) and `*_fingertip` (tracked bodies) links.
            merge_fixed_joints=False,
            collider_type="convex_hull",
            self_collision=True,
            # Zero here so the actuator cfgs are the single source of truth for
            # gains rather than silently inheriting URDF values.
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                target_type="position",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
            ),
        )
    )
    added = _apply_self_collision_filters(converter.usd_path, side)
    if added:
        print(f"[cfg] authored {added} self-collision filter targets -> {converter.usd_path}")
    return converter.usd_path


def usd_cache_dir(side: str = "right") -> str:
    """Where Isaac Lab caches the URDF -> USD conversion for this robot.

    Isaac Lab only reuses a converted USD when ``usd_dir`` is pinned: left at its
    default it picks a fresh ``/tmp/IsaacLab/usd_<timestamp>_<random>`` per
    process, so the ``.asset_hash`` check never hits and every run re-converts and
    re-writes ~25 MB. Override with ``DEXX_USD_CACHE``. The cache is shared, so
    warm it once (any single-env script) before launching parallel jobs.
    """
    root = os.environ.get(
        "DEXX_USD_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "dexx", "usd"),
    )
    return os.path.join(root, f"fr3_with_{side}_sharpa_wave")



@configclass
class FrankaSharpaEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    # Action space: arm control + hand joint control (22)
    # freeze_arm:              hand(22) = 22
    # use_pid_control:         pos_error(3) + rot_error_6d(6) + hand(22) = 31
    # use_joint_pos_control:   arm_joint_pos(7) + hand(22) = 29
    # use_joint_delta_control: arm_joint_delta(7) + hand(22) = 29
    # force/torque (default):  force(3) + torque(3) + hand(22) = 28
    action_space = 29  # Will be updated based on control mode
    # Observation space: proprioception + target states
    # Proprioception: q(22) + cos_q(22) + sin_q(22) + base_state(10) = 76
    # Target: depends on obs_future_length and number of joints
    # For obs_future_length=1: wrist states (3+3+3+4+4+3+3) + joint states (n_joints*3*3)
    # Actual dimension: 390 (calculated from runtime)
    observation_space = 583
    prop_hist_len = 30  # Required for ProprioAdapt: Conv1d needs at least 30 steps
    priv_info_dim = 40
    state_space = 0

    # Whether to include object BPS (128d static shape encoding) in actor obs.
    # Honored by FrankaSharpaForceEnv / FrankaSharpaForceCriticHorizonEnv
    # (and their deploy variants). When False, BPS is NOT appended to obs and
    # obs_dim is reduced by 128.
    # NOTE: under asymmetric_ac=True in the base force env, BPS is always
    # removed from actor regardless of this flag (treated as privileged).
    # In the critic-horizon variant, this flag solely controls actor BPS.
    enable_bps: bool = True

    hand_side: str = "right"
    robot_asset_override: str = None  # optional: path (abs or repo-relative) to a custom robot URDF (default is assets/generated/fr3_with_{side}_sharpa_wave.urdf)
    material_elastomer_ids: list = None  # collision-shape indices of the 5 fingertip elastomers for friction DR; None -> [27,28,30,32,33] (stock asset, 34 shapes). Re-calibrate after any asset change with tools/calibrate_elastomer_ids.py -- out-of-range ids are dropped silently.
    
    # Keypoint tracking parameters
    obs_future_length: int = 1  # Number of future steps for target observations
    use_quat_rot: bool = False  # Whether to use quaternion rotation in action space
    use_pid_control: bool = False  # Whether to use Diff IK for wrist (action: pos_error(3)+rot_error(6)=9 dims)
    use_osc_control: bool = False  # Whether to use OSC (Operational Space Control) for arm (action: pos_error(3)+rot_error(6)=9 dims)
    actions_moving_average: float = 0.4  # Moving average coefficient for hand/wrist actions
    arm_actions_moving_average: float = 0.15  # Separate (more aggressive) EMA for arm joint_pos_des to match real Franka impedance ~50ms LP bandwidth; sim2real arm-shake mitigation
    use_joint_pos_control: bool = False  # Whether to use absolute joint position control for arm (action: joint_pos(7) dims)
    use_joint_delta_control: bool = True  # Whether to use joint delta control for arm (action: joint_delta(7) dims, recommended for sim2real)
    joint_delta_scale: float = 0.2  # Scale for arm joint-delta actions (rad per step at 30Hz)
    freeze_arm: bool = False  # Freeze arm at reset position, action space = hand only (22)

    # Tracking reward mode: controls which reward terms are active
    # - wrist tracking: reward_eef_pos/rot/vel (world-frame wrist pose tracking)
    # - absolute hand tracking: reward_*_tip_pos (world-frame hand body tracking)
    # - relative hand tracking: reward_rel_*_tip (wrist-frame hand body tracking, decouples hand shape from wrist error)
    # All three can be independently enabled. For freeze_arm, use rel_hand only.
    use_wrist_tracking_reward: bool = True
    use_absolute_hand_tracking_reward: bool = True
    use_relative_hand_tracking_reward: bool = False

    # Tracking reference source. Controls which keys back `target_wrist_*` and
    # `target_joints_*` in `_build_data` (used by `_get_rewards` / obs).
    #   "auto"     — prefer retargeted `opt_*` when present in demo, fall back to MANO
    #   "retarget" — strict: require `opt_*`, raise if missing
    #   "mano"     — force raw MANO (`wrist_pos` / `mano_joints`), ignore `opt_*`
    reference_source: str = "auto"
    # Optional override for RoboToolBatch retarget pkl root. Empty keeps the
    # dataset default: data/retargeting/robotool_batch/mano2{dexhand}.
    robotool_batch_retarget_root: str = ""

    translation_scale: float = 0.02  # Scale for translation actions (±2cm, matching UWLab OSC scale)
    orientation_scale: float = 0.05  # Scale for orientation actions (±0.05 rad ≈ ±3°)

    # Gravity compensation in sim (real Franka has built-in gravity comp, sim PD does not)
    arm_gravity_compensation: bool = True  # Add gravity+coriolis compensation torques to arm PD

    # OSC controller parameters (only used when use_osc_control=True)
    osc_kp_xyz: float = 1500.0  # Position stiffness
    osc_kp_rot: float = 1500.0  # Rotation stiffness
    osc_damping_ratio: float = 1.0  # Damping ratio
    osc_partial_decoupling: bool = True  # Decouple translation/rotation inertia
    osc_nullspace_stiffness: float = 10.0  # Nullspace posture stiffness

    # Observation noise for sim2real (simulates sensor noise)
    obs_joint_pos_noise: float = 0.01  # Noise std for hand joint positions (rad, ~0.6°)
    obs_wrist_pos_noise: float = 0.002  # Noise std for wrist position (m, 2mm)
    obs_wrist_rot_noise: float = 0.017  # Noise std for wrist rotation (rad, ~1°)

    # Action delay for sim2real (simulates communication latency)
    action_delay_steps: int = 2  # Number of steps to delay actions (0=disabled, 2 steps @30Hz ≈ 67ms latency)
    # Action-delay Domain Randomization (sim2real). When randomize_action_delay=True,
    # each env gets its own delay sampled uniformly in [min, max] at reset time. The
    # FIFO buffer is always sized to `max` (upper bound). Real-hardware lag varies
    # 0-100ms across joints, so DR over the delay range makes the policy robust to
    # the actual per-run latency.
    randomize_action_delay: bool = True
    action_delay_min: int = 0          # inclusive, 0 = no delay for this env
    action_delay_max: int = 3          # inclusive, 3 steps @30Hz = 100ms

    # Tightening parameters (curriculum learning)
    tighten_method: str = "exp_decay"  # "None", "const", "linear_decay", "exp_decay", "cos"
    tighten_factor: float = 0.7  # Tightening factor
    tighten_steps: int = 3000  # Number of steps for tightening
    
    # Reset parameters 
    random_state_init: bool = True  # Whether to randomly initialize state
    # When random_state_init is False, default is demo frame 0. Set this to start every reset at a fixed demo index (clamped per-env to seq_len-1). Used for debugging a deploy start frame.
    fixed_reset_demo_frame: int | None = None
    loop_trajectory: bool = False  # Whether to loop reference trajectory (reset only on max_episode_length or terminate)

    # Reverse curriculum on initialization: start from near-grasp frames, gradually expand to full trajectory.
    # Tuned together with the success-reward settings below.
    init_curriculum_enabled: bool = True
    init_curriculum_method: str = "linear"  # "linear", "exp", "cos"
    init_curriculum_start: float = 0.3   # early training: sample from 30%+ of the demo
    init_curriculum_end: float = 0.0     # Late training: sample from 0%+ (full approach)
    init_curriculum_steps: int = 15000   # ramp length (steps)

    # Adaptive initialization from rollout state buffer
    adaptive_init_enabled: bool = False  # Master switch
    adaptive_init_prob: float = 0.3  # Probability of sampling from buffer vs demo
    adaptive_init_buffer_size: int = 8192  # Max entries in ring buffer
    adaptive_init_warmup: int = 500  # Training steps before buffer is used
    adaptive_init_capture_top_k: float = 0.1  # Top fraction of per-step rewards to capture
    adaptive_init_min_progress: int = 10  # Minimum running_progress_buf before capture

    # Adaptive trajectory-fraction sampling — biases reset frames toward bins
    # where the policy currently fails most. Inspired by whole_body_tracking
    # commands.py adaptive bin sampler. Operates on normalized [0, 1) trajectory
    # fraction so it works with multi-data_idx envs (different per-env seq_len).
    #
    # Behavior:
    #   - During the first `warmup_steps` env steps, the sampler is FORCED uniform
    #     (equivalent to random_state_init baseline). Without this, the very
    #     first failures on a random policy collapse the pmf onto a few hard
    #     bins, the policy never sees easy frames, and reward tanks.
    #   - When `compose_with_curriculum=True` AND `init_curriculum_enabled=True`,
    #     bins below the curriculum's `earliest_frac` are masked out — adaptive
    #     samples by failure bias *inside* the curriculum window so you keep
    #     the easy-first warmup while still focusing on the hardest frames in
    #     that window.
    #   - When adaptive_sampling_enabled=True, REPLACES the plain curriculum /
    #     uniform random branches inside _reset_idx (still gated by random_state_init).
    adaptive_sampling_enabled: bool = False
    adaptive_sampling_bins: int = 50            # number of fractional bins along [0,1)
    adaptive_sampling_kernel_size: int = 3      # smoothing kernel size (non-causal)
    adaptive_sampling_lambda: float = 0.8       # geometric kernel decay
    # uniform_ratio: total uniform mass mixed into the pmf (split across bins).
    # 0.5 means a meaningful 50/50 floor against failure spikes — much more
    # forgiving than the reference's 0.1 which lets a single hot bin take ~100%
    # of probability mass once EMA stabilizes.
    adaptive_sampling_uniform_ratio: float = 0.5
    # alpha: EMA mixing factor for bin_failed_count.
    # 0.0005 → effective window ~2000 resets, slow enough that single bad
    # batches don't permanently bias the sampler.
    adaptive_sampling_alpha: float = 0.0005
    # warmup_steps: number of training env steps during which sampler stays
    # uniform (or follows curriculum window if enabled). After this, full
    # adaptive sampling kicks in. Should be at least 1× the typical episode
    # length × num_envs / batch_size to let policy collect signal.
    adaptive_sampling_warmup_steps: int = 1000
    # compose_with_curriculum: when True and init_curriculum_enabled=True, the
    # adaptive sampler restricts bin sampling to the curriculum's expanding
    # window. Best practice for early training stability.
    adaptive_sampling_compose_with_curriculum: bool = True

    # ------------------------------------------------------------------
    # Raycaster-based depth (visual_raycaster.VisualRaycaster)
    # ------------------------------------------------------------------
    # The point-cloud env renders its [N, H, W] depth with
    # simple_raycaster.MultiMeshRaycaster (z-depth, like real depth cameras).
    # Hit clamps fed to MultiMeshRaycaster.raycast_fused.
    raycaster_min_dist: float = 0.01
    raycaster_max_dist: float = 5.0
    # Quadric decimation factors. 0.0 disables. Robot link meshes are already
    # simple, so leave at 0.0; object/scene meshes (cleaned_mesh_10000.obj) get
    # a small amount of simplification by default to keep BVH build/query cheap.
    raycaster_link_simplify_factor: float = 0.0
    raycaster_simplify_factor: float = 0.1
    # Pinhole intrinsics for the depth image.
    camera_focal_length: float = 21.77
    camera_horizontal_aperture: float = 36.0

    # Asymmetric Actor-Critic: actor sees only deployable obs, critic sees obs + priv_info
    asymmetric_ac: bool = True  # When True, removes privileged obs (tips_distance, BPS) from actor
    
    # Dexhand configuration
    dexhand: str = "sharpa"  # Hand type (used by DexHandFactory)
    # control
    decimation = 4  # Policy freq = 120/4 = 30Hz, matching demo data 30fps and deploy
    clip_actions = 1.0
    torque_control = False
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=2,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=8,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=18388608, # 2**23
            gpu_max_rigid_patch_count=5*2**18,
            enable_ccd=True
        ),
    )
    # Arm base pose (matches the real arm-base height above the table). Single
    # source: dexx.deploy_config, shared with retargeting and deploy.
    arm_base_pos = _dcfg.ARM_BASE_POS   # (-0.1, 0.0, 0.415) — edit in dexx/deploy_config.py
    arm_base_rot = _dcfg.ARM_BASE_ROT   # identity quaternion
    arm_init_joint_pos: list = [0.24435, 0.17453, -0.13963, -2.14675, -1.78024, 1.83260, -0.05236]  # Will be updated by update_cfg_for_hand_side()

    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            # Converted from the merged URDF and patched with the self-collision
            # filters; resolved per-side by update_cfg_for_hand_side().
            usd_path="",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.1,
                angular_damping=0.1,
                max_linear_velocity=1000.0,
                max_angular_velocity=64 / math.pi * 180.0,
                max_depenetration_velocity=1000.0,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0005,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.002,
                rest_offset=0.0
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=arm_base_pos,
            rot=arm_base_rot,
            joint_pos={
                "fr3_joint1": arm_init_joint_pos[0],
                "fr3_joint2": arm_init_joint_pos[1],
                "fr3_joint3": arm_init_joint_pos[2],
                "fr3_joint4": arm_init_joint_pos[3],
                "fr3_joint5": arm_init_joint_pos[4],
                "fr3_joint6": arm_init_joint_pos[5],
                "fr3_joint7": arm_init_joint_pos[6],
                # Hand joints will be updated by update_cfg_for_hand_side()
            },
        ),
        actuators={
            # Arm joints: use ImplicitActuator
            # With arm_gravity_compensation=True, PhysX PD drives are zeroed at runtime
            # and replaced by manual PD + gravity + coriolis torques (mimics real Franka).
            # Gains are higher than real (K*3~4x) to compensate for sim's lower control
            # bandwidth (120Hz PD vs real Franka's 1kHz torque loop).
            # Real joint-impedance controller: K=[200,200,200,200,100,100,50], D=[20,20,20,20,10,10,5]
            "arm_joints": ImplicitActuatorCfg(
                joint_names_expr=["fr3_joint.*"],
                # Per-joint step-response-tuned gains (gravity compensation ON);
                # see dexx.robot_constants.
                stiffness=dict(ARM_TUNED_KP),
                damping=dict(ARM_TUNED_KD),
                # Rotor inertia / joint friction. The URDF cannot express them,
                # so they are set explicitly here. See dexx.robot_constants.
                armature=ARM_ARMATURE,
                friction=ARM_FRICTION,
            ),
            # Hand joints: use IdealPDActuator (will be updated by update_cfg_for_hand_side())
            # Hand gains/armature/friction come from HAND_GAINS, not from the
            # asset: a URDF-imported hand has armature=0, which lets the finger
            # joints tunnel through their PhysX limits in a single step.
            # update_cfg_for_hand_side() re-keys these for the chosen side.
            "hand_joints": IdealPDActuatorCfg(
                joint_names_expr=["right_.*"],  # Will be updated by update_cfg_for_hand_side()
                **_hand_gain_dicts("right"),
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

    # contact_sensor, actuated_joint_names, and fingertip_body_names will be updated by update_cfg_for_hand_side()
    contact_sensor: list = []
    actuated_joint_names: list = []
    fingertip_body_names: list = []

    # table
    # Table dimensions:
    # - size: x=1.5, y=2.4, z=0.03
    # - position: x=0.1, y=0, z=0.4 (top surface at z=0.415)
    # - fix_base_link = True -> kinematic_enabled=True
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/table",
        spawn=sim_utils.CuboidCfg(
            size=(1.5, 2.4, 0.03),  # x, y, z dimensions
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,  # Fixed table (equivalent to fix_base_link=True)
                disable_gravity=True,
                enable_gyroscopic_forces=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.002,
                rest_offset=0.0
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),  # Mass doesn't matter for kinematic objects
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.1, 0.0, _dcfg.TABLE_SURFACE_Z - 0.015),  # box centre: top at TABLE_SURFACE_Z, 0.03 thick
            rot=(1.0, 0.0, 0.0, 0.0)  # Identity quaternion
        ),
    )


    # Object spawn template. The env points `asset_path` at the demo dataset's
    # object (`obj_urdf_path`) at runtime and keeps the physics/mass props below;
    # the OakInk path here is only a placeholder and is never loaded.
    obj_id = "O02@0015@00019"
    obj_scale = oakink2_obj_scale.get(obj_id, 1.0)
    obj_mass = oakink2_obj_mass.get(obj_id, 0.05)

    object_cfg: RigidObjectCfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/object",
            spawn=sim_utils.UrdfFileCfg(
                asset_path=os.path.join(get_workspace_root(), "data", "OakInk-v2", "coacd_object_preview", "align_ds", obj_id, "scan.urdf"),
                fix_base = False,
                joint_drive=None,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=False,
                    disable_gravity=False,
                    enable_gyroscopic_forces=True,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                    sleep_threshold=0.005,
                    stabilization_threshold=0.0025,
                    max_depenetration_velocity=1000.0,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_offset=0.002, 
                    rest_offset=0.0
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=obj_mass),
                scale=(obj_scale, obj_scale, obj_scale),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 0.0),
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
        )


    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=1.2, replicate_physics=False)
    data_indices = ["925aa@1"]
    # contact
    enable_tactile = True
    enable_contact_force = True   # 5d scalar contact force in obs
    binary_contact = False        # False = continuous force for training
    enable_contact_pos = False
    disable_tactile_ids = []
    contact_smooth = 0.5
    contact_threshold = 0.2
    contact_latency = 0.005
    bind_multiasset_object_root_link: bool = False
    multiasset_object_root_link_name: str = "base"
    # contact domain randomization for sim2real
    contact_force_noise: float = 0.2        # Multiplicative noise std (±20% of force value)
    contact_pos_noise: float = 0.003        # Additive noise std (3mm)
    contact_dropout_prob: float = 0.05      # Per-finger probability of dropping tactile signal
    # align real
    dof_limits_scale = 0.9
    # Tighten hand joint limits to the real-hand reachable range (measured on
    # the Sharpa Wave hand; cfg/Sharpa order). Needed for sim-real alignment of ring/pinky
    # MCP_AA etc. that have inter-finger mechanical coupling.
    use_real_hand_limits: bool = True
    # randomize
    randomize_pd_gains = True
    randomize_p_gain_scale_lower = 0.5
    randomize_p_gain_scale_upper = 2
    randomize_d_gain_scale_lower = 0.5
    randomize_d_gain_scale_upper = 2
    # Arm PD gain Domain Randomization (separate from hand's randomize_pd_gains,
    # which only touches hand joints). Per-joint PD already matches the real arm
    # closely; this DR covers residual multi-joint coupling and real hardware
    # variability, so ±20% is enough.
    randomize_arm_pd_gains: bool = True
    randomize_arm_p_scale_lower: float = 0.80   # ±20%, sim2real shake mitigation
    randomize_arm_p_scale_upper: float = 1.20
    randomize_arm_d_scale_lower: float = 0.80
    randomize_arm_d_scale_upper: float = 1.20
    randomize_friction = True
    randomize_friction_scale_lower = 1.0
    randomize_friction_scale_upper = 2.5
    elastomer_base_friction = 0.8
    metal_base_friction = 0.1
    object_base_friction = 0.5
    randomize_com = True
    randomize_com_lower = -0.02
    randomize_com_upper = 0.02
    # ---- Polymetis arm backend (deploy) ----------------------------------
    # The real arm runs a joint-impedance controller; these are the gains it is
    # started with. Leave kq/kqd None to use the Polymetis defaults. Sim's
    # counterpart is robot_cfg.actuators["arm_joints"].stiffness/damping — the
    # two must describe the same controller or the policy meets a different
    # plant on hardware than the one it was trained against.
    polymetis_server_ip: str = "localhost"
    polymetis_state_port: int = _dcfg.POLYMETIS_STATE_PORT
    polymetis_cmd_port: int = _dcfg.POLYMETIS_CMD_PORT
    polymetis_kq: tuple | None = None
    polymetis_kqd: tuple | None = None
    # ---- Arm backend selection (deploy) ----------------------------------
    # "polymetis" (the reference backend) or "ros2" (experimental: a ros2_control
    # joint-impedance controller whose gains live in its YAML on the robot PC,
    # wrist from deploy/ros2/wrist_state_publisher.py; see docs/DEPLOY.md).
    arm_backend: str = "polymetis"
    ros2_namespace: str = ""

    # Training-time object xy displacement, metres (uniform +-). Eval can widen
    # it per-run via env._eval_perturb_obj_xy (scripts/eval.py --perturb_obj_xy);
    # the env takes the larger.
    randomize_obj_xy: float = 0.0
    randomize_mass = True
    randomize_mass_lower = 0.01
    randomize_mass_upper = 0.15
    # Deploy-only gain on the real tactile force reading (read by the deploy envs
    # via getattr(cfg, "force_scale", 1.0)); not used in simulation.
    force_scale = 2
    # Gravity scheduler (curriculum) parameters
    gravity_scheduler_enabled: bool = False  # Whether to enable gravity scheduler
    gravity_scheduler_method: str = "linear"  # "None", "linear", "exp", "cos"
    gravity_initial: float = 0.1  # Initial gravity magnitude (positive value, will be negated for z-axis)
    gravity_final: float = 9.81  # Final gravity magnitude (positive value, will be negated for z-axis)
    gravity_scheduler_steps: int = 1000  # Number of steps to reach final gravity
    
    # Virtual object force curriculum parameters
    virtual_object_force_enabled: bool = False  # Whether to enable virtual object force curriculum
    virtual_object_force_decay_method: str = "linear"  # "None", "linear", "exp", "cos"
    virtual_object_kp_initial: float = 30.0  # Initial proportional gain for virtual force
    virtual_object_kp_final: float = 0.0  # Final proportional gain (decay to zero)
    virtual_object_kd_initial: float = 3.0  # Initial derivative gain for virtual force
    virtual_object_kd_final: float = 0.0  # Final derivative gain (decay to zero)
    virtual_object_force_decay_steps: int = 10000  # Number of steps to decay gains to final values
    virtual_force_scale = 0.4
    # debug visualize
    debug_draw = True

    # deploy params
    speed_coef = 0.5
    current_coef = 0.3
    control_freq = 30  # Deploy control frequency (Hz), matches sim: 120Hz / decimation=4 = 30Hz = demo 30fps

    # Emergency stop safety limits (deploy only)
    emergency_stop_enabled: bool = True
    # 2.5 rad/s leaves a buffer above the policy's natural peak (~2.0 rad/s
    # at 30Hz); a 2.0 limit would trip false e-stops on snappy motions.
    # FR3 hardware limit is 2.62 rad/s; 2.5 still leaves a safety margin to that.
    arm_joint_vel_limit: float = 2.5       # rad/s, max arm joint velocity before e-stop
    arm_joint_delta_limit: float = 0.3     # rad, max single-step arm joint change before e-stop
    # 1.5 rad covers thumb_IP demo-target vs real-current gap right after reset
    # (target ~1.31 rad = SDK upper limit, real hand may not reach it during the
    # reset settle window before the first rollout step → false e-stop).
    # The check semantics is target_command - measured_angle, so this is
    # really a "policy is asking for something the hand can't physically
    # reach yet" detector, not an instantaneous step-jump cap. 1.5 leaves
    # the loud-mistake band (>2 rad) intact while tolerating reset slop.
    hand_joint_delta_limit: float = 1.5    # rad, max single-step hand joint change before e-stop
    # Reset route to the demo start. After a rollout the arm is already on the
    # demo trajectory, so the via-home detour is a slow round trip; going
    # straight to the next init frame is the normal case. It is not always safe
    # — from an awkward pose a straight joint interpolation can sweep the hand
    # across the table — so the operator is asked each reset (Enter = direct).
    deploy_prompt_home_route: bool = True
    # Consulted only when the prompt is disabled, e.g. an unattended script.
    deploy_direct_init_move: bool = True
    # E-stop when the newest arm state, hand state (during a rollout) or depth
    # frame is older than this: the policy would otherwise act on stale input.
    deploy_max_sensor_age_s: float = 0.25
    # Deploy debug recording
    deploy_debug_record: bool = True
    deploy_debug_record_steps: int = 100
    deploy_debug_record_dir: str = "logs/deploy_debug"
    # Wrist offsets (deploy): the arm clients report right_hand_C_MC (sim's EE)
    # in the arm-base frame, by FK of the measured joints.
    #   wrist_pos_offset: translation [x,y,z] ADDED to it (arm base -> env-local)
    #   wrist_quat_offset: quaternion [w,x,y,z] PREMULTIPLYING its rotation
    #                      (R_offset * R_arm); identity, as the base is unrotated
    wrist_pos_offset: tuple = _dcfg.ARM_BASE_POS  # the FK wrist is in the arm-base frame
    wrist_quat_offset: tuple = (1.0, 0.0, 0.0, 0.0)

    # ============================================================
    # Final-frame "success" shaping
    # ============================================================
    # Three additive reward terms computed in compute_imitation_reward,
    # all keyed off obj_trajectory[:, seq_len-1] (and the last K frames):
    #
    #   reward_final_pos      = exp(-α_pos · ||cur_obj − final_target||)
    #   reward_final_rot      = exp(-α_rot · |Δrot|)
    #   reward_final_approach = exp(-α_pos · min_k ||cur_obj − last_K[k]||)
    #
    # Larger weight + slower α-decay + wider window combine to give a
    # denser final-pose signal that propagates back through the trajectory,
    # at some cost on mid-grasp stages (the policy becomes more goal-fixated).
    # If grasp/manipulation tasks suffer, dial pos_weight back to 20 first,
    # or shrink window to 7.
    success_pos_weight: float = 30.0
    success_rot_weight: float = 5.0
    success_approach_weight: float = 1.0
    success_alpha_pos: float = 15.0     # exp decay on position dist (m)
    success_alpha_rot: float = 3.0      # exp decay on rotation angle (rad)
    success_reward_window: int = 10     # K = how many trailing frames count for "approach"
    success_reward_ramp: bool = True    # ramp from 0→1 over progress ∈ [max_length−K, max_length]
                                         # prevents policy from "shortcutting" to endpoint early

    # ---- In-hand manipulation reward ----
    # Penalize fingertip-object relative velocity when in contact, so the policy
    # learns to hold the object STILL inside the grasp (matching demo trajectories
    # for rotate / spin / pour) rather than getting away with friction-only grip.
    # 0.0 = disabled; 0.5 = modest, below fingertip-force weight 3.0.
    no_slip_weight: float = 0.5

    # ---- Premature-contact failure (gates frame-0 reaching) -----------
    # When enabled, an episode fails if any fingertip <0.005m from object
    # AND demo target says no-contact AND running_progress >= 50. This can kill
    # frame-0 reaching episodes because the policy approaches faster than
    # demo and triggers fingertip-touch before demo's expected contact frame.
    # Knobs:
    #   premature_contact_enabled = False    → fully disable the check
    #   premature_contact_dist_threshold     → lower = stricter
    #   premature_contact_progress_threshold → delay activation
    premature_contact_enabled: bool = True
    premature_contact_dist_threshold: float = 0.005
    premature_contact_progress_threshold: int = 50

    # ---- Eval infra: disable ALL terminations for full-rollout eval ------
    # When True, every `fail/*` cause is computed and LOGGED but
    # the aggregated `failed_execute` is zeroed → episodes run to
    # `episode_length_buf >= max_length` (i.e., natural time-out at seq_len-1).
    # Useful for diagnostic eval: see what the policy actually does when not
    # killed early, and which fail/* fire diagnostically.
    # NEVER enable during training (policy would have no incentive to avoid
    # failures since they cost nothing).
    eval_no_terminate: bool = False




def update_cfg_for_hand_side(cfg: "FrankaSharpaEnvCfg", hand_side: str):
    """Update all configuration parameters that depend on hand_side.
    
    Args:
        cfg: The configuration object to update
        hand_side: Either "left" or "right"
    """
    cfg.hand_side = hand_side
    
    # Update arm initial joint positions based on hand side
    cfg.arm_init_joint_pos = (
        [0.24435, 0.17453, -0.13963, -2.14675, -1.78024, 1.83260, -0.05236] 
        if hand_side == "right" 
        else [0.24435, 0.13963, -0.27925, -2.30383, 1.97222, 1.69297, -0.62832]
    )
    
    # Update robot_cfg spawn: the merged FR3 + Sharpa Wave URDF for this side.
    workspace_root = get_workspace_root()
    cfg.robot_cfg.spawn.usd_path = franka_sharpa_robot_usd(hand_side)
    # Optional override (a different robot asset). Default None keeps stock behavior.
    _asset_override = getattr(cfg, "robot_asset_override", None)
    if _asset_override:
        cfg.robot_cfg.spawn.usd_path = _asset_override if os.path.isabs(_asset_override) \
            else os.path.join(workspace_root, _asset_override)
        print(f"[cfg] robot asset overridden -> {cfg.robot_cfg.spawn.usd_path}")
    
    # Update robot_cfg init_state joint_pos with hand_side prefix
    cfg.robot_cfg.init_state.joint_pos = {
        "fr3_joint1": cfg.arm_init_joint_pos[0],
        "fr3_joint2": cfg.arm_init_joint_pos[1],
        "fr3_joint3": cfg.arm_init_joint_pos[2],
        "fr3_joint4": cfg.arm_init_joint_pos[3],
        "fr3_joint5": cfg.arm_init_joint_pos[4],
        "fr3_joint6": cfg.arm_init_joint_pos[5],
        "fr3_joint7": cfg.arm_init_joint_pos[6],
        # Hand joints
        f"{hand_side}_thumb_CMC_FE": math.pi/180 * 94.33,
        f"{hand_side}_thumb_CMC_AA": math.pi/180 * -14.90,
        f"{hand_side}_thumb_MCP_FE": math.pi/180 * 27.79,
        f"{hand_side}_thumb_MCP_AA": math.pi/180 * -0.14,
        f"{hand_side}_thumb_IP": math.pi/180 * 10.44,
        f"{hand_side}_index_MCP_FE": math.pi/180 * 63.32, 
        f"{hand_side}_index_MCP_AA": math.pi/180 * -4.95,
        f"{hand_side}_index_PIP": math.pi/180 * 28.80,
        f"{hand_side}_index_DIP": math.pi/180 * 19.52,
        f"{hand_side}_middle_MCP_FE": math.pi/180 * 26.46,
        f"{hand_side}_middle_MCP_AA": math.pi/180 * -10.42,
        f"{hand_side}_middle_PIP": math.pi/180 * 45.27,
        f"{hand_side}_middle_DIP": math.pi/180 * 14.61,
        f"{hand_side}_ring_MCP_FE": math.pi/180 * 24.44,
        f"{hand_side}_ring_MCP_AA": math.pi/180 * 7.01,
        f"{hand_side}_ring_PIP": math.pi/180 * 36.34,
        f"{hand_side}_ring_DIP": math.pi/180 * 22.85,
        f"{hand_side}_pinky_CMC": math.pi/180 * 5.25,
        f"{hand_side}_pinky_MCP_FE": math.pi/180 * 51.80,
        f"{hand_side}_pinky_MCP_AA": math.pi/180 * 8.72,
        f"{hand_side}_pinky_PIP": math.pi/180 * 35.61,
        f"{hand_side}_pinky_DIP": math.pi/180 * 29.33,
    }
    
    # Update robot_cfg actuators hand_joints joint_names_expr
    cfg.robot_cfg.actuators["hand_joints"].joint_names_expr = [f"{hand_side}_.*"]
    # Re-key the hand gains/armature/friction for this side (they are authored
    # for "right" in the class body). Armature is load-bearing: see robot_constants.
    for _k, _v in _hand_gain_dicts(hand_side).items():
        setattr(cfg.robot_cfg.actuators["hand_joints"], _k, _v)
    
    # Update contact_sensor paths
    cfg.contact_sensor = [
        # elastomer
        ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/Robot/{hand_side}_thumb_elastomer",
            history_length=3,
            track_contact_points=True,
            max_contact_data_count_per_prim=32,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/Robot/{hand_side}_index_elastomer",
            history_length=3,
            track_contact_points=True,
            max_contact_data_count_per_prim=32,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/Robot/{hand_side}_middle_elastomer",
            history_length=3,
            track_contact_points=True,
            max_contact_data_count_per_prim=32,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/Robot/{hand_side}_ring_elastomer",
            history_length=3,
            track_contact_points=True,
            max_contact_data_count_per_prim=32,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/Robot/{hand_side}_pinky_elastomer",
            history_length=3,
            track_contact_points=True,
            max_contact_data_count_per_prim=32,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        # DP
        ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/Robot/{hand_side}_thumb_DP",
            history_length=3,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/Robot/{hand_side}_index_DP",
            history_length=3,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/Robot/{hand_side}_middle_DP",
            history_length=3,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/Robot/{hand_side}_ring_DP",
            history_length=3,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/Robot/{hand_side}_pinky_DP",
            history_length=3,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        )
    ]
    
    # Update actuated_joint_names
    cfg.actuated_joint_names = [
        f"{hand_side}_thumb_CMC_FE",
        f"{hand_side}_thumb_CMC_AA",
        f"{hand_side}_thumb_MCP_FE",
        f"{hand_side}_thumb_MCP_AA",
        f"{hand_side}_thumb_IP",
        f"{hand_side}_index_MCP_FE",
        f"{hand_side}_index_MCP_AA",
        f"{hand_side}_index_PIP",
        f"{hand_side}_index_DIP",
        f"{hand_side}_middle_MCP_FE",
        f"{hand_side}_middle_MCP_AA",
        f"{hand_side}_middle_PIP",
        f"{hand_side}_middle_DIP",
        f"{hand_side}_ring_MCP_FE",
        f"{hand_side}_ring_MCP_AA",
        f"{hand_side}_ring_PIP",
        f"{hand_side}_ring_DIP",
        f"{hand_side}_pinky_CMC",
        f"{hand_side}_pinky_MCP_FE",
        f"{hand_side}_pinky_MCP_AA",
        f"{hand_side}_pinky_PIP",
        f"{hand_side}_pinky_DIP",
    ]
    
    # Update fingertip_body_names
    cfg.fingertip_body_names = [
        f"{hand_side}_thumb_fingertip",
        f"{hand_side}_index_fingertip",
        f"{hand_side}_middle_fingertip",
        f"{hand_side}_ring_fingertip",
        f"{hand_side}_pinky_fingertip",
    ]
