"""Config for `FrankaSharpaPointCloudEnv`.

Inherits the base force/contact env cfg (= `FrankaSharpaEnvCfg` used by
`franka-sharpa-force`) and layers on:
  - camera intrinsics for the depth raycaster
  - point-cloud counts (scene/hand/tactile) + augmentation knobs
  - fusion-strategy switch for the policy encoder

We do NOT change `observation_space` here. The env keeps the parent's flat
proprio obs in `obs_buf`; the point-cloud tensors are exposed alongside it
via `env._get_observations()['scene_pc']`, etc.
"""
from __future__ import annotations

from isaaclab.utils import configclass

# Single source of truth for camera intrinsics + PC workspace crop.
from dexx import deploy_config as _dcfg

# Inherit from the poseobs cfg so our env's flat proprio obs (557d = 550d
# critic_horizon + 7d noisy obj_pose) matches the poseobs teacher
# checkpoint. DAgger uses the 557d-compatible env; deployable students drop
# the 7d obj_pose tail.
from .franka_sharpa_force_poseobs_cfg import FrankaSharpaPoseObsCfg


@configclass
class FrankaSharpaPointCloudEnvCfg(FrankaSharpaPoseObsCfg):
    # ------------------------------------------------------------------
    # Camera (eye-to-hand, RealSense D455 half-res, ~50-80 cm from workspace
    # via the raycaster extrinsics).
    # ------------------------------------------------------------------
    # intrinsics come from dexx.deploy_config (edit there, not here)
    camera_width: int = _dcfg.DEPTH_W
    camera_height: int = _dcfg.DEPTH_H
    camera_fx: float = _dcfg.SIM_INTRINSICS["fx"]
    camera_fy: float = _dcfg.SIM_INTRINSICS["fy"]
    camera_cx: float = _dcfg.SIM_INTRINSICS["cx"]
    camera_cy: float = _dcfg.SIM_INTRINSICS["cy"]

    # Camera extrinsic override (camera-in-armbase, 4x4, ROS optical). When set,
    # `visual_raycaster._build_camera_extrinsics` uses it instead of the
    # shipped calibration (calib/camera_align/current.npy via
    # dexx.deploy_config.default_camera_extrinsic), so the sim depth camera sits
    # at the real (deploy) camera pose and the student trains on a matching
    # viewpoint. Set one of these whenever the camera has been re-calibrated.
    # `_path` (.npy) wins over `_matrix`.
    camera_extrinsic_path: str | None = None
    camera_extrinsic_matrix = None

    # Depth clamps used by the raycaster (also by DepthToPointCloud to
    # de-normalize before back-projection).
    depth_min: float = 0.1
    depth_max: float = 1.5

    # ------------------------------------------------------------------
    # Curtain overrides (the scene shape the depth raycaster has been
    # validated against).
    # ------------------------------------------------------------------
    curtain_back_x: float = -0.75
    curtain_back_y_halflen: float = 0.85
    curtain_side_x_center: float = 0.20
    curtain_side_x_halflen: float = 1.30
    curtain_side_y: float = 0.55

    # ------------------------------------------------------------------
    # Depth domain randomization (only used at training, env disables on deploy).
    # ------------------------------------------------------------------
    enable_depth_noise: bool = True
    cam_pose_pos_noise: float = 0.005          # m, σ
    cam_pose_rot_noise: float = 0.01           # rad, σ
    depth_noise_quadratic_coef: float = 0.0027
    depth_noise_linear_coef: float = 0.0
    depth_const_bias_std: float = 0.002
    depth_dropout_prob: float = 0.02

    # ------------------------------------------------------------------
    # Point-cloud config
    # ------------------------------------------------------------------
    pc_num_scene_points: int = 1024
    pc_num_hand_points: int = 11
    pc_num_tactile_points: int = 25
    pc_tactile_feature_dim: int = 1
    pc_encoder_output_dim: int = 64
    # one of {"early_concat", "separate_encode", "scene_only"}
    pc_fusion_strategy: str = "early_concat"

    # If you want to pin specific hand bodies for the 11 hand points, list
    # their names here (without the `<side>_` prefix). Default = wrist + 5
    # MCP knuckles + 5 fingertips, an informative spread that the PointNet
    # can use to read out hand pose.
    pc_hand_body_names: list | None = None  # set in __post_init__ below

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        if self.pc_hand_body_names is None:
            # 1 wrist + 5 MCP knuckles + 5 fingertips
            self.pc_hand_body_names = [
                "hand_C_MC",
                "thumb_CMC_VL",
                "index_MCP_VL",
                "middle_MCP_VL",
                "ring_MCP_VL",
                "pinky_MCP_VL",
                "thumb_fingertip",
                "index_fingertip",
                "middle_fingertip",
                "ring_fingertip",
                "pinky_fingertip",
            ]

    # Whether to express scene/hand/tactile points in world frame (default
    # True) or arm-base frame. World frame is the simplest invariant; the
    # encoder is a PointNet so it doesn't care, but downstream debug viz needs
    # to know which frame to draw in.
    pc_in_world_frame: bool = True

    # --- Modality ablation (default off = no behaviour change) ---
    pc_ablate_scene_pc: bool = False        # drop the camera-derived scene points
    pc_ablate_tactile_pc: bool = False      # exclude the 25 tactile points from the encoder
    pc_ablate_tactile_force: bool = False   # zero the tactile_force channel
    pc_force_repr: str = "scalar"           # "scalar" | "binary"
    # Divide tactile_force by this constant before feeding PointNet. Must be the
    # same value in sim and real for sim2real consistency. Default 1.0 = no-op
    # (preserves existing trained ckpt behaviour). Recommended: pick the typical
    # max F6 force the Sharpa hand sees on a grasp (≈10-30 N).
    pc_force_scale: float = 1.0
    # Use 3D tactile force vector (elastomer-local frame) instead of scalar
    # magnitude. When True, env outputs tactile_force shape (N, 25, 3) and
    # pc_tactile_feature_dim should be set to 3 (encoder will error otherwise).
    # Default False = scalar magnitude.
    pc_tactile_use_vec3: bool = False
    # If >0: gate tactile points whose force is below this threshold. Pairs with
    # `pc_tactile_gate_mode`:
    #   "zero" (default): zero out xyz + force of gated points (PointNet
    #          must learn to ignore origin cluster).
    #   "mask": leave xyz + force unchanged; emit `tactile_mask` (N, n_tactile)
    #          bool in obs_dict; encoder applies it BEFORE the maxpool so gated
    #          points are excluded cleanly.
    # 0.0 (default) = no gating, all 25 surface points always present.
    pc_tactile_force_gate: float = 0.0
    pc_tactile_gate_mode: str = "zero"   # "zero" | "mask"

    # ------------------------------------------------------------------
    # Disable the reverse curriculum inherited from critic_horizon_cfg.
    # That cfg samples from `[init_curriculum_start*seq_len, 0.98*seq_len)`
    # early on and ramps over `init_curriculum_steps` (15000) env steps. DAgger
    # only takes ~960 env steps total, so with the curriculum the policy would
    # never see the early demo frames.
    # Disabling the curriculum reverts the sampler to uniform `[0, 0.98*seq_len)`.
    # ------------------------------------------------------------------
    init_curriculum_enabled: bool = False

    # ------------------------------------------------------------------
    # Workspace bbox crop (in env-local frame, meters).
    # Drops scene_pc points outside this box BEFORE subsampling, so the 1024
    # final points concentrate on the manipulation region instead of being
    # wasted on table + curtain hits. Tuned so:
    #   x: forward of arm base, where the object lives
    #   y: lateral, narrower than the curtain width
    #   z: from just above table surface up to ~camera reach
    # Set both to None to disable.
    #
    # Table top is at world z=0.415 (dexx.deploy_config.TABLE_SURFACE_Z).
    # env_origins are at z=0, so env-local z ≈ world z. We set z_min just
    # 5mm above the table surface so the table itself drops out of scene_pc
    # but objects sitting on the table (most demo objects are 3-5cm tall) are
    # fully preserved.
    # ------------------------------------------------------------------
    # The crop box comes from dexx.deploy_config (edit there, not here); each
    # student checkpoint records the box it was trained with.
    pc_workspace_min: tuple = _dcfg.PC_WORKSPACE_MIN
    pc_workspace_max: tuple = _dcfg.PC_WORKSPACE_MAX

    # ------------------------------------------------------------------
    # Augmentation (train-only). DISABLED by default: until the PointNet has
    # learned a robust representation, aggressive jitter / dropout / force
    # noise drown out the signal.
    # ------------------------------------------------------------------
    pc_jitter_std: float = 0.0
    pc_dropout_ratio: float = 0.0
    pc_hand_noise_std: float = 0.0
    pc_force_noise_ratio: float = 0.0
    pc_force_dropout_prob: float = 0.0

    # NOTE: do not set `observation_space` here — `obs_buf` stays at whatever
    # the parent env defines. Point clouds ride alongside as extra dict keys.
