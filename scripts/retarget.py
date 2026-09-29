"""
Mano2Dexhand retargeting tool — 2-stage optimization WITH object-pose augmentation.

A 2-stage Adam optimization (Stage 1: arm-only; Stage 2: hand+arm) wrapped with
an object-pose augmentation pipeline:

  * For each demo, the original + N augmented variants are retargeted. Each
    variant perturbs the object (and the wrist/hand targets) by a yaw rotation
    around the arm base + an xy translation.
  * A reachability gate (--reachability_th) marks variants the arm cannot reach
    as reachable=False and saves a partial pkl (the training loader skips them).
  * Joint 1 (the base yaw) is FIXED in Stage 1
    but FREE in Stage 2 — giving the arm extra yaw reach for augmented poses.

Usage:
    python scripts/retarget.py \\
        --side right --headless --data_idx rt/0416_grasp/cube_small_2 --aug_num 5
"""

import math
import os
import pickle
import argparse
import logging

import numpy as np
import torch

from isaaclab.app import AppLauncher

# ---- Side-specific configuration ----
# All parameters that differ between left and right hand are collected here.
SIDE_CONFIGS = {
    "left": {
        "fr3_joint3_init": 1.0,
        "arm_default_joint_pos": [0.24435, 0.17453, -0.13963, -2.14675, -1.78024, 1.83260, -0.05236],
        "target_offset_xy": [0.5, -0.4],
        "stage1_ratio": 1 / 8,       # fraction of max_iter for stage 1
        "arm_lr_stage2": 0.0016,
        "hand_joint_prefix": "left",
    },
    "right": {
        "fr3_joint3_init": -1.0,
        "arm_default_joint_pos": [-0.3435, 0.13963, -0.27925, -2.00383, 1.97222, 1.69297, -0.62832],
        "target_offset_xy": [0.45, 0.00],
        "stage1_ratio": 1 / 12,
        "arm_lr_stage2": 0.0008,
        "hand_joint_prefix": "right",
    },
}

# Resolve project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# scripts/ lives directly under the repo root (dexx_release), so up 1 level.
PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))

"""Parse command line arguments."""
parser = argparse.ArgumentParser(description="Unified Mano to Dexhand retargeting - 2 stage (IsaacLab)")
parser.add_argument("--iter", type=int, default=4000, help="Maximum iterations")
parser.add_argument("--data_idx", type=str, default=None, help="Data index (e.g. rt/blue_cup/blue_cup_1)")
parser.add_argument("--task", type=str, default=None,
                    help="Task name to retarget all sequences (e.g. blue_cup). Equivalent to --data_idx rt/{task}/*")
parser.add_argument("--dexhand", type=str, default="sharpa", help="Dexhand type")
parser.add_argument("--side", type=str, default="right", choices=["left", "right"], help="Hand side")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--sim_device", type=str, default="cuda:0", help="Simulation device")
parser.add_argument("--target_offset_xy", type=float, nargs=2, default=None,
                    help="Target XY offset (overrides side default). E.g. --target_offset_xy 0.3 0.1")
parser.add_argument("--z_offset", type=float, default=0.0,
                    help="Additional Z offset applied to trajectory (default: 0.0)")
parser.add_argument("--obj_scale", type=float, default=1.0,
                    help="Uniform scale applied to object mesh (default: 1.0). "
                         "Saved in output pkl as 'obj_scale' for training to use.")
parser.add_argument("--stage1_iter", type=int, default=None,
                    help="Stage-1 (arm-only) iteration count. If unset, uses "
                         "side_cfg['stage1_ratio'] * --iter (legacy default: 1/8 right, 1/12 left). "
                         "If set, overrides the ratio; stage 2 then runs for (--iter - stage1_iter).")
parser.add_argument("--debug_viz_stride", type=int, default=1,
                    help="Show debug-draw (target hand pose) on every Nth env. "
                         "Default 1 = env 0 only (legacy). E.g. 10 = envs 0,10,20,... "
                         "Useful when running many parallel envs to visually compare frames.")
parser.add_argument("--no_real_hand_clamp", action="store_true",
                    help="Disable clamping opt_dof_pos to the real-hand reachable range (measured on the "
                         "Sharpa Wave hand). "
                         "By default (sharpa dexhand only) the clamp is applied so deploy and sim "
                         "see the same target distribution.")
# ---- Augmentation (object-pose perturbation) ----
parser.add_argument("--aug_num", type=int, default=0,
                    help="Number of augmented variants per demo (object xy/yaw perturbed). 0 = original only.")
parser.add_argument("--aug_radius", type=float, default=0.05,
                    help="Augmentation xy translation radius (m), uniform in [-r, r].")
parser.add_argument("--aug_yaw_deg", type=float, default=10.0,
                    help="Augmentation yaw range (deg), uniform in [-y, y].")
parser.add_argument("--aug_seed", type=int, default=42,
                    help="RNG seed for augmentation sampling (same seed -> same variants).")
parser.add_argument("--reachability_th", type=float, default=0.08,
                    help="Reachability gate: if stage1+2 mean arm EE error > this (m), the variant "
                         "is marked reachable=False and a partial pkl is saved (training loader skips it). "
                         "Raised 0.05 -> 0.08 to admit borderline demos (e.g. ~5.8cm) whose arm EE error "
                         "slightly exceeds the old gate; lower it back via CLI for stricter tracking.")
parser.add_argument("--dump_root", type=str, default=None,
                    help="Optional output root for retargeted pkls. robotool_batch task subdirs are "
                         "created under this root. Default: data/retargeting/robotool_batch/mano2<dexhand>.")
parser.add_argument("--skip_existing", action="store_true",
                    help="Incremental mode: skip variants whose target pkl already exists. "
                         "Combine with a fresh --aug_seed to add NEW aug variants on top of an "
                         "existing set without re-retargeting what is already there.")
parser.add_argument("--force_overwrite", action="store_true",
                    help="Allow an unreachable partial result to overwrite an existing\n"
                         "reachable pkl. Off by default so a low --iter run cannot\n"
                         "destroy converged data.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, RigidObject, ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import pytorch_kinematics as pk
from dexx.tasks.hand_imitation.dataset.factory import ManipDataFactory
from dexx.tasks.hand_imitation.dataset.transform import (
    aa_to_rotmat,
    rot6d_to_aa,
    rotmat_to_quat,
    rotmat_to_rot6d,
)
from dexx.tasks.hand_imitation.envs.factory import DexHandFactory
from dexx.tasks.franka_sharpa.sim2real.real_hand_limits import (
    SHARPA_REAL_LIMITS_NP,
    clamp_to_real_limits_np,
)
import dexx.tasks.hand_imitation.envs.sharpa  # noqa: F401  (registers the Sharpa hand)
from dexx.utils.debug_draw import DebugDraw

logging.getLogger("isaaclab").setLevel(logging.WARNING)


def pack_data(data, dexhand):
    """Pack demo data for retargeting."""
    packed_data = {}
    for k in data[0].keys():
        if k == "mano_joints":
            mano_joints = []
            for d in data:
                mano_joints.append(
                    torch.concat(
                        [
                            d[k][dexhand.to_hand(j_name)[0]]
                            for j_name in dexhand.body_names
                            if dexhand.to_hand(j_name)[0] != "wrist"
                        ],
                        dim=-1,
                    )
                )
            packed_data[k] = torch.stack(mano_joints).squeeze()
        elif isinstance(data[0][k], torch.Tensor):
            packed_data[k] = torch.stack([d[k] for d in data]).squeeze()
        elif isinstance(data[0][k], np.ndarray):
            packed_data[k] = np.stack([d[k] for d in data]).squeeze()
        else:
            packed_data[k] = [d[k] for d in data]
    return packed_data


def soft_clamp(x, lower, upper):
    """Soft clamp function."""
    return lower + torch.sigmoid(4 / (upper - lower) * (x - (lower + upper) / 2)) * (upper - lower)


def _build_hand_joint_names(prefix):
    """Build hand joint name list for the given side prefix ('left' or 'right')."""
    return [
        f"{prefix}_thumb_CMC_FE",
        f"{prefix}_thumb_CMC_AA",
        f"{prefix}_thumb_MCP_FE",
        f"{prefix}_thumb_MCP_AA",
        f"{prefix}_thumb_IP",
        f"{prefix}_index_MCP_FE",
        f"{prefix}_index_MCP_AA",
        f"{prefix}_index_PIP",
        f"{prefix}_index_DIP",
        f"{prefix}_middle_MCP_FE",
        f"{prefix}_middle_MCP_AA",
        f"{prefix}_middle_PIP",
        f"{prefix}_middle_DIP",
        f"{prefix}_ring_MCP_FE",
        f"{prefix}_ring_MCP_AA",
        f"{prefix}_ring_PIP",
        f"{prefix}_ring_DIP",
        f"{prefix}_pinky_CMC",
        f"{prefix}_pinky_MCP_FE",
        f"{prefix}_pinky_MCP_AA",
        f"{prefix}_pinky_PIP",
        f"{prefix}_pinky_DIP",
    ]


# ============================================================
# Augmentation helpers (object-pose perturbation)
# ============================================================
def generate_aug_params(aug_num, aug_radius, aug_yaw_deg, seed):
    """Sample `aug_num` distinct (dx, dy, yaw_deg) perturbations."""
    if aug_num <= 0:
        return []
    rng = np.random.default_rng(seed)
    params, seen = [], set()
    for _ in range(aug_num * 300):
        if len(params) >= aug_num:
            break
        dx = float(rng.uniform(-aug_radius, aug_radius))
        dy = float(rng.uniform(-aug_radius, aug_radius))
        yaw = float(rng.uniform(-aug_yaw_deg, aug_yaw_deg))
        key = (round(dx, 3), round(dy, 3), round(yaw, 1))
        if key not in seen:
            seen.add(key)
            params.append((dx, dy, yaw))
    return params[:aug_num]


def yaw_rotmat_2x2(yaw_deg, device):
    t = math.radians(yaw_deg)
    c, s = math.cos(t), math.sin(t)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.float32, device=device)


def yaw_rotmat_3x3(yaw_deg, device):
    t = math.radians(yaw_deg)
    c, s = math.cos(t), math.sin(t)
    return torch.tensor([[c, -s, 0.0],
                         [s,  c, 0.0],
                         [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)


def rotate_xy_around(pts_xy, center_xy, R2):
    """Rotate [N,2] xy points around center_xy by the 2x2 matrix R2."""
    return (R2 @ (pts_xy - center_xy).T).T + center_xy


def aug_tag(dx, dy, yaw):
    """Human-readable tag, e.g. dxp2.7cm_dyn0.6cm_yawp7.2deg."""
    def fv(v, scale=100, unit="cm"):
        return f"{'p' if v >= 0 else 'n'}{abs(v) * scale:.1f}{unit}"
    return f"dx{fv(dx)}_dy{fv(dy)}_yaw{fv(yaw, 1, 'deg')}"


def make_partial_result(arm_base_pos, offset_xy, offset_z,
                        aug_xy_delta, aug_yaw_deg, mean_err, max_err):
    """Partial dict saved when a variant fails the reachability gate.

    All optimized values are False; aug + offset metadata kept; reachable=False.
    The training loader checks `reachable` and skips these.
    """
    def _np(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    return {
        "opt_wrist_pos": False,
        "opt_wrist_rot": False,
        "opt_dof_pos": False,
        "opt_arm_joint_pos": False,
        "opt_joints_pos": False,
        "xy_offset": _np(offset_xy),
        "z_offset": _np(offset_z),
        "arm_base_pos": arm_base_pos,
        "aug_xy_delta": list(aug_xy_delta),
        "aug_yaw_deg": float(aug_yaw_deg),
        "aug_arm_ee_loss": mean_err,
        "ik_mean_err": mean_err,
        "ik_max_err": max_err,
        "reachable": False,
    }


@configclass
class Mano2DexhandSceneCfg(InteractiveSceneCfg):
    """Scene configuration for Mano2Dexhand retargeting."""
    num_envs: int = 1
    env_spacing: float = 1.0
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


class Mano2Dexhand:
    """Unified IsaacLab Mano2Dexhand retargeting tool (2-stage optimization)."""

    def __init__(self, args, dexhand, obj_urdf_path, side_cfg, obj_scale=1.0):
        self.dexhand = dexhand
        self.headless = args.headless
        self.num_envs = args.num_envs
        self.sim_device = args.sim_device if hasattr(args, 'sim_device') else "cuda:0"
        self.side_cfg = side_cfg
        self.obj_scale = obj_scale
        # Debug-draw stride: show reference pose on every Nth env. 1 = env 0 only.
        self.debug_viz_stride = max(int(getattr(args, 'debug_viz_stride', 1)), 1)
        # Optional explicit stage-1 iter override (overrides side_cfg.stage1_ratio).
        self.stage1_iter_override = getattr(args, 'stage1_iter', None)

        # Setup simulation
        sim_cfg = SimulationCfg(
            dt=1.0 / 120.0,
            device=self.sim_device,
            physx=PhysxCfg(
                solver_type=1,
                max_position_iteration_count=16,
                max_velocity_iteration_count=0,
                bounce_threshold_velocity=0.2,
                gpu_max_rigid_contact_count=8388608,  # 2**23
                gpu_max_rigid_patch_count=5 * 2**18,
            ),
        )
        self.sim = sim_utils.SimulationContext(sim_cfg)

        # Setup scene
        scene_cfg = Mano2DexhandSceneCfg(num_envs=self.num_envs, env_spacing=1.0)
        self.scene = InteractiveScene(scene_cfg)

        # Add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        side = 'right' if dexhand.side == 'rh' else 'left'

        # Load combined robot URDF (Franka arm + Sharpa hand) — relative paths
        asset_dir = os.path.join(PROJECT_ROOT, "assets")
        if 'sharpa' in str(dexhand):
            robot_urdf_path = os.path.join(asset_dir, "generated", f"fr3_with_{side}_sharpa_wave.urdf")
        else:
            robot_urdf_path = os.path.join(
                asset_dir, "sharpa_wave",
                f"{side}_sharpa_wave", f"{side}_sharpa_wave.urdf",
            )

        # arm_base_pos single source of truth: dexx.deploy_config (currently (-0.1,0,0.415)).
        from dexx import deploy_config as _dcfg
        self.arm_base_pos = _dcfg.ARM_BASE_POS
        arm_base_rot = _dcfg.ARM_BASE_ROT
        arm_joint_pos = {
            "fr3_joint1": 0.24435,
            "fr3_joint2": 0.17453,
            "fr3_joint3": side_cfg["fr3_joint3_init"],
            "fr3_joint4": -2.14675,
            "fr3_joint5": -1.78024,
            "fr3_joint6": 1.83260,
            "fr3_joint7": -0.05236,
        }

        # Merged FR3 + Sharpa Wave URDF (built by scripts/build_merged_urdf.py).
        # Same converted+patched USD the training envs use. Going through this
        # helper (rather than spawning the URDF directly) is what applies the
        # self-collision filter pairs -- without them the palm blocks the finger
        # bases and the hand cannot close, silently. It also keeps a single
        # converter cfg per cache dir; two different cfgs pointed at the same
        # dir would each invalidate the other's .asset_hash and re-convert.
        from dexx.tasks.franka_sharpa.franka_sharpa_env_cfg import franka_sharpa_robot_usd

        robot_cfg = ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=franka_sharpa_robot_usd(side),
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
                    rest_offset=0.0,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=self.arm_base_pos,
                rot=arm_base_rot,
                joint_pos=arm_joint_pos,
            ),
            actuators={
                "joints": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=400.0,
                    damping=40.0,
                    effort_limit=100.0,
                ),
            },
        )

        self.robot = Articulation(robot_cfg)
        self.scene.articulations["robot"] = self.robot
        self.hand = self.robot  # Keep 'hand' reference for compatibility

        # Load object asset. Branch on extension: tool.usd preferred, .urdf fallback.
        # In both cases we MUST pass `rigid_props` so RigidBodyAPI is applied at
        # spawn time — convert_mesh.py without --mass produces USD without
        # RigidBodyAPI, which would crash Articulation init otherwise.
        s = self.obj_scale
        rigid_props = sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            retain_accelerations=False,
            kinematic_enabled=True,
            disable_gravity=True,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0025,
            max_depenetration_velocity=1000.0,
        )
        obj_ext = os.path.splitext(obj_urdf_path)[1].lower()
        if obj_ext in (".usd", ".usda", ".usdc"):
            obj_spawn = sim_utils.UsdFileCfg(
                usd_path=obj_urdf_path,
                scale=(s, s, s),
                rigid_props=rigid_props,
            )
            print(f"[INFO] Object asset = USD: {obj_urdf_path}")
        else:
            obj_spawn = sim_utils.UrdfFileCfg(
                asset_path=obj_urdf_path,
                fix_base=False,
                joint_drive=None,
                scale=(s, s, s),
                rigid_props=rigid_props,
            )
            print(f"[INFO] Object asset = URDF: {obj_urdf_path}")
        obj_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/object",
            spawn=obj_spawn,
        )
        self.object = RigidObject(obj_cfg)
        self.scene.rigid_objects["object"] = self.object

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions()

        # Reset simulation
        self.sim.reset()

        # Identify arm and hand joints (must be after robot is loaded)
        self._identify_joints()

        # Build kinematic chain for hand (from dexhand URDF)
        urdf_path = dexhand.urdf_path
        asset_root = os.path.split(urdf_path)[0]
        asset_file = os.path.split(urdf_path)[1]
        print(f"Loading hand URDF from: {asset_root}/{asset_file}")

        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"Hand URDF file not found: {urdf_path}")

        # Read as bytes: these URDFs carry an XML encoding declaration,
        # which lxml rejects when handed a str.
        self.chain = pk.build_chain_from_urdf(
            open(os.path.join(asset_root, asset_file), 'rb').read())
        self.chain = self.chain.to(dtype=torch.float32, device=self.sim_device)

        # Build kinematic chain for arm (from combined robot URDF)
        if not os.path.exists(robot_urdf_path):
            raise FileNotFoundError(f"Robot URDF file not found: {robot_urdf_path}")

        # Read as bytes: the generated URDF carries an XML encoding
        # declaration, which lxml rejects when handed a str.
        with open(robot_urdf_path, 'rb') as f:
            robot_urdf_string = f.read()

        self.arm_chain = pk.build_chain_from_urdf(robot_urdf_string)
        self.arm_chain = self.arm_chain.to(dtype=torch.float32, device=self.sim_device)

        # Get arm joint names in chain
        all_chain_joint_names = self.arm_chain.get_joint_parameter_names()
        self.arm_joint_names_in_chain = []
        for name in all_chain_joint_names:
            if 'fr3_joint' in name.lower():
                for i in range(1, 8):
                    if f'joint{i}' in name.lower() or f'fr3_joint{i}' in name.lower():
                        if name not in self.arm_joint_names_in_chain:
                            self.arm_joint_names_in_chain.append(name)
                        break

        if len(self.arm_joint_names_in_chain) < 7:
            self.arm_joint_names_in_chain = all_chain_joint_names[:7]

        self.arm_joint_names_in_chain = self.arm_joint_names_in_chain[:7]
        self.arm_end_effector_link = f"{side}_hand_C_MC"

        print(f"Arm joint names in chain: {self.arm_joint_names_in_chain}")

        # Create mapping from robot joints to arm_chain joints (for Stage 2 FK)
        all_robot_joint_names = self.robot.joint_names
        self.robot2arm_chain_order = []

        for chain_joint_name in all_chain_joint_names:
            if chain_joint_name in all_robot_joint_names:
                robot_joint_idx = all_robot_joint_names.index(chain_joint_name)
                self.robot2arm_chain_order.append(robot_joint_idx)
            else:
                found = False
                for robot_joint_name in all_robot_joint_names:
                    if chain_joint_name.lower() in robot_joint_name.lower() or robot_joint_name.lower() in chain_joint_name.lower():
                        robot_joint_idx = all_robot_joint_names.index(robot_joint_name)
                        self.robot2arm_chain_order.append(robot_joint_idx)
                        found = True
                        break
                if not found:
                    print(f"Warning: Chain joint {chain_joint_name} not found in robot joints, using index 0")
                    self.robot2arm_chain_order.append(0)

        print(f"robot2arm_chain_order length: {len(self.robot2arm_chain_order)}, arm_chain joints: {len(all_chain_joint_names)}")

        self.arm_chain_body_names = list(self.arm_chain.get_link_names())
        print(f"Arm chain body names (first 10): {self.arm_chain_body_names[:10]}")

        # Get joint limits and properties
        joint_pos_limits = self.hand.root_physx_view.get_dof_limits().to(self.sim_device)
        print("joint_pos_limits: ", joint_pos_limits.shape, joint_pos_limits)
        if joint_pos_limits.ndim == 3:
            all_joint_lower_limits = joint_pos_limits[0, :, 0]
            all_joint_upper_limits = joint_pos_limits[0, :, 1]
        else:
            all_joint_lower_limits = joint_pos_limits[:, 0]
            all_joint_upper_limits = joint_pos_limits[:, 1]

        self.arm_joint_lower_limits = all_joint_lower_limits[self.arm_joint_indices]
        self.arm_joint_upper_limits = all_joint_upper_limits[self.arm_joint_indices]

        if len(self.hand_joint_indices) > 0:
            self.dexhand_dof_lower_limits = all_joint_lower_limits[self.hand_joint_indices]
            self.dexhand_dof_upper_limits = all_joint_upper_limits[self.hand_joint_indices]
        else:
            self.dexhand_dof_lower_limits = all_joint_lower_limits
            self.dexhand_dof_upper_limits = all_joint_upper_limits

        self.limit_info = {
            "rh": {
                "lower": self.dexhand_dof_lower_limits.cpu().numpy().astype(np.float32),
                "upper": self.dexhand_dof_upper_limits.cpu().numpy().astype(np.float32),
            }
        }

        self._dexhand_dof_speed_limits = self.robot.data.joint_vel_limits.clone()

        # Default DOF positions for hand
        self.num_hand_dofs = len(self.hand_joint_indices)
        default_dof_pos = torch.ones(self.num_hand_dofs, device=self.sim_device) * np.pi / 50
        self.dexhand_default_dof_pos = default_dof_pos

        # Default arm joint positions (from side config)
        selected_arm_joint_pos = side_cfg["arm_default_joint_pos"]
        self.arm_default_joint_pos = torch.tensor(selected_arm_joint_pos, device=self.sim_device)

        # Transformation matrix
        table_half_height = 0.015
        table_surface_z = _dcfg.TABLE_SURFACE_Z  # table top; the box is table_half_height thick below it

        mujoco2gym_transf = torch.eye(4, dtype=torch.float32, device=self.sim_device)
        m1 = aa_to_rotmat(torch.tensor([0, 0, -np.pi / 2], dtype=torch.float32, device=self.sim_device))
        m2 = aa_to_rotmat(torch.tensor([np.pi / 2, 0, 0], dtype=torch.float32, device=self.sim_device))
        mujoco2gym_transf[:3, :3] = m1 @ m2
        mujoco2gym_transf[:3, 3] = torch.tensor([0, 0, table_surface_z], device=self.sim_device)
        self.mujoco2gym_transf = mujoco2gym_transf

        # Get body handles
        self.dexhand_handles = {}
        for body_name in self.dexhand.body_names:
            if body_name in self.robot.body_names:
                self.dexhand_handles[body_name] = self.robot.body_names.index(body_name)

        # Get DOF order for kinematic chain (hand only)
        if self.chain is not None:
            chain_joint_names = self.chain.get_joint_parameter_names()
            all_joint_names = self.robot.joint_names
            hand_joint_names = [all_joint_names[i] for i in self.hand_joint_indices]

            print(f"Chain joint names ({len(chain_joint_names)}): {chain_joint_names[:5]}...")
            print(f"Hand joint names ({len(hand_joint_names)}): {hand_joint_names[:5]}...")

            self.isaac2chain_order = []
            for j_name in chain_joint_names:
                if j_name in hand_joint_names:
                    self.isaac2chain_order.append(hand_joint_names.index(j_name))
                else:
                    found = False
                    for hand_joint in hand_joint_names:
                        if j_name.lower() in hand_joint.lower() or hand_joint.lower() in j_name.lower():
                            self.isaac2chain_order.append(hand_joint_names.index(hand_joint))
                            found = True
                            break
                    if not found:
                        print(f"Warning: Joint {j_name} not found in hand joints, using index 0")
                        self.isaac2chain_order.append(0)

            print(f"isaac2chain_order: {self.isaac2chain_order[:10]}... (length: {len(self.isaac2chain_order)})")

            if len(self.isaac2chain_order) != len(chain_joint_names):
                print(f"Warning: isaac2chain_order length ({len(self.isaac2chain_order)}) != chain_joint_names length ({len(chain_joint_names)})")
                if len(self.isaac2chain_order) > len(chain_joint_names):
                    self.isaac2chain_order = self.isaac2chain_order[:len(chain_joint_names)]
                else:
                    self.isaac2chain_order.extend([0] * (len(chain_joint_names) - len(self.isaac2chain_order)))
        else:
            self.isaac2chain_order = list(range(self.num_hand_dofs))

        # Initialize chamfer distance for contact optimization
        import chamfer_distance as chd
        self.ch_dist = chd.ChamferDistance()

        # Contact optimization parameters
        self.contact_d_th = 0.02
        self.contact_alpha = 100.0
        self.contact_loss_weight = 0.01
        self.force_loss_weight = 0.005
        self.obj_mass = 0.05
        self.gravity = 9.81
        self.contact_opt_start_iter = 6000

        # Initialize debug draw for visualization
        

        # Set camera view
        if not self.headless:
            self.debug_draw = DebugDraw()
            self.sim.set_camera_view([4.0, 3.0, 3.0], [-4.0, -3.0, 0.0])

    def _compute_rotation_error(self, R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
        """Compute rotation error between two rotation matrices."""
        if R1.ndim == 2:
            R1 = R1.unsqueeze(0)
        if R2.ndim == 2:
            R2 = R2.unsqueeze(0)
        R_error = torch.bmm(R2.transpose(-1, -2), R1)
        trace = R_error[:, 0, 0] + R_error[:, 1, 1] + R_error[:, 2, 2]
        angle = torch.acos(torch.clamp((trace - 1) / 2, -1 + 1e-6, 1 - 1e-6))
        return angle

    def fitting(self, max_iter, obj_trajectory, target_wrist_pos, target_wrist_rot,
                target_mano_joints, obj_verts=None, tip_list=None,
                target_offset_xy_override=None, z_offset_value=0.0,
                aug_xy_delta=(0.0, 0.0), aug_yaw_deg=0.0):
        """Perform 2-stage optimization to fit MANO data to robot hand.

        Stage 1: Arm-only optimization to reach target wrist position.
        Stage 2: Joint hand + arm optimization for fingertip matching.

        Args:
            max_iter: Maximum iterations
            obj_trajectory: Object trajectory [num_envs, 4, 4] or [num_envs, T, 4, 4]
            target_wrist_pos: Target wrist positions [num_envs, 3]
            target_wrist_rot: Target wrist rotations [num_envs, 3] (axis-angle)
            target_mano_joints: Target MANO joint positions [num_envs, N, 3]
            obj_verts: Object vertices in object frame [M, 3], optional
            tip_list: List of tip joint names, optional
            target_offset_xy_override: Override XY offset [2], optional
            z_offset_value: Additional Z offset applied to trajectory
        """
        assert target_mano_joints.shape[0] == self.num_envs

        if tip_list is None:
            tip_list = ["thumb_fingertip", "index_fingertip", "middle_fingertip", "ring_fingertip", "pinky_fingertip"]

        # Contact optimization setup
        if obj_verts is None:
            enable_contact_opt = False
        else:
            enable_contact_opt = True
            obj_verts = obj_verts.to(self.sim_device)
            if obj_verts.ndim == 1:
                obj_verts = obj_verts.view(-1, 3)
            elif obj_verts.ndim > 2:
                obj_verts = obj_verts.view(-1, 3)
            if self.obj_scale != 1.0:
                obj_verts = obj_verts * self.obj_scale
                print(f"[retarget] Scaled obj_verts by {self.obj_scale}")

        # Transform targets to simulation frame
        target_wrist_pos = (self.mujoco2gym_transf[:3, :3] @ target_wrist_pos.T).T + self.mujoco2gym_transf[:3, 3]
        target_wrist_rot = self.mujoco2gym_transf[:3, :3] @ aa_to_rotmat(target_wrist_rot)
        target_mano_joints = target_mano_joints.view(-1, 3)
        target_mano_joints = (self.mujoco2gym_transf[:3, :3] @ target_mano_joints.T).T + self.mujoco2gym_transf[:3, 3]
        target_mano_joints = target_mano_joints.view(self.num_envs, -1, 3)

        # Transform object trajectory
        if obj_trajectory.ndim == 4:
            obj_trajectory = obj_trajectory[:, 0]

        if obj_trajectory.ndim == 3:
            obj_trajectory_transformed = torch.zeros_like(obj_trajectory)
            for i in range(self.num_envs):
                obj_trajectory_transformed[i] = self.mujoco2gym_transf @ obj_trajectory[i]
            obj_trajectory = obj_trajectory_transformed
        elif obj_trajectory.ndim == 2:
            obj_trajectory = self.mujoco2gym_transf @ obj_trajectory
            obj_trajectory = obj_trajectory.unsqueeze(0)

        # Augmentation: yaw-rotate object + targets around the arm base (sim frame)
        if abs(aug_yaw_deg) > 1e-6:
            _R2 = yaw_rotmat_2x2(aug_yaw_deg, self.sim_device)
            _R3 = yaw_rotmat_3x3(aug_yaw_deg, self.sim_device)
            _center = torch.tensor([self.arm_base_pos[0], self.arm_base_pos[1]],
                                   device=self.sim_device, dtype=torch.float32)
            _twp_xy = rotate_xy_around(target_wrist_pos[:, :2], _center, _R2)
            target_wrist_pos = torch.cat([_twp_xy, target_wrist_pos[:, 2:3]], dim=-1)
            target_wrist_rot = _R3.unsqueeze(0) @ target_wrist_rot
            _mj_flat = target_mano_joints.reshape(-1, 3)
            _mj_xy = rotate_xy_around(_mj_flat[:, :2], _center, _R2)
            target_mano_joints = torch.cat([_mj_xy, _mj_flat[:, 2:3]], dim=-1).view(self.num_envs, -1, 3)
            obj_trajectory = obj_trajectory.clone()
            _obj_xy = rotate_xy_around(obj_trajectory[:, :2, 3], _center, _R2)
            obj_trajectory[:, :2, 3] = _obj_xy
            obj_trajectory[:, :3, :3] = _R3.unsqueeze(0) @ obj_trajectory[:, :3, :3]
            print(f"[aug] yaw-rotated targets by {aug_yaw_deg:.1f} deg around arm base")

        # Compute offset
        first_obj_pos = obj_trajectory[0, :3, 3]

        # Use override or side-default for XY offset
        if target_offset_xy_override is not None:
            target_offset_xy = torch.tensor(target_offset_xy_override, device=self.sim_device)
        else:
            target_offset_xy = torch.tensor(self.side_cfg["target_offset_xy"], device=self.sim_device)

        # Augmentation xy translation: shift where the object (and targets) land
        aug_dxy = torch.tensor(list(aug_xy_delta), device=self.sim_device, dtype=target_offset_xy.dtype)
        offset_xy = (target_offset_xy + aug_dxy) - first_obj_pos[:2]
        offset_z = torch.tensor([z_offset_value], device=self.sim_device)
        offset = torch.cat([offset_xy, offset_z])

        print(f"Side: {self.side_cfg['hand_joint_prefix']}")
        print(f"Target offset XY: {target_offset_xy.cpu().numpy()}")
        print(f"Applying XY offset: {offset_xy.cpu().numpy()}")
        print(f"Applying Z offset: {offset_z.item()}")

        # Apply offset
        target_wrist_pos_offset = target_wrist_pos.clone()
        target_wrist_pos_offset[:, :2] += offset_xy.unsqueeze(0)
        target_wrist_pos_offset[:, 2] += offset_z

        target_mano_joints_offset = target_mano_joints.clone()
        target_mano_joints_offset[:, :, :2] += offset_xy.unsqueeze(0).unsqueeze(0)
        target_mano_joints_offset[:, :, 2] += offset_z

        obj_trajectory_offset = obj_trajectory.clone()
        obj_trajectory_offset[:, :2, 3] += offset_xy.unsqueeze(0)
        obj_trajectory_offset[:, 2, 3] += offset_z

        # Initialize optimization variables
        opt_wrist_pos = target_wrist_pos_offset.clone().detach()
        opt_wrist_rot = rotmat_to_rot6d(target_wrist_rot).clone().detach()

        if self.dexhand_default_dof_pos.ndim > 1:
            self.dexhand_default_dof_pos = self.dexhand_default_dof_pos.flatten()[:self.num_hand_dofs]
        opt_hand_dof_pos = self.dexhand_default_dof_pos[None, :].repeat(self.num_envs, 1).clone().detach().requires_grad_(True)

        arm_joint2_7_indices = self.arm_joint_indices[1:7]
        arm_joint2_7_default = self.arm_default_joint_pos[1:7]
        opt_arm_joint_pos = arm_joint2_7_default[None, :].repeat(self.num_envs, 1).clone().detach().requires_grad_(True)

        joint1_initial_pos = self.arm_default_joint_pos[0]

        # Joint 1 (base yaw): fixed in Stage 1, FREE in Stage 2 — give it its own
        # optimization variable so Stage 2 can use the base yaw for extra reach.
        opt_joint1 = joint1_initial_pos.detach().reshape(1).repeat(self.num_envs).clone().detach().requires_grad_(True)

        print(f"Debug - dexhand_default_dof_pos shape: {self.dexhand_default_dof_pos.shape}")
        print(f"Debug - opt_hand_dof_pos shape: {opt_hand_dof_pos.shape}")
        print(f"Debug - opt_arm_joint_pos shape: {opt_arm_joint_pos.shape}")
        print(f"Debug - num_envs: {self.num_envs}, num_hand_dofs: {self.num_hand_dofs}")
        print(f"Debug - Joint1 (fixed) initial position: {joint1_initial_pos.item():.4f}")

        # Two-stage optimization
        stage1_ratio = self.side_cfg["stage1_ratio"]
        arm_lr_stage2 = self.side_cfg["arm_lr_stage2"]
        if self.stage1_iter_override is not None:
            stage1_iter = max(0, min(int(self.stage1_iter_override), max_iter))
            print(f"[stage1] Using --stage1_iter override: {stage1_iter} "
                  f"(side_cfg ratio would have given {int(max_iter * stage1_ratio)})")
        else:
            stage1_iter = int(max_iter * stage1_ratio)
        stage2_iter = max_iter - stage1_iter

        print(f"Optimization stages: Stage 1 (arm only) = {stage1_iter} iterations, Stage 2 (hand + arm) = {stage2_iter} iterations")

        # Stage 1 optimizer: only arm joints
        opti_stage1 = torch.optim.Adam(
            [{"params": [opt_arm_joint_pos], "lr": 0.0004}]
        )

        # Stage 2 optimizer: hand + arm joints 2-7 + the now-free joint 1
        opti_stage2 = torch.optim.Adam(
            [
                {"params": [opt_hand_dof_pos], "lr": 0.0004},
                {"params": [opt_arm_joint_pos], "lr": arm_lr_stage2},
                {"params": [opt_joint1], "lr": arm_lr_stage2},
            ]
        )

        # Compute weights for different joints.
        # Per-finger × per-level scheme: finger_w sets the per-finger importance
        # (thumb > index > middle > ring > pinky), scale modulates within a
        # finger (tip > distal > intermediate > proximal). This raises the
        # non-tip levels (distal/intermediate/proximal) so retarget tracks
        # the whole finger pose, not just the fingertip.
        FINGER_W = {"thumb": 25.0, "index": 15.0, "middle": 10.0, "ring": 7.0, "pinky": 5.0}
        LEVEL_SCALE = {"tip": 1.0, "distal": 0.6, "intermediate": 0.4, "proximal": 0.3}
        weight = []
        for k in self.dexhand.body_names:
            k_hand = self.dexhand.to_hand(k)[0]
            # Per-finger base
            finger_w = 1.0
            for finger, w in FINGER_W.items():
                if finger in k_hand:
                    finger_w = w
                    break
            # Per-level scale
            scale = 1.0
            for level, s in LEVEL_SCALE.items():
                if level in k_hand:
                    scale = s
                    break
            weight.append(finger_w * scale)
        weight = torch.tensor(weight, device=self.sim_device, dtype=torch.float32)

        iter = 0
        past_loss = 1e10
        stage1_past_loss = 1e10

        tip_indices = []
        for tip_name in tip_list:
            tip_body_name = None
            for body_name in self.dexhand.body_names:
                if tip_name in body_name:
                    tip_body_name = body_name
                    print(tip_body_name)
                    break
            if tip_body_name and tip_body_name in self.dexhand.body_names:
                tip_idx = self.dexhand.body_names.index(tip_body_name)
                tip_indices.append(tip_idx)

        arm_joint2_7_lower = self.arm_joint_lower_limits[1:7]
        arm_joint2_7_upper = self.arm_joint_upper_limits[1:7]
        arm_joint1_lower = self.arm_joint_lower_limits[0]
        arm_joint1_upper = self.arm_joint_upper_limits[0]

        enable_stage_1 = True

        # ============================================================
        # Stage 1: Optimize arm joints only to reach target wrist pos
        # ============================================================
        print(f"\n=== Stage 1: Arm-only optimization (iterations 1-{stage1_iter}) ===")
        print(f"Stage 1 target: target_wrist_pos_offset (with XY offset applied)")
        while iter < stage1_iter and enable_stage_1:
            iter += 1

            # Clamp arm joint positions (joints 2-7 only)
            opt_arm_joint_pos_clamped = torch.clamp(
                opt_arm_joint_pos,
                arm_joint2_7_lower[None, :],
                arm_joint2_7_upper[None, :],
            )

            # Combine arm joint positions (hand joints stay at default)
            all_joint_pos = torch.zeros(self.num_envs, self.robot.num_joints, device=self.sim_device)
            all_joint_pos[:, self.arm_joint_indices[0]] = joint1_initial_pos
            all_joint_pos[:, arm_joint2_7_indices] = opt_arm_joint_pos_clamped.detach()
            all_joint_pos[:, self.hand_joint_indices] = self.dexhand_default_dof_pos[None, :].repeat(self.num_envs, 1)

            # Set states in simulation
            self.robot.write_joint_state_to_sim(all_joint_pos.detach(), torch.zeros_like(all_joint_pos.detach()))
            self.robot.set_joint_position_target(all_joint_pos.detach())

            # Set object state (with offset applied)
            obj_root_state = self.object.data.default_root_state.clone()
            obj_root_state[:, :3] = obj_trajectory_offset[:, :3, 3] + self.scene.env_origins
            obj_root_state[:, 3:7] = rotmat_to_quat(obj_trajectory_offset[:, :3, :3])
            obj_root_state[:, 7:10] = 0
            obj_root_state[:, 10:13] = 0
            self.object.write_root_state_to_sim(obj_root_state)

            # Step simulation
            self.sim.step(render=not self.headless)
            self.scene.update(self.sim.get_physics_dt())

            # Compute arm forward kinematics to get end effector position
            try:
                full_arm_joint_pos = torch.zeros(self.num_envs, 7, device=self.sim_device)
                full_arm_joint_pos[:, 0] = joint1_initial_pos
                full_arm_joint_pos[:, 1:7] = opt_arm_joint_pos_clamped

                all_chain_joint_names = self.arm_chain.get_joint_parameter_names()
                full_chain_joint_pos = torch.zeros(self.num_envs, len(all_chain_joint_names), device=self.sim_device)

                for i, arm_joint_name in enumerate(self.arm_joint_names):
                    if arm_joint_name in all_chain_joint_names:
                        chain_idx = all_chain_joint_names.index(arm_joint_name)
                        full_chain_joint_pos[:, chain_idx] = full_arm_joint_pos[:, i]

                arm_fk_result = self.arm_chain.forward_kinematics(full_chain_joint_pos)

                if self.arm_end_effector_link in arm_fk_result:
                    arm_ee_transform = arm_fk_result[self.arm_end_effector_link].get_matrix()
                else:
                    found_ee = False
                    for link_name in arm_fk_result.keys():
                        if "hand_c_mc" in link_name.lower() or "c_mc" in link_name.lower():
                            arm_ee_transform = arm_fk_result[link_name].get_matrix()
                            found_ee = True
                            break
                    if not found_ee:
                        last_link = list(arm_fk_result.keys())[-1]
                        arm_ee_transform = arm_fk_result[last_link].get_matrix()
                        print(f"Warning: End effector link {self.arm_end_effector_link} not found in Stage 1, using {last_link}")

                arm_ee_pos = arm_ee_transform[:, :3, 3]
                arm_ee_rotmat = arm_ee_transform[:, :3, :3]
                arm_base_pos_tensor = torch.tensor(
                    self.arm_base_pos, device=self.sim_device, dtype=torch.float32
                ).unsqueeze(0).repeat(self.num_envs, 1)
                arm_ee_pos_world = arm_ee_pos + arm_base_pos_tensor + self.scene.env_origins
            except Exception as e:
                import traceback
                print(f"Arm forward kinematics error: {e}")
                print(traceback.format_exc())
                arm_ee_pos_world = target_wrist_pos_offset + self.scene.env_origins
                arm_ee_pos_world = arm_ee_pos_world.detach()
                arm_ee_rotmat = torch.eye(3, device=self.sim_device, dtype=torch.float32).unsqueeze(0).repeat(self.num_envs, 1, 1)

            # Stage 1 loss: arm end effector position and rotation
            target_wrist_pos_world = target_wrist_pos_offset + self.scene.env_origins
            arm_ee_pos_loss = torch.mean(torch.norm(arm_ee_pos_world - target_wrist_pos_world, dim=-1))
            arm_ee_rot_loss = torch.mean(self._compute_rotation_error(arm_ee_rotmat, target_wrist_rot))

            # Temporal smoothness: penalize arm joint velocity (demo fps = 30 Hz)
            if self.num_envs > 1:
                arm_vel = torch.gradient(opt_arm_joint_pos, spacing=1.0 / 30.0, dim=0)[0]
                arm_vel_loss = torch.mean(arm_vel ** 2)
            else:
                arm_vel_loss = torch.zeros((), device=self.sim_device)

            rot_loss_weight = 0.25
            arm_ee_pos_loss_weight = 0.5
            arm_vel_weight = 1e-3
            loss = (arm_ee_pos_loss_weight * arm_ee_pos_loss
                    + rot_loss_weight * arm_ee_rot_loss
                    + arm_vel_weight * arm_vel_loss)

            if iter % 100 == 0:
                print(f"Stage 1 - Iter {iter}/{stage1_iter}: arm_ee_pos_loss={arm_ee_pos_loss.item():.6f}, arm_ee_rot_loss={arm_ee_rot_loss.item():.6f}, arm_vel_loss={arm_vel_loss.item():.6f}, total_loss={loss.item():.6f}")
                print(f"  arm_ee_pos_world[0]={arm_ee_pos_world[0].detach().cpu().numpy()}, target_wrist_pos_offset[0]={target_wrist_pos_offset[0].detach().cpu().numpy()}")
                print(f"  arm_joint_pos[0]={full_arm_joint_pos[0].detach().cpu().numpy()}")

            opti_stage1.zero_grad()
            loss.backward()
            opti_stage1.step()

            if iter % 100 == 0:
                if iter > 1 and stage1_past_loss - loss.item() < 1e-5:
                    print(f"Stage 1 converged early at iteration {iter}")
                    break
                stage1_past_loss = loss.item()

        # ============================================================
        # Stage 2: Optimize hand and arm together
        # ============================================================
        print(f"\n=== Stage 2: Hand + Arm joint optimization (iterations {iter+1}-{max_iter}) ===")
        print(f"Stage 2 target: target_wrist_pos_offset (with XY offset applied) for both hand and arm")

        # Clamp helpers
        if self.dexhand_dof_lower_limits.ndim > 1:
            hand_lower_limits = self.dexhand_dof_lower_limits[0]
        else:
            hand_lower_limits = self.dexhand_dof_lower_limits
        if self.dexhand_dof_upper_limits.ndim > 1:
            hand_upper_limits = self.dexhand_dof_upper_limits[0]
        else:
            hand_upper_limits = self.dexhand_dof_upper_limits

        # Tighten hand joint limits to the real-hand reachable range (measured
        # on the Sharpa Wave hand) so IK
        # optimization searches inside the deployable set. Real limits are in
        # cfg order, which matches dexhand_dof_lower/upper_limits layout (built
        # from hand_joint_indices which follow _build_hand_joint_names order).
        if (not args_cli.no_real_hand_clamp) and args_cli.dexhand == "sharpa" \
                and hand_lower_limits.shape[0] == 22:
            _real_lower = torch.as_tensor(SHARPA_REAL_LIMITS_NP[:, 0],
                                          dtype=hand_lower_limits.dtype,
                                          device=hand_lower_limits.device)
            _real_upper = torch.as_tensor(SHARPA_REAL_LIMITS_NP[:, 1],
                                          dtype=hand_upper_limits.dtype,
                                          device=hand_upper_limits.device)
            hand_lower_limits = torch.maximum(hand_lower_limits, _real_lower)
            hand_upper_limits = torch.minimum(hand_upper_limits, _real_upper)
            print(f"[retarget_2stage] Tightened optimization-loop hand limits to real "
                  f"real-hand reachable range (measured on the Sharpa Wave hand, cfg order).")

        while iter < max_iter:
            iter += 1

            # Set object state
            obj_root_state = self.object.data.default_root_state.clone()
            obj_root_state[:, :3] = obj_trajectory_offset[:, :3, 3] + self.scene.env_origins
            obj_root_state[:, 3:7] = rotmat_to_quat(obj_trajectory_offset[:, :3, :3])
            obj_root_state[:, 7:10] = 0
            obj_root_state[:, 10:13] = 0

            # Clamp hand DOF positions
            opt_hand_dof_pos_clamped = torch.clamp(
                opt_hand_dof_pos,
                hand_lower_limits[None, :],
                hand_upper_limits[None, :],
            )

            # Clamp arm joint positions (joints 2-7 only)
            opt_arm_joint_pos_clamped = torch.clamp(
                opt_arm_joint_pos,
                arm_joint2_7_lower[None, :],
                arm_joint2_7_upper[None, :],
            )
            # Joint 1 is free in Stage 2 — clamp it to its joint limits
            opt_joint1_clamped = torch.clamp(opt_joint1, arm_joint1_lower, arm_joint1_upper)

            # Combine arm and hand joint positions
            all_joint_pos = torch.zeros(self.num_envs, self.robot.num_joints, device=self.sim_device)
            all_joint_pos[:, self.arm_joint_indices[0]] = opt_joint1_clamped.detach()
            all_joint_pos[:, arm_joint2_7_indices] = opt_arm_joint_pos_clamped.detach()
            all_joint_pos[:, self.hand_joint_indices] = opt_hand_dof_pos_clamped.detach()

            # Set states in simulation
            self.robot.write_joint_state_to_sim(all_joint_pos.detach(), torch.zeros_like(all_joint_pos.detach()))
            self.robot.set_joint_position_target(all_joint_pos.detach())
            self.object.write_root_state_to_sim(obj_root_state)

            # Step simulation
            self.sim.step(render=not self.headless)
            self.scene.update(self.sim.get_physics_dt())

            # Use complete arm_chain for forward kinematics
            wrist_pos_from_fk = None

            all_chain_joint_names = self.arm_chain.get_joint_parameter_names()

            # Build full_chain_joint_pos preserving gradients
            full_chain_joint_pos_list = []
            for chain_idx, robot_joint_idx in enumerate(self.robot2arm_chain_order):
                if chain_idx >= len(all_chain_joint_names):
                    continue

                if robot_joint_idx < self.robot.num_joints:
                    if robot_joint_idx == self.arm_joint_indices[0]:
                        # Joint 1: free in Stage 2 — gradient flows through opt_joint1
                        full_chain_joint_pos_list.append(opt_joint1_clamped)
                    elif robot_joint_idx in arm_joint2_7_indices:
                        if isinstance(arm_joint2_7_indices, torch.Tensor):
                            arm_joint_local_idx = (arm_joint2_7_indices == robot_joint_idx).nonzero(as_tuple=True)[0][0].item()
                        else:
                            arm_joint_local_idx = list(arm_joint2_7_indices).index(robot_joint_idx)
                        full_chain_joint_pos_list.append(opt_arm_joint_pos_clamped[:, arm_joint_local_idx])
                    elif robot_joint_idx in self.hand_joint_indices:
                        if isinstance(self.hand_joint_indices, torch.Tensor):
                            hand_joint_local_idx = (self.hand_joint_indices == robot_joint_idx).nonzero(as_tuple=True)[0][0].item()
                        else:
                            hand_joint_local_idx = list(self.hand_joint_indices).index(robot_joint_idx)
                        full_chain_joint_pos_list.append(opt_hand_dof_pos_clamped[:, hand_joint_local_idx])
                    else:
                        full_chain_joint_pos_list.append(torch.zeros(self.num_envs, device=self.sim_device))
                else:
                    full_chain_joint_pos_list.append(torch.zeros(self.num_envs, device=self.sim_device))

            if len(full_chain_joint_pos_list) == len(all_chain_joint_names):
                full_chain_joint_pos = torch.stack(full_chain_joint_pos_list, dim=1)
            else:
                full_chain_joint_pos = torch.zeros(self.num_envs, len(all_chain_joint_names), device=self.sim_device)
                for i, pos_tensor in enumerate(full_chain_joint_pos_list):
                    if i < len(all_chain_joint_names):
                        full_chain_joint_pos[:, i] = pos_tensor

            # Forward kinematics using complete arm_chain
            arm_chain_fk_result = self.arm_chain.forward_kinematics(full_chain_joint_pos)

            arm_base_pos_tensor = torch.tensor(
                self.arm_base_pos, device=self.sim_device, dtype=torch.float32
            ).unsqueeze(0).repeat(self.num_envs, 1)

            # Extract hand body positions from arm_chain FK result
            pk_joints_list = []
            for body_name in self.dexhand.body_names:
                found_body = None
                if body_name in arm_chain_fk_result:
                    found_body = body_name
                else:
                    for chain_body_name in arm_chain_fk_result.keys():
                        if body_name.lower() in chain_body_name.lower() or chain_body_name.lower() in body_name.lower():
                            found_body = chain_body_name
                            break

                if found_body is not None and found_body in arm_chain_fk_result:
                    body_pos = arm_chain_fk_result[found_body].get_matrix()[:, :3, 3]
                    pk_joints_list.append(body_pos)
                else:
                    if iter == stage1_iter + 1:
                        print(f"Warning: Body {body_name} not found in arm_chain FK result")

            if len(pk_joints_list) == 0:
                raise RuntimeError("No hand bodies found in arm_chain FK result")

            pk_joints = torch.stack(pk_joints_list, dim=1)
            pk_joints = pk_joints + arm_base_pos_tensor.unsqueeze(1)

            # Get end effector link position and rotation
            ee_link_transform = None
            if self.arm_end_effector_link in arm_chain_fk_result:
                ee_link_transform = arm_chain_fk_result[self.arm_end_effector_link].get_matrix()
                if iter == stage1_iter + 1:
                    print(f"Found end effector link '{self.arm_end_effector_link}' in FK result")
            else:
                found_ee = False
                if iter == stage1_iter + 1:
                    print(f"End effector link '{self.arm_end_effector_link}' not found, searching by pattern...")

                for link_name in arm_chain_fk_result.keys():
                    if "hand_c_mc" in link_name.lower() or "c_mc" in link_name.lower():
                        ee_link_transform = arm_chain_fk_result[link_name].get_matrix()
                        found_ee = True
                        if iter == stage1_iter + 1:
                            print(f"Found end effector link by pattern: '{link_name}'")
                        break

                if not found_ee:
                    for link_name in arm_chain_fk_result.keys():
                        if "link8" in link_name.lower() or "end" in link_name.lower() or "ee" in link_name.lower():
                            ee_link_transform = arm_chain_fk_result[link_name].get_matrix()
                            found_ee = True
                            if iter == stage1_iter + 1:
                                print(f"Found end effector link by fallback pattern: '{link_name}'")
                            break

                if not found_ee:
                    last_link = list(arm_chain_fk_result.keys())[-1]
                    ee_link_transform = arm_chain_fk_result[last_link].get_matrix()
                    if iter == stage1_iter + 1:
                        print(f"Warning: End effector link '{self.arm_end_effector_link}' not found, using last link: '{last_link}'")

            wrist_pos_from_fk = ee_link_transform[:, :3, 3]
            wrist_rotmat_from_fk = ee_link_transform[:, :3, :3]

            opt_wrist_pos = wrist_pos_from_fk.detach()
            opt_wrist_rot = rotmat_to_rot6d(wrist_rotmat_from_fk).detach()

            arm_ee_pos_world = wrist_pos_from_fk + arm_base_pos_tensor + self.scene.env_origins

            # Get current joint positions from simulation
            body_indices = [self.dexhand_handles[k] for k in self.dexhand.body_names if k in self.dexhand_handles]
            isaac_joints = self.robot.data.body_pos_w[:, body_indices]
            isaac_joints = isaac_joints - self.scene.env_origins.unsqueeze(1)

            # Compute combined loss
            target_joints = torch.cat([target_wrist_pos_offset[:, None], target_mano_joints_offset], dim=1)
            hand_loss = torch.mean(torch.norm(pk_joints - target_joints, dim=-1) * weight[None])

            target_wrist_pos_world = target_wrist_pos_offset + self.scene.env_origins
            arm_ee_pos_loss = torch.mean(torch.norm(arm_ee_pos_world - target_wrist_pos_world, dim=-1))
            arm_ee_rot_loss = torch.mean(self._compute_rotation_error(wrist_rotmat_from_fk, target_wrist_rot))

            # Temporal smoothness: penalize joint velocity (demo fps = 30 Hz);
            # arm velocity includes the now-free joint 1.
            if self.num_envs > 1:
                arm_vel = torch.gradient(opt_arm_joint_pos, spacing=1.0 / 30.0, dim=0)[0]
                j1_vel = torch.gradient(opt_joint1, spacing=1.0 / 30.0, dim=0)[0]
                hand_vel = torch.gradient(opt_hand_dof_pos, spacing=1.0 / 30.0, dim=0)[0]
                arm_vel_loss = torch.mean(arm_vel ** 2) + torch.mean(j1_vel ** 2)
                hand_vel_loss = torch.mean(hand_vel ** 2)
            else:
                arm_vel_loss = torch.zeros((), device=self.sim_device)
                hand_vel_loss = torch.zeros((), device=self.sim_device)

            arm_loss_weight = 0.05
            rot_loss_weight = 0.05
            arm_vel_weight = 1e-3
            hand_vel_weight = 1e-4
            loss = (hand_loss
                    + arm_loss_weight * arm_ee_pos_loss
                    + rot_loss_weight * arm_ee_rot_loss
                    + arm_vel_weight * arm_vel_loss
                    + hand_vel_weight * hand_vel_loss)

            if iter % 100 == 0:
                full_arm_joint_pos_debug = torch.zeros(self.num_envs, 7, device=self.sim_device)
                full_arm_joint_pos_debug[:, 0] = opt_joint1_clamped.detach()
                full_arm_joint_pos_debug[:, 1:7] = opt_arm_joint_pos_clamped.detach()

                wrist_pos_str = "N/A"
                if wrist_pos_from_fk is not None:
                    wrist_pos_str = str(wrist_pos_from_fk[0].detach().cpu().numpy())

                print(f"Stage 2 - Iter {iter}/{max_iter}: hand_loss={hand_loss.item():.6f}, arm_ee_pos_loss={arm_ee_pos_loss.item():.6f}, arm_ee_rot_loss={arm_ee_rot_loss.item():.6f}, arm_vel={arm_vel_loss.item():.6f}, hand_vel={hand_vel_loss.item():.6f}, total_loss={loss.item():.6f}")
                print(f"  wrist_pos_from_fk[0]={wrist_pos_str}, target_wrist_pos_offset[0]={target_wrist_pos_offset[0].detach().cpu().numpy()}")
                print(f"  arm_joint_pos[0]={full_arm_joint_pos_debug[0].detach().cpu().numpy()}")

            # Visualize reference hand pose
            if not self.headless and iter % 10 == 0:
                self._visualize_reference_pose(target_joints)

            # Optimize
            opti_stage2.zero_grad()
            loss.backward()
            opti_stage2.step()

            if iter % 100 == 0:
                if iter > stage1_iter + 1 and past_loss - loss.item() < 1e-5:
                    break
                past_loss = loss.item()

        # Get final joint positions
        self.sim.step(render=False)

        final_all_joint_pos = torch.zeros(self.num_envs, self.robot.num_joints, device=self.sim_device)
        final_all_joint_pos[:, self.arm_joint_indices[0]] = opt_joint1_clamped.detach()
        final_all_joint_pos[:, arm_joint2_7_indices] = opt_arm_joint_pos_clamped.detach()
        final_all_joint_pos[:, self.hand_joint_indices] = opt_hand_dof_pos_clamped.detach()

        body_indices = [self.dexhand_handles[k] for k in self.dexhand.body_names if k in self.dexhand_handles]
        if len(body_indices) > 0:
            isaac_joints = self.robot.data.body_pos_w[:, body_indices]
            isaac_joints = isaac_joints - self.scene.env_origins.unsqueeze(1)
        else:
            isaac_joints = self.robot.data.body_pos_w - self.scene.env_origins.unsqueeze(1)

        final_hand_dof_pos = final_all_joint_pos[:, self.hand_joint_indices]

        # Compute per-frame final loss at converged solution for downstream analysis/debugging.
        target_joints_final = torch.cat([target_wrist_pos_offset[:, None], target_mano_joints_offset], dim=1)
        final_joint_dist = torch.norm(pk_joints - target_joints_final, dim=-1)
        final_hand_loss_per_env = torch.mean(final_joint_dist * weight[None], dim=1)

        target_wrist_pos_world_final = target_wrist_pos_offset + self.scene.env_origins
        final_arm_ee_pos_loss_per_env = torch.norm(arm_ee_pos_world - target_wrist_pos_world_final, dim=-1)
        final_arm_ee_rot_loss_per_env = self._compute_rotation_error(wrist_rotmat_from_fk, target_wrist_rot)

        arm_loss_weight = 0.05
        rot_loss_weight = 0.05
        final_total_loss_per_env = (
            final_hand_loss_per_env
            + arm_loss_weight * final_arm_ee_pos_loss_per_env
            + rot_loss_weight * final_arm_ee_rot_loss_per_env
        )

        # opt_dof_pos is in cfg (Sharpa) order. On the real hand (range measured
        # on the Sharpa Wave hand), the URDF clamp above is looser than what the motors can actually reach
        # (esp. ring/pinky MCP_AA due to inter-finger coupling). Apply a second
        # clamp based on empirically measured real-hand range so sim and deploy
        # see the same target distribution.
        opt_dof_pos_np = final_hand_dof_pos.detach().cpu().numpy()
        apply_real_clamp = (
            (not args_cli.no_real_hand_clamp)
            and args_cli.dexhand == "sharpa"
            and opt_dof_pos_np.shape[-1] == 22
        )
        if apply_real_clamp:
            pre_clamp = opt_dof_pos_np.copy()
            opt_dof_pos_np = clamp_to_real_limits_np(opt_dof_pos_np)
            diff = np.abs(opt_dof_pos_np - pre_clamp)
            n_frames_clipped = int((diff.max(axis=-1) > 1e-4).sum())
            if n_frames_clipped > 0:
                clip_per_joint = (diff > 1e-4).reshape(-1, 22).sum(axis=0)
                max_clip_per_joint = diff.reshape(-1, 22).max(axis=0)
                print(f"[retarget_2stage] real-hand clamp: {n_frames_clipped} / "
                      f"{int(np.prod(pre_clamp.shape[:-1]))} frames had >= 1 joint clipped")
                for j in range(22):
                    if clip_per_joint[j] > 0:
                        print(f"  joint[{j:2d}]: {int(clip_per_joint[j])} frames clipped, "
                              f"max_clip={max_clip_per_joint[j]:.4f} rad "
                              f"(real limits: [{SHARPA_REAL_LIMITS_NP[j,0]:.3f}, {SHARPA_REAL_LIMITS_NP[j,1]:.3f}])")

        to_dump = {
            "opt_wrist_pos": opt_wrist_pos.detach().cpu().numpy(),
            "opt_wrist_rot": rot6d_to_aa(opt_wrist_rot).detach().cpu().numpy(),
            "opt_dof_pos": opt_dof_pos_np,
            "opt_dof_pos_unclamped": final_hand_dof_pos.detach().cpu().numpy(),
            "real_hand_clamp_applied": bool(apply_real_clamp),
            "opt_arm_joint_pos": final_all_joint_pos[:, self.arm_joint_indices].detach().cpu().numpy(),
            "opt_joints_pos": isaac_joints.detach().cpu().numpy(),
            "opt_final_loss_per_frame": final_total_loss_per_env.detach().cpu().numpy(),
            "opt_final_loss_mean": final_total_loss_per_env.mean().item(),
            "opt_final_hand_loss_per_frame": final_hand_loss_per_env.detach().cpu().numpy(),
            "opt_final_arm_pos_loss_per_frame": final_arm_ee_pos_loss_per_env.detach().cpu().numpy(),
            "opt_final_arm_rot_loss_per_frame": final_arm_ee_rot_loss_per_env.detach().cpu().numpy(),
            "xy_offset": offset_xy.detach().cpu().numpy(),
            "z_offset": offset_z.detach().cpu().numpy(),
            "arm_base_pos": self.arm_base_pos,
            "side": self.side_cfg["hand_joint_prefix"],
            "obj_scale": self.obj_scale,
            # NOTE: do NOT dump 'obj_traj' — the training data loader rebuilds
            # obj_trajectory from the source data and conflicts with a saved one.
        }

        return to_dump

    def _visualize_reference_pose(self, target_joints):
        """Visualize reference hand pose using debug draw.

        target_joints: (N, 33, 3) in `dexhand.body_names` order
            (wrist == body_names[0]) + body_names[1:] non-wrist bodies).
        Uses `gym_bone_links`, whose indices are in `body_names` order;
        `bone_links` uses a different layered indexing and would produce
        scrambled bones here.
        """
        self.debug_draw.clear()

        bone_links = getattr(self.dexhand, "gym_bone_links", None)

        # Visualize env 0 plus every `debug_viz_stride`-th env after that.
        # debug_viz_stride=1 (default) → env 0 only: special-cased, since
        # range(0, num_envs, 1) would draw every env.
        if self.debug_viz_stride <= 1:
            viz_env_ids = [0] if self.num_envs > 0 else []
        else:
            viz_env_ids = list(range(0, self.num_envs, self.debug_viz_stride))
        for env_id in viz_env_ids:
            mano_joints = target_joints[env_id]

            if bone_links is not None:
                for link in bone_links:
                    parent_idx, child_idx = link[0], link[1]
                    if parent_idx < mano_joints.shape[0] and child_idx < mano_joints.shape[0]:
                        parent_pos = mano_joints[parent_idx] + self.scene.env_origins[env_id]
                        child_pos = mano_joints[child_idx] + self.scene.env_origins[env_id]
                        line_points = torch.stack([parent_pos, child_pos], dim=0)
                        self.debug_draw.plot(line_points, size=2.0, color=(0.0, 1.0, 0.0, 1.0))

            joint_points = mano_joints + self.scene.env_origins[env_id]
            self.debug_draw.point(joint_points, color=(1.0, 0.0, 0.0, 1.0), size=8.0)

    def _identify_joints(self):
        """Identify arm and hand joint indices."""
        all_joint_names = self.robot.joint_names

        self.arm_joint_names = [f"fr3_joint{i}" for i in range(1, 8)]
        self.arm_joint_indices = []
        for joint_name in self.arm_joint_names:
            if joint_name in all_joint_names:
                self.arm_joint_indices.append(all_joint_names.index(joint_name))

        # Hand joints — prefix depends on side
        prefix = self.side_cfg["hand_joint_prefix"]
        self.hand_joint_names = _build_hand_joint_names(prefix)
        self.hand_joint_indices = []
        for joint_name in self.hand_joint_names:
            if joint_name in all_joint_names:
                self.hand_joint_indices.append(all_joint_names.index(joint_name))


if __name__ == "__main__":

    side = args_cli.side
    side_cfg = SIDE_CONFIGS[side]

    # Create dexhand
    dexhand = DexHandFactory.create_hand(args_cli.dexhand, side)

    def run(parser, idx):
        """Run retargeting for a single data index."""
        dataset_type = ManipDataFactory.dataset_type(idx)
        demo_d = ManipDataFactory.create_data(
            manipdata_type=dataset_type,
            side=parser.side,
            device=parser.sim_device,
            mujoco2gym_transf=torch.eye(4, device=parser.sim_device),
            dexhand=dexhand,
            verbose=False,
        )

        demo_data = pack_data([demo_d[idx]], dexhand)
        parser.num_envs = demo_data["mano_joints"].shape[0]

        # Object path (relative to project root)
        obj_path = os.path.join(PROJECT_ROOT, demo_data['obj_urdf_path'][0])

        print(f"Dataset type: {dataset_type}")

        obj_scale = parser.obj_scale if hasattr(parser, 'obj_scale') else 1.0
        print(f"Object scale: {obj_scale}")
        mano2dexhand = Mano2Dexhand(parser, dexhand, obj_path, side_cfg, obj_scale=obj_scale)

        # Get object vertices if available
        obj_verts = None
        if "obj_verts" in demo_data:
            obj_verts = demo_data["obj_verts"]
            if isinstance(obj_verts, list):
                obj_verts = obj_verts[0]
            if isinstance(obj_verts, torch.Tensor):
                obj_verts = obj_verts.to(parser.sim_device)

        tip_list = ["thumb_fingertip", "index_fingertip", "middle_fingertip", "ring_fingertip", "pinky_fingertip"]

        # Use CLI overrides for offsets if provided
        target_offset_xy_override = parser.target_offset_xy if hasattr(parser, 'target_offset_xy') else None
        z_offset_value = parser.z_offset if hasattr(parser, 'z_offset') else 0.0

        # Augmentation only targets the robotool_batch pipeline.
        if dataset_type != "robotool_batch":
            raise ValueError(
                f"retarget.py only supports robotool_batch (got '{dataset_type}').")
        parts = idx.split("/")
        rt_task, rt_exp = parts[1], parts[2].split("@")[0]
        base_result = idx.split("@", 1)[1] if "@" in idx else "0"

        # Variant list: original first, then sampled augmentations.
        variants = [((0.0, 0.0), 0.0, base_result)]
        for (dx, dy, yaw) in generate_aug_params(
                parser.aug_num, parser.aug_radius, parser.aug_yaw_deg, parser.aug_seed):
            variants.append(((dx, dy), yaw, f"{base_result}_{aug_tag(dx, dy, yaw)}"))
        print(f"[AUG] {len(variants)} run(s): original + {len(variants) - 1} aug(s)")

        out_root = (os.path.join(parser.dump_root, rt_task) if parser.dump_root
                    else f"data/retargeting/robotool_batch/mano2{str(dexhand)}/{rt_task}")

        # --skip_existing: drop variants whose target pkl already exists, so
        # incremental runs (new --aug_seed) only retarget the NEW augs.
        if parser.skip_existing:
            _kept = []
            for (_axy, _ayw, _res) in variants:
                _p = os.path.join(out_root, f"{rt_exp}@{_res}.pkl")
                if os.path.exists(_p):
                    print(f"  [skip-existing] already exists: {_p}")
                else:
                    _kept.append((_axy, _ayw, _res))
            print(f"[AUG] after --skip_existing filter: {len(_kept)}/{len(variants)} variants to retarget")
            variants = _kept
            if not variants:
                print("[AUG] nothing to do (all variants already exist).")
                return

        for vi, (aug_xy, aug_yaw, result) in enumerate(variants):
            print(f"\n{'=' * 60}")
            print(f"[AUG {vi}/{len(variants) - 1}] result='{result}'  "
                  f"aug_xy={aug_xy}  yaw={aug_yaw:.1f} deg")
            print(f"{'=' * 60}")

            to_dump = mano2dexhand.fitting(
                parser.iter,
                demo_data["obj_trajectory"],
                demo_data["wrist_pos"],
                demo_data["wrist_rot"],
                demo_data["mano_joints"].view(parser.num_envs, -1, 3),
                obj_verts=obj_verts,
                tip_list=tip_list,
                target_offset_xy_override=target_offset_xy_override,
                z_offset_value=z_offset_value,
                aug_xy_delta=aug_xy,
                aug_yaw_deg=aug_yaw,
            )

            # Reachability gate: mean arm EE position error over the trajectory.
            arm_err = np.asarray(to_dump["opt_final_arm_pos_loss_per_frame"]).reshape(-1)
            mean_err = float(arm_err.mean())
            max_err = float(arm_err.max())
            if mean_err <= parser.reachability_th:
                to_dump["reachable"] = True
                to_dump["aug_xy_delta"] = list(aug_xy)
                to_dump["aug_yaw_deg"] = float(aug_yaw)
                to_dump["aug_arm_ee_loss"] = mean_err
                to_dump["ik_mean_err"] = mean_err
                to_dump["ik_max_err"] = max_err
                print(f"  [OK] reachable: mean arm EE err = {mean_err:.4f} m "
                      f"<= {parser.reachability_th}")
            else:
                print(f"  [SKIP] unreachable: mean arm EE err = {mean_err:.4f} m "
                      f"> {parser.reachability_th} -> partial pkl")
                to_dump = make_partial_result(
                    mano2dexhand.arm_base_pos, to_dump["xy_offset"], to_dump["z_offset"],
                    aug_xy, aug_yaw, mean_err, max_err,
                )

            dump_path = os.path.join(out_root, f"{rt_exp}@{result}.pkl")
            os.makedirs(os.path.dirname(dump_path), exist_ok=True)

            # Never let an unreachable partial clobber a converged result. A short
            # --iter run does not converge, so a quickstart invocation would
            # otherwise overwrite the shipped example demo with a partial.
            # Reachable results always overwrite.
            if not to_dump.get("reachable", True) and os.path.exists(dump_path):
                try:
                    with open(dump_path, "rb") as f:
                        _prev_reachable = pickle.load(f).get("reachable", False)
                except Exception:
                    _prev_reachable = False
                if _prev_reachable and not parser.force_overwrite:
                    print(f"  [KEEP] {dump_path} already holds a REACHABLE result; "
                          f"refusing to overwrite it with this unreachable partial "
                          f"(mean arm EE err = {mean_err:.4f} m). Raise --iter, or pass "
                          f"--force_overwrite / --dump_root to write anyway.")
                    continue

            with open(dump_path, "wb") as f:
                pickle.dump(to_dump, f)
            print(f"  saved -> {dump_path}")

    # Resolve data indices: --task expands to all exps, --data_idx with task-only also expands
    data_indices = []
    if args_cli.task:
        data_indices = [f"rt/{args_cli.task}"]
    elif args_cli.data_idx:
        data_indices = [s for s in args_cli.data_idx.split(",") if s.strip()]
    else:
        parser.error("Must specify --data_idx or --task")

    # Expand task-only and glob patterns
    from dexx.tasks.hand_imitation.dataset.robotool_batch_dataset_dexhand import expand_rt_indices
    expanded = expand_rt_indices(data_indices)
    if len(expanded) != len(data_indices) or expanded != data_indices:
        data_indices = expanded
        print(f"[INFO] Expanded to {len(data_indices)} sequences: {data_indices}")

    for i, idx in enumerate(data_indices):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(data_indices)}] Retargeting: {idx}")
        print(f"{'='*60}")
        run(args_cli, idx)

    # Release the simulation context before the app, as env.close() does for
    # the gym scripts: without it simulation_app.close() hangs after the
    # results are already on disk.
    _sim = sim_utils.SimulationContext.instance()
    if _sim is not None:
        _sim.clear_all_callbacks()
        _sim.clear_instance()
    simulation_app.close()
