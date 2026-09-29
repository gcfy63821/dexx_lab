"""Deploy a PointCloud student on the real robot.

Arm through the NUC Polymetis joint bridge (ZMQ), depth over ZMQ from the
camera host, hand and tactile through the Sharpa SDK. Optionally (experimental,
not validated on the robot) the arm and/or depth go through ROS2 instead:
`--arm_backend ros2`, `--depth_backend ros2` (docs/DEPLOY.md, "ROS2 backend"). Deploys DAgger
PointCloudStudent checkpoints only (train_ppo_pc.py checkpoints read the sim-only
object-pose observation and are refused). See docs/DEPLOY.md for the full
startup sequence.

Pre-flight:
  1) NUC: Polymetis server + `python deploy/polymetis_joint_bridge.py`
  2) Camera host: `python deploy/realsense_depth_zmq_pub.py` (PUB :5562)
  3) (optional) `python deploy/move_to_frame_polymetis.py --ip <NUC_IP> --pkl ... --frame 0 --hold`

Usage (the shipped lean student):
    python deploy/deploy_pc.py \\
        --load_path checkpoints/student_lean_v6_L1.pth --side right \\
        --data_idx '["rt/0416_grasp/cube_small_2"]' \\
        --camera_extrinsic calib/camera_align/current.npy \\
        --pc_workspace_min 0.0,-0.40,0.422 \\
        --polymetis_ip <NUC_IP> --depth_zmq_addr tcp://<CAM_HOST>:5562 --headless

ROS2 backend (source /opt/ros/humble/setup.bash first; same ROS_DOMAIN_ID as the
robot PC; `python deploy/ros2/wrist_state_publisher.py` running): replace the last
line with
        --arm_backend ros2 --depth_backend ros2 --headless
"""
import argparse
import ast
import json
import os
import sys

from dexx import deploy_config as _dcfg  # single source of truth for ports/addrs

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Deploy a PointCloud student on the real robot.")
parser.add_argument("--task", type=str, default="franka-sharpa-pointcloud-polymetis-deploy")
parser.add_argument("--load_path", type=str, required=True)
parser.add_argument("--side", type=str, default="right")
parser.add_argument("--data_idx", type=str, default=None,
                    help="JSON list of demo idx — the env reads the reference wrist/joint "
                         "targets and the object's init pose from these demos.")
parser.add_argument("--max_steps", type=int, default=10_000_000)
parser.add_argument("--camera_extrinsic", type=str, required=True,
                    help="4x4 .npy camera-in-armbase transform (ROS optical). Use the one "
                         "the student was trained with — see calib/camera_align/README.md.")
parser.add_argument("--depth_height", type=int, default=_dcfg.DEPTH_H)
parser.add_argument("--depth_width", type=int, default=_dcfg.DEPTH_W)

# Env-side PC transforms. All default to None = "take it from the ckpt's
# pc_env_meta" (see algo/dagger/pc_env_meta.py). Pass one only to deliberately
# deviate from training — e.g. --pc_ablate_tactile_force for a real-robot
# tactile ablation of an otherwise unchanged policy.
parser.add_argument("--pc_ablate_tactile_pc", action="store_true", default=None)
parser.add_argument("--pc_ablate_tactile_force", action="store_true", default=None)
parser.add_argument("--pc_force_repr", type=str, default=None, choices=["scalar", "binary"])
parser.add_argument("--pc_tactile_force_gate", type=float, default=None,
                    help="If >0, gate tactile points whose force < this.")
parser.add_argument("--pc_tactile_gate_mode", type=str, default=None,
                    choices=["zero", "mask"])
parser.add_argument("--pc_force_scale", type=float, default=None,
                    help="Divisor applied to tactile_force. Restored from ckpt if unset. "
                         "MUST match training or the real hand's forces land in a "
                         "range the policy never saw.")
parser.add_argument("--pc_tactile_use_vec3", action="store_true", default=None,
                    help="Enable 3D tactile force (feat_dim=3). Restored from ckpt if unset.")
parser.add_argument("--no_contact_force", action="store_true", default=None,
                    help="Zero the 5d proprio contact force. Restored from ckpt if unset.")
parser.add_argument("--no_tactile", action="store_true", default=None,
                    help="Zero the whole 20d proprio tactile tail. Restored from ckpt if unset.")
# Crop box: restored from the ckpt; override per robot setup.
parser.add_argument("--pc_workspace_min", type=str, default=None,
                    help='Crop box min "x,y,z" (env-local), e.g. "0.0,-0.40,0.422": '
                         'the real table can sit a few mm higher in the cloud than in '
                         'sim, so raise z_min if table points get in.')
parser.add_argument("--pc_workspace_max", type=str, default=None,
                    help='Crop box max "x,y,z" (env-local).')
parser.add_argument("--pc_no_crop", action="store_true",
                    help="Disable the crop (box = ±10 m) to inspect what the camera sees. "
                         "Heavily out of distribution for the policy.")

# --- Action safety ramp ---
parser.add_argument("--action_ramp_steps", type=int, default=15,
                    help="Scale the action from 0 to 1 (smoothstep) over the first N policy "
                         "steps, against a large joint velocity right after reset. Default "
                         "15 (0.5 s at 30 Hz); 0 = off. The whole action is scaled, so the "
                         "hand targets move toward mid-range while it ramps.")

# ---- Arm backend ----
parser.add_argument("--arm_backend", choices=["polymetis", "ros2"], default="polymetis",
                    help="polymetis (reference) or ros2 (experimental: ros2_control "
                         "joint-impedance controller, gains in its YAML).")
# ---- Arm: Polymetis (NUC joint bridge over ZMQ) ----
parser.add_argument("--polymetis_ip", type=str, default=None,
                    help="IP of the NUC running polymetis_joint_bridge.py (wired link). "
                         "Required with --arm_backend polymetis.")
parser.add_argument("--polymetis_kq", type=str, default=None,
                    help='7 comma-separated joint-impedance stiffnesses, e.g. '
                         '"200,200,200,200,100,100,50". Unset = Polymetis defaults.')
parser.add_argument("--polymetis_kqd", type=str, default=None,
                    help='7 comma-separated joint-impedance dampings; must be given with --polymetis_kq.')
parser.add_argument("--polymetis_state_port", type=int, default=_dcfg.POLYMETIS_STATE_PORT)
parser.add_argument("--polymetis_cmd_port", type=int, default=_dcfg.POLYMETIS_CMD_PORT)
# ---- Arm: ROS2 ----
parser.add_argument("--ros2_namespace", type=str, default="",
                    help="Prefix of /joint_states and /franka_wrist_state (empty on the reference setup).")
# ---- Depth ----
parser.add_argument("--depth_backend", choices=["zmq", "ros2"], default="zmq")
parser.add_argument("--depth_zmq_addr", type=str, default=None,
                    help="ZMQ addr of the camera-host depth publisher, e.g. tcp://<CAM_HOST>:5562. "
                         "Required with --depth_backend zmq.")
parser.add_argument("--depth_topic", type=str, default="/camera/camera/depth/image_rect_raw",
                    help="ROS2 depth Image topic (--depth_backend ros2): 320x240, 16UC1 or 32FC1.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

if args_cli.arm_backend == "polymetis" and not args_cli.polymetis_ip:
    raise SystemExit("--polymetis_ip is required with --arm_backend polymetis")
if args_cli.depth_backend == "zmq" and not args_cli.depth_zmq_addr:
    raise SystemExit("--depth_zmq_addr is required with --depth_backend zmq")
if args_cli.arm_backend == "ros2" and (args_cli.polymetis_kq or args_cli.polymetis_kqd):
    raise SystemExit("--polymetis_kq/--polymetis_kqd do not apply to --arm_backend ros2: "
                     "its gains are in the robot-side controller YAML")
if not os.path.isfile(args_cli.camera_extrinsic):
    raise SystemExit(f"--camera_extrinsic not found: {args_cli.camera_extrinsic}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ------------------------------------------------------------------------- #
import time
import importlib

import numpy as np
import torch
import gymnasium as gym
from omegaconf import OmegaConf

import dexx.tasks.franka_sharpa  # noqa: F401

from dexx.algo.dagger.pc_env_meta import (
    align_pc_dims_to_ckpt, apply_pc_env_meta, is_dagger_student_ckpt, load_checkpoint,
)
from dexx.scripts.deploy.realsense_depth_zmq_subscriber import RealSenseDepthZmqSubscriber

_USE_ROS2 = "ros2" in (args_cli.arm_backend, args_cli.depth_backend)
if _USE_ROS2:
    try:
        import rclpy
    except ImportError as exc:
        raise SystemExit(f"a ROS2 backend was selected but rclpy is not importable ({exc!r}); "
                         f"`source /opt/ros/humble/setup.bash` first")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_entry_point(entry_point: str):
    module, target = entry_point.split(":")
    if target.endswith("Cfg") or target[0].isupper():
        mod = importlib.import_module(module)
        return getattr(mod, target)()
    if target.endswith(".yaml") or target.endswith(".yml"):
        mod = importlib.import_module(module)
        base_dir = os.path.dirname(mod.__file__)
        return OmegaConf.load(os.path.join(base_dir, target))
    raise ValueError(entry_point)


def _build_dagger_student(ckpt, device):
    from dexx.algo.dagger.pointcloud_student import PointCloudStudent
    # PointNet per-point input dim = 3 (xyz) + type_dim + tactile_feat_dim;
    # type_dim is 1 for "scalar" and 3 for "onehot" and must match training.
    pc_cfg = dict(
        fusion_strategy=ckpt["pc_fusion_strategy"],
        n_scene=ckpt["n_scene"], n_hand=ckpt["n_hand"], n_tactile=ckpt["n_tactile"],
        tactile_feat_dim=ckpt["tactile_feat_dim"], output_dim=ckpt["pc_output_dim"],
        ablate_tactile_pc=ckpt.get("ablate_tactile_pc", False),
        type_repr=ckpt.get("pc_type_repr", "scalar"),
    )
    model = PointCloudStudent(
        proprio_dim=ckpt["proprio_dim"], action_dim=ckpt["action_dim"],
        pc_encoder_cfg=pc_cfg,
        hidden=tuple(ckpt.get("student_hidden", (512, 256))),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, "DAgger PointCloudStudent"


def _parse_xyz(s: str | None):
    if s is None:
        return None
    xyz = tuple(float(x) for x in s.split(","))
    if len(xyz) != 3:
        raise SystemExit(f'crop box corner needs "x,y,z", got {s!r}')
    return xyz


def _parse7(v, name):
    if v is None:
        return None
    xs = [float(x) for x in v.split(",")]
    if len(xs) != 7:
        raise ValueError(f"{name} needs 7 values, got {len(xs)}")
    return xs


def main():
    # ---- Env cfg
    spec = gym.spec(args_cli.task)
    env_cfg = parse_entry_point(spec.kwargs["env_cfg_entry_point"])
    env_cfg.scene.num_envs = 1
    if args_cli.device:
        env_cfg.sim.device = args_cli.device
    env_cfg.hand_side = args_cli.side
    if args_cli.data_idx:
        try:
            di = json.loads(args_cli.data_idx)
        except json.JSONDecodeError:
            di = ast.literal_eval(args_cli.data_idx)
        env_cfg.data_indices = di

    # No domain randomisation or observation noise on the robot.
    for fl in ("randomize_pd_gains", "randomize_friction", "randomize_mass",
               "randomize_com", "enable_obj_pose_noise", "enable_depth_noise"):
        if hasattr(env_cfg, fl):
            setattr(env_cfg, fl, False)
    for fl in ("pc_jitter_std", "pc_dropout_ratio", "pc_hand_noise_std",
               "pc_force_noise_ratio", "pc_force_dropout_prob"):
        if hasattr(env_cfg, fl):
            setattr(env_cfg, fl, 0.0)
    if hasattr(env_cfg, "init_curriculum_enabled"):
        env_cfg.init_curriculum_enabled = False

    # ---- Student checkpoint: env point counts + PC transforms as trained
    print(f"[DeployPC] loading ckpt: {args_cli.load_path}", flush=True)
    ckpt = load_checkpoint(args_cli.load_path)
    if not is_dagger_student_ckpt(ckpt):
        _keys = list((ckpt.get("model") or {}).keys())[:5]
        raise SystemExit(
            f"{args_cli.load_path} is not a DAgger PointCloudStudent checkpoint "
            f"(first state-dict keys: {_keys}). deploy_pc.py deploys DAgger students "
            f"only; PPO (train_ppo_pc.py) checkpoints read the sim-only object-pose "
            f"observation and cannot run on the robot.")
    align_pc_dims_to_ckpt(env_cfg, ckpt, tag="DeployPC")
    ws_min, ws_max = _parse_xyz(args_cli.pc_workspace_min), _parse_xyz(args_cli.pc_workspace_max)
    if args_cli.pc_no_crop:
        ws_min, ws_max = (-10.0, -10.0, -10.0), (10.0, 10.0, 10.0)
        print("[DeployPC] !!! --pc_no_crop: crop box = ±10 m; the policy is OUT OF "
              "DISTRIBUTION — inspection only !!!", flush=True)
    # On the robot `pc_force_scale` matters most: the Sharpa SDK's F6 forces go
    # through the same divisor as in sim. CLI values win over the ckpt.
    apply_pc_env_meta(
        env_cfg,
        ckpt,
        overrides={
            "pc_workspace_min": ws_min,
            "pc_workspace_max": ws_max,
            "pc_force_scale": args_cli.pc_force_scale,
            "pc_tactile_force_gate": args_cli.pc_tactile_force_gate,
            "pc_tactile_gate_mode": args_cli.pc_tactile_gate_mode,
            "pc_force_repr": args_cli.pc_force_repr,
            "pc_ablate_tactile_pc": args_cli.pc_ablate_tactile_pc,
            "pc_ablate_tactile_force": args_cli.pc_ablate_tactile_force,
            "pc_tactile_use_vec3": args_cli.pc_tactile_use_vec3,
            "enable_contact_force": False if args_cli.no_contact_force else None,
            "enable_tactile": False if args_cli.no_tactile else None,
        },
        tag="DeployPC",
    )
    print(f"[DeployPC] crop box {tuple(env_cfg.pc_workspace_min)} .. "
          f"{tuple(env_cfg.pc_workspace_max)}", flush=True)

    # ---- Arm: Polymetis bridge on the NUC, or ROS2
    env_cfg.arm_backend = args_cli.arm_backend
    env_cfg.ros2_namespace = args_cli.ros2_namespace
    if args_cli.arm_backend == "ros2":
        print("[DeployPC] arm backend = ROS2 (EXPERIMENTAL: not validated on the robot; "
              "Polymetis is the reference backend)", flush=True)
    env_cfg.polymetis_server_ip = args_cli.polymetis_ip or "localhost"
    env_cfg.polymetis_state_port = args_cli.polymetis_state_port
    env_cfg.polymetis_cmd_port = args_cli.polymetis_cmd_port
    _kq = _parse7(args_cli.polymetis_kq, "--polymetis_kq")
    _kqd = _parse7(args_cli.polymetis_kqd, "--polymetis_kqd")
    if (_kq is None) != (_kqd is None):
        raise ValueError("--polymetis_kq and --polymetis_kqd must be given together.")
    if _kq is not None:
        env_cfg.polymetis_kq, env_cfg.polymetis_kqd = _kq, _kqd
        print(f"[DeployPC] polymetis joint impedance OVERRIDE Kq={_kq} Kqd={_kqd}", flush=True)

    # One rclpy context for both ROS2 clients, shut down after everything closed.
    if _USE_ROS2 and not rclpy.ok():
        rclpy.init()

    # Whatever happens from here on — a refused checkpoint, a missing depth
    # stream, an exception or Ctrl+C — close what was opened: that stops the hand
    # worker, releases the arm and restores the terminal.
    env_raw = depth_sub = None
    try:
        # ---- Env (connects the arm + Sharpa SDK, builds the pk FK chain)
        print(f"[DeployPC] creating env: {args_cli.task}", flush=True)
        env_raw = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        base_env = env_raw.unwrapped
        device = torch.device(str(base_env.device))
        T = np.load(args_cli.camera_extrinsic).astype(np.float32)
        if T.shape != (4, 4):
            raise ValueError(f"{args_cli.camera_extrinsic}: expected a 4x4 matrix, got {T.shape}")
        base_env.set_camera_extrinsic(T)

        # ---- Policy
        model, arch = _build_dagger_student(ckpt, device)
        print(f"[DeployPC] arch: {arch}", flush=True)

        # ---- The student's trained observation layout ---------------------------
        # A lean student (--student_drop_slots) was trained on a sliced proprio
        # vector; the env publishes the full one. Masking runs before slicing,
        # because mask indices refer to the env's original layout. The deploy env
        # emits no object-pose tail (dims 550+), so a student must not read it.
        _mask_idx = list(ckpt.get("student_obs_mask_idx", []) or [])
        _keep_idx = list(ckpt.get("student_keep_idx", []) or [])
        obs_width = int(base_env.cfg.observation_space)
        read_idx = _keep_idx or list(range(int(ckpt["proprio_dim"])))
        if max(read_idx + _mask_idx) >= obs_width:
            raise SystemExit(
                f"this checkpoint reads proprio dims up to {max(read_idx + _mask_idx)}, but the "
                f"deploy env provides {obs_width} (no object-pose observation on the robot). "
                f"Deploy a student trained with --student_drop_slots "
                f"obj_bps,tips_distance,obj_pose_tail.")
        if _keep_idx and len(_keep_idx) != int(ckpt["proprio_dim"]):
            raise RuntimeError(
                f"checkpoint says proprio_dim={ckpt['proprio_dim']} but its keep-index selects "
                f"{len(_keep_idx)} dims; the ckpt is inconsistent.")
        _mask_t = torch.as_tensor(_mask_idx, dtype=torch.long, device=device) if _mask_idx else None
        _keep_t = torch.as_tensor(_keep_idx, dtype=torch.long, device=device) if _keep_idx else None
        if _keep_t is not None:
            print(f"[DeployPC] LEAN STUDENT: proprio sliced to {len(_keep_idx)}d, "
                  f"dropped {list(ckpt.get('student_drop_slots', []) or [])}", flush=True)
        if _mask_t is not None:
            print(f"[DeployPC] SPARSE-REF ckpt: zeroing {len(_mask_idx)} proprio dims", flush=True)

        def _make_inp(d):
            obs_in = d["policy"]
            if _mask_t is not None:
                obs_in = obs_in.clone()
                obs_in[:, _mask_t] = 0.0
            if _keep_t is not None:
                obs_in = obs_in[:, _keep_t]
            return {
                "obs": obs_in,
                "scene_pc": d["scene_pc"], "scene_mask": d["scene_mask"],
                "hand_pc": d["hand_pc"],
                "tactile_pc": d["tactile_pc"], "tactile_force": d["tactile_force"],
            }

        # ---- Depth from the camera host (ZMQ) or a ROS2 topic
        if args_cli.depth_backend == "zmq":
            depth_sub = RealSenseDepthZmqSubscriber(
                addr=args_cli.depth_zmq_addr,
                height=args_cli.depth_height, width=args_cli.depth_width,
                device=str(device),
            )
            depth_src, depth_hint = args_cli.depth_zmq_addr, "is realsense_depth_zmq_pub.py running on the camera host?"
        else:
            from dexx.scripts.deploy.ros2_depth_subscriber import RealSenseDepthRos2Subscriber
            depth_sub = RealSenseDepthRos2Subscriber(
                topic=args_cli.depth_topic,
                height=args_cli.depth_height, width=args_cli.depth_width,
                device=str(device),
            )
            depth_src, depth_hint = args_cli.depth_topic, "is the realsense2_camera driver publishing it?"
        t0 = time.time()
        while depth_sub.n_received == 0 and time.time() - t0 < 5.0:
            time.sleep(0.1)
        if depth_sub.n_received == 0:
            if depth_sub.n_rejected:
                raise SystemExit(
                    f"{depth_sub.n_rejected} depth frame(s) from {depth_src} were "
                    f"rejected for their size (expected {args_cli.depth_height}x"
                    f"{args_cli.depth_width}); match --depth_height/--depth_width to the publisher.")
            raise SystemExit(f"no depth frame from {depth_src} within 5 s — {depth_hint}")
        base_env.set_depth_source(depth_sub.get_latest, age_fn=depth_sub.age)
        print(f"[DeployPC] depth = {args_cli.depth_backend.upper()} @ {depth_src}", flush=True)

        # ---- Reset + run.
        step = 0
        _ramp_N = int(args_cli.action_ramp_steps)
        if _ramp_N > 0:
            print(f"[DeployPC] action ramp-up enabled: first {_ramp_N} steps scaled "
                  f"smoothstep(0→1) (~{_ramp_N/30.0:.1f}s @ 30Hz)", flush=True)
        try:
            obs, _ = env_raw.reset()
            print("[DeployPC] entering control loop @ 30Hz (Ctrl+C to stop)", flush=True)
            while simulation_app.is_running() and step < args_cli.max_steps and not base_env.quit_requested:
                with torch.no_grad():
                    action = model.act_inference(_make_inp(obs))
                    action = torch.clamp(action, -1.0, 1.0)
                # Action ramp-up: smoothstep(0→1) over first N steps. Mitigates
                # large initial joint velocity on real-robot start.
                if _ramp_N > 0 and step < _ramp_N:
                    s = min(step / float(_ramp_N), 1.0)
                    scale = float(s * s * (3.0 - 2.0 * s))   # smoothstep
                    action = action * scale
                    if step % 5 == 0:
                        print(f"  [DeployPC-RAMP] step {step}/{_ramp_N} action_scale={scale:.3f}",
                              flush=True)
                obs, _r, done, _trunc, _info = env_raw.step(action)
                step += 1
                if step % 30 == 0:
                    print(f"  [DeployPC] step={step} depth_rx={depth_sub.n_received}", flush=True)
        except KeyboardInterrupt:
            print("\n[DeployPC] KeyboardInterrupt — leaving rollout loop.", flush=True)

    finally:
        print("[DeployPC] exiting", flush=True)
        for close in (getattr(depth_sub, "shutdown", None), getattr(env_raw, "close", None)):
            try:
                if close is not None:
                    close()
            except Exception as exc:  # noqa: BLE001
                print(f"[DeployPC] cleanup error: {exc!r}", flush=True)
        if _USE_ROS2 and rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
    simulation_app.close()
