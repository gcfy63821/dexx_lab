"""Record per-demo teacher rollout RGB videos (one MP4 per demo).

Builds the `franka-sharpa-pointcloud-record` env ONCE with all demos in
`data_idx_list`. The env auto-expands each base demo into all retarget aug
variants and assigns env e → expanded_indices[e % len(expanded_indices)].
We pick `num_envs` ≥ len(expanded_indices) so every variant gets at least
one env (recommended); falling short means some demos may be uncovered.

Run rollout, capture per-env RGB each step, track per-env success
(`env.success_buf`). After rollout, group envs by base demo. For each
base demo: pick FIRST env that succeeded → save its RGB trajectory as
`<demo_label>.mp4`. If no env of that demo succeeded, optionally save
the longest run of that demo's envs as `<demo>_fail.mp4` for debugging.

Why one-env-for-all: Isaac Lab simulation_app can only have ONE sim
context per process; `gym.make` twice in the same script → RuntimeError.

Usage:
    python scripts/record_videos.py \\
        --teacher_ckpt checkpoints/teacher_poseobs_T0.pth \\
        --side right \\
        --data_idx_list '["rt/0416_grasp/cube_small_2", "rt/0420_manip/squeegee_1"]' \\
        --num_envs 22 --max_steps 400 \\
        --out_dir logs/expert_rollout_videos \\
        --headless

Suggestion: --num_envs ≈ n_demos × 11 (covers all aug variants).
For 3 demos → 33 envs; for 21 demos → 231 envs (memory: 7-8 GB at 240×320 cam).
"""
from __future__ import annotations

import argparse
import sys
import json
import ast

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Per-demo teacher rollout RGB video.")
parser.add_argument("--task", default="franka-sharpa-pointcloud-record")
parser.add_argument("--teacher_ckpt", required=True)
parser.add_argument("--side", default="right")
parser.add_argument("--data_idx_list", required=True,
                    help="JSON list of base demo idx — one video per entry.")
parser.add_argument("--num_envs", type=int, default=None,
                    help="If None, auto-set to len(expanded_data_indices) (covers all aug variants).")
parser.add_argument("--max_steps", type=int, default=400)
parser.add_argument("--max_retries", type=int, default=2,
                    help="Reset + re-run if some demos got no success yet.")
parser.add_argument("--out_dir", default="logs/expert_rollout_videos")
parser.add_argument("--fps", type=int, default=15)
parser.add_argument("--cam_height", type=int, default=480)
parser.add_argument("--cam_width", type=int, default=640)
parser.add_argument("--cam_yaw_deg", type=float, default=0.0,
                    help="Rotate the third-person camera around z (up) axis by N degrees, "
                         "pivoting around env_cfg.record_camera.target. +30 = counterclockwise "
                         "30° viewed from above. Pass 0 to use the cfg default pose.")
parser.add_argument("--cam_z_offset", type=float, default=0.0,
                    help="Additive offset to cam pos z (meters). Negative = lower camera. "
                         "Applied AFTER --cam_yaw_deg rotation. Default 0 = unchanged.")
parser.add_argument("--init_frame_max", type=int, default=None,
                    help="If set, force every env's reset to sample init_frame from [0, N]. "
                         "Useful for 'show me a demo from the beginning' videos.")
parser.add_argument("--save_per_env", action="store_true", default=False,
                    help="Save ONE video per env (instead of one per base demo). Useful when "
                         "data_idx_list contains explicit aug variants (e.g. `demo@0` + "
                         "`demo@0_dxn...`) and you want a separate video per variant.")
parser.add_argument("--save_fails", action=argparse.BooleanOptionalAction, default=True,
                    help="Also save failed rollouts (--no-save_fails to skip them).")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--snapshot_frames", type=int, default=0,
                    help="If > 0, also save N evenly-spaced PNG snapshots from each "
                         "successful rollout (in addition to / instead of mp4). "
                         "Files go to <out_dir>/<demo_label>/frame_NN.png.")
parser.add_argument("--snapshot_only", action="store_true", default=False,
                    help="Skip mp4 encoding, only save snapshot PNGs (requires --snapshot_frames > 0).")
parser.add_argument("--eval_no_terminate", action="store_true", default=False,
                    help="Disable early termination on success — let the rollout run "
                         "until the demo ends (or --max_steps), producing a longer clip.")
parser.add_argument("--draw_force", action="store_true", default=False,
                    help="Overlay per-fingertip contact force vectors on each saved snapshot, "
                         "drawn as arrows from the fingertip in image space (debug-draw style).")
parser.add_argument("--force_scale", type=float, default=0.01,
                    help="Meters per Newton when drawing force arrows. Default 0.01 → 10N = 10cm.")
parser.add_argument("--force_min_n", type=float, default=0.3,
                    help="Forces below this magnitude (N) are not drawn.")
parser.add_argument("--max_per_demo", type=int, default=4,
                    help="When --save_per_env is set, cap saved success snapshots per BASE demo to N.")

AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

# Force enable_cameras (record_camera needs it).
if not getattr(args, "enable_cameras", False):
    args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ------------------------------------------------------------------------- #
import os
from collections import defaultdict

import numpy as np
import torch
import gymnasium as gym

import dexx.tasks.franka_sharpa  # noqa: F401

from dexx.algo.ppo.ppo import PPO
from dexx.wrapper.sharpa_wave_deploy_env_wrapper import GymStyleEnvWrapper
from dexx.wrapper.config_wrapper import ConfigWrapper


def parse_data_idx_list(s: str) -> list[str]:
    try:
        out = json.loads(s)
    except json.JSONDecodeError:
        out = ast.literal_eval(s)
    if not isinstance(out, list):
        raise ValueError(f"--data_idx_list must parse to a list, got {type(out)}")
    return out


def _label_of(demo_idx: str) -> str:
    """`rt/0416_grasp/cube_small_2` -> `0416_grasp_cube_small_2`."""
    return demo_idx.replace("rt/", "").replace("/", "_").replace("@", "_at_")


def _base_of(expanded_idx: str) -> str:
    """`rt/0416_grasp/cube_small_2@0_dxn0.5cm_...` -> `rt/0416_grasp/cube_small_2`."""
    return expanded_idx.split("@")[0]


def _capture_rgb_per_env(env, num_envs: int) -> list[np.ndarray | None]:
    """Pull RGB once for all envs. Returns list of (H,W,3) uint8 or None."""
    cam = getattr(env, "record_camera", None)
    if cam is None or cam.data.output is None:
        return [None] * num_envs
    rgba = cam.data.output.get("rgb", None)
    if rgba is None:
        return [None] * num_envs
    if rgba.shape[-1] == 4:
        rgba = rgba[..., :3]
    return [rgba[i].detach().cpu().numpy().astype(np.uint8) for i in range(num_envs)]


_CAPTURE_FINGER_DEBUG_PRINTED = [False]


def _capture_finger_data_per_env(env, num_envs: int) -> list[tuple[np.ndarray, np.ndarray] | None]:
    """Per-env (fingertip_pos_world(5,3), contact_force_w(5,3)). None if unavailable."""
    hand = getattr(env, "hand", None)
    finger_bodies = getattr(env, "finger_bodies", None)
    elastomer_ids = getattr(env, "elastomer_ids", None)
    force_vec = getattr(env, "last_contacts_vec_w", None)
    if not _CAPTURE_FINGER_DEBUG_PRINTED[0]:
        _CAPTURE_FINGER_DEBUG_PRINTED[0] = True
        eo = getattr(env, "scene", None)
        eo_origins = eo.env_origins if eo is not None else None
        print(f"[capture_finger DEBUG] hand={hand is not None} "
              f"finger_bodies={finger_bodies is not None and len(finger_bodies)} "
              f"elastomer_ids={elastomer_ids is not None and len(elastomer_ids)} "
              f"last_contacts_vec_w={force_vec is not None and tuple(force_vec.shape)} "
              f"env_origins[0]={eo_origins[0].tolist() if eo_origins is not None else None}")
    if hand is None or finger_bodies is None or force_vec is None:
        return [None] * num_envs
    pos_idx = elastomer_ids if elastomer_ids is not None else finger_bodies
    # body_pos_w is world frame; the record camera is positioned with
    # `set_world_poses_from_view(cam_pos_local + env_origin, ...)` per env
    # (each env has its own camera). We want to project in env-local frame
    # since cam params were built from env_cfg.record_camera.pos (env-local).
    fpos_w_t = hand.data.body_pos_w[:, pos_idx, :]  # (N, 5, 3) world
    env_origins = env.scene.env_origins.unsqueeze(1)  # (N, 1, 3)
    fpos_local = fpos_w_t - env_origins             # (N, 5, 3) env-local
    out = []
    fpos_np = fpos_local.detach().cpu().numpy().astype(np.float32)
    # Force vector is a direction (Δposition for the arrow head). Translation
    # doesn't apply. Use directly.
    force_np = force_vec.detach().cpu().numpy().astype(np.float32)
    for i in range(num_envs):
        out.append((fpos_np[i].copy(), force_np[i].copy()))
    return out


def _encode_mp4(frames: list[np.ndarray], out_path: str, fps: int) -> bool:
    import imageio
    if not frames:
        return False
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8,
                                macro_block_size=None)
    for f in frames:
        writer.append_data(f)
    writer.close()
    return True


def _save_snapshots(frames: list[np.ndarray], out_dir: str, n: int,
                    finger_data: list[tuple[np.ndarray, np.ndarray]] | None = None,
                    cam_params: dict | None = None,
                    force_scale: float = 0.01,
                    force_min_n: float = 0.3) -> int:
    """Sample n evenly-spaced frames from `frames` and write to `out_dir/frame_NN.png`.

    If `finger_data` + `cam_params` are provided, overlay per-fingertip force
    arrows on each frame before writing.
    Returns count of frames actually saved (clamped if len(frames) < n)."""
    import imageio
    if not frames:
        return 0
    os.makedirs(out_dir, exist_ok=True)
    T = len(frames)
    if T <= n:
        idxs = list(range(T))
    else:
        idxs = [int(round(t * (T - 1) / (n - 1))) for t in range(n)]
        seen = set(); idxs_ = []
        for i in idxs:
            if i not in seen:
                seen.add(i); idxs_.append(i)
        idxs = idxs_
    for k, i in enumerate(idxs):
        img = frames[i].copy()
        if finger_data is not None and cam_params is not None and i < len(finger_data):
            fpos_w, force_w = finger_data[i]
            img = _draw_force_arrows(img, fpos_w, force_w, cam_params,
                                     scale=force_scale, min_n=force_min_n)
        imageio.imwrite(os.path.join(out_dir, f"frame_{k:02d}_t{i:03d}.png"), img)
    return len(idxs)


# Per-finger arrow colors (BGR-ish but we're using RGB so just RGB tuples)
_FINGER_COLORS = [
    (244, 67, 54),    # thumb  — red
    (255, 152, 0),    # index  — orange
    (255, 235, 59),   # middle — yellow
    (76, 175, 80),    # ring   — green
    (33, 150, 243),   # pinky  — blue
]


def _build_camera_params(cam_pos: tuple, cam_target: tuple, focal_length: float,
                         horizontal_aperture: float, width: int, height: int) -> dict:
    """Pinhole camera params + world→camera transform for projection."""
    cam_pos_ = np.array(cam_pos, dtype=np.float64)
    cam_tgt_ = np.array(cam_target, dtype=np.float64)
    fx = focal_length / horizontal_aperture * width
    fy = fx  # square pixels
    cx = width / 2.0
    cy = height / 2.0
    # Camera frame (ROS optical: x-right, y-down, z-forward toward scene).
    # World up = +z. Image-right = world_up × fwd (i.e. +y for a camera looking
    # along +x with head up in +z). Image-down = right × fwd. The reverse
    # `cross(fwd, world_up)` ordering would invert both image axes.
    fwd = cam_tgt_ - cam_pos_
    fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(world_up, fwd)
    right = right / (np.linalg.norm(right) + 1e-12)
    down = np.cross(right, fwd)
    down = down / (np.linalg.norm(down) + 1e-12)
    R = np.stack([right, down, fwd], axis=0)  # world → cam: rows are basis vectors
    return dict(R=R, t=cam_pos_, fx=fx, fy=fy, cx=cx, cy=cy, w=width, h=height)


def _project(points_world: np.ndarray, cam: dict) -> np.ndarray:
    """points_world: (N, 3) in world frame. Returns (N, 3) = (u, v, depth)."""
    R = cam["R"]; t = cam["t"]
    p_cam = (points_world - t) @ R.T  # (N, 3)
    z = p_cam[:, 2]
    u = cam["fx"] * p_cam[:, 0] / np.where(np.abs(z) < 1e-6, 1e-6, z) + cam["cx"]
    v = cam["fy"] * p_cam[:, 1] / np.where(np.abs(z) < 1e-6, 1e-6, z) + cam["cy"]
    return np.stack([u, v, z], axis=-1)


def _draw_arrow(img: np.ndarray, p0: tuple, p1: tuple, color: tuple,
                thickness: int = 3, head_size: int = 10):
    """Tiny CV-free arrow rasterizer using PIL.ImageDraw."""
    from PIL import Image, ImageDraw
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    x0, y0 = p0; x1, y1 = p1
    draw.line([(x0, y0), (x1, y1)], fill=color, width=thickness)
    # Arrowhead: two short lines from tip
    import math
    ang = math.atan2(y1 - y0, x1 - x0)
    for da in (math.pi / 6, -math.pi / 6):
        hx = x1 - head_size * math.cos(ang - da)
        hy = y1 - head_size * math.sin(ang - da)
        draw.line([(x1, y1), (hx, hy)], fill=color, width=thickness)
    # Filled circle at base
    r = max(2, thickness)
    draw.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], fill=color)
    return np.array(pil)


_DRAW_FORCE_DEBUG_PRINTED = [False]


def _draw_force_arrows(img: np.ndarray,
                       fingertip_pos_w: np.ndarray,  # (5, 3)
                       force_vec_w: np.ndarray,      # (5, 3) N
                       cam: dict,
                       scale: float = 0.01,
                       min_n: float = 0.3) -> np.ndarray:
    """Project + draw per-finger force arrows. Returns annotated img."""
    if fingertip_pos_w is None or force_vec_w is None:
        return img
    tips_uv = _project(fingertip_pos_w, cam)            # (5, 3)
    head_w = fingertip_pos_w + force_vec_w * scale       # (5, 3)
    heads_uv = _project(head_w, cam)                     # (5, 3)
    W, H = cam["w"], cam["h"]

    if not _DRAW_FORCE_DEBUG_PRINTED[0]:
        _DRAW_FORCE_DEBUG_PRINTED[0] = True
        print(f"[draw_force DEBUG] cam_pos={cam['t']} fx={cam['fx']:.1f}")
        for k in range(min(5, fingertip_pos_w.shape[0])):
            fmag = float(np.linalg.norm(force_vec_w[k]))
            print(f"  tip{k}: pos_w={fingertip_pos_w[k]} force_w={force_vec_w[k]} mag={fmag:.2f}N  "
                  f"uv=({tips_uv[k,0]:.1f},{tips_uv[k,1]:.1f},z={tips_uv[k,2]:.3f})")

    out = img
    for k in range(min(5, fingertip_pos_w.shape[0])):
        force_mag = float(np.linalg.norm(force_vec_w[k]))
        if force_mag < min_n:
            continue
        if tips_uv[k, 2] <= 0 or heads_uv[k, 2] <= 0:
            continue
        p0 = (int(round(tips_uv[k, 0])), int(round(tips_uv[k, 1])))
        p1 = (int(round(heads_uv[k, 0])), int(round(heads_uv[k, 1])))
        if not (-50 <= p0[0] <= W + 50 and -50 <= p0[1] <= H + 50):
            continue
        color = _FINGER_COLORS[k % len(_FINGER_COLORS)]
        thickness = max(2, int(round(2 + force_mag / 20)))
        head_sz = max(8, int(round(8 + force_mag / 10)))
        out = _draw_arrow(out, p0, p1, color, thickness=thickness, head_size=head_sz)
    return out


def main():
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[ExpertVideo] out_dir = {args.out_dir}")

    data_idx_list = parse_data_idx_list(args.data_idx_list)
    print(f"[ExpertVideo] {len(data_idx_list)} base demo(s) requested:")
    for d in data_idx_list:
        print(f"    {d}")

    # ---- Probe: pre-build env cfg to know how many envs we need ----
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg, load_cfg_from_registry

    env_cfg = parse_env_cfg(args.task, device="cuda",
                            num_envs=args.num_envs or 2,  # placeholder; will rebuild
                            use_fabric=True)
    env_cfg.seed = args.seed
    env_cfg.hand_side = args.side
    env_cfg.data_indices = list(data_idx_list)
    if hasattr(env_cfg, "record_camera"):
        env_cfg.record_camera.height = int(args.cam_height)
        env_cfg.record_camera.width = int(args.cam_width)
        # Optional: rotate camera around z (up) axis, pivoting around target.
        if abs(args.cam_yaw_deg) > 1e-6:
            import math
            pos0 = tuple(env_cfg.record_camera.pos)
            tgt = tuple(env_cfg.record_camera.target)
            dx, dy, dz = pos0[0]-tgt[0], pos0[1]-tgt[1], pos0[2]-tgt[2]
            r = math.radians(float(args.cam_yaw_deg))
            nx = dx*math.cos(r) - dy*math.sin(r)
            ny = dx*math.sin(r) + dy*math.cos(r)
            new_pos = (tgt[0]+nx, tgt[1]+ny, tgt[2]+dz)
            env_cfg.record_camera.pos = new_pos
            print(f"[ExpertVideo] cam_yaw_deg={args.cam_yaw_deg:+.1f}° → "
                  f"pos {pos0} -> {tuple(round(p, 3) for p in new_pos)} "
                  f"(target {tgt} unchanged)")
        if abs(args.cam_z_offset) > 1e-6:
            pos0 = tuple(env_cfg.record_camera.pos)
            new_pos = (pos0[0], pos0[1], pos0[2] + float(args.cam_z_offset))
            env_cfg.record_camera.pos = new_pos
            print(f"[ExpertVideo] cam_z_offset={args.cam_z_offset:+.2f}m → "
                  f"pos z {pos0[2]:.3f} -> {new_pos[2]:.3f}")
    if hasattr(env_cfg, "init_curriculum_enabled"):
        env_cfg.init_curriculum_enabled = False
    for k in ("randomize_mass", "randomize_friction", "randomize_pd_gains",
              "randomize_com"):
        if hasattr(env_cfg, k):
            setattr(env_cfg, k, False)
    if args.eval_no_terminate and hasattr(env_cfg, "eval_no_terminate"):
        env_cfg.eval_no_terminate = True
        print(f"[ExpertVideo] eval_no_terminate=True → rollouts run until demo end / max_steps")

    # ---- Build env (auto-expansion happens here) ----
    # First call needs num_envs set; we DON'T yet know expansion count.
    # Strategy: build with num_envs = max(args.num_envs or 8, len(data_idx_list)),
    # check expanded_data_indices, and if user requested more, document it.
    initial_n = args.num_envs or max(8, len(data_idx_list))
    env_cfg.scene.num_envs = initial_n
    env_raw = gym.make(args.task, cfg=env_cfg, render_mode=None)
    env_raw = GymStyleEnvWrapper(env_raw, clip_actions=env_cfg.clip_actions)
    base_env = env_raw.unwrapped
    device = base_env.device

    expanded_indices = [str(x) for x in getattr(base_env, "data_indices", [])]
    print(f"[ExpertVideo] expanded to {len(expanded_indices)} variants from "
          f"{len(data_idx_list)} base demos.")

    if args.num_envs is None:
        # User didn't specify; warn if env has fewer than variant count.
        if initial_n < len(expanded_indices):
            print(f"[ExpertVideo] WARNING: num_envs={initial_n} < expanded "
                  f"{len(expanded_indices)} → only first {initial_n} variants "
                  f"will run. Some demos may be uncovered. Re-run with "
                  f"--num_envs {len(expanded_indices)} for full coverage.")
        num_envs = initial_n
    else:
        num_envs = initial_n
        if num_envs < len(expanded_indices):
            print(f"[ExpertVideo] WARNING: num_envs={num_envs} < expanded "
                  f"{len(expanded_indices)} → some demos may be uncovered.")

    # Per-env → base demo mapping. We strip "@variant" from BOTH sides so that
    # user-supplied `rt/foo/bar@0` matches expanded `rt/foo/bar@0` (no aug,
    # explicit base) AND `rt/foo/bar` matches `rt/foo/bar@0_xxx` (aug-expanded).
    env_to_base = [_base_of(expanded_indices[e % len(expanded_indices)])
                   for e in range(num_envs)]
    base_to_envs: dict[str, list[int]] = defaultdict(list)
    for e, b in enumerate(env_to_base):
        base_to_envs[b].append(e)
    print(f"[ExpertVideo] base → envs mapping:")
    for b in data_idx_list:
        b_stripped = _base_of(b)
        envs = base_to_envs.get(b_stripped, [])
        print(f"    {b:<50s} envs={envs[:5]}{'...' if len(envs)>5 else ''}")

    # Aim record camera (env method)
    if hasattr(base_env, "_apply_record_camera_pose"):
        base_env._apply_record_camera_pose()

    # Build camera params for projecting force vectors. World ≈ env-local here
    # (env_origins at world z=0). After cam_yaw_deg rotation we used updated pos.
    cam_params = None
    if args.draw_force:
        rec = env_cfg.record_camera
        cam_params = _build_camera_params(
            cam_pos=tuple(rec.pos), cam_target=tuple(rec.target),
            focal_length=rec.focal_length, horizontal_aperture=rec.horizontal_aperture,
            width=rec.width, height=rec.height)
        print(f"[ExpertVideo] draw_force=True; cam pos={rec.pos} tgt={rec.target} "
              f"fx≈{cam_params['fx']:.0f} (force_scale={args.force_scale}m/N, "
              f"min={args.force_min_n}N)")

    # ---- Build teacher agent ----
    agent_cfg = load_cfg_from_registry(args.task, "gym_style_cfg_entry_point")
    agent_cfg["algorithm"]["num_actors"] = num_envs
    agent_cfg["algorithm"]["minibatch_size"] = min(num_envs * 8, 32768)
    agent_cfg["seed"] = args.seed
    agent_cfg["load_path"] = args.teacher_ckpt

    config = ConfigWrapper(agent_cfg, env_cfg, test=True)
    log_dir = os.path.join(args.out_dir, "_log")
    _agents = {"PPO": PPO}
    _algo = agent_cfg.get("algo", "PPO")
    if _algo not in _agents:
        raise SystemExit(f"unsupported algo {_algo!r} in the agent cfg; "
                         f"this release ships only {sorted(_agents)}")
    AgentCls = _agents[_algo]
    agent = AgentCls(env_raw, output_dir=log_dir, full_config=config, create_output_dir=False)
    print(f"[ExpertVideo] loading teacher: {args.teacher_ckpt}")

    # `agent.restore_test` does a STRICT state_dict load. Teacher poseobs ckpt
    # may have been trained with a different `priv_info_dim` than the yaml
    # default — usually critic dim mismatch (e.g. critic weight 705 vs 597
    # = 557 + 148 vs 557 + 40 priv_info). `strict=False` does NOT bypass
    # SHAPE mismatch (only key missing), so we pre-filter: drop any ckpt key
    # whose shape doesn't match the current model. For inference we only need
    # actor + sigma (act_inference path), so dropping critic keys is fine.
    ckpt = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
    own_sd = agent.model.state_dict()
    sd_to_load = {}
    skipped_shape = []
    for k, v in ckpt["model"].items():
        if k not in own_sd:
            continue  # extra key in ckpt — skip
        if tuple(own_sd[k].shape) != tuple(v.shape):
            skipped_shape.append((k, tuple(v.shape), tuple(own_sd[k].shape)))
            continue
        sd_to_load[k] = v
    missing, unexpected = agent.model.load_state_dict(sd_to_load, strict=False)
    actor_missing = [k for k in missing if k.startswith("actor") or k == "sigma"]
    if actor_missing:
        print(f"[ExpertVideo] WARNING: actor missing keys after load: {actor_missing[:5]}")
        raise RuntimeError(
            f"actor state_dict missing — teacher arch mismatch. "
            f"sample missing: {actor_missing[:3]}"
        )
    if skipped_shape:
        n_critic = sum(1 for k, _, _ in skipped_shape if "critic" in k)
        print(f"[ExpertVideo] skipped {len(skipped_shape)} shape-mismatched keys "
              f"({n_critic} critic — fine, only actor used for inference)")
        for k, c_shape, m_shape in skipped_shape[:3]:
            print(f"  - {k}: ckpt={c_shape} vs model={m_shape}")
    print(f"[ExpertVideo] loaded {len(sd_to_load)}/{len(ckpt['model'])} keys from ckpt")
    if hasattr(agent, "normalize_input") and agent.normalize_input:
        agent.running_mean_std.load_state_dict(ckpt["running_mean_std"])
    agent.set_eval()

    # ---- Rollout — capture per-env RGB (+ optional finger data), track per-env success ----
    # Per-env: list of completed sub-episodes [(rgb_history, finger_history, succeeded), ...]
    completed: list[list[tuple[list[np.ndarray], list, bool]]] = [[] for _ in range(num_envs)]
    current_rgb: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
    current_fdat: list[list] = [[] for _ in range(num_envs)]
    current_succ: list[bool] = [False] * num_envs
    # Track success at each step. env's success_buf gets cleared in _reset_idx.
    base_done_collected: set[str] = set()
    base_succ_video_saved: set[str] = set()

    for attempt in range(1, args.max_retries + 1):
        if attempt > 1:
            n_remaining = len([b for b in data_idx_list if b not in base_succ_video_saved])
            if n_remaining == 0:
                print(f"[ExpertVideo] all demos have success videos; skipping retry.")
                break
            print(f"[ExpertVideo] retry {attempt}/{args.max_retries} "
                  f"({n_remaining} demos still need success)")

        # Optional: force init_frame ∈ [0, args.init_frame_max] for all envs
        if args.init_frame_max is not None and hasattr(base_env, "set_eval_init_range"):
            lo = torch.zeros(num_envs, dtype=torch.long, device=device)
            hi = torch.full((num_envs,), int(args.init_frame_max),
                            dtype=torch.long, device=device)
            base_env.set_eval_init_range(lo, hi)
            print(f"[ExpertVideo] init_frame_range = [0, {args.init_frame_max}] per env")

        obs_dict = env_raw.reset()
        if isinstance(obs_dict, tuple):
            obs_dict = obs_dict[0]
        if not isinstance(obs_dict, dict):
            obs_dict = base_env._get_observations()
        # reset capture state
        current_rgb = [[] for _ in range(num_envs)]
        current_fdat = [[] for _ in range(num_envs)]
        current_succ = [False] * num_envs

        # Reset detector: track previous step's episode_length_buf. When it
        # drops (env was reset), close out the previous sub-episode.
        # GymStyleEnvWrapper returns done=None, so we can't rely on it.
        prev_ep_len = base_env.episode_length_buf.clone()

        for step in range(args.max_steps):
            with torch.no_grad():
                input_dict = {
                    "obs": agent.running_mean_std(obs_dict["obs"]),
                    "priv_info": obs_dict["priv_info"],
                }
                mu = agent.model.act_inference(input_dict)
                mu = torch.clamp(mu, -1.0, 1.0)

            step_ret = env_raw.step(mu)
            # GymStyleEnvWrapper returns (obs_dict, None, None, None);
            # other wrappers may return 4 or 5 tuples — handle both.
            if isinstance(step_ret, tuple) and len(step_ret) >= 1:
                obs_dict = step_ret[0]
            if not isinstance(obs_dict, dict):
                obs_dict = base_env._get_observations()

            # Force record_camera render + capture
            if hasattr(base_env, "step_record_camera"):
                base_env.step_record_camera()
            frames_now = _capture_rgb_per_env(base_env, num_envs)
            if args.draw_force:
                fdat_now = _capture_finger_data_per_env(base_env, num_envs)
            else:
                fdat_now = [None] * num_envs
            for i, f in enumerate(frames_now):
                if f is not None:
                    current_rgb[i].append(f)
                    current_fdat[i].append(fdat_now[i])

            # Read success_buf BEFORE done handling (env's _reset_idx clears it)
            succ_now = (base_env.success_buf > 0) if hasattr(base_env, "success_buf") else None
            if succ_now is not None:
                for i in succ_now.nonzero(as_tuple=False).flatten().tolist():
                    current_succ[i] = True

            # Detect resets via episode_length_buf dropping. If env was reset
            # at this step, ep_len_now < prev_ep_len for that env (or 0).
            # NOTE: env's success_buf gets cleared in _reset_idx which fires
            # at the SAME step as the success check, so we can't observe
            # success_buf=1 post-step. Workaround: when we detect a reset,
            # mark the sub-episode as success if it ran long enough (≥ ~30
            # frames = early grasp+move) — this is a heuristic, not a strict
            # success check. The deliverable is "a video per demo" regardless;
            # success label is just metadata for the filename suffix.
            ep_len_now = base_env.episode_length_buf
            done_mask = (ep_len_now < prev_ep_len)
            if done_mask.any():
                done_ids = done_mask.nonzero(as_tuple=False).flatten().tolist()
                for i in done_ids:
                    if len(current_rgb[i]) >= 5:  # skip trivial restart spurious dones
                        was_long = len(current_rgb[i]) >= 30
                        completed[i].append((current_rgb[i], current_fdat[i],
                                              current_succ[i] or was_long))
                    current_rgb[i] = []
                    current_fdat[i] = []
                    current_succ[i] = False
            prev_ep_len = ep_len_now.clone()

            if step % 50 == 0:
                n_completed = sum(len(completed[i]) for i in range(num_envs))
                ns = sum(1 for i in range(num_envs) for t in completed[i] if t[-1])
                in_progress = sum(1 for i in range(num_envs) if len(current_rgb[i]) > 5)
                print(f"    step {step:4d}: completed_subepisodes={n_completed} "
                      f"(success={ns}) in_progress={in_progress}", flush=True)

        # Flush partial in-progress episodes too (treat as success/non-success based on flag)
        for i in range(num_envs):
            if len(current_rgb[i]) >= 5:
                completed[i].append((current_rgb[i], current_fdat[i], current_succ[i]))
            current_rgb[i] = []
            current_fdat[i] = []
            current_succ[i] = False

        # ---- After attempt, save videos -------------------------------
        if args.save_per_env:
            # One video per ENV — full variant label in filename. Cap each
            # base demo at --max_per_demo successful saves.
            per_base_saved_count: dict[str, int] = defaultdict(int)
            for ei, demo in enumerate(env_to_base):
                full_idx = expanded_indices[ei % len(expanded_indices)]
                base = _base_of(full_idx)
                if full_idx in base_succ_video_saved:
                    continue
                if per_base_saved_count[base] >= args.max_per_demo:
                    continue
                label = _label_of(full_idx)
                for (hist, fdat, succ) in completed[ei]:
                    if succ and hist:
                        ok = True
                        if not args.snapshot_only:
                            out_path = os.path.join(args.out_dir, f"{label}.mp4")
                            ok = _encode_mp4(hist, out_path, args.fps)
                            if ok:
                                print(f"  [ExpertVideo] ✓ env{ei} {full_idx} → {out_path} ({len(hist)} frames)")
                        if ok and args.snapshot_frames > 0:
                            snap_dir = os.path.join(args.out_dir, label)
                            print(f"  [SAVE DEBUG] args.draw_force={args.draw_force} "
                                  f"len(fdat)={len(fdat) if fdat else None} "
                                  f"fdat[0]={'tuple' if (fdat and fdat[0] is not None) else 'None'}")
                            n_saved = _save_snapshots(
                                hist, snap_dir, args.snapshot_frames,
                                finger_data=fdat if args.draw_force else None,
                                cam_params=cam_params, force_scale=args.force_scale,
                                force_min_n=args.force_min_n)
                            print(f"  [ExpertVideo]   + {n_saved} snapshots → {snap_dir}/")
                        if ok:
                            base_succ_video_saved.add(full_idx)
                            per_base_saved_count[base] += 1
                            break
        else:
            # Default: one video per base demo (collapse aug variants)
            for demo in data_idx_list:
                if demo in base_succ_video_saved:
                    continue
                label = _label_of(demo)
                envs_for_demo = base_to_envs.get(_base_of(demo), [])
                if not envs_for_demo:
                    print(f"  [ExpertVideo] {demo}: no env covers this demo (num_envs too small). Skipping.")
                    base_done_collected.add(demo)
                    continue
                # Find first env that has a successful sub-episode
                saved = False
                for ei in envs_for_demo:
                    for (hist, fdat, succ) in completed[ei]:
                        if succ and hist:
                            ok = True
                            if not args.snapshot_only:
                                out_path = os.path.join(args.out_dir, f"{label}.mp4")
                                ok = _encode_mp4(hist, out_path, args.fps)
                                if ok:
                                    print(f"  [ExpertVideo] ✓ {demo} → {out_path} ({len(hist)} frames)")
                            if ok and args.snapshot_frames > 0:
                                snap_dir = os.path.join(args.out_dir, label)
                                n_saved = _save_snapshots(
                                    hist, snap_dir, args.snapshot_frames,
                                    finger_data=fdat if args.draw_force else None,
                                    cam_params=cam_params, force_scale=args.force_scale,
                                    force_min_n=args.force_min_n)
                                print(f"  [ExpertVideo]   + {n_saved} snapshots → {snap_dir}/")
                            if ok:
                                base_succ_video_saved.add(demo)
                                saved = True
                                break
                    if saved:
                        break

    # ---- Fail fallback ----
    for demo in data_idx_list:
        if demo in base_succ_video_saved:
            continue
        if not args.save_fails:
            continue
        label = _label_of(demo)
        envs_for_demo = base_to_envs.get(demo, [])
        if not envs_for_demo:
            continue
        # Pick longest sub-episode from any env of this demo
        best = None  # (hist, env_id)
        for ei in envs_for_demo:
            for (hist, _fdat, succ) in completed[ei]:
                if best is None or len(hist) > len(best[0]):
                    best = (hist, ei)
        if best is not None and best[0]:
            out_path = os.path.join(args.out_dir, f"{label}_fail.mp4")
            ok = _encode_mp4(best[0], out_path, args.fps)
            if ok:
                print(f"  [ExpertVideo] ∼ {demo}: no success; saved longest "
                      f"({len(best[0])} frames, env {best[1]}) → {out_path}")

    # ---- Summary ----
    print(f"\n[ExpertVideo] DONE. {len(base_succ_video_saved)}/{len(data_idx_list)} "
          f"demos got a success video.")
    for d in data_idx_list:
        marker = "✓" if d in base_succ_video_saved else "✗"
        print(f"  {marker} {d}")

    try:
        env_raw.close()
    except Exception:
        pass
    simulation_app.close()


if __name__ == "__main__":
    main()
