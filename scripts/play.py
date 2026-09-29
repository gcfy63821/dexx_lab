"""Evaluate a PointCloud student ckpt (DAgger or PPO fine-tuned).

Auto-detects the architecture from the checkpoint:
  - "mlp.0.weight" key → DAgger `PointCloudStudent` (deterministic policy)
  - "actor_mlp.mlp.0.weight" key → PPO `ActorCriticPointCloud` (uses mu)

Runs N episodes, accumulates `extras["succeeded"]` (set in env's `_get_rewards`
before `_reset_idx` clears the per-env buffers), reports success / failure /
mean episode reward.

NOTE: this "success" is NOT the one `eval.py` reports, and the two numbers must
not be compared. Here it means the episode reached the end of the trajectory
without a failure termination; there it means the object finished within
--success_dist (5 cm) of its final target. A short --max_steps truncates
episodes and drives this number down without the policy being any worse. For a
number you would quote, use eval.py.

Usage:
    python scripts/play.py \
        --task franka-sharpa-pointcloud \
        --load_path logs/.../dagger_final.pth \
        --side right --data_idx '[...]' \
        --num_envs 16 --max_episodes 200 --headless
"""
import argparse
import sys
import json
import ast

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate PointCloud student.")
parser.add_argument("--task", type=str, default="franka-sharpa-pointcloud")
parser.add_argument("--load_path", type=str, required=True)
parser.add_argument("--camera_extrinsic", type=str, default=None,
                    help="Path to a 4x4 .npy camera-in-armbase extrinsic. Must match "
                         "what the student was TRAINED with, or its scene cloud arrives "
                         "from a viewpoint it never saw. Omitted -> the shipped default.")
parser.add_argument("--side", type=str, default="right")
parser.add_argument("--data_idx", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--max_episodes", type=int, default=200)
parser.add_argument("--max_steps", type=int, default=1_000_000)
parser.add_argument("--label", type=str, default=None,
                    help="Optional label printed in the results header")

# Env-side PC transforms. All default to None = "take it from the ckpt's
# pc_env_meta" (see algo/dagger/pc_env_meta.py). Pass one only to deliberately
# deviate from training, e.g. a modality ablation of an unchanged policy.
parser.add_argument("--pc_ablate_tactile_pc", action="store_true", default=None,
                    help="Exclude the tactile points from the PC encoder.")
parser.add_argument("--pc_ablate_tactile_force", action="store_true", default=None,
                    help="Zero the tactile_force channel.")
parser.add_argument("--pc_force_repr", type=str, default=None,
                    choices=["scalar", "binary"],
                    help="Tactile force representation: raw scalar or binary contact.")
parser.add_argument("--pc_tactile_force_gate", type=float, default=None,
                    help="If >0, gate tactile points whose force < this.")
parser.add_argument("--pc_tactile_gate_mode", type=str, default=None,
                    choices=["zero", "mask"])
parser.add_argument("--pc_force_scale", type=float, default=None,
                    help="Divisor applied to tactile_force. Restored from ckpt if unset.")
parser.add_argument("--pc_tactile_use_vec3", action="store_true", default=None,
                    help="Enable 3D tactile force (feat_dim=3). Restored from ckpt if unset.")
parser.add_argument("--no_contact_force", action="store_true", default=None,
                    help="Zero the 5d proprio contact force. Restored from ckpt if unset.")
parser.add_argument("--no_tactile", action="store_true", default=None,
                    help="Zero the whole 20d proprio tactile tail. Restored from ckpt if unset.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows."""

import os
import time
import importlib

import torch
import gymnasium as gym
from omegaconf import OmegaConf

import dexx.tasks.franka_sharpa  # noqa: F401

from dexx.algo.dagger.pc_env_meta import align_pc_dims_to_ckpt, apply_pc_env_meta, load_checkpoint


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


def _build_dagger_student(ckpt: dict, device: torch.device):
    from dexx.algo.dagger.pointcloud_student import PointCloudStudent
    pc_cfg = dict(
        fusion_strategy=ckpt["pc_fusion_strategy"],
        n_scene=ckpt["n_scene"],
        n_hand=ckpt["n_hand"],
        n_tactile=ckpt["n_tactile"],
        tactile_feat_dim=ckpt["tactile_feat_dim"],
        output_dim=ckpt["pc_output_dim"],
        ablate_tactile_pc=ckpt.get("ablate_tactile_pc", False),
        # Drives the per-point feature width (scalar vs one-hot type encoding).
        # Omitting it silently builds a 7-dim encoder and any one-hot checkpoint
        # fails to load with a size mismatch. Keep in sync with scripts/eval.py.
        type_repr=ckpt.get("pc_type_repr", "scalar"),
    )
    model = PointCloudStudent(
        proprio_dim=ckpt["proprio_dim"],
        action_dim=ckpt["action_dim"],
        pc_encoder_cfg=pc_cfg,
        hidden=tuple(ckpt.get("student_hidden", (512, 256))),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, "DAgger PointCloudStudent"


def _build_ppo_student(ckpt: dict, device: torch.device):
    from dexx.algo.ppo.actor_critic_pointcloud import ActorCriticPointCloud
    cfg = ckpt["cfg"]
    pc_cfg = dict(
        fusion_strategy=cfg.pc_fusion_strategy,
        n_scene=cfg.n_scene,
        n_hand=cfg.n_hand,
        n_tactile=cfg.n_tactile,
        tactile_feat_dim=cfg.tactile_feat_dim,
        output_dim=cfg.pc_output_dim,
        ablate_tactile_pc=getattr(cfg, "ablate_tactile_pc", False),
    )
    model = ActorCriticPointCloud(dict(
        actions_num=cfg.action_dim,
        input_shape=(cfg.proprio_dim,),
        actor_units=list(cfg.actor_units),
        priv_mlp_units=[256, 128, cfg.priv_info_dim],
        priv_info_dim=cfg.priv_info_dim,
        critic_units=list(cfg.critic_units),
        pc_config=pc_cfg,
    )).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, "PPO ActorCriticPointCloud (mu only)"


def main():
    spec = gym.spec(args_cli.task)
    env_cfg = parse_entry_point(spec.kwargs["env_cfg_entry_point"])
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device:
        env_cfg.sim.device = args_cli.device
    if args_cli.side:
        env_cfg.hand_side = args_cli.side
    if args_cli.data_idx:
        try:
            data_indices = json.loads(args_cli.data_idx)
        except json.JSONDecodeError:
            data_indices = ast.literal_eval(args_cli.data_idx)
        env_cfg.data_indices = data_indices

    if args_cli.camera_extrinsic:
        import os as _os
        assert _os.path.exists(args_cli.camera_extrinsic), \
            f"--camera_extrinsic not found: {args_cli.camera_extrinsic}"
        env_cfg.camera_extrinsic_path = _os.path.abspath(args_cli.camera_extrinsic)
        print(f"[PlayPC] sim camera extrinsic <- {env_cfg.camera_extrinsic_path}", flush=True)

    # Clean eval: disable DR + aug
    for fl in (
        "randomize_pd_gains", "randomize_friction", "randomize_mass",
        "randomize_com", "enable_obj_pose_noise", "enable_depth_noise",
    ):
        if hasattr(env_cfg, fl):
            setattr(env_cfg, fl, False)
    for fl in (
        "pc_jitter_std", "pc_dropout_ratio", "pc_hand_noise_std",
        "pc_force_noise_ratio", "pc_force_dropout_prob",
    ):
        if hasattr(env_cfg, fl):
            setattr(env_cfg, fl, 0.0)
    # Disable init curriculum so the env samples uniformly over [0, seq_len),
    # not biased toward the demo-end frames the curriculum picks. We keep
    # `random_state_init=True` so the distribution stays mixed (otherwise
    # everyone starts at frame 0 and the policy has to play the whole demo,
    # which is much harder than the training distribution). For a totally
    # fixed-frame evaluation, override via `env_cfg.fixed_reset_demo_frame`.
    if hasattr(env_cfg, "init_curriculum_enabled"):
        env_cfg.init_curriculum_enabled = False

    # Restore the env-side PC transforms (force scale / gate / repr / ablations)
    # the student was TRAINED with; without this the policy is played on a
    # different input distribution than it learned. Peek the ckpt before the env
    # is built so the cfg overrides take effect. CLI flags still win.
    _ckpt_peek = load_checkpoint(args_cli.load_path)
    # Make the env emit the point counts / tactile width the student was trained
    # with, before the env is built.
    align_pc_dims_to_ckpt(env_cfg, _ckpt_peek, tag="PlayPC")
    apply_pc_env_meta(
        env_cfg,
        _ckpt_peek,
        overrides={
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
        tag="PlayPC",
    )

    print(f"\n[PlayPC] env: {args_cli.task} num_envs={args_cli.num_envs}")
    env_raw = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = env_raw.unwrapped
    device = torch.device(str(env.device))

    print(f"[PlayPC] loading ckpt: {args_cli.load_path}")
    ckpt = load_checkpoint(args_cli.load_path, map_location=device)
    sd_keys = list(ckpt.get("model", {}).keys())

    # Detect architecture
    if any(k.startswith("actor_mlp.") for k in sd_keys):
        model, arch_label = _build_ppo_student(ckpt, device)
    elif any(k == "mlp.0.weight" or k.startswith("mlp.") for k in sd_keys):
        model, arch_label = _build_dagger_student(ckpt, device)
    else:
        raise RuntimeError(
            f"Unknown ckpt format; first 5 sd keys: {sd_keys[:5]}"
        )
    print(f"[PlayPC] architecture: {arch_label}")

    obs, _ = env_raw.reset()

    total_episodes = 0
    total_successes = 0.0
    total_failures = 0.0
    episode_rewards_sum = []
    episode_rewards = torch.zeros(args_cli.num_envs, device=device)
    total_steps = 0
    t_start = time.time()

    print(f"[PlayPC] running until {args_cli.max_episodes} episodes "
          f"(or {args_cli.max_steps} steps).")

    # Restore the student's trained observation layout: a lean student
    # (--student_drop_slots) was trained on a sliced proprio vector while the env
    # still publishes the full one. Mask first, then slice — mask indices refer
    # to the env's original layout.
    _mask_idx = list(ckpt.get("student_obs_mask_idx", []) or [])
    _keep_idx = list(ckpt.get("student_keep_idx", []) or [])
    _mask_t = (torch.as_tensor(_mask_idx, dtype=torch.long, device=device)
               if _mask_idx else None)
    _keep_t = (torch.as_tensor(_keep_idx, dtype=torch.long, device=device)
               if _keep_idx else None)
    if _keep_t is not None:
        print(f"[PlayPC] LEAN STUDENT ckpt: proprio sliced to {len(_keep_idx)}d, "
              f"dropped {list(ckpt.get('student_drop_slots', []) or [])}", flush=True)

    def _make_inp(d):
        obs_in = d["policy"]
        if _mask_t is not None:
            obs_in = obs_in.clone()
            obs_in[:, _mask_t] = 0.0
        if _keep_t is not None:
            obs_in = obs_in[:, _keep_t]
        return {
            "obs": obs_in,
            "scene_pc": d["scene_pc"],
            "scene_mask": d["scene_mask"],
            "hand_pc": d["hand_pc"],
            "tactile_pc": d["tactile_pc"],
            "tactile_force": d["tactile_force"],
        }

    while (total_episodes < args_cli.max_episodes
           and total_steps < args_cli.max_steps
           and simulation_app.is_running()):
        with torch.no_grad():
            inp = _make_inp(obs)
            action = model.act_inference(inp)
            action = torch.clamp(action, -1.0, 1.0)

        obs, reward, terminated, truncated, extras = env_raw.step(action)
        episode_rewards += reward
        total_steps += 1

        def _as_float(v):
            return v.item() if hasattr(v, "item") else float(v)
        total_successes += _as_float(extras.get("succeeded", 0.0)) * args_cli.num_envs
        total_failures += _as_float(extras.get("failed_execute", 0.0)) * args_cli.num_envs

        done = terminated | truncated
        if done.any():
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            for idx in done_ids.tolist():
                total_episodes += 1
                episode_rewards_sum.append(episode_rewards[idx].item())
                episode_rewards[idx] = 0.0
            if total_episodes % max(1, args_cli.num_envs) == 0:
                rate = total_successes / max(total_episodes, 1)
                f_rate = total_failures / max(total_episodes, 1)
                mean_r = (sum(episode_rewards_sum) / len(episode_rewards_sum)) if episode_rewards_sum else 0.0
                elapsed = time.time() - t_start
                print(f"  [{total_episodes:4d} eps | {total_steps:6d} steps | "
                      f"{elapsed:5.1f}s] success={rate*100:5.1f}% "
                      f"fail={f_rate*100:5.1f}% mean_r={mean_r:7.2f}")

    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"[PlayPC] Results  {args_cli.label or ''}")
    print(f"  ckpt:           {args_cli.load_path}")
    print(f"  arch:           {arch_label}")
    print(f"  task:           {args_cli.task}")
    print(f"  episodes:       {total_episodes}")
    print(f"  steps:          {total_steps}")
    print(f"  elapsed:        {elapsed:.1f}s")
    print(f"  success rate:   {total_successes / max(total_episodes, 1) * 100:5.1f}% "
          f"({total_successes:.1f}/{total_episodes})")
    print(f"  failure rate:   {total_failures / max(total_episodes, 1) * 100:5.1f}% "
          f"({total_failures:.1f}/{total_episodes})")
    if episode_rewards_sum:
        print(f"  mean ep reward: {sum(episode_rewards_sum) / len(episode_rewards_sum):.2f}")
    print("=" * 60)

    env_raw.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
