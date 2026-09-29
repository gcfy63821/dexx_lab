"""PPO fine-tune of a DAgger PointCloud student.

Loads the DAgger ckpt into `ActorCriticPointCloud`, then runs standard PPO
on top in the same `franka-sharpa-pointcloud` env.

Only full-observation students are supported: a lean student
(`--student_drop_slots`) is refused. The resulting PPO checkpoint is for sim
evaluation only; `deploy/deploy_pc.py` deploys DAgger students.
"""
import argparse
import sys
import json
import ast

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="PPO fine-tune for PointCloud student.")
parser.add_argument("--task", type=str, default="franka-sharpa-pointcloud")
parser.add_argument("--dagger_ckpt", type=str, required=True,
                    help="DAgger PointCloud student checkpoint.")
parser.add_argument("--side", type=str, default="right")
parser.add_argument("--data_idx", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--camera_extrinsic", type=str, default=None,
                    help="Path to a 4x4 .npy camera-in-armbase extrinsic (ROS optical). "
                         "Must match what the DAgger student was trained with. Omit to "
                         "use the env default (calib/camera_align/current.npy).")

# PPO knobs
parser.add_argument("--max_iters", type=int, default=200)
parser.add_argument("--horizon", type=int, default=32)
parser.add_argument("--mini_epochs", type=int, default=2)
parser.add_argument("--minibatch_size", type=int, default=1024)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--clip", type=float, default=0.2)
parser.add_argument("--entropy_coef", type=float, default=1e-4)
parser.add_argument("--init_logstd", type=float, default=-2.0,
                    help="Initial log(sigma) for action distribution. Smaller "
                         "values keep PPO closer to DAgger mu — for fine-tune "
                         "of an imitation-warm-start policy try -3 or -4.")
parser.add_argument("--save_every_n", type=int, default=10)
parser.add_argument("--out_dir", type=str, default="logs/ppo_pc")

# Modality ablation (default off = no behaviour change)
parser.add_argument("--pc_ablate_tactile_pc", action="store_true",
                    help="Exclude the tactile points from the PC encoder.")
parser.add_argument("--pc_ablate_tactile_force", action="store_true",
                    help="Zero the tactile_force channel.")
parser.add_argument("--pc_force_repr", type=str, default="scalar",
                    choices=["scalar", "binary"],
                    help="Tactile force representation: raw scalar or binary contact.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows."""

import os
import importlib

import gymnasium as gym
from omegaconf import OmegaConf

import dexx.tasks.franka_sharpa  # noqa: F401

from dexx.algo.ppo.ppo_pointcloud import PPOPointCloud, PPOPointCloudConfig
from dexx.algo.dagger.pc_env_meta import (
    align_pc_dims_to_ckpt, apply_pc_env_meta, lean_student_refusal, load_checkpoint,
)


def parse_entry_point(entry_point: str):
    module, target = entry_point.split(":")
    if target.endswith("Cfg") or target[0].isupper():
        mod = importlib.import_module(module)
        return getattr(mod, target)()
    if target.endswith(".yaml") or target.endswith(".yml"):
        mod = importlib.import_module(module)
        base_dir = os.path.dirname(mod.__file__)
        return OmegaConf.load(os.path.join(base_dir, target))
    raise ValueError(f"Unsupported entry_point: {entry_point}")


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
            di = json.loads(args_cli.data_idx)
        except json.JSONDecodeError:
            di = ast.literal_eval(args_cli.data_idx)
        env_cfg.data_indices = di
    if args_cli.camera_extrinsic:
        if not os.path.exists(args_cli.camera_extrinsic):
            raise SystemExit(f"--camera_extrinsic not found: {args_cli.camera_extrinsic}")
        env_cfg.camera_extrinsic_path = os.path.abspath(args_cli.camera_extrinsic)
        print(f"[PPOPC] sim camera extrinsic <- {env_cfg.camera_extrinsic_path}", flush=True)

    # The DAgger student fixes the point counts / hand bodies / tactile width
    # the env must produce so its encoder weights load.
    _ckpt_peek = load_checkpoint(args_cli.dagger_ckpt)
    _why = lean_student_refusal(_ckpt_peek)
    if _why:
        raise SystemExit(f"[PPOPC] {args_cli.dagger_ckpt}: {_why}")
    align_pc_dims_to_ckpt(env_cfg, _ckpt_peek, tag="PPOPC")

    # Inherit the env-side PC transforms (force scale / gate / repr / ablations)
    # from the DAgger ckpt, so the fine-tune sees the same input distribution
    # the warm-start weights were trained on. CLI flags still win.
    _pc_env_meta = apply_pc_env_meta(
        env_cfg,
        _ckpt_peek,
        overrides={
            "pc_ablate_tactile_pc": True if args_cli.pc_ablate_tactile_pc else None,
            "pc_ablate_tactile_force": True if args_cli.pc_ablate_tactile_force else None,
            "pc_force_repr": args_cli.pc_force_repr if args_cli.pc_force_repr != "scalar" else None,
        },
        tag="PPOPC",
    )

    print(f"\n[PPOPC] Creating env: {args_cli.task}, num_envs={args_cli.num_envs}")
    env_raw = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = env_raw.unwrapped
    device = str(env.device)

    obs_dim = env.cfg.observation_space
    action_dim = env.cfg.action_space
    priv_info_dim = int(getattr(env.cfg, "priv_info_dim", 148))

    # Read DAgger student layout from ckpt so PPO's actor MLP matches and
    # the warm-start weights actually load.
    dagger_ckpt = _ckpt_peek
    student_hidden = tuple(dagger_ckpt.get("student_hidden", (512, 256)))
    print(f"[PPOPC] reading student_hidden={student_hidden} from DAgger ckpt")

    cfg = PPOPointCloudConfig(
        proprio_dim=obs_dim,
        action_dim=action_dim,
        priv_info_dim=priv_info_dim,
        n_scene=int(env.cfg.pc_num_scene_points),
        n_hand=int(env.cfg.pc_num_hand_points),
        n_tactile=int(env.cfg.pc_num_tactile_points),
        tactile_feat_dim=int(env.cfg.pc_tactile_feature_dim),
        pc_fusion_strategy=str(env.cfg.pc_fusion_strategy),
        pc_output_dim=int(env.cfg.pc_encoder_output_dim),
        ablate_tactile_pc=bool(getattr(env.cfg, "pc_ablate_tactile_pc", False)),
        pc_type_repr=str(dagger_ckpt.get("pc_type_repr", "scalar")),
        # Carry the resolved env-side PC transforms into the PPO ckpt too.
        pc_env_meta=dict(_pc_env_meta),
        # Match DAgger student MLP exactly so warm-start loads end-to-end.
        actor_units=student_hidden,
        critic_units=student_hidden,
        num_envs=args_cli.num_envs,
        horizon=args_cli.horizon,
        mini_epochs=args_cli.mini_epochs,
        minibatch_size=args_cli.minibatch_size,
        lr=args_cli.lr,
        clip=args_cli.clip,
        entropy_coef=args_cli.entropy_coef,
        init_logstd=args_cli.init_logstd,
        max_iters=args_cli.max_iters,
        save_every_n=args_cli.save_every_n,
        out_dir=args_cli.out_dir,
        device=device,
    )

    print("=" * 72)
    print("[PPOPC] DIMENSION SUMMARY")
    print("=" * 72)
    print(f"  proprio={obs_dim}  action={action_dim}  priv={priv_info_dim}")
    print(f"  PC: scene={cfg.n_scene}  hand={cfg.n_hand}  tactile={cfg.n_tactile} "
          f"fusion={cfg.pc_fusion_strategy}")
    print(f"  PPO: horizon={cfg.horizon}  envs={cfg.num_envs}  "
          f"minibatch={cfg.minibatch_size}  mini_epochs={cfg.mini_epochs}  lr={cfg.lr}")
    print("=" * 72 + "\n")

    trainer = PPOPointCloud(cfg=cfg, env=env_raw, dagger_ckpt_path=args_cli.dagger_ckpt)
    trainer.run()
    env_raw.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
