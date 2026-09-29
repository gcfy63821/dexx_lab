# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with Gym-Style agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import shutil
import json
import ast
import re

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, write_through=True)

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent.")
parser.add_argument("--num_envs", type=int, default=16384, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--load_path", type=str, default=None, help="Checkpoint path.")
parser.add_argument("--max_agent_steps", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--resume", action="store_true", default=False, help="Resume training from checkpoint.")
parser.add_argument("--wandb-project-name", type=str, default="dex", help="the wandb's project name")
parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")
parser.add_argument("--wandb-name", type=str, default="dexmanip", help="the name of wandb's run")
parser.add_argument("--no-wandb", action="store_true", help="do not log to Weights & Biases")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=10000, help="Interval between video recordings (in agent steps).")
parser.add_argument("--side", type=str, default=None, help="Hand side (left or right)")
parser.add_argument("--data_idx", type=str, default=None, help="Data indices as a JSON/Python list string, e.g., '[\"925aa@1\", \"20aed@0\"]'")
parser.add_argument("--robot_asset", type=str, default=None, help="Optional custom robot URDF (abs or repo-relative). Default uses assets/generated/fr3_with_{side}_sharpa_wave.urdf.")
parser.add_argument("--material_elastomer_ids", type=str, default=None, help="JSON list of fingertip-elastomer collision-shape indices for friction DR (calibrate with tools/calibrate_elastomer_ids.py after any asset change). Default None -> [27,28,30,32,33] for the stock asset.")
parser.add_argument("--env_cfg", type=str, nargs="*", default=[], help="Override env_cfg fields, e.g., --env_cfg enable_tactile=False binary_contact=True force_reward_weight=0.0")
parser.add_argument("--no_contact_force", action="store_true", default=False, help="Modality ablation: disable the 5-d scalar contact force in the observation (env_cfg.enable_contact_force=False).")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from dexx.algo.ppo.ppo import PPO
from dexx.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper
from dexx.wrapper.config_wrapper import ConfigWrapper

from isaaclab.envs import DirectRLEnvCfg

import dexx.tasks.franka_sharpa  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
import wandb




@hydra_task_config(args_cli.task, "gym_style_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    # Do not delete Hydra's global outputs/ directory here. Multi-process
    # ablation launches race on this path and can crash each other during
    # startup; per-run logs are written under logs/gym_style instead.
    """Train with Gym-Style agent."""
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg["algorithm"]["max_agent_steps"] = args_cli.max_agent_steps if args_cli.max_agent_steps is not None else agent_cfg["algorithm"]["max_agent_steps"]
    agent_cfg["algorithm"]["num_actors"] = args_cli.num_envs if args_cli.num_envs is not None else agent_cfg["algorithm"]["num_actors"]
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg['seed']
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg["device"] = args_cli.device if args_cli.device is not None else agent_cfg["device"]
    agent_cfg["load_path"] = args_cli.load_path if args_cli.load_path is not None else agent_cfg["load_path"]
    if args_cli.robot_asset is not None and hasattr(env_cfg, "robot_asset_override"):
        env_cfg.robot_asset_override = args_cli.robot_asset
    if args_cli.material_elastomer_ids is not None and hasattr(env_cfg, "material_elastomer_ids"):
        env_cfg.material_elastomer_ids = json.loads(args_cli.material_elastomer_ids)
    agent_cfg["algorithm"]['minibatch_size'] = min([args_cli.num_envs * 8, 32768])

    # Set data_indices if provided
    if args_cli.data_idx is not None:
        if hasattr(env_cfg, 'data_indices'):
            # Parse JSON or Python list format
            try:
                # Try JSON format first (requires double quotes)
                data_indices = json.loads(args_cli.data_idx)
            except json.JSONDecodeError:
                # Fall back to Python literal evaluation (supports both single and double quotes)
                try:
                    data_indices = ast.literal_eval(args_cli.data_idx)
                except (ValueError, SyntaxError) as e:
                    raise ValueError(f"Invalid data_idx format: {args_cli.data_idx}. "
                                   f"Expected JSON or Python list format, e.g., '[\"925aa@1\", \"20aed@0\"]'. "
                                   f"Error: {e}")
            if not isinstance(data_indices, list):
                raise ValueError(f"data_idx must be a list, got {type(data_indices)}")
            env_cfg.data_indices = data_indices

    # Apply --env_cfg overrides (e.g., --env_cfg enable_tactile=False binary_contact=True)
    for override in args_cli.env_cfg:
        if "=" not in override:
            print(f"[WARN] Skipping invalid --env_cfg override (no '='): {override}")
            continue
        key, val_str = override.split("=", 1)
        if not hasattr(env_cfg, key):
            print(f"[WARN] env_cfg has no attribute '{key}', setting anyway")
        # Auto-cast value
        if val_str.lower() in ("true", "false"):
            val = val_str.lower() == "true"
        else:
            try:
                val = int(val_str)
            except ValueError:
                try:
                    val = float(val_str)
                except ValueError:
                    val = val_str
        setattr(env_cfg, key, val)
        print(f"[env_cfg override] {key} = {val} ({type(val).__name__})")

    # Modality ablation: zero the 5-d scalar contact force in the obs.
    if args_cli.no_contact_force:
        env_cfg.enable_contact_force = False
        print("[ablation] enable_contact_force = False (5d contact force zeroed in obs)")

    config = ConfigWrapper(agent_cfg, env_cfg)

    # specify directory for logging experiments
    log_root_path = os.path.abspath(os.path.join("logs", "gym_style", agent_cfg["algorithm"]["experiment_name"]))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    run_label = args_cli.wandb_name or args_cli.task or "run"
    run_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_label).strip("_") or "run"
    log_dir = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')}_{run_label}"
    log_dir = os.path.join(log_root_path, log_dir)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if args_cli.side is not None:
        if hasattr(env_cfg, 'hand_side'):
            env_cfg.hand_side = args_cli.side

    # create isaac environment
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    env = GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)
    
    # Store video recording info for agent (will be used in PPO training loop)
    if args_cli.video:
        video_folder = os.path.join(log_dir, "stage1_nn", "videos")
        os.makedirs(video_folder, exist_ok=True)
        print(f"[INFO] Video recording enabled")
        print(f"[INFO] Videos will be saved to: {video_folder}")
        print(f"[INFO] Video length: {args_cli.video_length} steps")
        print(f"[INFO] Video interval: {args_cli.video_interval} agent steps")
        agent_cfg["video"] = {
            "enabled": True,
            "folder": video_folder,
            "interval": args_cli.video_interval,
            "length": args_cli.video_length,
        }
    else:
        agent_cfg["video"] = {"enabled": False}
    _agents = {"PPO": PPO}
    if agent_cfg["algo"] not in _agents:
        raise SystemExit(f"unsupported algo {agent_cfg['algo']!r} in the agent cfg; "
                         f"this release ships only {sorted(_agents)}")
    agent = _agents[agent_cfg["algo"]](env, output_dir=log_dir, full_config=config)
    
    spec = gym.spec(args_cli.task)
    # Resolve cfg source files via the actual package location (works under the
    # src/ layout instead of assuming a cwd-relative dexx/... path).
    import importlib
    _env_mod = spec.kwargs.get("env_cfg_entry_point", None).split(":")[0]
    env_cfg_file = importlib.import_module(_env_mod).__file__
    _agent_ep = spec.kwargs.get("gym_style_cfg_entry_point", None)
    _agent_pkg, _agent_res = _agent_ep.split(":")
    agent_cfg_file = os.path.join(os.path.dirname(importlib.import_module(_agent_pkg).__file__), _agent_res)
    shutil.copy(env_cfg_file, os.path.join(log_dir, "env_cfg.py"))
    shutil.copy(agent_cfg_file, os.path.join(log_dir, "agent_cfg.yaml"))

    # Save runtime config snapshot (includes CLI overrides and actual values)
    runtime_config = {
        "cli_args": vars(args_cli),
        "agent_cfg": agent_cfg,
        "env_cfg": {
            k: getattr(env_cfg, k) for k in sorted(dir(env_cfg))
            if not k.startswith('_') and not callable(getattr(env_cfg, k, None))
            and not isinstance(getattr(env_cfg, k, None), type)
        },
        "task": args_cli.task,
        "log_dir": log_dir,
    }
    import yaml as _yaml
    with open(os.path.join(log_dir, "runtime_config.yaml"), "w") as f:
        _yaml.dump(runtime_config, f, default_flow_style=False, allow_unicode=True)
    print(f"[INFO] Saved runtime config to {os.path.join(log_dir, 'runtime_config.yaml')}")

    # load the checkpoint
    if args_cli.load_path is not None and not args_cli.resume:
        print("[WARN] --load_path was given without --resume, so the checkpoint "
              "is NOT loaded and training starts from scratch. Add --resume to "
              "actually restore it.", flush=True)

    if args_cli.resume:
        resume_path = agent_cfg["load_path"]
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        agent.restore_train(resume_path)
    
    
    is_main_process = (
        not torch.distributed.is_initialized()
        or torch.distributed.get_rank() == 0
    )

    if is_main_process and not args_cli.no_wandb:
        run_name = f"{args_cli.wandb_name}_{os.path.basename(log_dir)}"

        wandb.init(
            project=args_cli.wandb_project_name,
            entity=args_cli.wandb_entity,
            name=run_name,
            config=config.to_dict() if hasattr(config, "to_dict") else None,
        )

        agent.use_wandb = True
        agent.wandb = wandb
    else:
        agent.use_wandb = False

    # run training
    agent.train()

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
