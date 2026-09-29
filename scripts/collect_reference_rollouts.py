# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record rollouts of a trained franka-sharpa policy into pkl files.

Each rollout = one episode of one parallel env, captured frame-by-frame:
arm joint pos/vel, hand joint pos/vel (both sorted-USD order and cfg/Sharpa
order), wrist pose + velocities, object pose + velocities, action, demo frame
progress, demo idx. Episodes shorter than --min_length are dropped.

Example:
    python scripts/collect_reference_rollouts.py \\
        --task franka-sharpa-force-critic-horizon-simteacher \\
        --num_envs 32 \\
        --load_path logs/.../best.pth \\
        --data_idx '["rt/0416_grasp/cube_small_1"]' \\
        --num_rollouts 16 \\
        --min_length 100 \\
        --out_dir data/recordings/sim_teacher_$(date +%Y%m%d_%H%M%S)
"""

import argparse
import ast
import json
import os
import shutil
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record policy rollouts.")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--load_path", type=str, required=True, help="Policy ckpt.")
parser.add_argument("--side", type=str, default=None)
parser.add_argument("--data_idx", type=str, default=None,
                    help="JSON/Python list, e.g. '[\"rt/foo/bar\"]'")
parser.add_argument("--start_frame", type=int, default=None)
parser.add_argument("--num_rollouts", type=int, default=16,
                    help="Stop once this many rollouts of length>=min_length are saved.")
parser.add_argument("--min_length", type=int, default=100,
                    help="Minimum episode length (frames) to keep. Shorter are discarded.")
parser.add_argument("--env_filter", type=str, default=None,
                    help="Comma-separated env indices to record from (e.g. '0' for env-0 "
                         "only, or '0,1,2'). When set, episodes from any other env are "
                         "dropped — useful when you want to run headless with many envs "
                         "and only keep one env's trajectory for later replay/video.")
parser.add_argument("--max_steps", type=int, default=10000,
                    help="Hard cap on total env-steps (safety stop).")
parser.add_argument("--out_dir", type=str, default=None,
                    help="Output dir. Default: data/recordings/{task}_{timestamp}/")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import pickle  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import DirectRLEnvCfg  # noqa: E402

import dexx.tasks.franka_sharpa  # noqa: F401, E402

from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

from dexx.algo.ppo.ppo import PPO  # noqa: E402
from dexx.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from dexx.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _grab_state_per_env(env_unwrapped):
    """Snapshot full per-env state. Returns dict of CPU tensors shape [num_envs, ...]."""
    e = env_unwrapped
    # Make sure cached state buffers reflect current physics step.
    e._refresh_lab()

    # Full joint pos/vel in articulation's native joint order — what replay needs.
    full_joint_pos = e.hand.data.joint_pos.detach().cpu()         # [N, J_total]
    full_joint_vel = e.hand.data.joint_vel.detach().cpu()         # [N, J_total]

    state = {
        # Robot kinematics (env-local frame, env_origins already subtracted)
        "wrist_pos":        e.base_pos.detach().cpu(),            # [N, 3]
        "wrist_quat_wxyz":  e.base_quat.detach().cpu(),           # [N, 4] (Isaac is wxyz)
        "wrist_lin_vel":    e.base_lin_vel.detach().cpu(),        # [N, 3]
        "wrist_ang_vel":    e.base_ang_vel.detach().cpu(),        # [N, 3]
        "arm_joint_pos":    e.arm_joint_pos.detach().cpu(),       # [N, 7]
        "arm_joint_vel":    e.arm_joint_vel.detach().cpu(),       # [N, 7]
        # Hand DOFs in two orderings (USD/sorted vs cfg/Sharpa; see docs/JOINT_ORDERING.md)
        "hand_dof_pos_sorted":  e.hand_dof_pos.detach().cpu(),    # [N, 22] policy frame
        "hand_dof_vel_sorted":  e.hand_dof_vel.detach().cpu(),    # [N, 22]
        "hand_dof_pos_cfg":     e.real_hand_dof_pos.detach().cpu(),  # [N, 22] Sharpa
        "hand_dof_vel_cfg":     e.real_hand_dof_vel.detach().cpu(),  # [N, 22]
        # Full articulation joint vectors — replay reuses these directly.
        "full_joint_pos":   full_joint_pos,                       # [N, J_total]
        "full_joint_vel":   full_joint_vel,                       # [N, J_total]
        # Object kinematics (env-local)
        "object_pos":       e.object_pos.detach().cpu(),          # [N, 3]
        "object_quat_wxyz": e.object_rot.detach().cpu(),          # [N, 4]
        "object_lin_vel":   e.object_linvel.detach().cpu(),       # [N, 3]
        "object_ang_vel":   e.object_angvel.detach().cpu(),       # [N, 3]
        # Demo progress (frame index into the demo sequence this env is on)
        "progress_buf":     e.progress_buf.detach().cpu(),        # [N]
    }
    # opt_joints_pos: hand keypoint body positions in the SAME body order as the
    # retarget pkl's opt_joints_body_names, so the converter can drop them in.
    if _KEYPOINT_IDX is not None:
        kp = e.hand.data.body_pos_w[:, _KEYPOINT_IDX] - e.scene.env_origins.unsqueeze(1)
        state["opt_joints_pos"] = kp.detach().cpu()               # [N, K, 3]
    return state


_KEYPOINT_IDX = None   # set in main() after reset: body indices of opt_joints_body_names


def _stack_episode(frames):
    """List of per-step dicts (each [num_envs, ...]) sliced to one env -> dict of stacked tensors [T, ...]."""
    out = {}
    for k in frames[0].keys():
        out[k] = torch.stack([f[k] for f in frames], dim=0)  # [T, ...]
    return out


@hydra_task_config(args_cli.task, "gym_style_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    if os.path.isdir("outputs/"):
        shutil.rmtree("outputs/")

    # ---- Apply CLI overrides (mirror play.py so behavior is identical) ----
    env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg["seed"] = args_cli.seed
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device else env_cfg.sim.device
    agent_cfg["device"] = args_cli.device if args_cli.device else agent_cfg["device"]
    agent_cfg["load_path"] = args_cli.load_path
    agent_cfg["algorithm"]["num_actors"] = args_cli.num_envs
    agent_cfg["algorithm"]["minibatch_size"] = min([args_cli.num_envs * 8, 32768])

    # Same eval-time toggles as play.py
    env_cfg.randomize_pd_gains = False
    env_cfg.randomize_friction = True
    env_cfg.randomize_com = False
    env_cfg.randomize_mass = False
    env_cfg.sim.gravity = (0, 0, -9.81)

    if args_cli.side and hasattr(env_cfg, "hand_side"):
        env_cfg.hand_side = args_cli.side

    if args_cli.data_idx is not None and hasattr(env_cfg, "data_indices"):
        try:
            data_indices = json.loads(args_cli.data_idx)
        except json.JSONDecodeError:
            data_indices = ast.literal_eval(args_cli.data_idx)
        if not isinstance(data_indices, list):
            raise ValueError(f"--data_idx must be a list, got {type(data_indices)}")
        env_cfg.data_indices = data_indices

    if args_cli.start_frame is not None and hasattr(env_cfg, "fixed_reset_demo_frame"):
        env_cfg.fixed_reset_demo_frame = int(args_cli.start_frame)
        env_cfg.random_state_init = False

    config = ConfigWrapper(agent_cfg, env_cfg, test=True)

    # ---- Output dir ----
    out_dir = args_cli.out_dir or os.path.join(
        "data", "recordings",
        f"{args_cli.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Recording to: {out_dir}")

    # ---- Build env + agent ----
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)
    log_dir = os.path.join("logs", "gym_style", "_record_tmp",
                          datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    _agents = {"PPO": PPO}
    if agent_cfg["algo"] not in _agents:
        raise SystemExit(f"unsupported algo {agent_cfg['algo']!r} in the agent cfg; "
                         f"this release ships only {sorted(_agents)}")
    agent = _agents[agent_cfg["algo"]](env, output_dir=log_dir, full_config=config,
                                       create_output_dir=False)
    agent.restore_test(agent_cfg["load_path"])
    agent.set_eval()

    e = env.unwrapped  # FrankaSharpa* env
    num_envs = e.num_envs

    # Per-env demo label (which data_idx this env is on, fixed for whole run).
    env_data_indices = []
    for i in range(num_envs):
        env_data_indices.append(e.data_indices[i % len(e.data_indices)])

    # ---- Per-env episode buffers ----
    env_buffers = [list() for _ in range(num_envs)]
    env_actions_buf = [list() for _ in range(num_envs)]
    env_reward_acc = [0.0 for _ in range(num_envs)]   # cumulative episode reward per env

    # Parse env_filter into a set of allowed env indices (None = all).
    if args_cli.env_filter is not None:
        allowed_envs = set(int(x) for x in args_cli.env_filter.split(",") if x.strip() != "")
        bad = [i for i in allowed_envs if i < 0 or i >= num_envs]
        if bad:
            raise ValueError(f"--env_filter contains out-of-range env idx {bad} (num_envs={num_envs}).")
        print(f"[INFO] env_filter active: only saving rollouts from envs {sorted(allowed_envs)}")
    else:
        allowed_envs = None

    saved_count = 0
    discarded_short = 0
    discarded_filtered = 0
    step_idx = 0

    # ---- Init meta ----
    meta = {
        "task": args_cli.task,
        "load_path": args_cli.load_path,
        "data_indices": env_cfg.data_indices if hasattr(env_cfg, "data_indices") else None,
        "num_envs": num_envs,
        "min_length": args_cli.min_length,
        "joint_names": list(e.hand.joint_names),
        "arm_joint_indices": [int(x) for x in e.arm_joint_indices],
        "actuated_dof_indices_sorted": [int(x) for x in e.actuated_dof_indices],
        "hand_joint_indices_cfg": [int(x) for x in e.hand_joint_indices],
        "actuated_joint_names_cfg": list(e.cfg.actuated_joint_names),
        "control_mode": (
            "joint_delta" if getattr(e.cfg, "use_joint_delta_control", False)
            else "joint_pos" if getattr(e.cfg, "use_joint_pos_control", False)
            else "pid_ik" if getattr(e.cfg, "use_pid_control", False)
            else "osc" if getattr(e.cfg, "use_osc_control", False)
            else "force_torque"
        ),
        "joint_delta_scale": float(getattr(e.cfg, "joint_delta_scale", 0.0)),
        "actions_moving_average": float(getattr(e.cfg, "actions_moving_average", 0.0)),
        "decimation": int(e.cfg.decimation),
        "physics_dt": float(e.cfg.sim.dt),
        "control_dt": float(e.cfg.sim.dt) * int(e.cfg.decimation),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        # Stringify per-env data indices.
        meta_json = dict(meta)
        meta_json["env_data_indices"] = list(env_data_indices)
        json.dump(meta_json, f, indent=2, default=str)

    # ---- Resolve hand keypoint (opt_joints_pos) body indices ----
    # The retarget pkl stores opt_joints_pos in `opt_joints_body_names` order
    # (== dexhand.body_names[:K]). Capture the SAME bodies from the live
    # articulation so the converter can overwrite opt_joints_pos 1:1.
    global _KEYPOINT_IDX
    # Capture opt_joints_pos for the env's OWN hand_body_names (Isaac articulation
    # body names, guaranteed present). The env's reference loader maps
    # opt_joints_pos by name using target_body_names = hand_body_names\{wrist},
    # so storing opt_joints_body_names = hand_body_names lets it index 1:1.
    kp_names = list(getattr(e, "hand_body_names", []))
    kp_idx = list(getattr(e, "hand_body_indices", []))
    if kp_names and kp_idx and len(kp_names) == len(kp_idx):
        _KEYPOINT_IDX = [int(x) for x in kp_idx]
        print(f"[INFO] opt_joints_pos: capturing {len(_KEYPOINT_IDX)} hand keypoint bodies "
              f"({kp_names[:3]}... wrist first={kp_names[0]})")
        meta["opt_joints_body_names"] = list(kp_names)
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            mj = dict(meta); mj["env_data_indices"] = list(env_data_indices)
            json.dump(mj, f, indent=2, default=str)
    else:
        print("[WARN] env has no hand_body_names/indices; opt_joints_pos will be skipped.")

    # ---- Patch _get_rewards to snapshot per-env success and state BEFORE auto-reset ----
    # step() resets finished envs after _get_rewards, so the state it returns for
    # them is already the next episode's start; the true last frame is taken here.
    _orig_get_rewards = e._get_rewards
    def _patched_get_rewards():
        out = _orig_get_rewards()
        e._last_success_buf = e.success_buf.detach().clone()
        e._pre_reset_state = _grab_state_per_env(e)
        return out
    e._get_rewards = _patched_get_rewards

    # ---- Initial reset and grab pre-step snapshot for frame 0 ----
    obs_dict = env.reset()
    init_state = _grab_state_per_env(e)
    for i in range(num_envs):
        env_buffers[i].append({k: v[i].clone() for k, v in init_state.items()})

    print(f"[INFO] Recording up to {args_cli.num_rollouts} rollouts "
          f"(min_length={args_cli.min_length}). Hard cap: {args_cli.max_steps} steps.")

    while saved_count < args_cli.num_rollouts and step_idx < args_cli.max_steps:
        # ---- Policy inference ----
        input_dict = {
            "obs": agent.running_mean_std(obs_dict["obs"]),
            "priv_info": obs_dict.get("priv_info"),
        }
        mu = agent.model.act_inference(input_dict)
        mu = torch.clamp(mu, -1.0, 1.0)

        # Record action just chosen for each env
        action_cpu = mu.detach().cpu()

        # ---- Step env ----
        obs_dict, r, done, info = env.step(mu)
        step_idx += 1

        # The frame this action led to (pre-reset), and the state after step()
        # (for finished envs, the next episode's first frame).
        step_state = e._pre_reset_state
        post_state = _grab_state_per_env(e)
        r_cpu = r.detach().cpu().flatten()
        # Append to each env's buffer + accumulate reward
        for i in range(num_envs):
            env_actions_buf[i].append(action_cpu[i].clone())
            env_buffers[i].append({k: v[i].clone() for k, v in step_state.items()})
            env_reward_acc[i] += float(r_cpu[i])

        # ---- Handle resets (done==1) ----
        done_cpu = done.detach().cpu()
        for i in range(num_envs):
            if done_cpu[i].item():
                ep_len = len(env_buffers[i]) - 1  # steps of this episode
                if allowed_envs is not None and i not in allowed_envs:
                    discarded_filtered += 1
                    env_buffers[i] = [{k: v[i].clone() for k, v in post_state.items()}]
                    env_actions_buf[i] = []
                    env_reward_acc[i] = 0.0
                    continue
                if ep_len >= args_cli.min_length:
                    ep = _stack_episode(env_buffers[i])               # [T+1, ...] initial + one per step
                    actions = torch.stack(env_actions_buf[i], dim=0)  # [T, ...] action at frame k
                    out_path = os.path.join(out_dir, f"rollout_{saved_count:04d}.pkl")
                    with open(out_path, "wb") as f:
                        succeeded_i = (float(e._last_success_buf[i])
                                       if hasattr(e, "_last_success_buf") else 0.0)
                        n_fr = int(ep["wrist_pos"].shape[0])
                        total_rew = float(env_reward_acc[i])
                        pickle.dump({
                            "env_idx": i,
                            "data_idx": env_data_indices[i],
                            "num_frames": n_fr,
                            "succeeded": succeeded_i,     # env success flag at termination
                            "total_reward": total_rew,    # cumulative episode reward
                            "mean_reward": total_rew / max(n_fr - 1, 1),  # per-step reward
                            "actions": actions,           # [T, action_dim]
                            **{k: v for k, v in ep.items()},
                        }, f)
                    saved_count += 1
                    print(f"[OK ] env {i:3d}  len={n_fr}  succ={succeeded_i:.0f}  "
                          f"rew={total_rew:.1f}  data={env_data_indices[i]}  "
                          f"-> {os.path.basename(out_path)} ({saved_count}/{args_cli.num_rollouts})")
                    if saved_count >= args_cli.num_rollouts:
                        break
                else:
                    discarded_short += 1
                # Start a fresh buffer for this env from the state after step(),
                # which is the next episode's first frame (the env auto-resets).
                env_buffers[i] = [{k: v[i].clone() for k, v in post_state.items()}]
                env_actions_buf[i] = []
                env_reward_acc[i] = 0.0

    print(f"[DONE] saved={saved_count}, discarded_short={discarded_short}, "
          f"discarded_filtered={discarded_filtered}, steps={step_idx}, out={out_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
