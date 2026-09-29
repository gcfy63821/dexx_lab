"""Comprehensive policy evaluation script.

Measures success rate + survival across multiple init-distribution modes:

  1. early       — seq_idx ~ Uniform[0, early_max]      (default early_max=10)
                   tests "reach from very start"
  2. pre_contact — seq_idx ~ Uniform[0, first_contact[i] - 1]   per-env
                   tests "reach phase only" (frames before any fingertip touches obj)
  3. random      — seq_idx ~ Uniform[0, 0.98*seq_len[i]]   per-env
                   tests "full task distribution"
  4. per_stage   — N stages by percentile (e.g., 5 quantiles).
                   For each stage i: seq_idx ~ Uniform[i/N * seq_len, (i+1)/N * seq_len]
                   tests survival + success conditional on starting in stage i.

Success definition (RELAXED): episode reaches `||obj_pos - final_obj_pos|| < success_dist`
at any step during the rollout. Default threshold 0.05 m. Mid-episode
fail/* events DO terminate the episode (default behavior) unless
`--eval_no_terminate` is passed, in which case episode runs until
`episode_length_buf >= seq_len - 1`.

Per-episode records (saved to JSON):
  - init_frame
  - survival_len (step count from reset until done)
  - succeeded (bool)
  - min_final_dist (smallest obj-to-final distance seen this ep)
  - fail_cause (list of fail/* that fired)

Usage:
  python scripts/eval_teacher.py \
      --task franka-sharpa-force-poseobs \
      --load_path logs/.../best.pth \
      --side right \
      --data_idx '[...10 demos...]' \
      --num_envs 64 \
      --episodes_per_mode 200 \
      --modes early pre_contact random per_stage \
      --num_stages 5 \
      --success_dist 0.05 \
      --out_dir logs/policy_eval_$(date +%Y%m%d_%H%M%S) \
      --headless
"""

from __future__ import annotations
import argparse
import json
import os
import sys

# IMPORTANT: AppLauncher must come first
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Comprehensive policy evaluation.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--load_path", type=str, required=True)
parser.add_argument("--side", type=str, default="right")
parser.add_argument("--data_idx", type=str, required=True,
                    help="JSON list of data_idx strings")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes_per_mode", type=int, default=200,
                    help="Stop each mode after this many episode terminations.")
parser.add_argument("--max_steps_per_mode", type=int, default=4000,
                    help="Hard ceiling on play steps per mode (safety).")
parser.add_argument("--modes", nargs="+",
                    default=["early", "pre_contact", "random", "per_stage"],
                    choices=["early", "pre_contact", "random", "per_stage"])
parser.add_argument("--save_traj", action="store_true", default=False,
                    help="Store per-frame trajectory tracking arrays "
                         "(traj_pos_dist, traj_rot_deg) in records.json. "
                         "Max/mean summary stats are always saved regardless.")
parser.add_argument("--early_max_frame", type=int, default=10,
                    help="`early` mode: sample seq_idx from [0, early_max_frame]")
parser.add_argument("--num_stages", type=int, default=5,
                    help="Number of percentile stages for `per_stage` mode.")
parser.add_argument("--contact_threshold", type=float, default=0.02,
                    help="Per-fingertip distance to obj surface (m) considered 'contact'.")
parser.add_argument("--success_dist", type=float, default=0.05,
                    help="Episode succeeds if min(||obj_pos - final_obj_pos||) < this.")
parser.add_argument("--eval_no_terminate", action="store_true",
                    help="Disable env terminations; let episode run to time_out.")
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--expand_aug", action="store_true", default=False,
                    help="Evaluate every retarget augmentation variant ({demo}@{aug}) of each "
                         "demo. By default only the base demos are evaluated and reported "
                         "under their base IDs (DISABLE_AUG_EXPAND=1).")

AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Rest follows
# Base demos unless --expand_aug: the env reads this when it builds data_indices.
if args.expand_aug:
    os.environ.pop("DISABLE_AUG_EXPAND", None)
else:
    os.environ["DISABLE_AUG_EXPAND"] = "1"

import ast
import json as _json
import time
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
import torch

import dexx.tasks.franka_sharpa  # noqa: F401
from dexx.algo.ppo.ppo import PPO
from dexx.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper
from dexx.wrapper.config_wrapper import ConfigWrapper
from isaaclab.envs import DirectRLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config


def parse_data_idx(s: str) -> list[str]:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return ast.literal_eval(s)


@dataclass
class EpisodeRecord:
    mode: str
    env_id: int
    init_frame: int
    survival_len: int            # number of env.step() calls before this episode terminated
    succeeded: bool              # min_final_dist < success_dist during episode
    min_final_dist: float        # smallest obj-to-final distance over episode
    end_final_dist: float = -1.0 # obj-to-final distance at the step episode ended
    min_final_rot_deg: float = -1.0   # smallest |Δrot| over episode (deg)
    end_final_rot_deg: float = -1.0   # |Δrot| at the step episode ended (deg)
    # Per-frame trajectory tracking stats: |obj_pos(t) - demo_pos(t)| over episode.
    max_traj_pos_dist: float = -1.0    # worst-frame obj→demo position deviation (m)
    mean_traj_pos_dist: float = -1.0   # mean over episode (m)
    max_traj_rot_deg: float = -1.0     # worst-frame obj→demo orientation deviation (deg)
    mean_traj_rot_deg: float = -1.0    # mean over episode (deg)
    # Optional per-frame trajectory arrays (enabled by --save_traj).
    traj_pos_dist: list[float] = field(default_factory=list)
    traj_rot_deg: list[float] = field(default_factory=list)
    fail_causes: list[str] = field(default_factory=list)
    # Object pose (env-local frame, meters) at episode reset (start) and at
    # the step it terminated (end). 3-d each.
    obj_start: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    obj_end: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


# ---- main ---------------------------------------------------------------
@hydra_task_config(args.task, "gym_style_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[eval_policy] out_dir = {args.out_dir}")

    # Common env setup
    env_cfg.scene.num_envs = args.num_envs
    agent_cfg["algorithm"]["num_actors"] = args.num_envs
    agent_cfg["algorithm"]["minibatch_size"] = min(args.num_envs * 8, 32768)
    agent_cfg["seed"] = args.seed
    env_cfg.seed = args.seed
    agent_cfg["load_path"] = args.load_path
    if hasattr(env_cfg, "hand_side"):
        env_cfg.hand_side = args.side
    if hasattr(env_cfg, "data_indices"):
        env_cfg.data_indices = parse_data_idx(args.data_idx)
    # Disable randomness / curriculum for clean eval
    env_cfg.randomize_pd_gains = False
    env_cfg.randomize_friction = False
    env_cfg.randomize_com = False
    env_cfg.randomize_mass = False
    env_cfg.sim.gravity = (0, 0, -9.81)
    if hasattr(env_cfg, "init_curriculum_enabled"):
        env_cfg.init_curriculum_enabled = False
    env_cfg.random_state_init = True  # We override via _eval_init_seq_idx_*
    if args.eval_no_terminate and hasattr(env_cfg, "eval_no_terminate"):
        env_cfg.eval_no_terminate = True

    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    env = GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)
    base_env = env.unwrapped
    device = base_env.device

    # ---- Capture expanded data_indices for per-task breakdown ----
    # When `rt/` indices are passed, the env auto-expands each base demo into
    # all its retarget-augmentation variants (each variant = a different object
    # position / yaw, e.g. `peg_1@0_dxn1.3cm_dyn5.7cm_yawn1.8deg`). That aug
    # spread IS the object-position randomization for this eval. Env `e` is
    # bound to `data_indices[e % len(data_indices)]`.
    expanded_indices = [str(x) for x in getattr(base_env, "data_indices", [])]
    print(f"[eval_policy] expanded data_indices: {len(expanded_indices)} aug "
          f"variants; env e -> data_indices[e % {len(expanded_indices)}]")
    if expanded_indices and args.num_envs < len(expanded_indices):
        print(f"[eval_policy] WARNING: num_envs ({args.num_envs}) < aug variants "
              f"({len(expanded_indices)}); variants {args.num_envs}.. are NOT covered.")

    def _demo_category(idx_str: str) -> str:
        """`rt/0420_manip/squeegee_1@0_dxn1.3cm...` -> `squeegee` (strip path/@aug/_N)."""
        base = idx_str.split("@")[0].rstrip("/")
        exp = base.split("/")[-1]
        head, _, tail = exp.rpartition("_")
        return head if (head and tail.isdigit()) else exp

    env_category = [
        _demo_category(expanded_indices[e % len(expanded_indices)])
        if expanded_indices else "unknown"
        for e in range(args.num_envs)
    ]
    with open(os.path.join(args.out_dir, "expanded_data_indices.json"), "w", encoding="utf-8") as f:
        _json.dump({"data_indices": expanded_indices,
                    "env_category": env_category}, f, indent=2)

    # Build agent
    config = ConfigWrapper(agent_cfg, env_cfg, test=True)
    log_dir = os.path.join(args.out_dir, "_log")
    _agents = {"PPO": PPO}
    _algo = agent_cfg.get("algo", "PPO")
    if _algo not in _agents:
        raise SystemExit(f"unsupported algo {_algo!r} in the agent cfg; "
                         f"this release ships only {sorted(_agents)}")
    AgentCls = _agents[_algo]
    agent = AgentCls(env, output_dir=log_dir, full_config=config, create_output_dir=False)
    print(f"[eval_policy] loading ckpt: {args.load_path}")
    agent.restore_test(args.load_path)
    agent.set_eval()

    # ---- Compute per-env first-contact frame from demo tips_distance ----
    # tips_distance: (num_envs, T, 5) — per-frame per-tip distance to obj surface.
    tips_dist = base_env.demo_data["tips_distance"]  # tensor (N, T, 5)
    seq_lens = base_env.demo_data["seq_len"].long()  # (N,)
    in_contact = (tips_dist < args.contact_threshold).any(dim=-1)  # (N, T) bool
    # First True frame per env. If never contact, fallback to seq_len // 2.
    first_contact = torch.zeros(args.num_envs, dtype=torch.long, device=device)
    for i in range(args.num_envs):
        L = int(seq_lens[i].item())
        ic = in_contact[i, :L]
        if ic.any():
            first_contact[i] = int(ic.float().argmax().item())
        else:
            first_contact[i] = max(1, L // 2)  # fallback
    print(f"[eval_policy] first-contact frames (sample): {first_contact[:8].tolist()}")
    print(f"[eval_policy] seq_lens          (sample): {seq_lens[:8].tolist()}")

    # ---- Helpers ----
    def eval_one_mode(mode_name: str, lo: torch.Tensor, hi: torch.Tensor) -> list[EpisodeRecord]:
        """Run episodes until `args.episodes_per_mode` terminations OR
        `args.max_steps_per_mode` steps. Returns per-episode records."""
        print("\n" + "="*68)
        print(f"[mode={mode_name}] lo[:8]={lo[:8].tolist()}  hi[:8]={hi[:8].tolist()}")
        print("="*68)

        base_env.set_eval_init_range(lo, hi)
        # Force a reset with the new range so episode 0 starts in the right place.
        obs_dict = env.reset()
        per_env_init = base_env.progress_buf.clone()  # captured at reset
        per_env_t0 = torch.zeros(args.num_envs, dtype=torch.long, device=device)
        per_env_min_final = torch.full((args.num_envs,), float("inf"), device=device)
        per_env_min_final_rot = torch.full((args.num_envs,), float("inf"), device=device)
        per_env_last_final_rot = torch.full((args.num_envs,), -1.0, device=device)
        prev_final_rot = torch.full((args.num_envs,), -1.0, device=device)
        # Per-frame trajectory tracking (obj_pos / obj_rot vs demo current frame).
        per_env_traj_pos = [[] for _ in range(args.num_envs)]
        per_env_traj_rot = [[] for _ in range(args.num_envs)]
        per_env_fail_causes: list[set[str]] = [set() for _ in range(args.num_envs)]
        # Object position (env-local frame) captured at episode reset.
        # base_env.object_pos: (num_envs, 3) = object.data.root_pos_w - env_origins
        def _obj_pos_now() -> torch.Tensor:
            op = getattr(base_env, "object_pos", None)
            if op is None:
                return torch.zeros((args.num_envs, 3), device=device)
            return op.detach().clone()
        per_env_obj_start = _obj_pos_now()
        # Track dist_to_final per env: per_env_last_final_dist updated each step;
        # prev_final_dist snapshots it for the dying episode's final dist (same
        # timing convention as prev_obj_pos / obj_end).
        per_env_last_final_dist = torch.full((args.num_envs,), -1.0, device=device)
        prev_final_dist = torch.full((args.num_envs,), -1.0, device=device)
        records: list[EpisodeRecord] = []
        fail_keys = [
            "fail/hand_tracking", "fail/obj_pos_drift", "fail/obj_rot_drift",
            "fail/premature_contact", "fail/arm_below_table",
            "fail/error_buf_velocity_explosion",
        ]
        step_counter = 0
        t_start = time.time()
        # object_pos from the previous step. The env reports done=True the step
        # AFTER the reset already ran (terminated[:] = at_reset_buf), so by the
        # time we see `done` the env's object_pos reflects the NEW episode. The
        # terminating episode's end pose is the object_pos from the prior step.
        prev_obj_pos = _obj_pos_now()

        while step_counter < args.max_steps_per_mode and len(records) < args.episodes_per_mode:
            with torch.no_grad():
                input_dict = {
                    "obs": agent.running_mean_std(obs_dict["obs"]),
                    "priv_info": obs_dict["priv_info"],
                }
                mu = agent.model.act_inference(input_dict)
                mu = torch.clamp(mu, -1.0, 1.0)
            obs_dict, _r, done, info = env.step(mu)
            step_counter += 1
            cur_obj_pos = _obj_pos_now()

            # Per-env diagnostics: track min final-pos distance + fail/* fired
            rd = getattr(base_env, "reward_dict", None)
            if rd is not None:
                if "diag/final_pos_dist" in rd:
                    cur_dist = rd["diag/final_pos_dist"]
                    per_env_min_final = torch.minimum(per_env_min_final, cur_dist)
                    per_env_last_final_dist = cur_dist
                if "diag/final_rot_angle" in rd:
                    cur_rot = rd["diag/final_rot_angle"]  # radians
                    per_env_min_final_rot = torch.minimum(per_env_min_final_rot, cur_rot)
                    per_env_last_final_rot = cur_rot
                # Trajectory tracking: obj→demo_current per-frame distance.
                if "diag/obj_pos_dist" in rd:
                    _pd = rd["diag/obj_pos_dist"].detach().cpu().tolist()
                    for _i in range(args.num_envs):
                        per_env_traj_pos[_i].append(float(_pd[_i]))
                if "diag/obj_rot_angle" in rd:
                    import math as _math
                    _rd_arr = rd["diag/obj_rot_angle"].detach().cpu().tolist()
                    for _i in range(args.num_envs):
                        per_env_traj_rot[_i].append(_math.degrees(float(_rd_arr[_i])))
                for k in fail_keys:
                    if k in rd:
                        fired = (rd[k] > 0.5).nonzero(as_tuple=False).flatten()
                        for env_id in fired.tolist():
                            per_env_fail_causes[env_id].add(k.split("/")[-1])

            # Check episode terminations.
            # NOTE: env's step() override does `terminated[:] = at_reset_buf`,
            # which makes done=True at the step IMMEDIATELY AFTER reset (since
            # at_reset_buf is set inside _reset_idx). This produces a "fake"
            # done event at survival_len=1 per env per cycle. We filter these
            # out via `survival_len > min_survival_len`. Real episodes survive
            # at least until first fail check (running_progress_buf >= 20) or
            # time_out (seq_len - 1 ≥ 144 for our demos), so >= 5 is safe.
            done_t = done if torch.is_tensor(done) else torch.tensor(done, device=device)
            if done_t.any():
                done_ids = done_t.nonzero(as_tuple=False).flatten().tolist()
                for env_id in done_ids:
                    survival_len = step_counter - int(per_env_t0[env_id].item())
                    # Skip fake startup-reset dones (terminated[:] = at_reset_buf
                    # signaling = previous step's reset).
                    if survival_len < 5:
                        per_env_init[env_id] = base_env.progress_buf[env_id]
                        per_env_t0[env_id] = step_counter
                        per_env_min_final[env_id] = float("inf")
                        per_env_min_final_rot[env_id] = float("inf")
                        per_env_traj_pos[env_id] = []
                        per_env_traj_rot[env_id] = []
                        per_env_fail_causes[env_id] = set()
                        # env already reset -> cur_obj_pos is new episode's start
                        per_env_obj_start[env_id] = cur_obj_pos[env_id]
                        continue
                    min_dist = float(per_env_min_final[env_id].item())
                    success = (min_dist < args.success_dist) and \
                              (not torch.isinf(per_env_min_final[env_id]).item())
                    end_dist = float(prev_final_dist[env_id].item())
                    import math as _math
                    min_rot = float(per_env_min_final_rot[env_id].item())
                    end_rot = float(prev_final_rot[env_id].item())
                    # Trajectory stats: max + mean over per-frame obj-vs-demo dist.
                    _tp = per_env_traj_pos[env_id]
                    _tr = per_env_traj_rot[env_id]
                    _max_p = max(_tp) if _tp else -1.0
                    _mean_p = sum(_tp)/len(_tp) if _tp else -1.0
                    _max_r = max(_tr) if _tr else -1.0
                    _mean_r = sum(_tr)/len(_tr) if _tr else -1.0
                    rec = EpisodeRecord(
                        mode=mode_name,
                        env_id=env_id,
                        init_frame=int(per_env_init[env_id].item()),
                        survival_len=survival_len,
                        succeeded=success,
                        min_final_dist=min_dist if min_dist != float("inf") else -1.0,
                        end_final_dist=end_dist if end_dist >= 0 else -1.0,
                        min_final_rot_deg=_math.degrees(min_rot) if min_rot != float("inf") else -1.0,
                        end_final_rot_deg=_math.degrees(end_rot) if end_rot >= 0 else -1.0,
                        max_traj_pos_dist=_max_p,
                        mean_traj_pos_dist=_mean_p,
                        max_traj_rot_deg=_max_r,
                        mean_traj_rot_deg=_mean_r,
                        traj_pos_dist=([round(x, 4) for x in _tp] if args.save_traj else []),
                        traj_rot_deg=([round(x, 2) for x in _tr] if args.save_traj else []),
                        fail_causes=sorted(list(per_env_fail_causes[env_id])),
                        # obj_end = pose at the step BEFORE the reset that
                        # produced this `done` (terminated[:] = at_reset_buf).
                        obj_start=per_env_obj_start[env_id].tolist(),
                        obj_end=prev_obj_pos[env_id].tolist(),
                    )
                    records.append(rec)
                    # Reset per-env trackers
                    per_env_init[env_id] = base_env.progress_buf[env_id]
                    per_env_t0[env_id] = step_counter
                    per_env_min_final[env_id] = float("inf")
                    per_env_min_final_rot[env_id] = float("inf")
                    per_env_traj_pos[env_id] = []
                    per_env_traj_rot[env_id] = []
                    per_env_fail_causes[env_id] = set()
                    # env already reset -> cur_obj_pos is new episode's start
                    per_env_obj_start[env_id] = cur_obj_pos[env_id]
                    if len(records) >= args.episodes_per_mode:
                        break

            # Heartbeat
            if step_counter % 100 == 0:
                ok = sum(1 for r in records if r.succeeded)
                pct = 100.0 * ok / max(1, len(records))
                print(f"  step={step_counter:5d}  eps={len(records):4d}  succ={ok}/{len(records)} ({pct:.1f}%)  elapsed={time.time()-t_start:.0f}s")

            # Advance previous-step trackers (object pose + dist_to_final + rot).
            prev_obj_pos = cur_obj_pos
            prev_final_dist = per_env_last_final_dist.clone()
            prev_final_rot = per_env_last_final_rot.clone()

        t_elapsed = time.time() - t_start
        print(f"[mode={mode_name}] DONE — {len(records)} eps in {t_elapsed:.0f}s")
        base_env.set_eval_init_range(None, None)  # restore default sampling
        return records

    # ---- Define each mode's per-env (lo, hi) ----
    def mode_ranges():
        out = {}
        if "early" in args.modes:
            lo = torch.zeros(args.num_envs, dtype=torch.long, device=device)
            hi = torch.full((args.num_envs,), args.early_max_frame, dtype=torch.long, device=device)
            hi = torch.minimum(hi, (seq_lens - 1).clamp(min=0))
            out["early"] = (lo, hi)
        if "pre_contact" in args.modes:
            lo = torch.zeros(args.num_envs, dtype=torch.long, device=device)
            hi = (first_contact - 1).clamp(min=0)
            out["pre_contact"] = (lo, hi)
        if "random" in args.modes:
            lo = torch.zeros(args.num_envs, dtype=torch.long, device=device)
            hi = (seq_lens.float() * 0.98).long().clamp(min=0)
            out["random"] = (lo, hi)
        if "per_stage" in args.modes:
            N = args.num_stages
            for s in range(N):
                lo_f = s / N
                hi_f = (s + 1) / N
                lo = (seq_lens.float() * lo_f).long()
                hi = (seq_lens.float() * hi_f).long().clamp(max=seq_lens - 1)
                hi = torch.maximum(hi, lo)
                out[f"per_stage/{s}_of_{N}"] = (lo, hi)
        return out

    # ---- Run all modes ----
    all_records: dict[str, list[EpisodeRecord]] = {}
    for mode_name, (lo, hi) in mode_ranges().items():
        recs = eval_one_mode(mode_name, lo, hi)
        all_records[mode_name] = recs

    # ---- Aggregate & save ----
    print("\n" + "="*68)
    print("[eval_policy] SUMMARY")
    print("="*68)
    summary = {}
    for mode, recs in all_records.items():
        if not recs:
            summary[mode] = {"n": 0}
            print(f"  {mode:30s}: 0 eps")
            continue
        n = len(recs)
        succ = sum(1 for r in recs if r.succeeded)
        succ_pct = 100.0 * succ / n
        # Survival len stats
        surv = np.array([r.survival_len for r in recs])
        min_dists = np.array([r.min_final_dist for r in recs if r.min_final_dist >= 0])
        cause_counts: dict[str, int] = {}
        for r in recs:
            for c in r.fail_causes:
                cause_counts[c] = cause_counts.get(c, 0) + 1
        summary[mode] = {
            "n": n,
            "success": succ,
            "success_pct": round(succ_pct, 2),
            "survival_mean": float(surv.mean()),
            "survival_p50": float(np.percentile(surv, 50)),
            "survival_p90": float(np.percentile(surv, 90)),
            "min_final_dist_mean": float(min_dists.mean()) if len(min_dists) > 0 else None,
            "min_final_dist_p50": float(np.percentile(min_dists, 50)) if len(min_dists) > 0 else None,
            "fail_causes": cause_counts,
            "init_frame_mean": float(np.mean([r.init_frame for r in recs])),
            "init_frame_range": [int(min(r.init_frame for r in recs)),
                                  int(max(r.init_frame for r in recs))],
        }
        print(f"\n  --- {mode} ({n} eps) ---")
        print(f"    success_rate    : {succ_pct:.1f}% ({succ}/{n})")
        print(f"    survival len    : mean={surv.mean():.0f}  p50={np.percentile(surv,50):.0f}  p90={np.percentile(surv,90):.0f}")
        print(f"    init_frame range: [{int(min(r.init_frame for r in recs))}, {int(max(r.init_frame for r in recs))}]")
        if len(min_dists) > 0:
            print(f"    min final_dist  : mean={min_dists.mean():.3f}m  p50={np.percentile(min_dists,50):.3f}m")
        if cause_counts:
            print(f"    fail causes     : {cause_counts}")

    # Save JSON
    out_summary = os.path.join(args.out_dir, "summary.json")
    out_records = os.path.join(args.out_dir, "records.json")
    with open(out_summary, "w", encoding="utf-8") as f:
        _json.dump({
            "ckpt": args.load_path,
            "task": args.task,
            "num_envs": args.num_envs,
            "success_dist": args.success_dist,
            "eval_no_terminate": args.eval_no_terminate,
            "modes": summary,
        }, f, indent=2)
    def _env_cat(env_id: int) -> str:
        return env_category[env_id] if env_id < len(env_category) else "unknown"

    def _env_demo(env_id: int) -> str:
        if expanded_indices:
            return expanded_indices[env_id % len(expanded_indices)]
        return "unknown"

    with open(out_records, "w", encoding="utf-8") as f:
        per_mode_records = {
            mode: [
                {
                    "env_id": r.env_id,
                    "category": _env_cat(r.env_id),
                    "demo_idx": _env_demo(r.env_id),
                    "init_frame": r.init_frame,
                    "survival_len": r.survival_len,
                    "succeeded": r.succeeded,
                    "min_final_dist": r.min_final_dist,
                    "end_final_dist": r.end_final_dist,
                    "min_final_rot_deg": r.min_final_rot_deg,
                    "end_final_rot_deg": r.end_final_rot_deg,
                    "max_traj_pos_dist": r.max_traj_pos_dist,
                    "mean_traj_pos_dist": r.mean_traj_pos_dist,
                    "max_traj_rot_deg": r.max_traj_rot_deg,
                    "mean_traj_rot_deg": r.mean_traj_rot_deg,
                    "traj_pos_dist": r.traj_pos_dist,
                    "traj_rot_deg": r.traj_rot_deg,
                    "fail_causes": r.fail_causes,
                    "obj_start": r.obj_start,
                    "obj_end": r.obj_end,
                }
                for r in recs
            ]
            for mode, recs in all_records.items()
        }
        _json.dump(per_mode_records, f, indent=2)
    print(f"\n[eval_policy] saved summary to {out_summary}")
    print(f"[eval_policy] saved records to {out_records}")

    # ---- Per-task (category) + per-stage breakdown (the deliverable table) ----
    def _agg(recs):
        n = len(recs)
        s = sum(1 for r in recs if r.succeeded)
        return s, n, (100.0 * s / n if n else 0.0)

    cats = sorted(set(env_category))
    mode_names = list(all_records.keys())

    # cell[(mode, category)] -> records
    cell: dict[tuple[str, str], list] = {}
    for mode, recs in all_records.items():
        for r in recs:
            c = env_category[r.env_id] if r.env_id < len(env_category) else "unknown"
            cell.setdefault((mode, c), []).append(r)

    by_category: dict[str, Any] = {}
    for c in cats:
        all_c = [r for mode in mode_names for r in cell.get((mode, c), [])]
        s, n, pct = _agg(all_c)
        by_category[c] = {
            "overall": {"success": s, "n": n, "success_pct": round(pct, 2)},
            "by_mode": {
                mode: dict(zip(("success", "n", "success_pct"),
                               (lambda t: (t[0], t[1], round(t[2], 2)))(_agg(cell.get((mode, c), [])))))
                for mode in mode_names
            },
        }
    by_stage = {
        mode: dict(zip(("success", "n", "success_pct"),
                       (lambda t: (t[0], t[1], round(t[2], 2)))(_agg(recs))))
        for mode, recs in all_records.items()
    }

    # Markdown deliverable table
    md = []
    md.append(f"# Policy eval — per-task & per-stage success rate")
    md.append("")
    md.append(f"- ckpt: `{args.load_path}`")
    md.append(f"- task: `{args.task}`  ·  success = obj reaches goal pose within "
              f"**{args.success_dist*100:.0f} cm**  ·  num_envs: {args.num_envs}")
    md.append(f"- object-position randomization: **{len(expanded_indices)} retarget-aug "
              f"variants** across {len(cats)} task categories")
    md.append(f"- eval_no_terminate: {args.eval_no_terminate}")
    md.append("")
    md.append("## Per-task success rate")
    md.append("")
    md.append("| task category | overall | " + " | ".join(mode_names) + " |")
    md.append("|" + "---|" * (2 + len(mode_names)))
    for c in cats:
        ov = by_category[c]["overall"]
        cells = [f"{ov['success_pct']:.1f}% ({ov['success']}/{ov['n']})"]
        for m in mode_names:
            x = by_category[c]["by_mode"][m]
            cells.append(f"{x['success_pct']:.1f}% ({x['success']}/{x['n']})")
        md.append(f"| {c} | " + " | ".join(cells) + " |")
    # pooled overall row
    pooled = [r for recs in all_records.values() for r in recs]
    ps, pn, ppct = _agg(pooled)
    ovcells = [f"{ppct:.1f}% ({ps}/{pn})"]
    for m in mode_names:
        ovcells.append(f"{by_stage[m]['success_pct']:.1f}% ({by_stage[m]['success']}/{by_stage[m]['n']})")
    md.append(f"| **ALL** | " + " | ".join(ovcells) + " |")
    md.append("")
    md.append("## Per-stage success rate (all task categories pooled)")
    md.append("")
    md.append("| stage / mode | success rate |")
    md.append("|---|---|")
    for mode in mode_names:
        x = by_stage[mode]
        md.append(f"| {mode} | {x['success_pct']:.1f}% ({x['success']}/{x['n']}) |")
    md_text = "\n".join(md)

    out_table = os.path.join(args.out_dir, "eval_table.md")
    with open(out_table, "w", encoding="utf-8") as f:
        f.write(md_text + "\n")
    # append the breakdown into summary.json
    with open(out_summary, "w", encoding="utf-8") as f:
        _json.dump({
            "ckpt": args.load_path,
            "task": args.task,
            "num_envs": args.num_envs,
            "success_dist": args.success_dist,
            "eval_no_terminate": args.eval_no_terminate,
            "n_aug_variants": len(expanded_indices),
            "modes": summary,
            "by_category": by_category,
            "by_stage": by_stage,
        }, f, indent=2)

    print("\n" + "="*68)
    print(md_text)
    print("="*68)
    print(f"[eval_policy] saved deliverable table to {out_table}")
    # Close the env before the app: without it simulation_app.close() hangs.
    base_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
