"""Per-episode eval for a DAgger / PPO PointCloud student.

Like `play.py` (auto-detects DAgger PointCloudStudent vs PPO
ActorCriticPointCloud from the ckpt) but records `end_final_dist`,
`fail_causes`, `demo_idx` per episode and dumps `records.json` — the same
record schema as `eval_teacher.py`.

Usage:
    python scripts/eval.py \
        --task franka-sharpa-pointcloud \
        --load_path checkpoints/student_lean_v6_L1.pth \
        --side right --data_idx '[...]' \
        --num_envs 128 --max_episodes 8000 --out_dir logs/eval_dagger_pc/<tag>
"""
import argparse
import sys
import json
import ast

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="franka-sharpa-pointcloud")
parser.add_argument("--load_path", type=str, required=True)
parser.add_argument("--side", type=str, default="right")
parser.add_argument("--data_idx", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=128)
# Defaults are the reference evaluation protocol:
# every retarget augmentation variant, first-to-finish, physics DR on.
parser.add_argument("--max_episodes", type=int, default=8000)
parser.add_argument("--max_steps", type=int, default=30000)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--success_dist", type=float, default=0.05,
                    help="Closest-approach success threshold (min_final_dist<this). "
                         "Post-hoc strict thresholds use end_final_dist, not this.")
parser.add_argument("--label", type=str, default=None)
parser.add_argument("--save_traj", action="store_true", default=False,
                    help="Save per-frame trajectory arrays (traj_pos_dist, traj_rot_deg) "
                         "in records.json. Max/mean summary stats are always saved.")
# Env-side PC transforms. All default to None = "take it from the ckpt's
# pc_env_meta" (see algo/dagger/pc_env_meta.py). Pass one only to deliberately
# deviate from training — e.g. --pc_ablate_tactile_force for a modality ablation
# of an otherwise unchanged policy.
parser.add_argument("--pc_ablate_tactile_pc", action="store_true", default=None)
parser.add_argument("--pc_ablate_tactile_force", action="store_true", default=None)
parser.add_argument("--pc_force_repr", type=str, default=None,
                    choices=["scalar", "binary"])
parser.add_argument("--pc_tactile_force_gate", type=float, default=None,
                    help="If >0, gate tactile points whose force < this at inference.")
parser.add_argument("--pc_tactile_gate_mode", type=str, default=None,
                    choices=["zero", "mask"])
parser.add_argument("--pc_force_scale", type=float, default=None,
                    help="Divisor applied to tactile_force. Restored from ckpt if unset.")
parser.add_argument("--pc_tactile_use_vec3", action="store_true", default=None,
                    help="Enable 3D tactile force (sets feat_dim=3). Restored from ckpt if unset.")
parser.add_argument("--no_contact_force", action="store_true", default=None,
                    help="Zero the 5d proprio contact force. Restored from ckpt if unset.")
parser.add_argument("--no_tactile", action="store_true", default=None,
                    help="Zero the whole 20d proprio tactile tail. Restored from ckpt if unset.")
# Optional PC noise INJECTION at eval time (sim2real preview).
# By default the eval script forcibly zeros all PC noise for clean eval — use
# these flags to override and inject noise back in to gauge sim2real robustness
# without retraining.
parser.add_argument("--inject_jitter", type=float, default=None,
                    help="Override pc_jitter_std at eval time (e.g. 0.005 = 5mm).")
parser.add_argument("--inject_dropout", type=float, default=None,
                    help="Override pc_dropout_ratio at eval time (e.g. 0.10 = 10%%).")
parser.add_argument("--inject_hand_noise", type=float, default=None,
                    help="Override pc_hand_noise_std at eval time (e.g. 0.003 = 3mm).")
parser.add_argument("--perturb_obj_xy", type=float, default=0.0,
                    help="Eval-time random xy perturbation of obj init pos (meters). "
                         "0=off, 0.05=±5cm uniform. Hooks env._eval_perturb_obj_xy.")
parser.add_argument("--per_demo_quota", type=int, default=0,
                    help="Episodes to collect per demo. 0 (default, reference protocol) = "
                         "first-to-finish until --max_episodes; N > 0 = N per demo; -1 = "
                         "balanced, ceil(max_episodes / n_demos). First-to-finish weights the "
                         "aggregate toward demos whose episodes end sooner; per-demo rates "
                         "are reported either way.")
parser.add_argument("--expand_aug", action=argparse.BooleanOptionalAction, default=True,
                    help="Evaluate every retarget augmentation variant ({demo}@{aug}) of each "
                         "demo (default, reference protocol). --no-expand_aug evaluates only the "
                         "base demos and reports them under their base IDs (DISABLE_AUG_EXPAND=1).")
parser.add_argument("--keep_physics_dr", action=argparse.BooleanOptionalAction, default=True,
                    help="Keep the physical domain randomisation on during eval (default, "
                         "reference protocol): object mass 0.01-0.15 kg, friction x1.0-2.5, "
                         "COM +-2 cm, hand PD gains x0.5-2. --no-keep_physics_dr holds them at "
                         "nominal for a clean run; that also removes the variation contact "
                         "force is meant to help with.")
parser.add_argument("--camera_extrinsic", type=str, default=None,
                    help="Path to a 4x4 .npy camera-in-armbase extrinsic (ROS optical). "
                         "Must match what the student was TRAINED with, or its scene cloud "
                         "arrives from a different viewpoint than it ever saw.")
parser.add_argument("--ref_root", type=str, default=None,
                    help="Override robotool_batch retarget reference root.")
parser.add_argument("--pc_ablate_scene_pc", action="store_true", default=None,
                    help="Modality ablation: drop the camera-derived scene points "
                         "(default: whatever the checkpoint was trained with).")
parser.add_argument("--mask_obs_slots", type=str, default=None,
                    help="Comma-separated obs-slot ranges to zero before feeding student. "
                         "Format: 'lo:hi,lo:hi,...' e.g. '390:397,550:557' — take the bounds from the [obs-slots] line the env prints at startup, never from a remembered number; the block order has changed before. "
                         "Useful for K=1 demo-target field ablations (eval-only).")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ------------------------------------------------------------------------- #
import os
import time
import collections
import importlib

# Base demos unless --expand_aug: the env reads this when it builds data_indices.
if args_cli.expand_aug:
    if os.environ.pop("DISABLE_AUG_EXPAND", None) is not None:
        print("[EvalPC] NOTE: ignoring DISABLE_AUG_EXPAND from the environment because "
              "--expand_aug is on (default); pass --no-expand_aug for base demos only.",
              flush=True)
else:
    os.environ["DISABLE_AUG_EXPAND"] = "1"

from dataclasses import dataclass, asdict

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


def _build_dagger_student(ckpt, device):
    from dexx.algo.dagger.pointcloud_student import PointCloudStudent
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


def _build_ppo_student(ckpt, device):
    from dexx.algo.ppo.actor_critic_pointcloud import ActorCriticPointCloud
    cfg = ckpt["cfg"]
    pc_cfg = dict(
        fusion_strategy=cfg.pc_fusion_strategy,
        n_scene=cfg.n_scene, n_hand=cfg.n_hand, n_tactile=cfg.n_tactile,
        tactile_feat_dim=cfg.tactile_feat_dim, output_dim=cfg.pc_output_dim,
        ablate_tactile_pc=getattr(cfg, "ablate_tactile_pc", False),
        type_repr=getattr(cfg, "pc_type_repr", "scalar"),
    )
    model = ActorCriticPointCloud(dict(
        actions_num=cfg.action_dim, input_shape=(cfg.proprio_dim,),
        actor_units=list(cfg.actor_units),
        priv_mlp_units=[256, 128, cfg.priv_info_dim],
        priv_info_dim=cfg.priv_info_dim, critic_units=list(cfg.critic_units),
        pc_config=pc_cfg,
    )).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, "PPO ActorCriticPointCloud (mu only)"


@dataclass
class EpisodeRecord:
    env_id: int
    demo_idx: str
    init_frame: int
    survival_len: int
    succeeded: bool
    min_final_dist: float
    end_final_dist: float
    min_final_rot_deg: float
    end_final_rot_deg: float
    max_traj_pos_dist: float
    mean_traj_pos_dist: float
    max_traj_rot_deg: float
    mean_traj_rot_deg: float
    traj_pos_dist: list
    traj_rot_deg: list
    fail_causes: list
    obj_start: list
    obj_end: list


def main():
    os.makedirs(args_cli.out_dir, exist_ok=True)

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

    # Clean eval: DR + aug off, init_curriculum off (uniform), random_state_init on.
    _off = ["enable_obj_pose_noise", "enable_depth_noise"]
    if not args_cli.keep_physics_dr:
        _off += ["randomize_pd_gains", "randomize_friction", "randomize_mass",
                 "randomize_com"]
    else:
        print("[EvalPC] PHYSICS DR KEPT ON: mass / friction / COM / PD gains stay "
              "randomised for this eval", flush=True)
    for fl in _off:
        if hasattr(env_cfg, fl):
            setattr(env_cfg, fl, False)
    for fl in ("pc_jitter_std", "pc_dropout_ratio", "pc_hand_noise_std",
               "pc_force_noise_ratio", "pc_force_dropout_prob"):
        if hasattr(env_cfg, fl):
            setattr(env_cfg, fl, 0.0)
    if hasattr(env_cfg, "init_curriculum_enabled"):
        env_cfg.init_curriculum_enabled = False
    if args_cli.camera_extrinsic:
        assert os.path.exists(args_cli.camera_extrinsic), \
            f"--camera_extrinsic not found: {args_cli.camera_extrinsic}"
        env_cfg.camera_extrinsic_path = os.path.abspath(args_cli.camera_extrinsic)
        print(f"[EvalPC] sim camera extrinsic <- {env_cfg.camera_extrinsic_path}", flush=True)
    if args_cli.ref_root:
        env_cfg.robotool_batch_retarget_root = args_cli.ref_root
        print(f"[EvalPC] robotool_batch_retarget_root <- {args_cli.ref_root}", flush=True)
    # NOTE: the env-side PC transforms (force scale / gate / repr / ablations)
    # are restored from the ckpt further down, once `_ckpt_peek` is loaded.
    # Sim2real preview: inject PC noise back AFTER the clean-eval zeroing above.
    if args_cli.inject_jitter is not None:
        env_cfg.pc_jitter_std = float(args_cli.inject_jitter)
        print(f"[EvalPC][noise] pc_jitter_std = {env_cfg.pc_jitter_std}", flush=True)
    if args_cli.inject_dropout is not None:
        env_cfg.pc_dropout_ratio = float(args_cli.inject_dropout)
        print(f"[EvalPC][noise] pc_dropout_ratio = {env_cfg.pc_dropout_ratio}", flush=True)
    if args_cli.inject_hand_noise is not None:
        env_cfg.pc_hand_noise_std = float(args_cli.inject_hand_noise)
        print(f"[EvalPC][noise] pc_hand_noise_std = {env_cfg.pc_hand_noise_std}", flush=True)

    # ----- pre-load ckpt so the env emits exactly the point counts / tactile
    # width the student was trained with. Shared with play.py; see
    # dexx.algo.dagger.pc_env_meta.
    _ckpt_peek = load_checkpoint(args_cli.load_path)
    align_pc_dims_to_ckpt(env_cfg, _ckpt_peek, tag="EvalPC")
    # Restore the env-side PC transforms the student was TRAINED with. Without
    # this a policy trained on e.g. force/10 with 0.05 gating is evaluated on
    # raw force with every tactile point active — a silent input-distribution
    # mismatch that moves success rate. CLI flags (if passed) still win.
    apply_pc_env_meta(
        env_cfg,
        _ckpt_peek,
        overrides={
            "pc_force_scale": args_cli.pc_force_scale,
            "pc_tactile_force_gate": args_cli.pc_tactile_force_gate,
            "pc_tactile_gate_mode": args_cli.pc_tactile_gate_mode,
            "pc_force_repr": args_cli.pc_force_repr,
            "pc_ablate_scene_pc": args_cli.pc_ablate_scene_pc,
            "pc_ablate_tactile_pc": args_cli.pc_ablate_tactile_pc,
            "pc_ablate_tactile_force": args_cli.pc_ablate_tactile_force,
            "pc_tactile_use_vec3": args_cli.pc_tactile_use_vec3,
            "enable_contact_force": False if args_cli.no_contact_force else None,
            "enable_tactile": False if args_cli.no_tactile else None,
        },
        tag="EvalPC",
    )

    # (the tactile_feat_dim -> vec3 fallback lives in align_pc_dims_to_ckpt
    # above, so an explicit pc_env_meta block wins over it)
    print(f"[EvalPC] env: {args_cli.task} num_envs={args_cli.num_envs}", flush=True)
    env_raw = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    base_env = env_raw.unwrapped
    # Eval-time object perturbation
    if args_cli.perturb_obj_xy > 0:
        base_env._eval_perturb_obj_xy = float(args_cli.perturb_obj_xy)
        print(f"[EvalPC] OBJ PERTURBATION ENABLED: ±{args_cli.perturb_obj_xy*100:.1f}cm "
              f"random uniform in xy at every reset", flush=True)
    device = torch.device(str(base_env.device))

    expanded_indices = [str(x) for x in getattr(base_env, "data_indices", [])]
    print(f"[EvalPC] expanded data_indices: {len(expanded_indices)} variants; "
          f"env e -> data_indices[e % {len(expanded_indices)}]", flush=True)
    if len(expanded_indices) > args_cli.num_envs:
        print(f"[EvalPC] WARNING: {len(expanded_indices)} variants but only "
              f"{args_cli.num_envs} envs; {len(expanded_indices) - args_cli.num_envs} variants "
              f"(data_indices[{args_cli.num_envs}:]) will NOT be evaluated. Raise --num_envs "
              f"to at least {len(expanded_indices)} or pass --no-expand_aug.", flush=True)
    with open(os.path.join(args_cli.out_dir, "expanded_data_indices.json"), "w") as f:
        json.dump({"data_indices": expanded_indices}, f, indent=2)

    def _env_demo(env_id: int) -> str:
        if expanded_indices:
            return expanded_indices[env_id % len(expanded_indices)]
        return "unknown"

    print(f"[EvalPC] loading ckpt: {args_cli.load_path}", flush=True)
    ckpt = load_checkpoint(args_cli.load_path, map_location=device)
    sd_keys = list(ckpt.get("model", {}).keys())
    if any(k.startswith("actor_mlp.") for k in sd_keys):
        model, arch_label = _build_ppo_student(ckpt, device)
    elif any(k == "mlp.0.weight" or k.startswith("mlp.") for k in sd_keys):
        model, arch_label = _build_dagger_student(ckpt, device)
    else:
        raise RuntimeError(f"Unknown ckpt format; first 5 sd keys: {sd_keys[:5]}")
    print(f"[EvalPC] architecture: {arch_label}", flush=True)

    # ---- Restore the student's trained proprio layout ----------------------
    # A lean student was trained on a sliced obs; feeding it the env's full
    # vector would be a silent distribution shift (or a shape error).
    _student_mask_idx = list(_ckpt_peek.get("student_obs_mask_idx", []) or [])
    if _student_mask_idx:
        print(f"[EvalPC] SPARSE-REF ckpt: zeroing {len(_student_mask_idx)} proprio dims",
              flush=True)
    _student_keep_idx = list(_ckpt_peek.get("student_keep_idx", []) or [])
    _keep_t = None
    if _student_keep_idx:
        print(f"[EvalPC] LEAN STUDENT ckpt: slicing proprio to "
              f"{len(_student_keep_idx)}d, dropped "
              f"{list(_ckpt_peek.get('student_drop_slots', []) or [])}", flush=True)
        # The slot map is recorded during the first obs assembly; nothing has
        # forced one yet at this point in the script.
        _live = getattr(base_env, "actor_obs_slots", None)
        if _live is None:
            base_env._get_observations()
            _live = getattr(base_env, "actor_obs_slots", None)
        _saved = _ckpt_peek.get("student_obs_slots") or {}
        if _live and _saved and any(_live.get(k) != tuple(v) for k, v in _saved.items()):
            raise RuntimeError(
                f"actor obs layout changed since training: ckpt {_saved} vs env "
                f"{_live}. The saved keep-indices would select the wrong channels.")
        _keep_t = torch.as_tensor(_student_keep_idx, dtype=torch.long, device=device)

    _mask_ranges = []
    if args_cli.mask_obs_slots:
        for part in args_cli.mask_obs_slots.split(","):
            lo, hi = part.strip().split(":")
            _mask_ranges.append((int(lo), int(hi)))
        print(f"[EvalPC] OBS MASK ACTIVE: zeroing slots {_mask_ranges}", flush=True)

    def _make_inp(d):
        obs_in = d["policy"]
        if _mask_ranges:
            obs_in = obs_in.clone()
            for lo, hi in _mask_ranges:
                obs_in[:, lo:hi] = 0.0
        # Reduce to the student's trained layout LAST: mask indices refer to the
        # env's original obs, slicing changes the width.
        if _student_mask_idx:
            if obs_in is d["policy"]:
                obs_in = obs_in.clone()
            obs_in[:, _student_mask_idx] = 0.0
        if _keep_t is not None:
            obs_in = obs_in[:, _keep_t]
        return {"obs": obs_in, "scene_pc": d["scene_pc"],
                "scene_mask": d["scene_mask"], "hand_pc": d["hand_pc"],
                "tactile_pc": d["tactile_pc"], "tactile_force": d["tactile_force"]}

    def _obj_pos_now():
        op = getattr(base_env, "object_pos", None)
        if op is None:
            return torch.zeros((args_cli.num_envs, 3), device=device)
        return op.detach().clone()

    obs, _ = env_raw.reset()
    per_env_init = base_env.progress_buf.clone()
    per_env_t0 = torch.zeros(args_cli.num_envs, dtype=torch.long, device=device)
    per_env_min_final = torch.full((args_cli.num_envs,), float("inf"), device=device)
    per_env_min_final_rot = torch.full((args_cli.num_envs,), float("inf"), device=device)
    per_env_fail_causes = [set() for _ in range(args_cli.num_envs)]
    per_env_obj_start = _obj_pos_now()
    per_env_last_final_dist = torch.full((args_cli.num_envs,), -1.0, device=device)
    per_env_last_final_rot = torch.full((args_cli.num_envs,), -1.0, device=device)
    per_env_traj_pos = [[] for _ in range(args_cli.num_envs)]
    per_env_traj_rot = [[] for _ in range(args_cli.num_envs)]
    prev_obj_pos = _obj_pos_now()
    records: list = []
    fail_keys = ["fail/hand_tracking", "fail/obj_pos_drift", "fail/obj_rot_drift",
                 "fail/premature_contact", "fail/arm_below_table",
                 "fail/error_buf_velocity_explosion"]
    step_counter = 0
    t_start = time.time()

    # ---- Per-demo quota ----------------------------------------------------
    # 0 = first-to-finish (reference protocol). A fixed share per demo instead
    # keeps a demo whose episodes end sooner from dominating the aggregate.
    _demo_universe = sorted({_env_demo(i) for i in range(args_cli.num_envs)})
    if args_cli.per_demo_quota < 0:
        _demo_quota = -(-args_cli.max_episodes // max(1, len(_demo_universe)))
    else:
        _demo_quota = int(args_cli.per_demo_quota)
    _per_demo_counts = collections.Counter()
    if _demo_quota > 0:
        print(f"[EvalPC] per-demo quota = {_demo_quota} x {len(_demo_universe)} demos "
              f"= {_demo_quota * len(_demo_universe)} episodes", flush=True)
    else:
        print("[EvalPC] first-to-finish collection (reference protocol): demos whose "
              "episodes end sooner contribute more episodes; read the per-demo rates",
              flush=True)

    print(f"[EvalPC] running until {args_cli.max_episodes} eps or "
          f"{args_cli.max_steps} steps", flush=True)

    def _collection_done() -> bool:
        if _demo_quota > 0:
            return all(_per_demo_counts[d] >= _demo_quota for d in _demo_universe)
        return len(records) >= args_cli.max_episodes

    while (step_counter < args_cli.max_steps
           and not _collection_done()
           and simulation_app.is_running()):
        with torch.no_grad():
            action = model.act_inference(_make_inp(obs))
            action = torch.clamp(action, -1.0, 1.0)
        obs, _r, done, _trunc, _info = env_raw.step(action)
        step_counter += 1
        cur_obj_pos = _obj_pos_now()

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
            # Trajectory tracking: per-frame obj-vs-demo-current distance.
            if "diag/obj_pos_dist" in rd:
                _pd = rd["diag/obj_pos_dist"].detach().cpu().tolist()
                for _i in range(args_cli.num_envs):
                    per_env_traj_pos[_i].append(float(_pd[_i]))
            if "diag/obj_rot_angle" in rd:
                import math as _math
                _rd_arr = rd["diag/obj_rot_angle"].detach().cpu().tolist()
                for _i in range(args_cli.num_envs):
                    per_env_traj_rot[_i].append(_math.degrees(float(_rd_arr[_i])))
            for k in fail_keys:
                if k in rd:
                    fired = (rd[k] > 0.5).nonzero(as_tuple=False).flatten()
                    for env_id in fired.tolist():
                        per_env_fail_causes[env_id].add(k.split("/")[-1])

        done_t = done if torch.is_tensor(done) else torch.tensor(done, device=device)
        if done_t.any():
            done_ids = done_t.nonzero(as_tuple=False).flatten().tolist()
            for env_id in done_ids:
                survival_len = step_counter - int(per_env_t0[env_id].item())
                if survival_len < 5:
                    per_env_init[env_id] = base_env.progress_buf[env_id]
                    per_env_t0[env_id] = step_counter
                    per_env_min_final[env_id] = float("inf")
                    per_env_min_final_rot[env_id] = float("inf")
                    per_env_traj_pos[env_id] = []
                    per_env_traj_rot[env_id] = []
                    per_env_fail_causes[env_id] = set()
                    per_env_obj_start[env_id] = cur_obj_pos[env_id]
                    continue
                min_dist = float(per_env_min_final[env_id].item())
                success = (min_dist < args_cli.success_dist) and \
                          (not torch.isinf(per_env_min_final[env_id]).item())
                # Reward diagnostics retain this terminal step's pre-reset
                # measurements; the previous loop's values are one step old.
                end_dist = float(per_env_last_final_dist[env_id].item())
                import math
                min_rot = float(per_env_min_final_rot[env_id].item())
                end_rot = float(per_env_last_final_rot[env_id].item())
                _tp = per_env_traj_pos[env_id]
                _tr = per_env_traj_rot[env_id]
                _max_p = max(_tp) if _tp else -1.0
                _mean_p = sum(_tp)/len(_tp) if _tp else -1.0
                _max_r = max(_tr) if _tr else -1.0
                _mean_r = sum(_tr)/len(_tr) if _tr else -1.0
                rec = EpisodeRecord(
                    env_id=env_id,
                    demo_idx=_env_demo(env_id),
                    init_frame=int(per_env_init[env_id].item()),
                    survival_len=survival_len,
                    succeeded=bool(success),
                    min_final_dist=min_dist if min_dist != float("inf") else -1.0,
                    end_final_dist=end_dist if end_dist >= 0 else -1.0,
                    min_final_rot_deg=math.degrees(min_rot) if min_rot != float("inf") else -1.0,
                    end_final_rot_deg=math.degrees(end_rot) if end_rot >= 0 else -1.0,
                    max_traj_pos_dist=_max_p,
                    mean_traj_pos_dist=_mean_p,
                    max_traj_rot_deg=_max_r,
                    mean_traj_rot_deg=_mean_r,
                    traj_pos_dist=([round(x, 4) for x in _tp] if args_cli.save_traj else []),
                    traj_rot_deg=([round(x, 2) for x in _tr] if args_cli.save_traj else []),
                    fail_causes=sorted(list(per_env_fail_causes[env_id])),
                    obj_start=per_env_obj_start[env_id].tolist(),
                    obj_end=prev_obj_pos[env_id].tolist(),
                )
                # Drop episodes from a demo that already filled its share, so
                # the aggregate weights every demo equally.
                if _demo_quota > 0 and _per_demo_counts[rec.demo_idx] >= _demo_quota:
                    pass
                else:
                    records.append(rec)
                    _per_demo_counts[rec.demo_idx] += 1
                per_env_init[env_id] = base_env.progress_buf[env_id]
                per_env_t0[env_id] = step_counter
                per_env_min_final[env_id] = float("inf")
                per_env_min_final_rot[env_id] = float("inf")
                per_env_traj_pos[env_id] = []
                per_env_traj_rot[env_id] = []
                per_env_fail_causes[env_id] = set()
                per_env_obj_start[env_id] = cur_obj_pos[env_id]
                if _demo_quota > 0:
                    if all(_per_demo_counts[d] >= _demo_quota for d in _demo_universe):
                        break
                elif len(records) >= args_cli.max_episodes:
                    break

        if step_counter % 100 == 0:
            ok = sum(1 for r in records if r.succeeded)
            pct = 100.0 * ok / max(1, len(records))
            _lag = ""
            if _demo_quota > 0 and _demo_universe:
                _d = min(_demo_universe, key=lambda d: _per_demo_counts[d])
                _lag = (f" slowest={_d.split('/')[-1]}"
                        f" {_per_demo_counts[_d]}/{_demo_quota}")
            print(f"  step={step_counter:5d} eps={len(records):5d} "
                  f"closest_succ({100 * args_cli.success_dist:g}cm)={ok}/{len(records)} ({pct:.1f}%){_lag} "
                  f"elapsed={time.time()-t_start:.0f}s", flush=True)

        prev_obj_pos = cur_obj_pos

    elapsed = time.time() - t_start

    out_records = os.path.join(args_cli.out_dir, "records.json")
    with open(out_records, "w") as f:
        json.dump({"single": [asdict(r) for r in records]}, f, indent=2)
    print(f"[EvalPC] saved {len(records)} records → {out_records}", flush=True)

    # ---- strictN: the metric the protocol says to report -------------------
    # An episode counts only if the object finished within N cm of the demo's
    # final pose, AND object-position drift never fired, AND the episode was not
    # a bad init. Deliberately excludes ONLY obj_pos_drift, not the other failure
    # causes — ORing them all in would silently change what the number means.
    # See docs/EVAL.md §strict3.
    _BAD_INIT_SURVIVAL = 5

    def _strict(recs, cm):
        kept = [r for r in recs if r.survival_len > _BAD_INIT_SURVIVAL]
        if not kept:
            return 0.0, 0, 0
        ok = sum(1 for r in kept
                 if 0.0 <= r.end_final_dist < cm / 100.0
                 and "obj_pos_drift" not in r.fail_causes)
        return ok / len(kept), ok, len(kept)

    _strict_rates = {}
    for _cm in (2, 3, 5):
        rate, ok, n = _strict(records, _cm)
        _strict_rates[f"strict{_cm}"] = {"rate": rate, "successes": ok, "episodes": n}
    _n_bad_init = sum(1 for r in records if r.survival_len <= _BAD_INIT_SURVIVAL)

    print("[EvalPC] strict success (end_final_dist < N cm, no obj_pos_drift, "
          f"bad inits excluded: {_n_bad_init}/{len(records)}):", flush=True)
    for _k, _v in _strict_rates.items():
        print(f"    {_k:8s} {_v['successes']:4d}/{_v['episodes']:<4d} "
              f"({100 * _v['rate']:5.1f}%)", flush=True)

    _by_demo = {}
    for d in sorted({r.demo_idx for r in records}):
        _rs = [r for r in records if r.demo_idx == d]
        _ok = sum(1 for r in _rs if r.succeeded)
        _s3, _s3ok, _s3n = _strict(_rs, 3)
        _by_demo[d] = {"episodes": len(_rs), "succeeded": _ok,
                       "success_rate": _ok / max(1, len(_rs)),
                       "strict3": _s3, "strict3_successes": _s3ok,
                       "strict3_episodes": _s3n}
    _rates = [v["success_rate"] for v in _by_demo.values()]
    _macro = sum(_rates) / len(_rates) if _rates else 0.0
    _micro = sum(1 for r in records if r.succeeded) / max(1, len(records))
    print("[EvalPC] per-demo (closest-approach | strict3):", flush=True)
    for d, v in _by_demo.items():
        print(f"    {d.split('/')[-1]:24s} "
              f"{v['succeeded']:4d}/{v['episodes']:<4d} ({100*v['success_rate']:5.1f}%)  |  "
              f"{v['strict3_successes']:4d}/{v['strict3_episodes']:<4d} "
              f"({100*v['strict3']:5.1f}%)", flush=True)
    print(f"[EvalPC] success  macro (demo-averaged) = {100*_macro:.1f}%   "
          f"micro (episode-weighted) = {100*_micro:.1f}%", flush=True)

    summary = {
        "ckpt": args_cli.load_path, "label": args_cli.label,
        "per_demo_quota": _demo_quota,
        # The headline number. docs/EVAL.md says to report strict3, not the
        # closest-approach rate below.
        "strict": _strict_rates,
        "bad_init_excluded": _n_bad_init,
        "success_rate_per_demo": _by_demo,
        "success_rate_macro": _macro,
        "success_rate_micro": _micro,
        "arch": arch_label, "task": args_cli.task,
        "num_envs": args_cli.num_envs, "max_episodes": args_cli.max_episodes,
        "actual_episodes": len(records), "steps": step_counter,
        "elapsed_sec": elapsed,
    }
    with open(os.path.join(args_cli.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[EvalPC] done: {len(records)} eps in {elapsed:.1f}s", flush=True)
    env_raw.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
