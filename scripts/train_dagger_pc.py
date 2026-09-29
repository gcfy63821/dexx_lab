"""DAgger entry point for the point-cloud student.

Pipeline:
    state-expert teacher (e.g. franka-sharpa-force-poseobs ckpt)
        ↓ DAgger
    point-cloud student (proprio + scene/hand/tactile PC → action)

Usage:
    python scripts/train_dagger_pc.py \
        --task franka-sharpa-pointcloud \
        --teacher_ckpt checkpoints/teacher_poseobs.pth \
        --side right \
        --data_idx '[...]' \
        --num_envs 64 --dagger_iters 20 --rollout_steps 4096 \
        --beta_init 1.0 --beta_decay 0.8 \
        --train_epochs 5 --batch_size 512 --lr 5e-5 \
        --max_buffer 200000 \
        --out_dir logs/dagger_pc_$(date +%Y%m%d_%H%M%S) \
        --headless

See docs/DISTILLATION.md for the full lean-student (deployable) command.
"""
import argparse
import sys
import json
import ast

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="DAgger point-cloud student training.")
parser.add_argument("--task", type=str, default="franka-sharpa-pointcloud")
parser.add_argument("--teacher_ckpt", type=str, required=True)
parser.add_argument("--student_ckpt", type=str, default=None,
                    help="Optional pretrained student to resume from.")
parser.add_argument("--side", type=str, default="right")
parser.add_argument("--data_idx", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=64)

# DAgger schedule
parser.add_argument("--dagger_iters", type=int, default=20)
parser.add_argument("--rollout_steps", type=int, default=4096)
parser.add_argument("--beta_init", type=float, default=1.0)
parser.add_argument("--beta_decay", type=float, default=0.8)
parser.add_argument("--train_epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--max_buffer", type=int, default=200_000)
parser.add_argument("--out_dir", type=str, default="logs/dagger_pc")

# Modality ablation (default off = no behaviour change)
parser.add_argument("--pc_ablate_tactile_pc", action="store_true",
                    help="Exclude the tactile points from the PC encoder.")
parser.add_argument("--pc_ablate_tactile_force", action="store_true",
                    help="Zero the tactile_force channel.")
parser.add_argument("--pc_force_repr", type=str, default="scalar",
                    choices=["scalar", "binary"],
                    help="Tactile force representation: raw scalar or binary contact.")
parser.add_argument("--pc_tactile_force_gate", type=float, default=0.0,
                    help="If >0, zero (xyz + force) of tactile points whose force < this. "
                         "Gives 'no contact → no point' semantics. 0 = always-on 25 pts.")
parser.add_argument("--pc_force_scale", type=float, default=1.0,
                    help="Divide tactile_force by this constant (sim-real consistency). "
                         "Default 1.0 = no-op. Use ~10-30 to bring force into xyz (m) range.")
parser.add_argument("--pc_type_repr", type=str, default="scalar",
                    choices=["scalar", "onehot"],
                    help="PointNet type embedding: scalar (1 dim, type∈{0,1,2}) or "
                         "one-hot (3 dims). Affects encoder input_dim, not env shape.")
parser.add_argument("--pc_tactile_gate_mode", type=str, default="zero",
                    choices=["zero", "mask"],
                    help="When --pc_tactile_force_gate > 0: 'zero' (legacy, zeroes "
                         "xyz+force of gated points) or 'mask' (leaves xyz/force "
                         "unchanged; encoder skips via tactile_mask).")
parser.add_argument("--pc_tactile_use_vec3", action="store_true",
                    help="Use 3D tactile force vector in elastomer-local frame "
                         "(env outputs (N,25,3) instead of (N,25,1)). Also sets "
                         "pc_tactile_feature_dim=3 so encoder agrees.")
parser.add_argument("--seed", type=int, default=0,
                    help="Random seed (torch / numpy / env) — for multi-seed ablation runs.")
parser.add_argument("--no_contact_force", action="store_true", default=False,
                    help="Modality ablation: zero the 5-d scalar contact force in the obs.")

# PC augmentation (training-time only; defaults preserve env_cfg = no aug).
parser.add_argument("--pc_aug", action="store_true", default=False,
                    help="Shortcut: enable PC jitter+dropout+hand_noise with sane "
                         "defaults (jitter 2mm, dropout 5%%, hand_noise 1mm). "
                         "Overridden by explicit --pc_jitter_std/--pc_dropout_ratio/"
                         "--pc_hand_noise_std flags if given.")
parser.add_argument("--pc_jitter_std", type=float, default=None,
                    help="Gaussian jitter std (m) added to scene+hand+tactile points.")
parser.add_argument("--pc_dropout_ratio", type=float, default=None,
                    help="Random per-point dropout fraction (0..1).")
parser.add_argument("--pc_hand_noise_std", type=float, default=None,
                    help="Extra jitter std (m) on hand points only.")
parser.add_argument("--hand_body_subset", type=str, default=None,
                    choices=["minimal5", "minimal6", "default11", "dense22"],
                    help="Override pc_hand_body_names: minimal5=5fingertips only (no wrist), "
                         "minimal6=wrist+5fingertips, "
                         "default11=wrist+5MCP+5fingertips (cfg default), "
                         "dense22=11 + 5 PIP + 5 DIP + thumb_IP.")

parser.add_argument("--camera_extrinsic", type=str, default=None,
                    help="Path to a 4x4 .npy camera-in-armbase extrinsic (ROS optical). "
                         "Places the sim depth camera at this pose so the point cloud the "
                         "student trains on matches the real (deploy) camera. Omit to use "
                         "the shipped calibration calib/camera_align/current.npy.")
parser.add_argument("--ref_root", type=str, default=None,
                    help="Override robotool_batch retarget reference root (match the teacher's "
                         "training refs, so the teacher sees the obs distribution it was "
                         "trained on when labeling DAgger rollouts).")
parser.add_argument("--pc_ablate_scene_pc", action="store_true", default=False,
                    help="Modality ablation: drop the camera-derived scene points.")
parser.add_argument("--perturb_obj_xy", type=float, default=None,
                    help="Training-time object xy displacement in metres (uniform +-).")
parser.add_argument("--env_cfg", type=str, nargs="*", default=[],
                    help="Generic env_cfg overrides, KEY=VALUE, e.g. "
                         "--env_cfg obs_vel_from_noisy_pos=True force_reward_weight=0.0. "
                         "Values are cast to bool/int/float, else kept as a string. An "
                         "unknown KEY is a hard error: this script uses parse_known_args, "
                         "so a typo would otherwise be dropped in silence and the run "
                         "would look like it applied a setting it never did.")
parser.add_argument("--student_drop_slots", type=str, default=None,
                    help="Comma-separated NAMES of actor-obs channels to REMOVE from the "
                         "student's proprio (teacher keeps all of them). Names come from "
                         "`env.actor_obs_slots`, printed at startup — e.g. "
                         "'obj_bps,tips_distance,obj_pose_tail' leaves a 417-d student. "
                         "Unlike --student_mask_slots this slices the dims out rather than "
                         "zeroing them, so the student has no dead inputs and a "
                         "train/deploy layout mismatch fails loudly on a shape error "
                         "instead of silently feeding unlearned signal.")
parser.add_argument("--student_mask_slots", type=str, default=None,
                    help="Zero these proprio obs slots for the STUDENT only (teacher keeps "
                         "the full obs, so its DAgger labels stay in-distribution). "
                         "Format 'lo:hi,lo:hi'. Stored in the ckpt as `student_obs_mask_idx` "
                         "and re-applied by eval and deploy.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows."""

import os
import importlib

import torch
import gymnasium as gym
from omegaconf import OmegaConf

import dexx.tasks.franka_sharpa  # noqa: F401

from dexx.algo.dagger.dagger_pointcloud import (
    DAggerPointCloud, DAggerPointCloudConfig,
)
from dexx.algo.dagger.pc_env_meta import collect_pc_env_meta
from dexx.algo.dagger.teacher_utils import load_teacher


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
            data_indices = json.loads(args_cli.data_idx)
        except json.JSONDecodeError:
            data_indices = ast.literal_eval(args_cli.data_idx)
        env_cfg.data_indices = data_indices

    # Modality ablation overrides (CLI flags override the env_cfg defaults).
    if args_cli.pc_ablate_tactile_pc:
        env_cfg.pc_ablate_tactile_pc = True
    if args_cli.pc_ablate_tactile_force:
        env_cfg.pc_ablate_tactile_force = True
    if args_cli.pc_force_repr != "scalar":
        env_cfg.pc_force_repr = args_cli.pc_force_repr
    if args_cli.pc_tactile_force_gate > 0:
        env_cfg.pc_tactile_force_gate = args_cli.pc_tactile_force_gate
        env_cfg.pc_tactile_gate_mode = args_cli.pc_tactile_gate_mode
        print(f"[ablation] pc_tactile_force_gate = {args_cli.pc_tactile_force_gate}, "
              f"gate_mode={args_cli.pc_tactile_gate_mode}", flush=True)
    if args_cli.pc_force_scale != 1.0:
        env_cfg.pc_force_scale = args_cli.pc_force_scale
        print(f"[ablation] pc_force_scale = {args_cli.pc_force_scale} "
              f"(tactile_force /= scale before encoder)", flush=True)
    if args_cli.pc_tactile_use_vec3:
        env_cfg.pc_tactile_use_vec3 = True
        env_cfg.pc_tactile_feature_dim = 3
        print(f"[ablation] pc_tactile_use_vec3=True → tactile_force shape (N,25,3) "
              f"(elastomer-local frame), pc_tactile_feature_dim=3", flush=True)
    if args_cli.no_contact_force:
        env_cfg.enable_contact_force = False
        print("[ablation] enable_contact_force = False (5d contact force zeroed in obs)")

    # PC augmentation (training-time, default off). --pc_aug = preset, individual
    # flags override the preset. Eval/play scripts still default to OFF aug.
    _aug_defaults = {"jitter": 0.002, "dropout": 0.05, "hand_noise": 0.001}
    if args_cli.pc_aug:
        env_cfg.pc_jitter_std = _aug_defaults["jitter"]
        env_cfg.pc_dropout_ratio = _aug_defaults["dropout"]
        env_cfg.pc_hand_noise_std = _aug_defaults["hand_noise"]
    if args_cli.pc_jitter_std is not None:
        env_cfg.pc_jitter_std = args_cli.pc_jitter_std
    if args_cli.pc_dropout_ratio is not None:
        env_cfg.pc_dropout_ratio = args_cli.pc_dropout_ratio
    if args_cli.pc_hand_noise_std is not None:
        env_cfg.pc_hand_noise_std = args_cli.pc_hand_noise_std
    if any(getattr(env_cfg, k, 0.0) > 0.0
           for k in ("pc_jitter_std", "pc_dropout_ratio", "pc_hand_noise_std")):
        print(f"[ablation] PC aug ON: jitter={env_cfg.pc_jitter_std} "
              f"dropout={env_cfg.pc_dropout_ratio} hand_noise={env_cfg.pc_hand_noise_std}")

    # Hand body subset override (default = whatever env_cfg.__post_init__ sets).
    if args_cli.hand_body_subset:
        _subsets = {
            "minimal5": [
                "thumb_fingertip", "index_fingertip", "middle_fingertip",
                "ring_fingertip", "pinky_fingertip",
            ],
            "minimal6": [
                "hand_C_MC",
                "thumb_fingertip", "index_fingertip", "middle_fingertip",
                "ring_fingertip", "pinky_fingertip",
            ],
            "default11": [
                "hand_C_MC",
                "thumb_CMC_VL", "index_MCP_VL", "middle_MCP_VL",
                "ring_MCP_VL", "pinky_MCP_VL",
                "thumb_fingertip", "index_fingertip", "middle_fingertip",
                "ring_fingertip", "pinky_fingertip",
            ],
            "dense22": [
                "hand_C_MC",
                "thumb_CMC_VL", "index_MCP_VL", "middle_MCP_VL",
                "ring_MCP_VL", "pinky_MCP_VL",
                "thumb_MCP_VL", "index_PP", "middle_PP", "ring_PP", "pinky_PP",
                "thumb_MC", "index_MP", "middle_MP", "ring_MP", "pinky_MP",
                "thumb_fingertip", "index_fingertip", "middle_fingertip",
                "ring_fingertip", "pinky_fingertip",
                "thumb_IP",
            ],
        }
        env_cfg.pc_hand_body_names = _subsets[args_cli.hand_body_subset]
        env_cfg.pc_num_hand_points = len(env_cfg.pc_hand_body_names)
        print(f"[ablation] pc_hand_body_names = {args_cli.hand_body_subset} "
              f"({len(env_cfg.pc_hand_body_names) if env_cfg.pc_hand_body_names else 'all'} bodies)")

    if args_cli.pc_ablate_scene_pc:
        env_cfg.pc_ablate_scene_pc = True
        print("[ablation] pc_ablate_scene_pc = True (scene points dropped)")

    if args_cli.perturb_obj_xy is not None:
        env_cfg.randomize_obj_xy = float(args_cli.perturb_obj_xy)
        print(f"[dagger_pc] randomize_obj_xy = {env_cfg.randomize_obj_xy} m")

    if args_cli.camera_extrinsic:
        assert os.path.exists(args_cli.camera_extrinsic), \
            f"--camera_extrinsic not found: {args_cli.camera_extrinsic}"
        env_cfg.camera_extrinsic_path = os.path.abspath(args_cli.camera_extrinsic)
        print(f"[dagger_pc] sim camera extrinsic <- {env_cfg.camera_extrinsic_path}", flush=True)

    if args_cli.ref_root:
        env_cfg.robotool_batch_retarget_root = args_cli.ref_root
        print(f"[dagger_pc] robotool_batch_retarget_root <- {args_cli.ref_root} "
              f"(match the teacher's training refs so obs distribution aligns)", flush=True)

    for _ov in args_cli.env_cfg:
        if "=" not in _ov:
            raise ValueError(f"--env_cfg entry without '=': {_ov!r}")
        _k, _v = _ov.split("=", 1)
        if not hasattr(env_cfg, _k):
            raise AttributeError(
                f"--env_cfg: env_cfg has no attribute {_k!r}. Refusing to set it — a "
                f"silently ignored override produces a run that differs from its label.")
        if _v.lower() in ("true", "false"):
            _val = _v.lower() == "true"
        else:
            try:
                _val = int(_v)
            except ValueError:
                try:
                    _val = float(_v)
                except ValueError:
                    _val = _v
        setattr(env_cfg, _k, _val)
        print(f"[env_cfg] {_k} = {_val!r}")

    # Reproducibility: seed torch / numpy / env (for multi-seed ablation runs).
    import random as _random
    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)
    _random.seed(args_cli.seed)
    try:
        import numpy as _np
        _np.random.seed(args_cli.seed)
    except Exception:
        pass
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = args_cli.seed
    print(f"[DAggerPC] seed = {args_cli.seed}")

    print(f"\n[DAggerPC] Creating env: {args_cli.task}, num_envs={args_cli.num_envs}")
    env_raw = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = env_raw.unwrapped

    obs_dim = env.cfg.observation_space        # parent's flat proprio obs
    action_dim = env.cfg.action_space
    device = str(env.device)

    teacher_model, teacher_rms = load_teacher(
        ckpt_path=args_cli.teacher_ckpt,
        obs_dim=obs_dim,
        action_dim=action_dim,
        priv_info_dim=None,
        asymmetric=True,
        device=device,
    )

    # Verify teacher obs_dim matches env obs_dim
    teacher_obs_in = teacher_model.actor_mlp.mlp[0].in_features
    if teacher_obs_in != obs_dim:
        raise RuntimeError(
            f"Teacher actor_mlp expects {teacher_obs_in}-d obs but env produces "
            f"{obs_dim}-d obs. Use a teacher trained on the same proprio layout."
        )
    print(f"[DAggerPC] ✓ teacher obs_dim ({teacher_obs_in}) matches env ({obs_dim})")

    # ---- Named-slot DROP: slice channels out of the student's proprio -------
    # The teacher keeps the full obs above; only the student's copy shrinks.
    student_keep_idx = ()
    if args_cli.student_drop_slots:
        slots = getattr(env, "actor_obs_slots", None)
        if not slots:
            # The map is recorded during the first obs assembly, which has not
            # run yet at this point in the script.
            env_raw.reset()
            slots = getattr(env, "actor_obs_slots", None)
        if not slots:
            raise RuntimeError(
                "--student_drop_slots needs `env.actor_obs_slots`, which this task "
                "does not publish. Only the critic-horizon obs family builds it.")
        want = [n.strip() for n in args_cli.student_drop_slots.split(",") if n.strip()]
        unknown = [n for n in want if n not in slots]
        if unknown:
            raise ValueError(
                f"--student_drop_slots: unknown channel(s) {unknown}. "
                f"Available: {sorted(slots)}")
        drop = set()
        for n in want:
            lo, hi = slots[n]
            drop.update(range(lo, hi))
        student_keep_idx = tuple(i for i in range(obs_dim) if i not in drop)
        print(f"[dagger_pc] STUDENT DROP: removing {len(drop)} of {obs_dim} proprio dims "
              f"-> student proprio {len(student_keep_idx)}d "
              f"({', '.join(f'{n}{slots[n]}' for n in want)}); teacher unaffected",
              flush=True)

    student_obs_mask_idx = ()
    if args_cli.student_mask_slots:
        _idx = []
        for _part in args_cli.student_mask_slots.split(","):
            _lo, _hi = _part.strip().split(":")
            _lo, _hi = int(_lo), int(_hi)
            if not (0 <= _lo < _hi <= obs_dim):
                raise ValueError(
                    f"--student_mask_slots range {_lo}:{_hi} out of bounds for "
                    f"obs_dim={obs_dim}")
            _idx.extend(range(_lo, _hi))
        student_obs_mask_idx = tuple(sorted(set(_idx)))
        print(f"[dagger_pc] STUDENT MASK: zeroing {len(student_obs_mask_idx)} proprio dims "
              f"{args_cli.student_mask_slots} (teacher unaffected)", flush=True)

    dagger_cfg = DAggerPointCloudConfig(
        dagger_iters=args_cli.dagger_iters,
        rollout_steps_per_iter=args_cli.rollout_steps,
        beta_init=args_cli.beta_init,
        beta_decay=args_cli.beta_decay,
        train_epochs=args_cli.train_epochs,
        batch_size=args_cli.batch_size,
        lr=args_cli.lr,
        max_buffer_size=args_cli.max_buffer,
        # With --student_drop_slots the proprio is sliced before it ever reaches
        # the student, so the network is built at the reduced width.
        proprio_dim=len(student_keep_idx) if student_keep_idx else obs_dim,
        action_dim=action_dim,
        n_scene=int(env.cfg.pc_num_scene_points),
        n_hand=int(env.cfg.pc_num_hand_points),
        n_tactile=int(env.cfg.pc_num_tactile_points),
        tactile_feat_dim=int(env.cfg.pc_tactile_feature_dim),
        pc_fusion_strategy=str(env.cfg.pc_fusion_strategy),
        pc_output_dim=int(env.cfg.pc_encoder_output_dim),
        ablate_tactile_pc=bool(getattr(env.cfg, "pc_ablate_tactile_pc", False)),
        pc_type_repr=str(args_cli.pc_type_repr),
        # Snapshot the env-side PC transforms so eval/deploy reproduce the
        # exact input distribution the student trains on.
        pc_env_meta=collect_pc_env_meta(env.cfg),
        student_keep_idx=student_keep_idx,
        student_obs_mask_idx=student_obs_mask_idx,
        student_obs_slots=dict(getattr(env, "actor_obs_slots", None) or {}),
        student_drop_slots=tuple(
            n.strip() for n in (args_cli.student_drop_slots or "").split(",") if n.strip()),
        out_dir=args_cli.out_dir,
        device=device,
    )

    print("\n" + "=" * 72)
    print("[DAggerPC] DIMENSION SUMMARY")
    print("=" * 72)
    print(f"  teacher  obs_dim = {obs_dim}")
    print(f"  student  proprio = {dagger_cfg.proprio_dim}  + PC feature ({dagger_cfg.pc_output_dim})")
    print(f"  fusion strategy  = {dagger_cfg.pc_fusion_strategy}")
    print(f"  PC counts: scene={dagger_cfg.n_scene}  hand={dagger_cfg.n_hand}  "
          f"tactile={dagger_cfg.n_tactile}")
    print("=" * 72 + "\n")

    dagger = DAggerPointCloud(
        cfg=dagger_cfg,
        env=env_raw,
        teacher_model=teacher_model,
        teacher_running_mean_std=teacher_rms,
        student_ckpt_path=args_cli.student_ckpt,
    )

    dagger.run()
    env_raw.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
