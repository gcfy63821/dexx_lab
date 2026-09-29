"""DAgger trainer for the point-cloud student.

The replay buffer is small — ~16 KB per transition — so we can afford large
replay sizes without choking RAM.

Storage layout (per transition, CPU):
    obs            : (proprio_dim,) fp32      ~1-2 KB
    scene_pc       : (1024, 3) fp32           ~12 KB
    scene_mask     : (1024,) bool             ~1 KB
    hand_pc        : (11, 3) fp32              ~132 B
    tactile_pc     : (25, 3) fp32              ~300 B
    tactile_force  : (25, 1) fp32              ~100 B
    expert_action  : (act_dim,) fp32           ~120 B
Total ≈ 16 KB → 200k transitions ≈ 3.2 GB.
"""
from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .pc_env_meta import load_checkpoint
from .pointcloud_student import PointCloudStudent


@dataclass
class DAggerPointCloudConfig:
    # DAgger schedule
    dagger_iters: int = 10
    rollout_steps_per_iter: int = 4096
    beta_init: float = 1.0
    beta_decay: float = 0.7
    beta_min: float = 0.0

    # Student training
    train_epochs: int = 5
    batch_size: int = 512
    lr: float = 5e-5
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    # Buffer
    max_buffer_size: int = 200_000

    # Dimensions (filled by train_dagger_pc.py from env)
    proprio_dim: int = 0
    action_dim: int = 0
    n_scene: int = 1024
    n_hand: int = 11
    n_tactile: int = 25
    tactile_feat_dim: int = 1

    # Encoder
    pc_fusion_strategy: str = "early_concat"
    pc_output_dim: int = 64
    ablate_tactile_pc: bool = False
    pc_type_repr: str = "scalar"  # "scalar" | "onehot"

    # Env-side PC transforms (force scale / gate / repr / ablations). These
    # shape the student's input distribution but live in env_cfg, so they must
    # travel with the ckpt or eval/deploy silently uses different inputs than
    # training did. Filled by scripts/train_dagger_pc.py via collect_pc_env_meta().
    # See algo/dagger/pc_env_meta.py.
    pc_env_meta: dict = field(default_factory=dict)
    student_hidden: tuple = (512, 256)
    action_clip: float = 1.0

    # Lean / sparse student. The TEACHER always sees the env's full actor obs —
    # only the student's copy is reduced, so DAgger labels stay in-distribution.
    #   student_obs_mask_idx : dims zeroed in place (width unchanged)
    #   student_keep_idx     : dims kept, everything else sliced out (width shrinks)
    #   student_obs_slots    : the env's named slot map, stored for eval/deploy
    #   student_drop_slots   : the slot NAMES that produced student_keep_idx
    # Dropping 'obj_bps,tips_distance,obj_pose_tail' from a 557-d actor obs is
    # what yields the 417-d student.
    student_obs_mask_idx: tuple = ()
    student_keep_idx: tuple = ()
    student_obs_slots: dict = field(default_factory=dict)
    student_drop_slots: tuple = ()

    # Output
    out_dir: str = "logs/dagger_pointcloud"
    device: str = "cuda"


# ----------------------------------------------------------------------------
# Replay buffer (CPU, fp32)
# ----------------------------------------------------------------------------
class PCReplayBuffer:
    """Fixed-size FIFO of (proprio, scene_pc, scene_mask, hand_pc, tactile_pc,
    tactile_force, expert_action). Stored CPU-side in fp32; loaded GPU-side
    per minibatch."""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.obs = deque(maxlen=max_size)
        self.scene_pc = deque(maxlen=max_size)
        self.scene_mask = deque(maxlen=max_size)
        self.hand_pc = deque(maxlen=max_size)
        self.tactile_pc = deque(maxlen=max_size)
        self.tactile_force = deque(maxlen=max_size)
        self.tactile_mask = deque(maxlen=max_size)
        self.actions = deque(maxlen=max_size)

    def __len__(self):
        return len(self.obs)

    def add_batch(
        self,
        obs: torch.Tensor,
        scene_pc: torch.Tensor,
        scene_mask: torch.Tensor,
        hand_pc: torch.Tensor,
        tactile_pc: torch.Tensor,
        tactile_force: torch.Tensor,
        expert_action: torch.Tensor,
        tactile_mask: torch.Tensor | None = None,
    ) -> None:
        # Store as fp32, not fp16: at scene-PC coordinates >1m the fp16
        # mantissa drops resolution to ~1mm, which degrades PointNet learning.
        # 3.2 GB at max_buffer=200k fits comfortably on a 64 GB box.
        obs_np = obs.detach().cpu().numpy().astype(np.float32)
        sc_pc = scene_pc.detach().to(torch.float32).cpu()
        sc_mask = scene_mask.detach().cpu()
        hp = hand_pc.detach().to(torch.float32).cpu()
        tp = tactile_pc.detach().to(torch.float32).cpu()
        tf = tactile_force.detach().to(torch.float32).cpu()
        if tactile_mask is None:
            tm = torch.ones(tactile_force.shape[:2], dtype=torch.bool)
        else:
            tm = tactile_mask.detach().cpu().bool()
        act_np = expert_action.detach().cpu().numpy().astype(np.float32)
        B = obs_np.shape[0]
        for i in range(B):
            self.obs.append(obs_np[i])
            self.scene_pc.append(sc_pc[i])
            self.scene_mask.append(sc_mask[i])
            self.hand_pc.append(hp[i])
            self.tactile_pc.append(tp[i])
            self.tactile_force.append(tf[i])
            self.tactile_mask.append(tm[i])
            self.actions.append(act_np[i])

    def to_tensors(self):
        return (
            torch.from_numpy(np.stack(list(self.obs))),                    # (N, proprio_dim) fp32
            torch.stack(list(self.scene_pc)),                              # (N, n_scene, 3)   fp32
            torch.stack(list(self.scene_mask)),                            # (N, n_scene)     bool
            torch.stack(list(self.hand_pc)),                               # (N, n_hand, 3)    fp32
            torch.stack(list(self.tactile_pc)),                            # (N, n_tactile,3)  fp32
            torch.stack(list(self.tactile_force)),                         # (N, n_tactile,F)  fp32
            torch.stack(list(self.tactile_mask)),                          # (N, n_tactile)   bool
            torch.from_numpy(np.stack(list(self.actions))),                # (N, act_dim)      fp32
        )


# ----------------------------------------------------------------------------
# DAgger trainer
# ----------------------------------------------------------------------------
class DAggerPointCloud:
    """DAgger online distillation: state teacher → point-cloud student.

    Standard DAgger schedule (β-mix expert/student rollout, MSE on expert
    actions, β-decay) with the point-cloud student.
    """

    def __init__(
        self,
        cfg: DAggerPointCloudConfig,
        env,
        teacher_model: nn.Module,
        teacher_running_mean_std,
        student_ckpt_path: str | None = None,
    ):
        self.cfg = cfg
        self.env = env
        self.device = torch.device(cfg.device)

        # ---- Teacher (frozen) ----
        self.teacher = teacher_model
        self.teacher.eval()
        self.teacher_rms = teacher_running_mean_std
        if self.teacher_rms is not None:
            self.teacher_rms.eval()

        # ---- Student ----
        pc_encoder_cfg = dict(
            fusion_strategy=cfg.pc_fusion_strategy,
            n_scene=cfg.n_scene,
            n_hand=cfg.n_hand,
            n_tactile=cfg.n_tactile,
            tactile_feat_dim=cfg.tactile_feat_dim,
            output_dim=cfg.pc_output_dim,
            ablate_tactile_pc=cfg.ablate_tactile_pc,
            type_repr=cfg.pc_type_repr,
        )
        if cfg.student_keep_idx:
            self.student_keep_idx = torch.tensor(
                list(cfg.student_keep_idx), dtype=torch.long, device=self.device)
            print(f"[DAggerPC] LEAN STUDENT: proprio sliced to "
                  f"{len(cfg.student_keep_idx)}d, dropped "
                  f"{list(cfg.student_drop_slots)} (teacher unaffected)")
        else:
            self.student_keep_idx = None

        if cfg.student_obs_mask_idx:
            self.student_obs_mask_idx = torch.tensor(
                list(cfg.student_obs_mask_idx), dtype=torch.long, device=self.device)
            print(f"[DAggerPC] SPARSE-REF: masking {len(cfg.student_obs_mask_idx)} "
                  f"student proprio dims (teacher unaffected)")
        else:
            self.student_obs_mask_idx = None

        self.student = PointCloudStudent(
            proprio_dim=cfg.proprio_dim,
            action_dim=cfg.action_dim,
            pc_encoder_cfg=pc_encoder_cfg,
            hidden=tuple(cfg.student_hidden),
            action_clip=cfg.action_clip,
        ).to(self.device)

        if student_ckpt_path is not None:
            ckpt = load_checkpoint(student_ckpt_path, map_location=self.device)
            sd = ckpt.get("model", ckpt)
            self.student.load_state_dict(sd)
            print(f"[DAggerPC] Loaded student from {student_ckpt_path}")

        # ---- Optim ----
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            fused=True,
        )

        self.buffer = PCReplayBuffer(cfg.max_buffer_size)
        self.beta = cfg.beta_init

        os.makedirs(cfg.out_dir, exist_ok=True)
        n_params = sum(p.numel() for p in self.student.parameters())
        print(
            f"[DAggerPC] Initialized:\n"
            f"  fusion={cfg.pc_fusion_strategy}  n_scene={cfg.n_scene} "
            f"n_hand={cfg.n_hand} n_tactile={cfg.n_tactile}\n"
            f"  student params: {n_params/1e6:.2f}M\n"
            f"  buffer max: {cfg.max_buffer_size:,} transitions"
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _expert_action(self, obs_full: torch.Tensor) -> torch.Tensor:
        """Teacher sees the full env obs (normalize if rms provided)."""
        x = obs_full
        if self.teacher_rms is not None:
            x = self.teacher_rms(x)
        action = self.teacher.act_inference({"obs": x})
        return torch.clamp(action, -self.cfg.action_clip, self.cfg.action_clip)

    @torch.no_grad()
    def _student_action(self, obs_dict: dict) -> torch.Tensor:
        self.student.eval()
        return self.student.act_inference(obs_dict)

    # ------------------------------------------------------------------
    def rollout(self) -> dict:
        cfg = self.cfg
        env = self.env
        n_steps = 0
        n_added = 0
        n_success = 0.0
        n_episodes = 0.0

        obs_dict = env.reset()
        if isinstance(obs_dict, tuple):
            obs_dict = obs_dict[0]

        while n_steps < cfg.rollout_steps_per_iter:
            proprio = obs_dict["policy"]
            scene_pc = obs_dict["scene_pc"]
            scene_mask = obs_dict["scene_mask"]
            hand_pc = obs_dict["hand_pc"]
            tactile_pc = obs_dict["tactile_pc"]
            tactile_force = obs_dict["tactile_force"]
            tactile_mask = obs_dict.get("tactile_mask", None)  # opt-in

            # The teacher labels from the FULL obs; only the student's copy is
            # reduced. Masking runs before slicing so mask indices refer to the
            # env's original layout either way.
            expert_action = self._expert_action(proprio)

            proprio_student = proprio
            if self.student_obs_mask_idx is not None:
                proprio_student = proprio.clone()
                proprio_student[:, self.student_obs_mask_idx] = 0.0
            if self.student_keep_idx is not None:
                proprio_student = proprio_student[:, self.student_keep_idx]

            student_obs = {
                "obs": proprio_student,
                "scene_pc": scene_pc,
                "scene_mask": scene_mask,
                "hand_pc": hand_pc,
                "tactile_pc": tactile_pc,
                "tactile_force": tactile_force,
                "tactile_mask": tactile_mask,
            }
            student_action = self._student_action(student_obs)

            if self.beta > 0:
                action = self.beta * expert_action + (1 - self.beta) * student_action
            else:
                action = student_action
            action = torch.clamp(action, -cfg.action_clip, cfg.action_clip)

            self.buffer.add_batch(
                proprio_student, scene_pc, scene_mask, hand_pc,
                tactile_pc, tactile_force, expert_action,
                tactile_mask=tactile_mask,
            )
            n_added += proprio.shape[0]

            result = env.step(action)
            if len(result) == 5:
                obs_dict, _, terminated, truncated, extras = result
                done = terminated | truncated
            else:
                obs_dict, _, done, extras = result
            n_steps += proprio.shape[0]

            # Episode success rate of the ROLLOUT policy, over the episodes that
            # END inside this rollout. The env flags success on the step an
            # episode ends (reached the demo's end without a failure). Each
            # rollout starts from reset and lasts rollout_steps_per_iter /
            # num_envs steps per env, usually far shorter than an episode, so it
            # only counts episodes short enough to end inside that window, which
            # skews toward early terminations. It is a trend signal, not a
            # success rate: even a strong teacher (iteration 1, beta = 1) reads
            # far lower here than in eval_teacher.py. Use eval.py for students.
            if isinstance(extras, dict) and "succeeded_per_env" in extras:
                done = done.bool()
                n_success += float(extras["succeeded_per_env"][done].sum())
                n_episodes += float(done.sum())

        return {"steps": n_steps, "buffer_size": len(self.buffer), "added": n_added,
                "rollout_succ": (n_success / n_episodes) if n_episodes else float("nan"),
                "rollout_episodes": int(n_episodes)}

    # ------------------------------------------------------------------
    def train_student(self) -> dict:
        self.student.train()
        obs_t, sc_t, sm_t, hp_t, tp_t, tf_t, tm_t, act_t = self.buffer.to_tensors()
        try:
            obs_t = obs_t.pin_memory()
            act_t = act_t.pin_memory()
        except Exception:
            pass

        dataset = TensorDataset(obs_t, sc_t, sm_t, hp_t, tp_t, tf_t, tm_t, act_t)
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            pin_memory=False,
        )

        total_loss = 0.0
        n_batches = 0
        for _ in range(self.cfg.train_epochs):
            for obs_b, sc_b, sm_b, hp_b, tp_b, tf_b, tm_b, act_b in loader:
                obs_b = obs_b.to(self.device, non_blocking=True)
                sc_b = sc_b.to(self.device, non_blocking=True).float()
                sm_b = sm_b.to(self.device, non_blocking=True)
                hp_b = hp_b.to(self.device, non_blocking=True).float()
                tp_b = tp_b.to(self.device, non_blocking=True).float()
                tf_b = tf_b.to(self.device, non_blocking=True).float()
                tm_b = tm_b.to(self.device, non_blocking=True)
                act_b = act_b.to(self.device, non_blocking=True)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred = self.student(obs_b, sc_b, sm_b, hp_b, tp_b, tf_b, tm_b)
                    loss = F.mse_loss(pred.float(), act_b)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.student.parameters(), self.cfg.grad_clip)
                self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        return {"avg_loss": total_loss / max(n_batches, 1), "n_batches": n_batches}

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save(
            {
                "model": self.student.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "beta": self.beta,
                "cfg": self.cfg,
                # mirror critical dims at top level for safe reload
                "proprio_dim": self.cfg.proprio_dim,
                "action_dim": self.cfg.action_dim,
                "n_scene": self.cfg.n_scene,
                "n_hand": self.cfg.n_hand,
                "n_tactile": self.cfg.n_tactile,
                "tactile_feat_dim": self.cfg.tactile_feat_dim,
                "pc_fusion_strategy": self.cfg.pc_fusion_strategy,
                "pc_output_dim": self.cfg.pc_output_dim,
                "ablate_tactile_pc": self.cfg.ablate_tactile_pc,
                "pc_type_repr": self.cfg.pc_type_repr,
                # env-side PC transforms — eval/deploy restore these so the
                # student sees the same input distribution it trained on.
                "pc_env_meta": dict(self.cfg.pc_env_meta),
                "student_hidden": list(self.cfg.student_hidden),
                # Student obs layout — eval and deploy re-apply these so the
                # student is fed exactly the vector it was trained on.
                "student_obs_mask_idx": list(self.cfg.student_obs_mask_idx),
                "student_keep_idx": list(self.cfg.student_keep_idx),
                "student_obs_slots": dict(self.cfg.student_obs_slots),
                "student_drop_slots": list(self.cfg.student_drop_slots),
            },
            path,
        )

    def run(self):
        print(f"\n[DAggerPC] Starting {self.cfg.dagger_iters} iterations")
        for it in range(1, self.cfg.dagger_iters + 1):
            t0 = time.time()
            rollout_info = self.rollout()
            train_info = self.train_student()
            dt = time.time() - t0
            print(
                f"  Iter {it:2d}/{self.cfg.dagger_iters} | "
                f"β={self.beta:.3f} | "
                f"loss={train_info['avg_loss']:.6f} | "
                f"buffer={rollout_info['buffer_size']:,} | "
                f"roll_succ={rollout_info['rollout_succ']:.3f} "
                f"(n={rollout_info['rollout_episodes']}) | "
                f"{dt:.1f}s"
            )
            self.save(os.path.join(self.cfg.out_dir, f"dagger_iter_{it:02d}.pth"))
            self.beta = max(self.cfg.beta_min, self.beta * self.cfg.beta_decay)

        self.save(os.path.join(self.cfg.out_dir, "dagger_final.pth"))
        print(f"\n[DAggerPC] Done. Checkpoints saved to {self.cfg.out_dir}")
