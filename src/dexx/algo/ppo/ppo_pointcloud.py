"""Minimal PPO fine-tune trainer for `ActorCriticPointCloud`.

Deliberately small, but enough to do a useful fine-tune from a DAgger
initialization. Standard clipped PPO with GAE,
fp32 forward/backward, AdamW + fused.

Buffer layout (on GPU, since num_envs * horizon * PC size is small):
    obses       (num_envs, horizon, proprio_dim) fp32
    scene_pc    (num_envs, horizon, n_scene, 3) fp16
    scene_mask  (num_envs, horizon, n_scene) bool
    hand_pc     (num_envs, horizon, n_hand, 3) fp16
    tactile_pc  (num_envs, horizon, n_tactile, 3) fp16
    tactile_force (num_envs, horizon, n_tactile, F) fp16
    priv_info   (num_envs, horizon, priv_dim) fp32
    actions     (num_envs, horizon, action_dim) fp32
    log_probs / values / rewards / dones / returns / advantages

Total @ 64 envs × 32 horizon: ~50 MB for the PCs + ~5 MB for the rest. Fits.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from dexx.algo.ppo.actor_critic_pointcloud import ActorCriticPointCloud


@dataclass
class PPOPointCloudConfig:
    # Env / dims
    proprio_dim: int = 0
    action_dim: int = 0
    priv_info_dim: int = 148
    n_scene: int = 1024
    n_hand: int = 11
    n_tactile: int = 25
    tactile_feat_dim: int = 1
    # PC encoder
    pc_fusion_strategy: str = "early_concat"
    pc_output_dim: int = 64
    ablate_tactile_pc: bool = False
    pc_type_repr: str = "scalar"  # "scalar" | "onehot"
    # Env-side PC transforms (force scale / gate / repr / ablations) — see
    # algo/dagger/pc_env_meta.py. Saved with the ckpt via `cfg` so eval/deploy
    # can restore the training-time input distribution.
    pc_env_meta: dict = field(default_factory=dict)
    actor_units: tuple = (256, 512, 128, 64)
    critic_units: tuple = (256, 512, 128, 64)
    # logstd init: -2.0 → sigma ≈ 0.14 (tight enough to keep DAgger mu signal
    # at warm-up; PPO can still loosen via gradient if entropy bonus pushes).
    init_logstd: float = -2.0
    # PPO
    horizon: int = 32
    num_envs: int = 64
    gamma: float = 0.99
    tau: float = 0.95           # GAE lambda
    clip: float = 0.2
    value_clip: bool = True
    entropy_coef: float = 1e-4
    critic_coef: float = 0.5            # critic is random-init at PPO start and
                                         # raw rewards push critic loss to thousands
    critic_loss_clip: float = 10.0      # clamp per-sample (values - returns)^2 to keep
                                         # gradient bounded under bf16 autocast
    grad_clip: float = 1.0
    lr: float = 5e-5
    weight_decay: float = 1e-4
    mini_epochs: int = 2
    minibatch_size: int = 1024
    # Run
    max_iters: int = 200
    save_every_n: int = 10
    out_dir: str = "logs/ppo_pc"
    device: str = "cuda"
    action_clip: float = 1.0


class PPOPointCloud:
    def __init__(
        self,
        cfg: PPOPointCloudConfig,
        env,
        dagger_ckpt_path: str | None = None,
    ):
        self.cfg = cfg
        self.env = env
        self.device = torch.device(cfg.device)

        # ---- Model ----
        pc_cfg = dict(
            fusion_strategy=cfg.pc_fusion_strategy,
            n_scene=cfg.n_scene,
            n_hand=cfg.n_hand,
            n_tactile=cfg.n_tactile,
            tactile_feat_dim=cfg.tactile_feat_dim,
            output_dim=cfg.pc_output_dim,
            ablate_tactile_pc=cfg.ablate_tactile_pc,
            type_repr=getattr(cfg, "pc_type_repr", "scalar"),
        )
        self.model = ActorCriticPointCloud(dict(
            actions_num=cfg.action_dim,
            input_shape=(cfg.proprio_dim,),
            actor_units=list(cfg.actor_units),
            priv_mlp_units=[256, 128, cfg.priv_info_dim],
            priv_info_dim=cfg.priv_info_dim,
            critic_units=list(cfg.critic_units),
            pc_config=pc_cfg,
            init_logstd=cfg.init_logstd,
        )).to(self.device)

        # ---- Load DAgger init (just the actor + PC encoder weights) ----
        if dagger_ckpt_path is not None:
            self._load_dagger_init(dagger_ckpt_path)

        # ---- Optim ----
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            fused=True,
        )

        # ---- Storage (per-step rollout buffer, allocated lazily) ----
        self._storage: dict[str, torch.Tensor] = {}

        os.makedirs(cfg.out_dir, exist_ok=True)
        n_params = sum(p.numel() for p in self.model.parameters())
        print(
            f"[PPOPC] Initialized:\n"
            f"  fusion={cfg.pc_fusion_strategy}  proprio={cfg.proprio_dim}  "
            f"act={cfg.action_dim}  priv={cfg.priv_info_dim}\n"
            f"  PC counts: scene={cfg.n_scene}  hand={cfg.n_hand}  "
            f"tactile={cfg.n_tactile}\n"
            f"  params: {n_params/1e6:.2f}M  | horizon={cfg.horizon}  "
            f"num_envs={cfg.num_envs}  minibatch={cfg.minibatch_size}"
        )

    # ------------------------------------------------------------------
    def _load_dagger_init(self, path: str) -> None:
        """DAgger ckpt has the `PointCloudStudent` state_dict; map matching
        keys (pc_encoder + actor_mlp / mu) into our `ActorCriticPointCloud`.
        Critic weights are random-init (PPO learns critic from scratch)."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        sd = ckpt.get("model", ckpt)

        own = self.model.state_dict()
        loaded = 0
        skipped = []
        for k, v in sd.items():
            # PointCloudStudent uses self.pc_encoder.* and self.mlp.*; map mlp.* → actor_mlp.*
            # PC encoder name matches.
            tk = None
            if k.startswith("pc_encoder."):
                tk = k
            elif k.startswith("mlp."):
                # mlp.0.weight (Linear) → actor_mlp.mlp.0.weight
                # mlp.<last>.weight (output head) → mu.weight
                # Figure out last index from the student.
                idx = k.split(".")[1]
                rest = ".".join(k.split(".")[2:])
                # Student MLP: [Linear, ELU, Linear, ELU, Linear(action)]
                # Indices: 0=lin, 1=ELU, 2=lin, 3=ELU, 4=lin (output head)
                # We need to know how many hidden layers. Detect: a key with
                # the largest even idx is the output. The student saves only
                # Linear layers (no ELU params), so even-idx Linears.
                # Easier: build a parallel mapping table.
                pass
            if tk is not None and tk in own and own[tk].shape == v.shape:
                own[tk] = v
                loaded += 1
            else:
                skipped.append(k)

        # Handle student MLP → actor mlp head mapping properly: traverse the
        # student MLP's parameter list in order. Easiest: just iterate keys
        # matching `mlp.\d+\.` and match by ordinal to our actor's `actor_mlp.mlp.\d+.`.
        student_mlp_layers = sorted(
            {int(k.split(".")[1]) for k in sd if k.startswith("mlp.")}
        )
        # Student layers e.g. [0, 2, 4] for two hidden + output
        # Actor mlp has same hidden layout (actor_mlp.mlp.[0, 2, 4, ...])
        actor_mlp_layers = sorted(
            {int(k.split(".")[2]) for k in own if k.startswith("actor_mlp.mlp.")}
        )
        # Map the first len(actor_mlp_layers) student layers into actor_mlp;
        # the last student layer is the action head → goes to self.mu.
        # i.e. student[0,2,..] -> actor_mlp.mlp.[0,2,..]; student[last] -> mu
        if student_mlp_layers and actor_mlp_layers:
            head_layer_idx = student_mlp_layers[-1]
            hidden_student = student_mlp_layers[:-1]
            for s_idx, a_idx in zip(hidden_student, actor_mlp_layers):
                for suffix in ("weight", "bias"):
                    src = f"mlp.{s_idx}.{suffix}"
                    dst = f"actor_mlp.mlp.{a_idx}.{suffix}"
                    if src in sd and dst in own and sd[src].shape == own[dst].shape:
                        own[dst] = sd[src]
                        loaded += 1
            # head
            for suffix in ("weight", "bias"):
                src = f"mlp.{head_layer_idx}.{suffix}"
                dst = f"mu.{suffix}"
                if src in sd and dst in own and sd[src].shape == own[dst].shape:
                    own[dst] = sd[src]
                    loaded += 1

        self.model.load_state_dict(own)
        print(f"[PPOPC] Loaded {loaded} params from DAgger ckpt ({path})")

    # ------------------------------------------------------------------
    def _alloc_storage(self, obs: dict) -> None:
        H, N = self.cfg.horizon, self.cfg.num_envs
        D = self.device
        self._storage = {
            "obs": torch.zeros(N, H, self.cfg.proprio_dim, device=D),
            "scene_pc": torch.zeros(N, H, self.cfg.n_scene, 3, dtype=torch.float16, device=D),
            "scene_mask": torch.zeros(N, H, self.cfg.n_scene, dtype=torch.bool, device=D),
            "hand_pc": torch.zeros(N, H, self.cfg.n_hand, 3, dtype=torch.float16, device=D),
            "tactile_pc": torch.zeros(N, H, self.cfg.n_tactile, 3, dtype=torch.float16, device=D),
            "tactile_force": torch.zeros(N, H, self.cfg.n_tactile, self.cfg.tactile_feat_dim, dtype=torch.float16, device=D),
            "priv_info": torch.zeros(N, H, self.cfg.priv_info_dim, device=D),
            "actions": torch.zeros(N, H, self.cfg.action_dim, device=D),
            "values": torch.zeros(N, H, 1, device=D),
            "log_probs": torch.zeros(N, H, device=D),
            "rewards": torch.zeros(N, H, device=D),
            "dones": torch.zeros(N, H, device=D),
            "mus": torch.zeros(N, H, self.cfg.action_dim, device=D),
            "sigmas": torch.zeros(N, H, self.cfg.action_dim, device=D),
        }

    def _stash(self, n: int, obs: dict, action_res: dict, reward: torch.Tensor, done: torch.Tensor) -> None:
        s = self._storage
        s["obs"][:, n] = obs["policy"]
        s["scene_pc"][:, n] = obs["scene_pc"].to(torch.float16)
        s["scene_mask"][:, n] = obs["scene_mask"]
        s["hand_pc"][:, n] = obs["hand_pc"].to(torch.float16)
        s["tactile_pc"][:, n] = obs["tactile_pc"].to(torch.float16)
        s["tactile_force"][:, n] = obs["tactile_force"].to(torch.float16)
        s["priv_info"][:, n] = obs.get("priv_info", torch.zeros(self.cfg.num_envs, self.cfg.priv_info_dim, device=self.device))
        s["actions"][:, n] = action_res["actions"]
        s["values"][:, n] = action_res["values"]
        s["log_probs"][:, n] = -action_res["neglogpacs"]
        s["rewards"][:, n] = reward
        s["dones"][:, n] = done.float()
        s["mus"][:, n] = action_res["mus"]
        s["sigmas"][:, n] = action_res["sigmas"]

    def _compute_gae(self, last_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s = self._storage
        N, H = s["rewards"].shape
        gamma, tau = self.cfg.gamma, self.cfg.tau
        adv = torch.zeros_like(s["rewards"])
        last_gae = 0.0
        for t in reversed(range(H)):
            next_value = last_value.squeeze(-1) if t == H - 1 else s["values"][:, t + 1].squeeze(-1)
            mask = 1.0 - s["dones"][:, t]
            delta = s["rewards"][:, t] + gamma * next_value * mask - s["values"][:, t].squeeze(-1)
            last_gae = delta + gamma * tau * mask * last_gae
            adv[:, t] = last_gae
        returns = adv + s["values"].squeeze(-1)
        # Normalize advantage (standard PPO trick).
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, returns

    # ------------------------------------------------------------------
    def _make_obs_dict(self, env_obs: dict) -> dict:
        """Translate env step output → model input dict."""
        return {
            "obs": env_obs["policy"],
            "scene_pc": env_obs["scene_pc"],
            "scene_mask": env_obs["scene_mask"],
            "hand_pc": env_obs["hand_pc"],
            "tactile_pc": env_obs["tactile_pc"],
            "tactile_force": env_obs["tactile_force"],
            "priv_info": env_obs.get("priv_info"),
        }

    @torch.no_grad()
    def play_steps(self, env_obs: dict) -> dict:
        """One PPO rollout (horizon steps). Returns last obs (for next iter)."""
        if not self._storage:
            self._alloc_storage(env_obs)

        cum_rew = 0.0
        cum_succ = 0.0
        n_ep = 0
        for t in range(self.cfg.horizon):
            mdl_in = self._make_obs_dict(env_obs)
            res = self.model.act(mdl_in)
            # Store the raw sample (its log-prob is what `neglogpacs` holds);
            # clamp only what goes to the env, as ppo.py does.
            actions = torch.clamp(res["actions"], -self.cfg.action_clip, self.cfg.action_clip)

            step_result = self.env.step(actions)
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, extras = step_result
            else:
                next_obs, reward, dones_full, extras = step_result
                terminated = dones_full
                truncated = torch.zeros_like(dones_full)
            done = terminated | truncated

            self._stash(t, env_obs, res, reward, done)
            cum_rew += reward.mean().item()
            if isinstance(extras, dict):
                v = extras.get("succeeded", 0.0)
                cum_succ += float(v.item() if hasattr(v, "item") else v)
            if done.any():
                n_ep += int(done.sum().item())

            env_obs = next_obs

        # Bootstrap value at horizon end.
        with torch.no_grad():
            last_res = self.model.act(self._make_obs_dict(env_obs))
            last_val = last_res["values"]

        adv, returns = self._compute_gae(last_val)
        self._storage["advantages"] = adv
        self._storage["returns"] = returns

        return {
            "last_obs": env_obs,
            "mean_step_reward": cum_rew / self.cfg.horizon,
            "succeeded_per_step": cum_succ / self.cfg.horizon,
            "episodes_completed": n_ep,
        }

    # ------------------------------------------------------------------
    def update(self) -> dict:
        cfg = self.cfg
        s = self._storage
        N, H = cfg.num_envs, cfg.horizon
        flat = lambda x: x.reshape(N * H, *x.shape[2:])
        b = {
            "obs": flat(s["obs"]),
            "scene_pc": flat(s["scene_pc"]),
            "scene_mask": flat(s["scene_mask"]),
            "hand_pc": flat(s["hand_pc"]),
            "tactile_pc": flat(s["tactile_pc"]),
            "tactile_force": flat(s["tactile_force"]),
            "priv_info": flat(s["priv_info"]),
            "actions": flat(s["actions"]),
            "old_log_probs": flat(s["log_probs"]),
            "old_values": flat(s["values"]).squeeze(-1),
            "advantages": flat(s["advantages"]),
            "returns": flat(s["returns"]),
            "old_mus": flat(s["mus"]),
            "old_sigmas": flat(s["sigmas"]),
        }
        total = N * H
        idx = torch.arange(total, device=self.device)

        losses = {"actor": 0.0, "critic": 0.0, "entropy": 0.0, "kl": 0.0}
        n_batches = 0

        for epoch in range(cfg.mini_epochs):
            perm = idx[torch.randperm(total, device=self.device)]
            for start in range(0, total, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb = perm[start:end]
                if mb.numel() < 8:
                    continue

                batch_dict = {
                    "obs": b["obs"][mb],
                    "scene_pc": b["scene_pc"][mb].float(),
                    "scene_mask": b["scene_mask"][mb],
                    "hand_pc": b["hand_pc"][mb].float(),
                    "tactile_pc": b["tactile_pc"][mb].float(),
                    "tactile_force": b["tactile_force"][mb].float(),
                    "priv_info": b["priv_info"][mb],
                    "prev_actions": b["actions"][mb],
                }

                # Keep the PPO update in fp32 (no autocast): early actor-loss
                # spikes from large unclipped ratios overflow bf16 into NaN
                # gradients.
                out = self.model(batch_dict)
                new_log_probs = -out["prev_neglogp"]
                values = out["values"].squeeze(-1)
                entropy = out["entropy"]

                ratio = torch.exp(new_log_probs - b["old_log_probs"][mb])
                # Clamp ratio: standard PPO already clips via surr2, but the
                # unclipped surr1 = ratio * adv can still hit 1e5+ before the
                # gradient pulls it back, which overflows in bf16. Capping
                # the absolute ratio at a safe ceiling keeps surr1 bounded
                # without changing the optimization direction (the clamped
                # values are exactly the cases torch.min picks against anyway).
                ratio = ratio.clamp(max=10.0)
                surr1 = ratio * b["advantages"][mb]
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * b["advantages"][mb]
                actor_loss = -torch.min(surr1, surr2).mean()

                # Huber loss for the critic: gradient bounded at ±1 for
                # |residual| > beta, so a fresh critic starting far from
                # the returns doesn't blow gradients.
                if cfg.value_clip:
                    v_clipped = b["old_values"][mb] + torch.clamp(values - b["old_values"][mb], -cfg.clip, cfg.clip)
                    l1 = F.smooth_l1_loss(values, b["returns"][mb], reduction="none", beta=1.0)
                    l2 = F.smooth_l1_loss(v_clipped, b["returns"][mb], reduction="none", beta=1.0)
                    critic_loss = torch.max(l1, l2).mean()
                else:
                    critic_loss = F.smooth_l1_loss(values, b["returns"][mb], beta=1.0)

                entropy_loss = -entropy.mean()
                loss = actor_loss + cfg.critic_coef * critic_loss + cfg.entropy_coef * entropy_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self.optimizer.step()

                losses["actor"] += actor_loss.item()
                losses["critic"] += critic_loss.item()
                losses["entropy"] += entropy.mean().item()
                # KL between old and new policy
                with torch.no_grad():
                    new_mus = out["mus"].float()
                    new_sigmas = out["sigmas"].float()
                    old_mus = b["old_mus"][mb].float()
                    old_sigmas = b["old_sigmas"][mb].float() + 1e-8
                    kl = (torch.log(new_sigmas / old_sigmas + 1e-8)
                          + (old_sigmas ** 2 + (old_mus - new_mus) ** 2) / (2 * new_sigmas ** 2)
                          - 0.5).sum(dim=-1).mean()
                    losses["kl"] += kl.item()
                n_batches += 1

        for k in losses:
            losses[k] /= max(n_batches, 1)
        return losses

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "cfg": self.cfg,
            },
            path,
        )

    def run(self) -> None:
        env_result = self.env.reset()
        if isinstance(env_result, tuple):
            env_obs = env_result[0]
        else:
            env_obs = env_result

        t_global = time.time()
        for it in range(1, self.cfg.max_iters + 1):
            t0 = time.time()
            roll = self.play_steps(env_obs)
            env_obs = roll["last_obs"]
            losses = self.update()
            dt = time.time() - t0
            print(
                f"  iter {it:4d}/{self.cfg.max_iters} | "
                f"a_loss={losses['actor']:+.4f} c_loss={losses['critic']:+.4f} "
                f"H={losses['entropy']:+.3f} kl={losses['kl']:.4f} | "
                f"r̄={roll['mean_step_reward']:+.3f} "
                f"succ_rate={roll['succeeded_per_step']*100:5.2f}% "
                f"ep={roll['episodes_completed']:3d} | "
                f"{dt:.1f}s"
            )
            if it % self.cfg.save_every_n == 0:
                self.save(os.path.join(self.cfg.out_dir, f"ppo_iter_{it:04d}.pth"))

        self.save(os.path.join(self.cfg.out_dir, "ppo_final.pth"))
        elapsed = time.time() - t_global
        print(f"\n[PPOPC] Done. {self.cfg.max_iters} iters in {elapsed/60:.1f} min. "
              f"Saved to {self.cfg.out_dir}")
