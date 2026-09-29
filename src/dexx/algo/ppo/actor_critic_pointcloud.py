"""`ActorCriticPointCloud` — PPO actor-critic with a PointCloud actor.

Drop-in for `ActorCriticAsymmetric` but the actor consumes proprio + 3-source
point cloud through a `PointCloudEncoder`. Critic still uses proprio +
priv_info (priv_info has GT object pose so no PC needed for value baseline).

Lives outside `models.py`; `ppo_pointcloud.PPOPointCloud` instantiates it.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from dexx.algo.models.models import MLP
from dexx.tasks.franka_sharpa.pointcloud import PointCloudEncoder


class ActorCriticPointCloud(nn.Module):
    """Asymmetric AC with a PointNet-encoded actor.

    Expected `kwargs` keys (consistent with `ActorCriticAsymmetric`):
        actions_num         : int
        input_shape         : (proprio_dim,)
        actor_units         : list[int]            (MLP after concat)
        priv_mlp_units      : unused (kept for cfg compatibility)
        priv_info_dim       : int
        pc_config           : dict — kwargs forwarded to PointCloudEncoder
        critic_units        : list[int]            (optional; defaults to actor_units)
    """

    def __init__(self, kwargs: dict):
        super().__init__()
        actions_num = kwargs.pop("actions_num")
        input_shape = kwargs.pop("input_shape")
        self.units = kwargs.pop("actor_units")
        _ = kwargs.pop("priv_mlp_units", None)  # unused (no env_mlp)
        self.priv_info_dim = int(kwargs["priv_info_dim"])

        proprio_dim = int(input_shape[0])
        pc_config = kwargs.get("pc_config", {})
        if not pc_config:
            raise ValueError(
                "ActorCriticPointCloud needs `pc_config` in kwargs "
                "(forwarded to PointCloudEncoder)."
            )
        self.pc_encoder = PointCloudEncoder(**pc_config)
        pc_feat_dim = int(pc_config.get("output_dim", 64))

        actor_obs_dim = proprio_dim + pc_feat_dim
        critic_obs_dim = proprio_dim + self.priv_info_dim

        actor_out_size = self.units[-1]
        critic_units = kwargs.get("critic_units", self.units)
        critic_out_size = critic_units[-1]

        self.actor_mlp = MLP(units=self.units, input_size=actor_obs_dim)
        self.critic_mlp = MLP(units=critic_units, input_size=critic_obs_dim)

        self.mu = nn.Linear(actor_out_size, actions_num)
        self.sigma = nn.Parameter(
            torch.zeros(actions_num, dtype=torch.float32), requires_grad=True
        )
        self.value = nn.Linear(critic_out_size, 1)

        # Init biases to 0 (same as ActorCriticAsymmetric).
        for m in self.modules():
            if isinstance(m, nn.Linear) and getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
        # Init logstd. Default 0.0 = sigma 1.0 is too wide for fine-tuning a
        # DAgger student whose action units are roughly [-1, 1]: at sigma=1 the
        # sampled action is essentially random noise and DAgger's mu signal is
        # drowned. Pass `init_logstd` in kwargs (e.g. -2.0 → sigma 0.135) for
        # PPO fine-tune; leave at 0 for from-scratch PPO if ever needed.
        init_logstd = float(kwargs.get("init_logstd", 0.0))
        nn.init.constant_(self.sigma, init_logstd)

    # ------------------------------------------------------------------
    def _encode_pc(self, obs_dict: dict) -> torch.Tensor:
        return self.pc_encoder(
            obs_dict["scene_pc"],
            obs_dict["scene_mask"],
            obs_dict["hand_pc"],
            obs_dict["tactile_pc"],
            obs_dict["tactile_force"],
            obs_dict.get("tactile_mask", None),
        )

    def _actor_critic(self, obs_dict: dict):
        proprio = obs_dict["obs"]
        pc_feat = self._encode_pc(obs_dict)
        actor_in = torch.cat([proprio, pc_feat], dim=-1)
        mu = self.mu(self.actor_mlp(actor_in))

        priv_info = obs_dict.get("priv_info", None)
        if priv_info is None:
            priv_info = torch.zeros(
                proprio.shape[0], self.priv_info_dim, device=proprio.device
            )
        critic_in = torch.cat([proprio, priv_info], dim=-1)
        value = self.value(self.critic_mlp(critic_in))
        return mu, mu * 0 + self.sigma, value

    # ------------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs_dict: dict) -> dict:
        mu, logstd, value = self._actor_critic(obs_dict)
        # Clamp logstd to keep sigma in (e^-5 ≈ 0.0067, e^2 ≈ 7.4). Prevents the
        # exp() from blowing up under bf16 autocast + bad gradient signal.
        logstd = torch.clamp(logstd, min=-5.0, max=2.0)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        action = distr.sample()
        return {
            "neglogpacs": -distr.log_prob(action).sum(1),
            "values": value,
            "actions": action,
            "mus": mu,
            "sigmas": sigma,
        }

    @torch.no_grad()
    def act_inference(self, obs_dict: dict) -> torch.Tensor:
        """Deploy: only the actor (no priv_info needed). PC tensors required."""
        proprio = obs_dict["obs"]
        pc_feat = self._encode_pc(obs_dict)
        actor_in = torch.cat([proprio, pc_feat], dim=-1)
        return self.mu(self.actor_mlp(actor_in))

    def forward(self, input_dict: dict) -> dict:
        prev_actions = input_dict.get("prev_actions", None)
        mu, logstd, value = self._actor_critic(input_dict)
        logstd = torch.clamp(logstd, min=-5.0, max=2.0)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        entropy = distr.entropy().sum(dim=-1)
        prev_neglogp = -distr.log_prob(prev_actions).sum(1) if prev_actions is not None else None
        return {
            "prev_neglogp": torch.squeeze(prev_neglogp) if prev_neglogp is not None else None,
            "values": value,
            "entropy": entropy,
            "mus": mu,
            "sigmas": sigma,
            "extrin": None,
            "extrin_gt": None,
        }
