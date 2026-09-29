# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Deploy counterpart of FrankaSharpaForceCriticHorizonEnv: sizes the actor obs
and its history buffers for the critic-horizon additions (target_obj_pos,
target_obj_quat, tips_distance — all read from the demo pkl, so available at
deploy). The observations themselves are built by V3.
"""
from __future__ import annotations

import torch
import gymnasium as gym

from isaaclab.envs.utils.spaces import spec_to_gym_space

from .franka_sharpa_force_deploy_env_v2 import FrankaSharpaForceDeployEnvV2
from .franka_sharpa_critic_horizon_cfg import FrankaSharpaCriticHorizonCfg


class FrankaSharpaForceCriticHorizonDeployEnv(FrankaSharpaForceDeployEnvV2):
    cfg: FrankaSharpaCriticHorizonCfg

    def __init__(self, cfg: FrankaSharpaCriticHorizonCfg, render_mode: str | None = None, **kwargs):
        # Parent deploy env calls ForceEnv.__init__ which may run asymmetric_ac
        # reduce (obs 543 -> 410). We extend afterward — same pattern as training
        # variant, but here actor obs must match the training one exactly.
        super().__init__(cfg, render_mode, **kwargs)

        _asymmetric_ac = getattr(self.cfg, 'asymmetric_ac', False)
        _enable_bps = getattr(self.cfg, 'enable_bps', True)
        bps_dim = self.obj_bps.shape[-1] if (self.obj_bps is not None and _enable_bps) else 0
        if _asymmetric_ac:
            new_obs_dim = 410 + 12 + bps_dim                 # 422 or 550
        else:
            # parent obs-dim already reflects enable_bps
            new_obs_dim = self.cfg.observation_space + 7     # 422 or 550

        self.cfg.observation_space = new_obs_dim
        self.num_obs = new_obs_dim
        self.single_observation_space["policy"] = spec_to_gym_space(new_obs_dim)
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space["policy"], self.num_envs
        )

        self.proprio_hist_dim = new_obs_dim // 3
        self.obs_buf_lag_history = torch.zeros(
            (self.num_envs, max(80, self.cfg.prop_hist_len + 10), self.proprio_hist_dim),
            device=self.device, dtype=torch.float,
        )
        self.proprio_hist_buf = torch.zeros(
            (self.num_envs, self.cfg.prop_hist_len, self.proprio_hist_dim),
            device=self.device, dtype=torch.float,
        )

        print(
            f"[CriticHorizonDeploy] asymmetric_ac={_asymmetric_ac}, "
            f"obs_dim={new_obs_dim}, priv_info_dim={self.cfg.priv_info_dim}"
        )
