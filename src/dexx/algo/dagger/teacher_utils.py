"""Teacher checkpoint loading utilities.

Free of module-level argparse, so any script that needs `load_teacher` can
import it without this module consuming `sys.argv`.
"""
from __future__ import annotations

import torch

from dexx.algo.models.models import ActorCritic, ActorCriticAsymmetric
from dexx.algo.models.running_mean_std import RunningMeanStd


def infer_dims_from_ckpt(state_dict: dict) -> dict:
    """Infer (obs_dim, priv_info_dim, asymmetric, action_dim) from ckpt shapes.

    Reads the shape of `actor_mlp.mlp.0.weight` (first actor layer) and
    `critic_mlp.mlp.0.weight` (first critic layer, asymmetric) to deduce dims.
    """
    inferred: dict = {}

    actor_w_key = next(
        (k for k in state_dict.keys() if k.endswith("actor_mlp.mlp.0.weight")), None,
    )
    if actor_w_key is None:
        raise RuntimeError(
            f"Could not find actor_mlp.mlp.0.weight in ckpt. "
            f"Available keys (first 10): {list(state_dict.keys())[:10]}"
        )
    inferred["obs_dim"] = state_dict[actor_w_key].shape[1]

    critic_w_key = next(
        (k for k in state_dict.keys() if k.endswith("critic_mlp.mlp.0.weight")), None,
    )
    if critic_w_key is not None:
        critic_in = state_dict[critic_w_key].shape[1]
        inferred["priv_info_dim"] = critic_in - inferred["obs_dim"]
        inferred["asymmetric"] = True
    else:
        inferred["priv_info_dim"] = None
        inferred["asymmetric"] = False

    mu_w_key = next((k for k in state_dict.keys() if k.endswith("mu.weight")), None)
    if mu_w_key is not None:
        inferred["action_dim"] = state_dict[mu_w_key].shape[0]

    return inferred


def load_teacher(
    ckpt_path: str,
    obs_dim: int | None = None,
    action_dim: int | None = None,
    priv_info_dim: int | None = None,
    asymmetric: bool = True,
    device: str = "cuda",
):
    """Load a state-expert checkpoint into ActorCriticAsymmetric (or ActorCritic).

    All dimension args are OPTIONAL — if omitted, they are inferred from the
    checkpoint's state_dict shapes. `asymmetric` is auto-detected from the
    presence of `critic_mlp` keys (so an asymmetric ckpt is loaded with the
    right class even if you forget the flag).
    """
    checkpoint = torch.load(ckpt_path, map_location=device)
    sd = checkpoint["model"] if "model" in checkpoint else checkpoint

    inferred = infer_dims_from_ckpt(sd)
    if obs_dim is None:
        obs_dim = inferred["obs_dim"]
    if action_dim is None and "action_dim" in inferred:
        action_dim = inferred["action_dim"]
    if priv_info_dim is None and inferred["priv_info_dim"] is not None:
        priv_info_dim = inferred["priv_info_dim"]
    if inferred["asymmetric"]:
        asymmetric = True

    print(
        f"[load_teacher] {ckpt_path}\n"
        f"    inferred: obs_dim={inferred['obs_dim']}  "
        f"priv_info_dim={inferred['priv_info_dim']}  "
        f"asymmetric={inferred['asymmetric']}  "
        f"action_dim={inferred.get('action_dim')}\n"
        f"    using:    obs_dim={obs_dim}  priv_info_dim={priv_info_dim}  "
        f"asymmetric={asymmetric}  action_dim={action_dim}"
    )
    if priv_info_dim is None:
        priv_info_dim = 40  # final fallback (teachers without critic-horizon priv_info)

    if asymmetric:
        model = ActorCriticAsymmetric({
            "actions_num": action_dim,
            "input_shape": (obs_dim,),
            "actor_units": [256, 512, 128, 64],
            "priv_mlp_units": [256, 128, priv_info_dim],
            "priv_info": True,
            "priv_info_dim": priv_info_dim,
            "proprio_adapt": False,
        })
    else:
        model = ActorCritic({
            "actions_num": action_dim,
            "input_shape": (obs_dim,),
            "actor_units": [256, 512, 128, 64],
            "priv_mlp_units": [256, 128, priv_info_dim],
            "priv_info": True,
            "priv_info_dim": priv_info_dim,
            "proprio_adapt": False,
        })

    model.load_state_dict(sd)
    model.to(device)
    model.eval()

    running_mean_std = RunningMeanStd((obs_dim,)).to(device)
    if "running_mean_std" in checkpoint:
        running_mean_std.load_state_dict(checkpoint["running_mean_std"])
    running_mean_std.eval()

    return model, running_mean_std
