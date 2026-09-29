"""Calibrate fingertip-elastomer collision-shape indices for a robot USD.

The env's friction domain-randomization assigns a softer "elastomer" friction to the
5 fingertip elastomer collision shapes and a metal friction to the rest. Those shapes
are addressed by their index into the PhysX material-properties array, which is
hand-USD dependent (the stock Sharpa Wave hand = 64 shapes) and is NOT simply
derivable from the USD (PhysX convex-decomposition changes the shape count/order).

This tool determines the indices empirically and robustly: it spawns the robot, then
for each elastomer body it sets a unique marker friction via that body's own PhysX
rigid-body view and reads back which rows of the articulation's global material array
changed. Those rows are that body's shapes.

Usage:
    python tools/calibrate_elastomer_ids.py \
        --side right --headless

Then set the printed list in your run, e.g.:
    python scripts/train_teacher.py ... \
        --material_elastomer_ids '[57,58,60,62,63]'
(or set env_cfg.material_elastomer_ids in code.)
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--usd", default=None,
                    help="Robot USD (abs or cwd-relative). Default: the converted+patched "
                         "USD for --side, same one the training envs spawn.")
parser.add_argument("--side", default="right", choices=["right", "left"], help="Hand side (prefix of the elastomer body names).")
parser.add_argument("--marker", type=float, default=987.0, help="Marker friction value used to tag a body's shapes.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

if args.usd:
    _usd_path = args.usd
else:
    from dexx.tasks.franka_sharpa.franka_sharpa_env_cfg import franka_sharpa_robot_usd
    _usd_path = franka_sharpa_robot_usd(args.side)
print(f"[calibrate] robot USD: {_usd_path}")

# FR3 rest pose so the articulation passes joint-limit validation on spawn.
_FR3_REST = {"fr3_joint1": 0.0, "fr3_joint2": -0.785, "fr3_joint3": 0.0,
             "fr3_joint4": -2.356, "fr3_joint5": 0.0, "fr3_joint6": 1.571, "fr3_joint7": 0.785}


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device="cuda:0"))
    cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=_usd_path),
        init_state=ArticulationCfg.InitialStateCfg(joint_pos=_FR3_REST),
        actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=None, damping=None)},
    )
    robot = Articulation(cfg)
    sim.reset()

    art_view = robot.root_physx_view
    n_shapes = art_view.get_material_properties().shape[1]
    sim_view = sim.physics_sim_view
    side = args.side
    elastomer_bodies = [f"{side}_{finger}_elastomer" for finger in ("thumb", "index", "middle", "ring", "pinky")]

    per_body = {}
    for body in elastomer_bodies:
        try:
            bview = sim_view.create_rigid_body_view(f"/World/Robot/{body}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [WARN] {body}: could not create rigid-body view: {exc}")
            continue
        base = bview.get_material_properties().clone()
        marked = base.clone()
        marked[..., 0] = args.marker
        bview.set_material_properties(marked, torch.tensor([0], device="cpu"))
        glob = art_view.get_material_properties()
        idxs = (glob[0, :, 0] == args.marker).nonzero().flatten().tolist()
        per_body[body] = idxs
        bview.set_material_properties(base, torch.tensor([0], device="cpu"))  # restore
        print(f"  {body:26s} -> shapes {idxs}")

    all_ids = sorted(i for v in per_body.values() for i in v)
    print("\n" + "=" * 56)
    print(f"n_shapes = {n_shapes}   side = {side}")
    print(f"material_elastomer_ids = {all_ids}")
    print("=" * 56)
    print("Set this via --material_elastomer_ids or env_cfg.material_elastomer_ids.")
    simulation_app.close()


if __name__ == "__main__":
    main()
