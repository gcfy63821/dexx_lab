"""Build the shipped robot USD from the merged URDF.

Isaac Lab can convert the URDF at spawn time, but the conversion is not the whole
story: the hand's self-collision filter pairs cannot be expressed in URDF and are
authored onto the converted stage afterwards (see
`franka_sharpa_env_cfg.SELF_COLLISION_FILTER_PAIRS`). Reconstructing that on every
run means any code path that spawns the URDF directly silently loses it.

So we build the USD once, apply the filters, and commit the result. Runtime then
just loads an asset that already carries the right physics.

    python scripts/build_robot_usd.py --side both

Output: ``assets/robot/fr3_with_<side>_sharpa_wave/`` holding the Isaac Lab layer
layout (stub + ``configuration/{base,physics,sensor}.usd``) plus a
``.source_hash`` recording which URDF it came from. Re-run after
``scripts/build_merged_urdf.py``; `franka_sharpa_robot_usd()` compares the hash
and warns when the two have drifted.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, ".."))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--side", default="both", choices=["left", "right", "both"])
parser.add_argument("--out_dir", default=os.path.join(_REPO, "assets", "robot"),
                    help="Where the built USD trees are written.")

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

sys.path.insert(0, os.path.join(_REPO, "src"))

from dexx.tasks.franka_sharpa.franka_sharpa_env_cfg import (  # noqa: E402
    _apply_self_collision_filters,
    franka_sharpa_urdf,
    urdf_source_hash,
)
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402


def build(side: str, out_dir: str) -> str:
    urdf = franka_sharpa_urdf(side)
    dest = os.path.join(out_dir, f"fr3_with_{side}_sharpa_wave")

    # Convert into a scratch dir, then move into place, so a failed build never
    # leaves a half-written asset behind for the next run to load.
    staging = dest + ".building"
    shutil.rmtree(staging, ignore_errors=True)

    converter = UrdfConverter(
        UrdfConverterCfg(
            asset_path=urdf,
            usd_dir=staging,
            fix_base=True,
            # MUST stay False: merging deletes the `*_elastomer` (tactile sensor
            # prims) and `*_fingertip` (tracked bodies) links.
            merge_fixed_joints=False,
            collider_type="convex_hull",
            self_collision=True,
            # Zeroed so the actuator cfgs remain the single source of truth for
            # gains rather than the asset silently supplying them.
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                target_type="position",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
            ),
        )
    )
    n = _apply_self_collision_filters(converter.usd_path, side)

    # The converter's own cache markers describe the staging path; drop them so
    # nothing downstream mistakes this for a live conversion cache.
    for junk in (".asset_hash", "config.yaml"):
        p = os.path.join(staging, junk)
        if os.path.isfile(p):
            os.remove(p)
    with open(os.path.join(staging, ".source_hash"), "w") as f:
        f.write(urdf_source_hash(side))

    shutil.rmtree(dest, ignore_errors=True)
    os.replace(staging, dest)

    total = sum(os.path.getsize(os.path.join(r, f))
                for r, _, fs in os.walk(dest) for f in fs)
    print(f"[build_robot_usd] {side} -> {dest}\n"
          f"    from        : {os.path.relpath(urdf, _REPO)}\n"
          f"    filter pairs: {n} authored\n"
          f"    size        : {total / 1e6:.1f} MB")
    return dest


def main() -> None:
    sides = ["right", "left"] if args.side == "both" else [args.side]
    os.makedirs(args.out_dir, exist_ok=True)
    for s in sides:
        build(s, args.out_dir)
    simulation_app.close()


if __name__ == "__main__":
    main()
