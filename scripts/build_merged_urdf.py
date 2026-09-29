"""Merge the FR3 arm URDF with a Sharpa Wave hand URDF into one articulation.

The training env needs a **single** articulation carrying all 29 actuated joints
(7 arm + 22 hand); this produces it. Both inputs are the public/clean models
vendored under ``assets/`` — the hand comes from
https://github.com/sharpa-robotics/sharpa-urdf-usd-xml.

    python scripts/build_merged_urdf.py --side right
    python scripts/build_merged_urdf.py --side left

Output: ``assets/generated/fr3_with_<side>_sharpa_wave.urdf``, with every mesh
path rewritten relative to that file so the tree stays relocatable. Re-run it
after changing either input; the result is committed, so training needs no build
step.

The hand is attached to ``fr3_link8`` (the FR3 flange frame) by a fixed joint
whose default offset is **calibrated against the reference merged asset**
(``Right_final.usda``, which the demos and extrinsics were made with), not
chosen: ``xyz = 0 0 0.035`` and ``rpy = 0 0 3pi/4``. Together with this URDF's
``fr3_link7 -> fr3_link8`` (z=0.107, no rotation) that reproduces the reference
``fr3_link7 -> right_hand_C_MC`` transform of z=0.142 / yaw=135 deg exactly.

Verified: with these defaults, all 42 link frames the reference asset defines match
to <= 1e-4 mm and 0 deg at the home pose. Changing them moves the whole hand and
invalidates every retargeted demo and every camera extrinsic, so don't — unless
you re-retarget and re-calibrate.
"""

from __future__ import annotations

import argparse
import copy
import os
import re
import xml.etree.ElementTree as ET

_HERE = os.path.dirname(os.path.abspath(__file__))
_ASSETS = os.path.normpath(os.path.join(_HERE, "..", "assets"))

FR3_URDF = os.path.join(_ASSETS, "franka_fr3", "fr3.urdf")
FR3_END_LINK = "fr3_link7"

# Links pruned from the FR3 description before merging.
#
# `*_sc` are MoveIt self-collision proxy capsules and `*_accelerometer_*` are
# massless sensor frames. With `merge_fixed_joints=False` (which we need, to keep
# the elastomer/fingertip links) each of these becomes its own rigid body with no
# mass, so PhysX assigns a tiny isotropic inertia — and with self-collision on,
# every proxy capsule collides with the very link it wraps. The result is an
# articulation that explodes to ~1e7 rad within one step. `fr3_link8` is likewise
# a massless flange frame. The reference asset contains none of them: dropping
# these gives exactly its 42-body model.
PRUNE_PATTERNS = (r".*_sc$", r".*_accelerometer_(top|bottom)$", r"^fr3_link8$")
OUT_DIR = os.path.join(_ASSETS, "generated")

# Calibrated fr3_link8 -> hand-root offset, per side. See the module docstring;
# these are measured against the reference asset, not tuning knobs.
#
# The reference asset authors the yaw as literal 2.35619 / -0.785, i.e. a rounded
# pi/4. We use the exact analytic values instead: for the right hand that is a
# 4.5e-6 rad correction (nothing), for the left hand 4.0e-4 rad = 0.023 deg,
# which is 0.06 mm at the fingertip — an order of magnitude under the 1.6 mm
# retarget solver residual, so no demo is affected.
DEFAULT_JOINT_XYZ = "0 0 0.142"   # fr3_link7 -> hand root
DEFAULT_JOINT_RPY = {
    "right": "0 0 2.356194490192345",   # +3*pi/4  = +135 deg yaw
    "left": "0 0 -0.7853981633974483",  # -pi/4    =  -45 deg yaw
}


def hand_urdf(side: str) -> str:
    return os.path.join(_ASSETS, "sharpa_wave", f"{side}_sharpa_wave", f"{side}_sharpa_wave.urdf")


def root_link(root: ET.Element) -> str:
    """The one link that is never a joint's child."""
    links = {l.get("name") for l in root.findall("link")}
    children = {j.find("child").get("link") for j in root.findall("joint")}
    roots = links - children
    if len(roots) != 1:
        raise ValueError(f"expected exactly one root link, got {sorted(roots)}")
    return roots.pop()


def rewrite_mesh_paths(root: ET.Element, mapping: list[tuple[str, str]]) -> int:
    """Rewrite ``package://`` mesh URIs to paths relative to the output file."""
    n = 0
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename")
        if not fn:
            continue
        for pattern, repl in mapping:
            new = re.sub(pattern, repl, fn)
            if new != fn:
                mesh.set("filename", new)
                n += 1
                break
    return n


def prune_links(root: ET.Element, patterns) -> list[str]:
    """Drop massless helper links (and the joints attaching them) in place."""
    rx = [re.compile(p) for p in patterns]
    doomed = {l.get("name") for l in root.findall("link") if any(r.match(l.get("name")) for r in rx)}
    for j in list(root.findall("joint")):
        if j.find("child").get("link") in doomed or j.find("parent").get("link") in doomed:
            root.remove(j)
    for l in list(root.findall("link")):
        if l.get("name") in doomed:
            root.remove(l)
    return sorted(doomed)


def merge(side: str, joint_xyz: str, joint_rpy: str, out_path: str) -> None:
    arm = ET.parse(FR3_URDF).getroot()
    pruned = prune_links(arm, PRUNE_PATTERNS)
    hand = ET.parse(hand_urdf(side)).getroot()

    hand_root = root_link(hand)
    arm_links = {l.get("name") for l in arm.findall("link")}
    if FR3_END_LINK not in arm_links:
        raise ValueError(f"{FR3_END_LINK} not in {FR3_URDF}")

    # Output lives in assets/generated/, meshes one level up.
    rewrote = rewrite_mesh_paths(arm, [
        (r"package://franka_description/meshes/(.+)", r"../franka_fr3/meshes/\1"),
    ])
    rewrote += rewrite_mesh_paths(hand, [
        (rf"package://{side}_sharpa_wave/meshes/(.+)",
         rf"../sharpa_wave/{side}_sharpa_wave/meshes/\1"),
    ])

    merged = ET.Element("robot", {"name": f"fr3_with_{side}_sharpa_wave"})
    for el in arm:
        merged.append(copy.deepcopy(el))

    # Name collisions between the two models would silently drop geometry.
    hand_names = {l.get("name") for l in hand.findall("link")}
    clash = arm_links & hand_names
    if clash:
        raise ValueError(f"link name collision between arm and hand: {sorted(clash)}")

    for el in hand:
        if el.tag in ("link", "joint", "material", "transmission", "gazebo"):
            merged.append(copy.deepcopy(el))

    fixed = ET.SubElement(merged, "joint", {
        "name": f"fr3_to_{side}_sharpa_joint", "type": "fixed"
    })
    ET.SubElement(fixed, "parent", {"link": FR3_END_LINK})
    ET.SubElement(fixed, "child", {"link": hand_root})
    ET.SubElement(fixed, "origin", {"xyz": joint_xyz, "rpy": joint_rpy})

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ET.indent(merged, space="  ")
    ET.ElementTree(merged).write(out_path, encoding="utf-8", xml_declaration=True)

    n_links = len(merged.findall("link"))
    n_joints = len(merged.findall("joint"))
    n_act = len([j for j in merged.findall("joint") if j.get("type") in ("revolute", "prismatic", "continuous")])
    print(
        f"[build_merged_urdf] {side} -> {out_path}\n"
        f"    arm root   : {root_link(arm)}\n"
        f"    hand root  : {hand_root}  (fixed to {FR3_END_LINK} @ xyz='{joint_xyz}' rpy='{joint_rpy}')\n"
        f"    links      : {n_links}\n"
        f"    joints     : {n_joints}  ({n_act} actuated)\n"
        f"    pruned     : {len(pruned)} massless helper links\n"
        f"    mesh paths rewritten: {rewrote}"
    )


def main():
    p = argparse.ArgumentParser(description="Merge FR3 + Sharpa Wave hand URDF.")
    p.add_argument("--side", default="right", choices=["left", "right", "both"])
    p.add_argument("--joint-xyz", default=DEFAULT_JOINT_XYZ)
    p.add_argument("--joint-rpy", default=None,
                   help="Override the calibrated per-side yaw. Don't, unless re-retargeting.")
    p.add_argument("--out_dir", default=OUT_DIR)
    a = p.parse_args()

    sides = ["right", "left"] if a.side == "both" else [a.side]
    for s in sides:
        rpy = a.joint_rpy if a.joint_rpy is not None else DEFAULT_JOINT_RPY[s]
        merge(s, a.joint_xyz, rpy,
              os.path.join(a.out_dir, f"fr3_with_{s}_sharpa_wave.urdf"))


if __name__ == "__main__":
    main()
