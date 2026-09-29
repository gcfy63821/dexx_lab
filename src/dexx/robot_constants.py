"""Actuation constants for the FR3 + Sharpa Wave robot.

Deliberately free of ``isaaclab`` imports, so offline tooling can read these
without launching a simulator.

Geometry constants (arm base pose, table) live in :mod:`dexx.deploy_config`,
which is the single source shared with the retarget and deploy sides.
"""

from __future__ import annotations

# Per-joint hand PD gains (N*m/rad, N*m*s/rad), rotor inertia and joint friction.
#
# Read out of the vendor's shipped USD and converted from USD's per-DEGREE drive
# convention (x 180/pi). The vendor's `Right_final.usda` and the public hand
# release carry identical values, so this is one calibration, not two.
#
# ``armature`` is the load-bearing entry. URDF cannot express rotor inertia, so a
# URDF-imported hand gets armature=0 -- and these finger links have inertias on
# the order of 1e-6 kg*m^2, so even a 0.2 N*m torque produces enough angular
# acceleration to tunnel straight through the PhysX joint limit within one
# 1/120 s step. Measured with armature=0: 14 of 29 joints outside their limits
# after 100 zero-action steps, one at 15 rad against a [0, 1.75] limit. With
# these values: 0 of 29.
#
# The robot asset is built from the merged URDF (see
# ``scripts/build_merged_urdf.py``), so these values are the ONLY place the
# gains come from. Do not set them to None.
HAND_GAINS = {                    # suffix -> (stiffness, damping, armature, friction)
    "thumb_CMC_FE": (6.955, 0.2410, 0.00320, 0.132000),
    "thumb_CMC_AA": (13.201, 0.4516, 0.00320, 0.132000),
    "thumb_MCP_FE": (4.760, 0.1830, 0.00265, 0.104000),
    "thumb_MCP_AA": (6.622, 0.2080, 0.00265, 0.104000),
    "thumb_IP": (0.908, 0.0400, 0.00061, 0.024760),
    "pinky_CMC": (1.380, 0.0392, 0.00012, 0.013000),
}
_FINGER_ROLE_GAINS = {            # the four 4-finger roles
    "MCP_FE": (4.760, 0.1830, 0.00265, 0.104000),
    "MCP_AA": (6.622, 0.2080, 0.00265, 0.104000),
    "PIP": (0.908, 0.0400, 0.00061, 0.024760),
    "DIP": (0.904, 0.0315, 0.00042, 0.000418),
}
for _f in ("index", "middle", "ring", "pinky"):
    for _role, _g in _FINGER_ROLE_GAINS.items():
        HAND_GAINS.setdefault(f"{_f}_{_role}", _g)

# Arm rotor inertia / joint friction, from `Right_final.usda`'s authored
# `physxJoint:armature` / `jointFriction` (uniform across the 7 FR3 joints).
ARM_ARMATURE = 1.0
ARM_FRICTION = 0.2

# ---------------------------------------------------------------------------
# Arm joint-impedance gains (sim PD) — the single source of truth. Used by the
# arm actuator in `FrankaSharpaEnvCfg` and re-applied by
# `FrankaSharpaCriticHorizonCfg.__post_init__`.
#
# Step-response tuned against the real arm; `tools/sysid/` replays a motion in
# both and reports the gap. EVERY SHIPPED CHECKPOINT WAS TRAINED AGAINST THESE,
# so changing them invalidates the checkpoints — it is a retrain, not a tweak.
#
# j1-j3 damping (145/135/110) comes from a ROS2 system-ID: higher damping left
# sim over-damped, lagging the real arm. On Polymetis, j4-j7 still lag the real
# arm; ARM_KD_POLYMETIS_IT2 below is a re-fit of that damping which reduces the
# lag in the tools/sysid replay, but training does NOT use it. Adopting it
# means retraining. Pass it to the replay tools with --arm_kd to compare.
ARM_TUNED_KP = {
    "fr3_joint1": 1600.0, "fr3_joint2": 1600.0, "fr3_joint3": 1200.0,
    "fr3_joint4": 800.0,  "fr3_joint5": 500.0,  "fr3_joint6": 300.0,
    "fr3_joint7": 150.0,
}
ARM_TUNED_KD = {
    "fr3_joint1": 145.0, "fr3_joint2": 135.0, "fr3_joint3": 110.0,
    "fr3_joint4": 100.0, "fr3_joint5": 50.0,  "fr3_joint6": 30.0,
    "fr3_joint7": 15.0,
}

# Candidate damping from the Polymetis realignment. Recorded, not active.
ARM_KD_POLYMETIS_IT2 = {
    "fr3_joint1": 85.0, "fr3_joint2": 135.0, "fr3_joint3": 110.0,
    "fr3_joint4": 25.0, "fr3_joint5": 18.0,  "fr3_joint6": 10.0,
    "fr3_joint7": 5.0,
}

# Joint effort/velocity limits are deliberately absent: the merged URDF's own
# values (hand 0.19-3.3 N*m, arm 87 N*m) are correct and match
# `Right_final.usda`'s `maxForce` exactly. Overriding them is what lets bad gains act.


def hand_gain_dicts(side: str) -> dict:
    """Side-prefixed {stiffness,damping,armature,friction} dicts for an actuator cfg."""
    return {
        "stiffness": {f"{side}_{k}": v[0] for k, v in HAND_GAINS.items()},
        "damping": {f"{side}_{k}": v[1] for k, v in HAND_GAINS.items()},
        "armature": {f"{side}_{k}": v[2] for k, v in HAND_GAINS.items()},
        "friction": {f"{side}_{k}": v[3] for k, v in HAND_GAINS.items()},
    }
