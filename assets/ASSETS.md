# Asset provenance and licensing

The robot is a **Franka FR3 arm (7 DOF) + Sharpa Wave hand (22 DOF)**, 29 DOF in
total.

This directory holds **two public upstream models**, the **merged URDF** generated
from them, and the **robot USDs** built from that URDF. Read this before changing
any asset.

## Overview

| path | source | license | used by |
|---|---|---|---|
| `franka_fr3/` | Franka FR3 description (URDF + meshes) | Apache-2.0 (see `LICENSE`) | arm input to the merged URDF |
| `sharpa_wave/` | left/right-hand subset of `github.com/sharpa-robotics/sharpa-urdf-usd-xml` | Apache-2.0 (see `LICENSE.txt` / `NOTICE.txt`) | hand input to the merged URDF; retarget FK in `hand_imitation/envs/sharpa.py` |
| `generated/fr3_with_{right,left}_sharpa_wave.urdf` | **generated** by `scripts/build_merged_urdf.py` | MIT (this repo) | input to the robot USD build; `pytorch_kinematics` chain for retarget and deploy |
| `robot/fr3_with_{right,left}_sharpa_wave/` | **generated** by `scripts/build_robot_usd.py` (~26 MB each) | MIT (this repo) | **what the simulator spawns** |

The generated files are committed — the robot USDs included — so training needs
no build step. After changing any input, rebuild:

```bash
python scripts/build_merged_urdf.py --side both     # after changing an upstream model
python scripts/build_robot_usd.py  --side both      # after any URDF change — always
python scripts/check_asset_equivalence.py --reference_usd <previous robot USD> --side right  # optional gate
```

## Why the hand is the Wave

The Sharpa Wave has a public Apache-2.0 model repository, so the release can ship
it: 22 actuated joints, root link `{side}_hand_C_MC`, the joint and link names
every name-based lookup in the code uses.

The robot is built from a merged URDF rather than a hand-assembled USD so that
it is reproducible from the upstream models and checked by a regression gate.

## URDF and USD: who does what

- The **URDF** (40 KB per side, readable, diffable) is the **source**. Edit the robot there.
- The **USD** is the **built artifact that actually spawns**.

The USD is committed not to save a conversion, but because **the self-collision
filter pairs (next section) cannot be expressed in URDF** and can only be written
onto the USD after conversion. If it were rebuilt at every run, any code path that
spawned the URDF directly would silently drop them. Baking the filters into the
committed artifact puts the semantics in the file.

The built USD records the sha256 of its source URDF in `.source_hash`.
`shipped_robot_usd()` compares it on every load and **warns loudly** (but
continues) on a mismatch, so assets cannot drift silently from their source.

> Without the built USD the code falls back to a runtime conversion (which also
> applies the filter pairs), so a checkout without USDs still runs — slower the
> first time, and dependent on the local Isaac Lab producing the same result. The
> fallback cache directory is fixed by `usd_cache_dir()` and can be overridden with
> `DEXX_USD_CACHE`. The cache is shared; warm it with a single env before
> launching parallel processes.

Size: upstream models 49 MB + robot USDs 50 MB ≈ 99 MB; the largest single file
is 26 MB, under GitHub's large-file limit.

## Self-collision filters: the semantics URDF loses

The original assets excluded **7 pairs** of self-collisions (palm `hand_C_MC` ↔
the index / middle / ring / pinky proximal links and the thumb metacarpal; pinky
metacarpal ↔ pinky proximal; thumb metacarpal ↔ thumb proximal), 14 directed
records. These links touch each other at rest.

**URDF cannot express collision filtering, and Isaac Lab's `UrdfConverterCfg` has
no field for it** (only a `self_collision` boolean). A converted asset therefore
loses them, and with `enabled_self_collisions=True` the palm jams against the
finger roots — **the fingers physically cannot close**.

The failure is **completely silent**: no NaN, no warning, no limit violation;
episodes survive normally and the object simply does not move, so the success
rate collapses while every other signal looks normal. Turning self-collision off
(a diagnostic only) lets the fingers close again; writing the filter pairs back
does the same with self-collision on, which is the fix.

The fix is in `franka_sharpa_env_cfg.py`: `SELF_COLLISION_FILTER_PAIRS` and
`franka_sharpa_robot_usd()`. The latter drives `UrdfConverter` explicitly (keeping
lazy conversion and the `.asset_hash` cache), writes the pairs in both directions
with `UsdPhysics.FilteredPairsAPI`, then spawns via `UsdFileCfg`.

⚠️ **The filters must be written into the USD file, not patched after spawn:**
`spawn_from_urdf` carries a `@clone` decorator, so cloning to every env has already
happened and patching the source prim afterwards does not propagate.

⚠️ After swapping the hand or renaming links, re-check that the 7 pairs are still
right. `_apply_self_collision_filters()` raises if it cannot find a prim; it never
skips silently.

> Asset equivalence **cannot be established statically**. Two assets can be
> identical in gains, armature, torque limits, joint limits, collision
> approximation, mass and inertia and still differ in the filter pairs alone —
> enough to stop the fingers closing. `check_asset_equivalence.py` verifies kinematics only —
> after an asset change, also run a success-rate regression against a known
> checkpoint (`scripts/eval_teacher.py`, [EVAL.md](../docs/EVAL.md)).

## Armature is load-bearing — do not remove it

URDF cannot express rotor inertia, so a URDF-imported hand has **armature = 0**.
Finger link inertias are around 1e-6 kg·m², and a 0.2 N·m torque then drives a
joint **straight through its PhysX limit** within one 1/120 s step; even with zero
actions, joints end up far outside their limits.

Gains, armature and friction are therefore injected explicitly into the actuator
cfgs from `src/dexx/robot_constants.py` — `HAND_GAINS`, `ARM_ARMATURE`,
`ARM_FRICTION` — **not inherited from the asset**. The hand values come from the
vendor USD, converted from its per-degree drive convention (×180/π).

The arm's stiffness and damping are `ARM_TUNED_KP` / `ARM_TUNED_KD` in the same
file: a step-response fit to the real arm, and the values the shipped checkpoints
were trained with ([tutorial/07](../tutorial/07_dynamics_alignment/)).

## Elastomer collision-shape ids

Friction randomisation gives the 5 fingertip elastomers a softer friction than the
rest of the hand. They are addressed by **shape index** into the PhysX material
array (`env_cfg.material_elastomer_ids`, consumed in `franka_sharpa_env.py`).

The ids **cannot be read off the USD by counting collision prims**: PhysX convex
decomposition makes the shape count differ from the CollisionAPI prim count, and
the USD carries no friction authoring that identifies the elastomers.
`tools/calibrate_elastomer_ids.py` probes instead: for each elastomer body it sets
a marker friction through that body's own PhysX rigid-body view and reads back
which rows of the global material array changed.

```bash
python tools/calibrate_elastomer_ids.py --side right --headless
# -> material_elastomer_ids = [...]
```

Pass a new list with `scripts/train_teacher.py --material_elastomer_ids '[...]'`
or set it in the cfg. ⚠️ **Re-calibrate after any asset change** — the ids are
relative to the current asset's shape layout. The committed build (merged URDF →
built USD) has 34 shapes (arm 8 + hand 26), and the default is
**`material_elastomer_ids = [27,28,30,32,33]`**.

`franka_sharpa_env.py` filters the list with `i < n_shapes`: **if every id is out
of range, friction DR silently does nothing** — no error, no warning. Do not
assume it is active until you have calibrated.

## Mount transform: a calibration, not a tuning knob

The hand attaches to `fr3_link7` with a fixed joint, `xyz = 0 0 0.142`, yaw
`+3π/4` (right) / `−π/4` (left). It is a calibration: change it and the whole
hand moves, invalidating **every retargeted demo and every camera extrinsic**.

`check_asset_equivalence.py` is the cheap regression gate (~0.2 s, no simulator):
it compares every link frame of the merged URDF with a reference USD and prints
`EQUIVALENT` or `MISMATCH`. The left-hand mount uses the exact yaw `-π/4`; a
reference asset that rounds it to `-0.785` differs by about 0.02° (well under a
tenth of a millimetre at the fingertip) and reports `MISMATCH` at the default
tolerance. That difference is intentional and affects no demo.

## Known limitations

- The tactile UV map (taxel → contact position on the finger pad) is calibrated
  per hand. No calibrated map ships for the Wave yet.
  Without one the deploy env degrades gracefully: contact **position** is zeroed,
  contact **force** is unaffected (`DEXX_TACTILE_MAP_DIR` points at a
  calibration). Current policies are trained with `enable_contact_pos=False` and
  do not read contact position, so they are unaffected.
- **Left-hand training is untested:** the shipped demos are right-hand only.

## Third-party licenses

Both upstream directories are Apache-2.0; their `LICENSE` / `LICENSE.txt` /
`NOTICE.txt` are kept verbatim and must not be removed. The project's own code is
MIT (see the root `LICENSE`); see
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for vendored code and data.
