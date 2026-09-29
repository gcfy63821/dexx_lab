# 00 — Setup, repo map, and a smoke test

**Goal:** get the environment installed and prove it by running something, not by
reading a version number.

## Install

```bash
bash tutorial/00_setup/setup_env.sh \
    --isaacsim /path/to/isaac-sim \
    --isaaclab /path/to/IsaacLab
```

The installer targets **Isaac Sim 4.5 + Isaac Lab v2.2.1 + Python 3.10**.
First prepare the pinned Lab checkout and system prerequisites in
**[MANUAL_SETUP.md](MANUAL_SETUP.md)**. Lab installs Torch 2.7/cu128, and the
script matches PyTorch3D to its exact build. Project dependencies are pinned
where API compatibility requires it and installed under shared constraints.

**uv alternative** — no conda; Isaac Sim comes from NVIDIA's pip wheels
(needs glibc >= 2.34), or from a binary download with `--isaacsim`:

```bash
bash tutorial/00_setup/setup_uv.sh --isaaclab /path/to/IsaacLab --venv .venv --accept-eula
#   older glibc (e.g. Ubuntu 20.04): add --isaacsim /path/to/isaac-sim
source .venv/bin/activate
```

`--accept-eula` accepts the NVIDIA Omniverse EULA in the venv's activate
script; without it the Sim wheels prompt on first import. Behind a slow
PyPI, point uv at a mirror with `UV_DEFAULT_INDEX=<mirror>/simple`.
See [MANUAL_SETUP.md](MANUAL_SETUP.md#uv-route).

```bash
--name myenv                  # another dedicated conda environment name
--verify-only                 # check an existing env; install nothing
PYTORCH3D_BASE=0.7.8+5043d15   # override the PyTorch3D source/build prefix
```

The reference combination is constrained, not a guarantee of runtime success
on every GPU/driver. `requirements-full.txt` is a compatibility alias, not a
frozen environment. Paths are chosen by the user; no cluster-specific setup is
required. Headless still needs the simulator's system and graphics libraries.

## Check it

```bash
python tutorial/00_setup/check_install.py
```

This checks package files, assets, demonstrations, camera calibrations and the
pretrained teacher. It does not import the simulator task registry. It does **not** start Isaac Sim, so it takes a second
and tells you about missing files before a twenty-minute run does.

## Check the code's dependencies

```bash
python tutorial/00_setup/check_imports.py
```

Checks simulation imports and critical compiled modules. Real import failures
are errors; explicit Isaac startup deferrals remain unverified until the runtime
check. For real-robot transport tools, install `requirements-deploy.txt` and use
`--include-deploy`. Hardware SDKs are installed on their respective hosts.

```bash
python tutorial/00_setup/check_versions.py
python -m pip check
bash tutorial/run_acceptance.sh
```

Acceptance starts a headless API/CUDA preflight, then training and evaluation,
and requires all four demo IDs with ten episodes each. Static checks alone do
not prove the environment is ready. See the manual for the isolated preflight
command and how to verify its success marker.

## Check it can move

```bash
python tutorial/00_setup/check_portable.py
```

Answers a different question: if you copied this directory to another machine,
would it still work? It looks for symlinks that leave the repository, absolute
paths baked into code or config, and demonstrations whose source data or meshes
are not actually committed.

Run it after adding a demonstration or an asset. A repository can pass every
other check and still work only on the machine it was built on; nothing fails
until someone else tries to use it.

## What runs where

Three machines are involved once you reach hardware. In simulation, only the
first exists.

| | runs | lessons |
|---|---|---|
| **workstation** | training, evaluation, the policy at deploy time | 00–08 |
| **NUC** | Polymetis server + the ZMQ joint bridge, wired to the Franka | 09 |
| **camera host** | RealSense depth publisher (ZMQ) | 06, 09 |

## The shape of the pipeline

```
human demo (MANO)
   │  lesson 03 — retarget
   ▼
robot reference trajectory  ────────┐
   │  lesson 04 — PPO                │ the reference is an input to
   ▼                                 │ every stage after this one
state expert  (sees privileged state)│
   │  lesson 05 — DAgger             │
   ▼                                 │
point-cloud student  ◄───────────────┘
   │  lesson 08 — evaluate
   │  lessons 06, 07 — calibrate the camera, align the dynamics
   ▼
real robot  (lesson 09)
```

The expert is never deployed. It exists to label the student, because it is
allowed to see things a camera cannot — object velocity, contact state, the
randomized physical parameters. Lesson 04 makes that split concrete.

## Check

`check_install.py` prints `ALL CHECKS PASSED`. If it does not, it names the file
it wanted and the lesson that explains what that file is for.
