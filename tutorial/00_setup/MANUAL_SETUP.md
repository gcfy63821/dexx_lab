# Manual environment setup

The installer targets Linux x86_64, Isaac Sim 4.5 and Isaac Lab v2.2.1.
Paths below are placeholders chosen by you; no shared filesystem, cluster,
proxy or particular GPU model is required. An NVIDIA GPU supported by Isaac Sim
and the selected CUDA build is required.

## Compatibility baseline

| Component | Baseline |
|---|---|
| Python | 3.10 |
| Isaac Sim | 4.5.0 binary distribution |
| Isaac Lab | v2.2.1, commit `0f00ca2b4b2d54d5f90006a92abb1b00a72b2f20` (package 0.45.9) |
| Torch / torchvision | 2.7.0 / 0.22.0; the official Lab installer selects cu128 |
| Gymnasium / NumPy | 1.2.0 / 1.26.4 |
| PyTorch3D | 0.7.8+5043d15 built for the exact Torch/CUDA pair |
| simple-raycaster | commit `7bab59c56e9a340f20b7af29e4b769108cb697fd` |

This is a constrained installation baseline, **not a claim that every target
machine has passed training**. Run the headless acceptance test below.
`requirements-full.txt` is a compatibility alias to the curated requirements,
not a lock file.

The [official Lab compatibility table](https://github.com/isaac-sim/IsaacLab/tree/v2.2.1)
supports Sim 4.5. This Lab version supplies the contact-point and articulation
APIs the tasks need. Older Lab versions cannot be substituted merely because
`import isaaclab` succeeds.

## 1. System prerequisites

Install Git, Conda, a compiler toolchain and CMake. Use the
[Isaac Sim 4.5 system requirements](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/requirements.html)
and compatibility checker to select a supported driver/GPU. The driver must
also support the CUDA build chosen for Torch; see
[NVIDIA CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/).

On Ubuntu 22.04, typical system prerequisites include:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git libsm6 libxt6 libxrender1 libxi6 libgl1 libvulkan1 vulkan-tools
```

These packages do not install the NVIDIA graphics driver. In a container,
configure NVIDIA Container Toolkit to expose compute **and graphics** libraries
and the Vulkan ICD. `nvidia-smi` or a successful Torch CUDA operation alone does
not establish that Vulkan works; check `vulkaninfo --summary` and the simulator
logs as well. `--headless` removes the window, not these dependencies.

Download the Isaac Sim 4.5 binary distribution and unpack it. Set
`ISAACSIM_PATH` to that directory. The automatic installer uses this binary
route. A pip-based Sim install additionally requires its supported glibc
version (at least 2.34 for the 4.5 wheels); changing Python alone does not fix
an incompatible glibc.

## 2. Pin Isaac Lab and create the environment

From your chosen workspace:

```bash
git clone --branch v2.2.1 --depth 1 https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git rev-parse HEAD  # must equal the commit in the table
ln -s "$ISAACSIM_PATH" _isaac_sim
cd ../dexx_release  # this repository
```

For the automatic path:

```bash
bash tutorial/00_setup/setup_env.sh --isaaclab ../IsaacLab --isaacsim "$ISAACSIM_PATH" --name dexx
```

Or run the equivalent installation manually, starting in this repository:

```bash
export DEXX_REPO="$PWD"
export ISAACLAB_PATH="$(cd ../IsaacLab && pwd)"
export PIP_CONSTRAINT="$DEXX_REPO/constraints-sim45.txt"
export PIP_BUILD_CONSTRAINT="$PIP_CONSTRAINT"
conda create -y -n dexx python=3.10
conda activate dexx
python -m pip install 'setuptools<81' toml
cd "$ISAACLAB_PATH"
# The upstream installer calls `tabs`; noninteractive shells may use TERM=dumb.
if [ -z "${TERM:-}" ] || [ "$TERM" = dumb ]; then export TERM=xterm; fi
./isaaclab.sh -c dexx
./isaaclab.sh -i none
conda activate dexx
cd "$DEXX_REPO"
```

`-i none` omits optional RL frameworks. The official Lab installer can replace
an existing Torch build with 2.7/cu128, so use a dedicated environment. The
constraints apply during Lab installation as well as project installation;
`pin==2.7.0` avoids newer Pinocchio wheels pulling NumPy 2. The setuptools cap
keeps the `pkg_resources` build interface required by Lab's flatdict dependency;
`PIP_BUILD_CONSTRAINT` applies it to isolated builds on newer pip releases.

The automatic path checks the Lab checkout, imported source path and versions
before reusing an existing environment. It refuses conflicting Sim links.
For a preconfigured CUDA variant, keep Torch/torchvision paired and rebuild or
select matching PyTorch3D; that variant needs its own runtime acceptance. Do
not run Lab's installer expecting it to preserve a different CUDA build.

## 3. Install matching PyTorch3D and the project

For the default 2.7/cu128 build:

```bash
python -c "import torch; print(torch.__version__)"
python -m pip install --extra-index-url https://miropsota.github.io/torch_packages_builder \
    pytorch3d==0.7.8+5043d15pt2.7.0cu128
python -m pip install -e . -r requirements.txt
# Same package version can hide a different Git revision; replace its code.
python -m pip install --force-reinstall --no-deps \
    "$(sed -n '/^simple-raycaster @ /p' requirements.txt)"
python -m pip check
```

The PyTorch3D index is a third-party build service. If its matching wheel is
unavailable for your Python/platform, build the pinned source using a matching
Torch/CUDA toolkit; do not substitute an incompatible binary. PyTorch3D is
required by the dataset loader, including its compiled operators.

The automatic installer additionally constrains the exact installed Torch
CUDA build during later pip operations. The Git dependencies are pinned
in `requirements.txt` and their dependencies are resolved with ordinary pip.
The raycaster code is then explicitly reinstalled: pip otherwise may retain a
different Git commit with the same package version, even with `--upgrade`.
This one targeted `--no-deps` reinstall follows dependency resolution and is
checked by `pip check` and Git provenance verification. It is not a way to
hide dependency conflicts.

The pinned raycaster supports dynamic per-environment mesh subsets used by
multi-object training. Do not replace it with an arbitrary 0.2.0/HEAD: the same
version label can have different fused-call shape and return contracts.

Real-robot/camera transport dependencies are separate:

```bash
python -m pip install -r requirements-deploy.txt
python tutorial/00_setup/check_imports.py --include-deploy
```

Polymetis, the RealSense SDK and the Sharpa hand SDK remain specific to their
respective hosts. The shipped training scripts use the project's own PPO/DAgger
implementations.

## uv route

`setup_uv.sh` installs the same baseline into a uv virtual environment instead
of conda, and installs every Isaac Lab `source/` package editable, like
`./isaaclab.sh -i none`. Isaac Sim comes from one of two places:

- **pip (default):** the `isaacsim[all,extscache]==4.5.0` wheels from
  `pypi.nvidia.com`. Needs glibc >= 2.34; the Lab checkout must not contain an
  `_isaac_sim` link, and `check_versions.py` reads the Sim version from the wheel.
- **binary:** pass `--isaacsim /path/to/isaac-sim` (or keep an existing
  `_isaac_sim` link). The venv's `activate` then sources
  `_isaac_sim/setup_conda_env.sh`, exactly as Lab's conda environments do. Use
  this on older glibc such as Ubuntu 20.04 (2.31). Isaac Sim's
  `exts/omni.isaac.ml_archive/pip_prebundle` holds Torch 2.5.1/cu118, which Kit
  imports at startup ahead of the venv's Torch 2.7; the installer moves it to
  `pip_prebundle.disabled-by-dexx` (the venv provides every package in it).
  Move it back to restore the stock Sim. On glibc < 2.32 PyTorch3D is built
  from source at the pinned commit, which needs the CUDA 12.8 `nvcc`, `cudart`
  and `cccl` (e.g. apt `cuda-nvcc-12-8 cuda-cudart-dev-12-8 cuda-cccl-12-8`).

```bash
git clone --branch v2.2.1 --depth 1 https://github.com/isaac-sim/IsaacLab.git ../IsaacLab
bash tutorial/00_setup/setup_uv.sh --isaaclab ../IsaacLab --venv .venv --accept-eula
# glibc < 2.34: add --isaacsim "$ISAACSIM_PATH" to use the binary Sim instead
source .venv/bin/activate
bash tutorial/run_acceptance.sh
```

The constraints file is passed as `UV_CONSTRAINT` and `UV_BUILD_CONSTRAINT`,
and `UV_INDEX_STRATEGY=unsafe-best-match` keeps the PyTorch/PyTorch3D indexes
authoritative for their CUDA builds. uv's own variables still apply, e.g.
`UV_DEFAULT_INDEX` for a PyPI mirror or `UV_CACHE_DIR` for a shared cache.

## 4. Verify in stages

```bash
python tutorial/00_setup/check_versions.py
python -m pip check
python tutorial/00_setup/check_install.py
python tutorial/00_setup/check_portable.py
python tutorial/00_setup/check_imports.py
python tutorial/01_frames_and_constants/check_frames.py
```

These check versions, files and imports. The import checker reports explicit
Isaac runtime deferrals as **unverified**, and fails real dependency/ABI errors.
It does not prove simulator API or GPU readiness.

For an isolated headless API/CUDA check, choose a new result filename:

```bash
python -u tutorial/00_setup/check_runtime.py --headless --result logs/runtime-check.json
```

Require the result JSON to contain `"ok": true`; simulator startup failures may
return exit code zero without completing Python code. This checks task config,
contact APIs, PyTorch3D CUDA KNN and moving per-environment raycaster subsets.
It does not validate a full robot physics rollout or RTX camera.

Then run the complete acceptance test (it includes runtime preflight):

```bash
bash tutorial/run_acceptance.sh
```

It uses a fresh output directory, trains a student, waits for the final summary
and requires exactly four demonstration IDs with ten episodes each. Each stage
must also exit successfully: a shutdown timeout fails acceptance even if an
artifact exists. A failed preflight stops the run. Keep the output logs and
installed versions when reporting results. Runtime depends on the target hardware and first-run caches.
For videos/RTX cameras, additionally validate `--enable_cameras` on that target.

## Troubleshooting

- **Dependency conflict:** use the pinned Lab checkout and constraints from the
  start, in a dedicated environment.
- **Missing contact-point API:** check the imported Lab source and commit;
  removing contact fields changes the observations and is not a compatibility fix.
- **PyTorch3D import/CUDA error:** match Torch, CUDA, Python and the compiled wheel.
- **Vulkan/graphics failure with headless:** inspect the driver, loader and ICD
  inside the actual runtime/container; CUDA visibility is insufficient.
- **Conda activation fails under `set -u`:** Isaac Sim activation hooks may read
  unset shell variables. The installer temporarily relaxes strict shell flags
  and checks activation status explicitly.
