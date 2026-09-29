#!/usr/bin/env bash
# Install the documented Dex-X baseline into a uv virtual environment.
#
#   bash tutorial/00_setup/setup_uv.sh --isaaclab /path/to/IsaacLab [--venv .venv] --accept-eula
#   bash tutorial/00_setup/setup_uv.sh --isaaclab /path/to/IsaacLab --isaacsim /path/to/isaac-sim ...
#   bash tutorial/00_setup/setup_uv.sh --venv /path/to/venv --verify-only
#
# Same baseline as setup_env.sh (Isaac Lab v2.2.1, Python 3.10, Torch
# 2.7.0/cu128, matching PyTorch3D) in a uv virtual environment instead of conda.
# Isaac Sim 4.5 comes from one of two places:
#   pip     (default) the isaacsim wheels from pypi.nvidia.com; needs glibc >= 2.34
#   binary  --isaacsim, or an existing _isaac_sim link in the Lab checkout; the
#           venv's activate script then sources _isaac_sim/setup_conda_env.sh,
#           as Lab's conda environments do. Works on older glibc (Ubuntu 20.04).
#
# --accept-eula records OMNI_KIT_ACCEPT_EULA=YES in the venv's activate script;
# the Isaac Sim wheels otherwise prompt for the NVIDIA EULA on first import.
# --verify-only skips every install step and runs the checks against the venv.
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO=$PWD

VENV=${VIRTUAL_ENV:-$REPO/.venv}
ISAACLAB=${ISAACLAB_PATH:-}
ISAACSIM=${ISAACSIM_PATH:-}
VERIFY_ONLY=0
ACCEPT_EULA=0
SIM_MODE=
PYTORCH3D_BASE=${PYTORCH3D_BASE:-0.7.8+5043d15}
PYTORCH3D_SRC=https://github.com/facebookresearch/pytorch3d.git@5043d15361d16a7093b4b60572c5f730c6c83308
PY_VER=3.10
SIM_VER=4.5.0
LAB_SHA=0f00ca2b4b2d54d5f90006a92abb1b00a72b2f20
# Override any index with a mirror serving the same wheels, e.g.
#   TORCH_INDEX_URL=https://mirror.nju.edu.cn/pytorch/whl/cu128
#   UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple
TORCH_INDEX=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}
NVIDIA_INDEX=${NVIDIA_INDEX_URL:-https://pypi.nvidia.com}
PYTORCH3D_INDEX=${PYTORCH3D_INDEX_URL:-https://miropsota.github.io/torch_packages_builder}
INSTALL_CONSTRAINTS=
trap '[[ -z "$INSTALL_CONSTRAINTS" ]] || rm -f "$INSTALL_CONSTRAINTS"' EXIT

while [ $# -gt 0 ]; do
  case "$1" in
    --venv)        VENV=$2; shift 2 ;;
    --isaaclab)    ISAACLAB=$2; shift 2 ;;
    --isaacsim)    ISAACSIM=$2; shift 2 ;;
    --accept-eula) ACCEPT_EULA=1; shift ;;
    --verify-only) VERIFY_ONLY=1; shift ;;
    -h|--help)     sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1"; exit 1 ;;
  esac
done

step () { echo; echo "=== $*"; }
die  () { echo "ERROR: $*" >&2; exit 1; }

step "0/6  prerequisites"
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die "This installer targets Linux x86_64. See MANUAL_SETUP.md."
command -v uv >/dev/null || die "uv not found. Install it: https://docs.astral.sh/uv/getting-started/installation/"
echo "  uv: $(uv --version)"
GLIBC=$(ldd --version 2>/dev/null | head -n 1 | grep -oE '[0-9]+\.[0-9]+$' || true)
[ -n "$GLIBC" ] || die "Could not determine the glibc version."
echo "  glibc: $GLIBC"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/  GPU: /'
else
  echo "  WARNING: nvidia-smi not found. Installation can proceed, but Isaac Sim,"
  echo "           the runtime check and training need an NVIDIA GPU with Vulkan."
fi
if [ "$VERIFY_ONLY" -eq 0 ]; then
  [ -n "$ISAACLAB" ] || die "--isaaclab is required (or set ISAACLAB_PATH).
       Clone Isaac Lab v2.2.1 first; see tutorial/00_setup/MANUAL_SETUP.md step 2."
  [ -d "$ISAACLAB/source/isaaclab" ] || die "no source/isaaclab under $ISAACLAB"
  ISAACLAB=$(cd "$ISAACLAB" && pwd)
  [ "$(git -C "$ISAACLAB" rev-parse HEAD)" = "$LAB_SHA" ] || die "Isaac Lab v2.2.1 ($LAB_SHA) is required. See MANUAL_SETUP.md."
  git -C "$ISAACLAB" diff --quiet HEAD -- source isaaclab.sh || die "Isaac Lab has local source changes; use a clean baseline checkout."
  command -v cmake >/dev/null || die "Install cmake and build tools before installing Isaac Lab."
  echo "  IsaacLab: $ISAACLAB"
  if [ -n "$ISAACSIM" ]; then
    [ -d "$ISAACSIM" ] || die "--isaacsim path does not exist: $ISAACSIM"
    if [ -e "$ISAACLAB/_isaac_sim" ]; then
      [ "$ISAACLAB/_isaac_sim" -ef "$ISAACSIM" ] || die "_isaac_sim differs from --isaacsim; resolve the link before installing."
    else
      ln -s "$(cd "$ISAACSIM" && pwd)" "$ISAACLAB/_isaac_sim"
      echo "  _isaac_sim: linked -> $ISAACSIM"
    fi
  fi
  if [ -e "$ISAACLAB/_isaac_sim" ]; then
    SIM_MODE=binary
    [ -f "$ISAACLAB/_isaac_sim/setup_conda_env.sh" ] || die "$ISAACLAB/_isaac_sim is not a binary Isaac Sim (no setup_conda_env.sh)."
    SIM_VERSION=$(head -n 1 "$ISAACLAB/_isaac_sim/VERSION")
    [[ "$SIM_VERSION" == 4.5.* ]] || die "Expected Isaac Sim 4.5, found $SIM_VERSION."
    echo "  Isaac Sim: binary $SIM_VERSION"
  else
    SIM_MODE=pip
    [ "$(printf '%s\n2.34\n' "$GLIBC" | sort -V | head -n 1)" = 2.34 ] || die "The Isaac Sim $SIM_VER wheels need glibc >= 2.34 (found $GLIBC).
       Download the binary Isaac Sim 4.5 and pass --isaacsim /path/to/isaac-sim."
    echo "  Isaac Sim: pip wheels $SIM_VER"
  fi
fi

step "1/6  uv environment $VENV (python $PY_VER)"
if [ -x "$VENV/bin/python" ]; then
  echo "  exists — reusing it"
elif [ "$VERIFY_ONLY" -eq 1 ]; then
  die "no virtual environment at $VENV (--verify-only creates nothing)"
else
  uv venv --python "$PY_VER" --seed "$VENV"
fi
VENV=$(cd "$VENV" && pwd)
# shellcheck disable=SC1091
set +u; source "$VENV/bin/activate"; set -u
echo "  python: $(python --version)"
python -c 'import sys; assert sys.version_info[:2] == (3, 10), "This baseline requires Python 3.10"'
export OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA:-}
if [ "$ACCEPT_EULA" -eq 1 ]; then
  grep -q '^export OMNI_KIT_ACCEPT_EULA=YES' "$VENV/bin/activate" \
    || echo 'export OMNI_KIT_ACCEPT_EULA=YES' >> "$VENV/bin/activate"
  export OMNI_KIT_ACCEPT_EULA=YES
fi
if [ "$VERIFY_ONLY" -eq 0 ] && [ "$SIM_MODE" = binary ]; then
  # What `isaaclab.sh -c` writes into a conda env's activate.d, for this venv.
  sed -i '/^# >>> dexx isaac sim >>>$/,/^# <<< dexx isaac sim <<<$/d' "$VENV/bin/activate"
  cat >> "$VENV/bin/activate" <<ACT
# >>> dexx isaac sim >>>
export ISAACLAB_PATH="$ISAACLAB"
export RESOURCE_NAME=IsaacSim
source "$ISAACLAB/_isaac_sim/setup_conda_env.sh"
# Sim's PYTHONPATH carries its own Torch 2.5.1/Gymnasium 0.29 (ml_archive
# pip_prebundle), which would shadow the pinned baseline in this venv.
export PYTHONPATH="$VENV/lib/python$PY_VER/site-packages:\$PYTHONPATH"
# <<< dexx isaac sim <<<
ACT
  set +u; source "$VENV/bin/activate"; set -u
  # Kit also puts that prebundle on sys.path and imports Torch 2.5.1 from it
  # while starting, before any script runs; compiled extensions built for the
  # venv's Torch 2.7 then fail to load. The venv provides every package it
  # holds, so set it aside (restore: move the .disabled-by-dexx dir back).
  ML_PREBUNDLE=$(cd "$ISAACLAB/_isaac_sim" && pwd -P)/exts/omni.isaac.ml_archive/pip_prebundle
  if [ -n "$(ls -A "$ML_PREBUNDLE" 2>/dev/null)" ]; then
    [ ! -e "$ML_PREBUNDLE.disabled-by-dexx" ] || die "$ML_PREBUNDLE.disabled-by-dexx already exists; resolve it by hand."
    mv "$ML_PREBUNDLE" "$ML_PREBUNDLE.disabled-by-dexx"
    mkdir "$ML_PREBUNDLE"
    echo "  set aside Isaac Sim's bundled Torch 2.5.1: $ML_PREBUNDLE.disabled-by-dexx"
  fi
fi
INSTALL_CONSTRAINTS=$(mktemp)
cat "$REPO/constraints-sim45.txt" > "$INSTALL_CONSTRAINTS"
# Lab's flatdict is built from source and still imports pkg_resources.
export UV_CONSTRAINT="$INSTALL_CONSTRAINTS"
export UV_BUILD_CONSTRAINT="$INSTALL_CONSTRAINTS"
# The pinned Torch/PyTorch3D builds live on dedicated indexes; let every index
# compete so a same-named package on PyPI cannot shadow the CUDA build.
export UV_INDEX_STRATEGY=unsafe-best-match
UVPIP=(uv pip install --python "$VENV/bin/python")

if [ "$VERIFY_ONLY" -eq 0 ]; then

step "2/6  Isaac Sim $SIM_VER ($SIM_MODE) + Torch 2.7.0/cu128"
SIM_PKGS=()
[ "$SIM_MODE" = pip ] && SIM_PKGS=("isaacsim[all,extscache]==$SIM_VER")
"${UVPIP[@]}" "${SIM_PKGS[@]}" torch==2.7.0 torchvision==0.22.0 \
  'setuptools<81' toml pip \
  --extra-index-url "$NVIDIA_INDEX" --extra-index-url "$TORCH_INDEX"

step "3/6  Isaac Lab v2.2.1 (editable, no optional RL frameworks)"
# Mirrors `./isaaclab.sh -i none`: every source/ extension, plus the
# rl/mimic packages without their framework extras.
LAB_PKGS=()
for pkg in "$ISAACLAB"/source/*/; do
  [ -f "$pkg/setup.py" ] && LAB_PKGS+=(-e "${pkg%/}")
done
"${UVPIP[@]}" "${LAB_PKGS[@]}" torch==2.7.0 torchvision==0.22.0 \
  --extra-index-url "$NVIDIA_INDEX" --extra-index-url "$TORCH_INDEX"
python tutorial/00_setup/check_versions.py --core-only
python - >> "$INSTALL_CONSTRAINTS" <<'PYPINS'
from importlib.metadata import PackageNotFoundError, version
for name in ("torch", "torchvision", "isaaclab", "isaacsim", "gymnasium", "numpy"):
    try:
        print(f"{name}=={version(name)}")
    except PackageNotFoundError:  # isaacsim is not a distribution with a binary Sim
        pass
PYPINS

step "4/6  pytorch3d, matched to the installed torch"
PYTORCH3D_WANT=$(python - "$PYTORCH3D_BASE" <<'PY'
import sys, torch
ver, _, cuda = torch.__version__.partition("+")
if not cuda.startswith("cu"):
    sys.exit(f"torch {torch.__version__} has no CUDA suffix; a CPU build cannot run this.")
print(f"{sys.argv[1]}pt{ver}{cuda}")
PY
)
if [ "$(printf '%s\n2.32\n' "$GLIBC" | sort -V | head -n 1)" = 2.32 ]; then
  echo "  pytorch3d -> $PYTORCH3D_WANT"
  "${UVPIP[@]}" "pytorch3d==$PYTORCH3D_WANT" --extra-index-url "$PYTORCH3D_INDEX" \
    --extra-index-url "$TORCH_INDEX" || die "pytorch3d==$PYTORCH3D_WANT is not available on the prebuilt index.
Browse $PYTORCH3D_INDEX for a build matching torch, then re-run with
  PYTORCH3D_BASE=<base> bash tutorial/00_setup/setup_uv.sh ...
See tutorial/00_setup/MANUAL_SETUP.md step 3."
elif (cd /tmp && python -c "import torch, pytorch3d._C") 2>/dev/null \
     && python -c "from importlib.metadata import distribution as d; import sys; sys.exit('${PYTORCH3D_SRC##*@}' not in (d('pytorch3d').read_text('direct_url.json') or ''))"; then
  echo "  pytorch3d: source build of ${PYTORCH3D_SRC##*@} already installed"
else
  # The prebuilt wheels link against glibc 2.32; build the same commit instead.
  CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
  [ -x "$CUDA_HOME/bin/nvcc" ] || die "glibc $GLIBC is too old for the prebuilt pytorch3d wheels, and
       no nvcc was found under CUDA_HOME=$CUDA_HOME to build it. Install the CUDA
       toolkit matching torch (e.g. apt cuda-nvcc-12-8 cuda-cudart-dev-12-8 cuda-cccl-12-8)."
  echo "  pytorch3d -> source $PYTORCH3D_SRC (glibc $GLIBC < 2.32; nvcc from $CUDA_HOME)"
  # Torch's headers need cuSPARSE/cuBLAS/... headers: take the ones its own
  # nvidia-* wheels installed, so only nvcc/cudart/cccl come from the toolkit.
  NV_INCLUDES=$(python -c "import glob, os, nvidia; print(os.pathsep.join(sorted(glob.glob(os.path.join(nvidia.__path__[0], '*', 'include')))))")
  # ninja + MAX_JOBS: without them torch's extension builder compiles one file
  # at a time (~20 min instead of ~2). Each nvcc/cc1plus job can take ~4 GB, so
  # bound the parallelism by the memory this container may actually use.
  "${UVPIP[@]}" ninja
  if [ -z "${MAX_JOBS:-}" ]; then
    MEM_KB=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes; do
      LIM=$(cat "$f" 2>/dev/null || true)
      if [[ "$LIM" =~ ^[0-9]+$ ]] && [ $((LIM / 1024)) -lt "$MEM_KB" ]; then MEM_KB=$((LIM / 1024)); fi
    done
    MAX_JOBS=$(( MEM_KB / (4 * 1024 * 1024) ))
    [ "$MAX_JOBS" -ge 1 ] || MAX_JOBS=1
    [ "$MAX_JOBS" -le "$(nproc)" ] || MAX_JOBS=$(nproc)
  fi
  echo "  building with MAX_JOBS=$MAX_JOBS"
  CUDA_HOME="$CUDA_HOME" FORCE_CUDA=1 PATH="$CUDA_HOME/bin:$PATH" CPATH="$NV_INCLUDES${CPATH:+:$CPATH}" \
    MAX_JOBS="$MAX_JOBS" \
    "${UVPIP[@]}" --no-build-isolation --reinstall-package pytorch3d "pytorch3d @ git+$PYTORCH3D_SRC"
fi
(cd /tmp && python -c "import torch, pytorch3d._C") || die "pytorch3d is installed but its compiled extension does not load."

step "5/6  Dex-X and pinned dependencies"
# Git dependencies build C++/CUDA extensions against the installed Torch.
"${UVPIP[@]}" --no-build-isolation -e . -r requirements.txt \
  --extra-index-url "$NVIDIA_INDEX" --extra-index-url "$TORCH_INDEX"
# A same-version wheel from another Git commit would otherwise be kept.
RAYCASTER_REQUIREMENT=$(sed -n '/^simple-raycaster @ /p' requirements.txt)
[ -n "$RAYCASTER_REQUIREMENT" ] || die "Missing pinned raycaster requirement"
"${UVPIP[@]}" --reinstall-package simple-raycaster --no-deps "$RAYCASTER_REQUIREMENT"
echo "  installed dexx (editable) + requirements.txt under baseline constraints"

else
  echo; echo "=== 2-5/6  install steps skipped (--verify-only)"
fi

step "6/6  verify"
ok=0
python tutorial/00_setup/check_install.py  || ok=1
python tutorial/00_setup/check_portable.py || ok=1
python tutorial/00_setup/check_imports.py  || ok=1
python tutorial/01_frames_and_constants/check_frames.py || ok=1
python tutorial/00_setup/check_versions.py || ok=1
# A binary Sim puts its own bundled packages on PYTHONPATH; check this env.
PYTHONPATH= python -m pip check || ok=1

echo
if [ "$ok" -eq 0 ]; then
  cat <<MSG
STATIC AND DEPENDENCY CHECKS PASSED. GPU runtime remains unverified.

  source $VENV/bin/activate
  bash tutorial/run_acceptance.sh      # headless runtime, training and evaluation

Only the acceptance test proves the environment is correct. Then start at
tutorial/README.md
MSG
else
  echo "Some checks failed — see the output above, and"
  echo "tutorial/00_setup/MANUAL_SETUP.md for the same steps done by hand."
  exit 1
fi
