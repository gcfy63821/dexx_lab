#!/usr/bin/env bash
# Install the documented Dex-X baseline and run static/dependency checks.
#
#   bash tutorial/00_setup/setup_env.sh --isaaclab /path/to/IsaacLab [--name dexx]
#   bash tutorial/00_setup/setup_env.sh --isaacsim /path/to/isaac-sim --isaaclab /path/to/IsaacLab
#   bash tutorial/00_setup/setup_env.sh --name <existing> --verify-only
#
# Baseline: Isaac Lab v2.2.1, Isaac Sim 4.5, Python 3.10. Lab installs Torch
# 2.7/cu128; PyTorch3D must match. See MANUAL_SETUP.md for system requirements.
# Full headless acceptance is required before treating a machine as ready.
#
# --verify-only skips every install step and just runs the checks against an
# environment you already have.
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO=$PWD

ENV_NAME=dexx
ISAACLAB=${ISAACLAB_PATH:-}
ISAACSIM=${ISAACSIM_PATH:-}
VERIFY_ONLY=0
PYTORCH3D_BASE=${PYTORCH3D_BASE:-0.7.8+5043d15}
PY_VER=3.10
LAB_SHA=0f00ca2b4b2d54d5f90006a92abb1b00a72b2f20
INSTALL_CONSTRAINTS=
trap '[[ -z "$INSTALL_CONSTRAINTS" ]] || rm -f "$INSTALL_CONSTRAINTS"' EXIT

while [ $# -gt 0 ]; do
  case "$1" in
    --name)        ENV_NAME=$2; shift 2 ;;
    --isaaclab)    ISAACLAB=$2; shift 2 ;;
    --isaacsim)    ISAACSIM=$2; shift 2 ;;
    --verify-only) VERIFY_ONLY=1; shift ;;
    -h|--help)     sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1"; exit 1 ;;
  esac
done

step () { echo; echo "=== $*"; }
die  () { echo "ERROR: $*" >&2; exit 1; }

step "0/6  prerequisites"
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die "This installer targets Linux x86_64. See MANUAL_SETUP.md."
command -v conda >/dev/null || die "conda not found. Install Miniconda first."
command -v nvidia-smi >/dev/null || die "nvidia-smi not found — this needs an NVIDIA GPU."
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/  GPU: /'
echo "  conda: $(conda --version)"
if [ "$VERIFY_ONLY" -eq 0 ]; then
  [ -n "$ISAACLAB" ] || die "--isaaclab is required (or set ISAACLAB_PATH).
       Isaac Sim and Isaac Lab are installed separately; see
       tutorial/00_setup/MANUAL_SETUP.md steps 1-2."
  [ -x "$ISAACLAB/isaaclab.sh" ] || die "no isaaclab.sh under $ISAACLAB"
  ISAACLAB=$(cd "$ISAACLAB" && pwd)
  [ "$(git -C "$ISAACLAB" rev-parse HEAD)" = "$LAB_SHA" ] || die "Isaac Lab v2.2.1 ($LAB_SHA) is required. See MANUAL_SETUP.md."
  git -C "$ISAACLAB" diff --quiet HEAD -- source isaaclab.sh || die "Isaac Lab has local source changes; use a clean baseline checkout."
  command -v cmake >/dev/null || die "Install cmake and build tools before running Isaac Lab's installer."
  echo "  IsaacLab: $ISAACLAB"
  if [ -n "$ISAACSIM" ]; then
    [ -d "$ISAACSIM" ] || die "--isaacsim path does not exist: $ISAACSIM"
    if [ -e "$ISAACLAB/_isaac_sim" ]; then
      [ "$ISAACLAB/_isaac_sim" -ef "$ISAACSIM" ] || die "_isaac_sim differs from --isaacsim; resolve the link before installing."
      echo "  _isaac_sim: matches --isaacsim"
    else
      ISAACSIM=$(cd "$ISAACSIM" && pwd)
      ln -s "$ISAACSIM" "$ISAACLAB/_isaac_sim"
      echo "  _isaac_sim: linked -> $ISAACSIM"
    fi
  elif [ ! -e "$ISAACLAB/_isaac_sim" ]; then
    die "$ISAACLAB/_isaac_sim is missing and --isaacsim was not given.
       Isaac Lab needs Isaac Sim linked in as _isaac_sim."
  fi
  [ -f "$ISAACLAB/_isaac_sim/VERSION" ] || die "Isaac Sim VERSION file missing; this installer expects a binary Sim 4.5 installation."
  SIM_VERSION=$(head -n 1 "$ISAACLAB/_isaac_sim/VERSION")
  [[ "$SIM_VERSION" == 4.5.* ]] || die "Expected Isaac Sim 4.5, found $SIM_VERSION."
fi

step "1/6  conda environment '$ENV_NAME' (python $PY_VER)"
# conda activation hooks are hostile to `set -eu`: Isaac Sim's reads
# $ZSH_VERSION unguarded, and an env's deactivate.d may `unalias isaaclab`,
# which returns non-zero when the alias is absent and silently kills the script.
# Relax both flags around every conda call.
set +eu
source "$(conda info --base)/etc/profile.d/conda.sh"
set -eu
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "  exists — reusing it"
elif [ "$VERIFY_ONLY" -eq 1 ]; then
  die "environment '$ENV_NAME' does not exist (--verify-only creates nothing)"
else
  conda create -y -n "$ENV_NAME" "python=$PY_VER"
fi
set +eu; conda activate "$ENV_NAME"; activation_status=$?; set -eu
[ "$activation_status" -eq 0 ] || die "Could not activate $ENV_NAME"
echo "  python: $(python --version)"
python -c 'import sys; assert sys.version_info[:2] == (3, 10), "This baseline requires Python 3.10"'
INSTALL_CONSTRAINTS=$(mktemp)
cat "$REPO/constraints-sim45.txt" > "$INSTALL_CONSTRAINTS"
# Apply the same constraints to pip calls made inside the upstream installer.
export PIP_CONSTRAINT="$INSTALL_CONSTRAINTS"
# pip >=25.3 isolates build constraints; older pip inherits PIP_CONSTRAINT.
export PIP_BUILD_CONSTRAINT="$INSTALL_CONSTRAINTS"

if [ "$VERIFY_ONLY" -eq 0 ]; then

step "2/6  Isaac Lab v2.2.1 (official Torch 2.7/cu128 default)"
python -m pip install 'setuptools<81' toml
if python -c "import isaaclab" 2>/dev/null; then
  python - "$ISAACLAB" <<'PYLAB'
import isaaclab, sys
from pathlib import Path
expected = Path(sys.argv[1]) / "source/isaaclab/isaaclab/__init__.py"
if Path(isaaclab.__file__).resolve() != expected.resolve():
    sys.exit("Installed isaaclab differs from --isaaclab. Use a fresh environment or correct its editable installation.")
print("  reusing pinned checkout:", isaaclab.__version__)
PYLAB
else
  echo "  running: ./isaaclab.sh -c $ENV_NAME  &&  ./isaaclab.sh -i none"
  echo "  (-i none = no extra RL libraries; they can drag in a conflicting torch)"
  (
    cd "$ISAACLAB"
    # The pinned upstream script runs `tabs 4` under set -e. A headless
    # shell commonly has TERM=dumb; give only this subprocess a usable type.
    if [[ -z "${TERM:-}" || "$TERM" == dumb ]]; then
      export TERM=xterm
    fi
    ./isaaclab.sh -c "$ENV_NAME" && ./isaaclab.sh -i none
  )
  set +eu; conda activate "$ENV_NAME"; activation_status=$?; set -eu
  [ "$activation_status" -eq 0 ] || die "Could not reactivate $ENV_NAME"
fi

# Validate before choosing a binary extension, and preserve the exact CUDA build
# throughout subsequent dependency resolution (not just the public version).
python tutorial/00_setup/check_versions.py --core-only
python - >> "$INSTALL_CONSTRAINTS" <<'PYPINS'
from importlib.metadata import version
for name in ("torch", "torchvision", "isaaclab", "gymnasium", "numpy"):
    print(f"{name}=={version(name)}")
PYPINS

step "3/6  pytorch3d, matched to the torch Isaac Lab installed"
python - "$PYTORCH3D_BASE" <<'PY'
import subprocess, sys
base = sys.argv[1]
try:
    import torch
except ImportError:
    sys.exit("torch is not importable — Isaac Lab did not install. See MANUAL_SETUP.md step 2.")
full = torch.__version__                      # e.g. 2.5.1+cu118
ver, _, cuda = full.partition("+")
if not cuda.startswith("cu"):
    sys.exit(f"torch {full} has no CUDA suffix; a CPU build cannot run this. "
             f"Reinstall Isaac Lab against a CUDA build.")
want = f"{base}pt{ver}{cuda}"
print(f"  torch      = {full}")
print(f"  pytorch3d -> {want}")
try:
    import pytorch3d
    have = getattr(pytorch3d, "__version__", "")
    # the installed build encodes the torch it was compiled against
    import importlib.metadata as md
    dist = md.version("pytorch3d")
    if f"pt{ver}{cuda}" in dist:
        print(f"  already installed and matching: {dist}")
        sys.exit(0)
    print(f"  installed {dist} does NOT match torch {full} — reinstalling")
except ImportError:
    pass
cmd = [sys.executable, "-m", "pip", "install", "--extra-index-url",
       "https://miropsota.github.io/torch_packages_builder", f"pytorch3d=={want}"]
print("  " + " ".join(cmd))
r = subprocess.run(cmd)
if r.returncode != 0:
    sys.exit(f"""
pytorch3d=={want} is not available on the prebuilt index.
Browse https://miropsota.github.io/torch_packages_builder for a build matching
torch {full}, then either install it by hand or re-run with
  PYTORCH3D_BASE=<base> bash tutorial/00_setup/setup_env.sh ...
See tutorial/00_setup/MANUAL_SETUP.md step 3.""")
PY

step "4-5/6  Dex-X and pinned dependencies"
python -m pip install --quiet -e . -r requirements.txt
# pip can keep an installed VCS package from another commit when its version
# number is identical. Replace this package's code explicitly after resolving
# all dependencies above; the final pip/provenance checks still apply.
RAYCASTER_REQUIREMENT=$(sed -n '/^simple-raycaster @ /p' requirements.txt)
[ -n "$RAYCASTER_REQUIREMENT" ] || die "Missing pinned raycaster requirement"
python -m pip install --quiet --force-reinstall --no-deps "$RAYCASTER_REQUIREMENT"
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

  conda activate $ENV_NAME
  bash tutorial/run_acceptance.sh      # headless runtime, training and evaluation

Only the acceptance test proves the environment is correct. Then start at
tutorial/README.md
MSG
else
  echo "Some checks failed — see the output above, and"
  echo "tutorial/00_setup/MANUAL_SETUP.md for the same steps done by hand."
  exit 1
fi
