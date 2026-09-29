"""Validate the documented baseline without starting Isaac Sim."""
import argparse
from importlib.metadata import PackageNotFoundError, distribution, version
import json
from pathlib import Path
import subprocess
import sys

from packaging.requirements import Requirement

LAB_SHA = "0f00ca2b4b2d54d5f90006a92abb1b00a72b2f20"


def same_git_repo(a: str, b: str) -> bool:
    """uv records a Git URL without its .git suffix; pip keeps it."""
    def norm(url):
        url = url.rstrip("/")
        return url[:-4] if url.endswith(".git") else url
    return norm(a) == norm(b)


def check_raycaster_revision(root: Path) -> None:
    """Same-version wheels can contain incompatible code; verify the Git pin."""
    requirement = next(
        Requirement(line) for line in (root / "requirements.txt").read_text().splitlines()
        if line.strip().startswith("simple-raycaster @")
    )
    expected_url, _, expected_sha = requirement.url.removeprefix("git+").rpartition("@")
    try:
        direct = json.loads(distribution(requirement.name).read_text("direct_url.json") or "null")
        valid = (isinstance(direct, dict) and isinstance(direct.get("url"), str)
                 and same_git_repo(direct["url"], expected_url)
                 and isinstance(direct.get("vcs_info"), dict)
                 and direct["vcs_info"].get("vcs") == "git"
                 and direct["vcs_info"].get("commit_id") == expected_sha)
    except (ValueError, TypeError):
        valid = False
    if not valid:
        raise RuntimeError(
            "simple-raycaster does not match the pinned Git revision. Reinstall with "
            f"python -m pip install --force-reinstall --no-deps '{requirement}'"
        )


def isaac_sim_version(checkout: Path) -> str:
    """Binary Sim linked as _isaac_sim (setup_env.sh), else the pip wheels (setup_uv.sh)."""
    link = checkout / "_isaac_sim"
    if link.exists():
        return (link / "VERSION").read_text().strip()
    try:
        return version("isaacsim")
    except PackageNotFoundError:
        raise RuntimeError(
            "Isaac Sim not found: link a binary Sim 4.5 as _isaac_sim in the Lab checkout "
            "or install the isaacsim==4.5.0 wheels (tutorial/00_setup/setup_uv.sh)"
        ) from None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-only", action="store_true",
                        help="check Lab's baseline before project dependencies are installed")
    args = parser.parse_args(argv)
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("The Sim 4.5 baseline requires Python 3.10")
    root = Path(__file__).resolve().parents[2]
    for line in (root / "constraints-sim45.txt").read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        req = Requirement(line)
        if args.core_only and req.name == "opencv-python":
            continue
        if req.marker and not req.marker.evaluate():
            continue
        installed = version(req.name)
        if installed not in req.specifier:
            raise RuntimeError(f"Expected {req}; found {installed}")

    import isaaclab
    import torch
    import torchvision

    checkout = Path(isaaclab.__file__).resolve().parents[3]
    sha = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if sha != LAB_SHA:
        raise RuntimeError("Use the Isaac Lab v2.2.1 checkout documented in MANUAL_SETUP.md")
    subprocess.run(
        ["git", "-C", str(checkout), "diff", "--exit-code", "HEAD", "--", "source", "isaaclab.sh"],
        check=True, stdout=subprocess.DEVNULL,
    )
    sim_version = isaac_sim_version(checkout)
    if not sim_version.startswith("4.5."):
        raise RuntimeError(f"Expected Isaac Sim 4.5; found {sim_version}")
    if not torch.version.cuda:
        raise RuntimeError("A CUDA-enabled Torch build is required")
    if torch.__version__.partition("+")[2] != torchvision.__version__.partition("+")[2]:
        raise RuntimeError("Torch and torchvision must use the same CUDA build")
    if not args.core_only:
        check_raycaster_revision(root)
    print(f"BASELINE VERSIONS MATCH: Torch {torch.__version__}; GPU runtime still needs validation")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        sys.exit(f"BASELINE CHECK FAILED: {exc}")
