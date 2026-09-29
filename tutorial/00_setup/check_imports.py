"""Check imports used by simulation, without starting Isaac Sim.

    python tutorial/00_setup/check_imports.py
    python tutorial/00_setup/check_imports.py --include-deploy

Hardware SDKs and Isaac Sim runtime imports are reported separately. A passing
static check does not verify the simulator, CUDA operators, or training; run
check_runtime.py --headless --result runtime.json and tutorial/run_acceptance.sh
for those checks.
"""
from __future__ import annotations

import argparse
import ast
import importlib
import importlib.machinery
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCAN = ("src/dexx", "scripts", "tools", "deploy")
SKIP_DIRS = {"__pycache__", ".git"}
FIRST_PARTY = {"dexx"}
# These tools acquire data from live hardware, rather than process saved data.
DEPLOY_TOOLS = {"tools/calib/capture_multiframe_zmq.py", "tools/calib/live_calibrate_extrinsic.py"}
# ROS2 (the optional deploy backend) comes from a system install, not pip.
HARDWARE_ONLY = {"polymetis", "pyrealsense2", "sharpa",
                 "rclpy", "sensor_msgs", "geometry_msgs", "std_msgs", "franka_msgs"}
SIM_RUNTIME = {"omni", "pxr", "isaacsim", "carb", "usdrt"}
LAB_PACKAGES = {"isaaclab", "isaaclab_tasks", "isaaclab_rl"}
LAB_RUNTIME = {"isaaclab.app", "isaaclab.assets", "isaaclab.actuators", "isaaclab.controllers",
               "isaaclab.envs", "isaaclab.managers", "isaaclab.scene", "isaaclab.sensors",
               "isaaclab.sim", "isaaclab_tasks"}
PIP_NAME = {
    "warp": "warp-lang",
    "simple_raycaster": "the pinned simple-raycaster in requirements.txt",
    "bps_torch": "the pinned bps_torch in requirements.txt",
    "chamfer_distance": "the pinned chamfer_distance in requirements.txt",
    "pytorch3d": "see MANUAL_SETUP.md (must match Torch and CUDA)",
    "cv2": "opencv-python", "yaml": "pyyaml", "PIL": "pillow",
    "sklearn": "scikit-learn", "zmq": "pyzmq", "msgpack_numpy": "msgpack-numpy",
}


def source_imports(include_deploy: bool = False) -> dict[str, set[str]]:
    """Actual module names, including submodules, mapped to their callers."""
    found: dict[str, set[str]] = {}
    for rel in SCAN:
        if rel == "deploy" and not include_deploy:
            continue
        base = os.path.join(ROOT, rel)
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS
                           and (include_deploy or d != "deploy")]
            for filename in filenames:
                if not filename.endswith(".py"):
                    continue
                path = os.path.join(dirpath, filename)
                caller = os.path.relpath(path, ROOT)
                if not include_deploy and ("deploy" in filename or caller in DEPLOY_TOOLS):
                    continue
                with open(path, encoding="utf-8") as source:
                    tree = ast.parse(source.read(), filename=caller)
                for node in ast.walk(tree):
                    names = []
                    if isinstance(node, ast.Import):
                        names = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                        names = [node.module]
                    for name in names:
                        found.setdefault(name, set()).add(caller)
    return found


def module_spec_without_import(name: str):
    """Resolve package files without executing parent package initializers."""
    parts = name.split(".")
    spec = importlib.util.find_spec(parts[0])
    for index in range(1, len(parts)):
        if spec is None or spec.submodule_search_locations is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(
            ".".join(parts[:index + 1]), spec.submodule_search_locations)
    return spec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-deploy", action="store_true",
                        help="also check deployment and live-camera Python dependencies")
    args = parser.parse_args(argv)
    stdlib = getattr(sys, "stdlib_module_names", set())
    third_party = {
        name: files for name, files in source_imports(args.include_deploy).items()
        if name.split(".")[0] not in stdlib | FIRST_PARTY and not name.startswith("_")
    }
    if any(name.split(".")[0] == "pytorch3d" for name in third_party):
        third_party["pytorch3d._C"] = {"PyTorch3D compiled extension"}

    failed, hardware, runtime, ok = [], [], [], []
    for name in sorted(third_party):
        root = name.split(".")[0]
        if root in HARDWARE_ONLY:
            hardware.append(name)
            continue
        if root in SIM_RUNTIME:
            runtime.append(name)
            continue
        try:
            if any(name == prefix or name.startswith(prefix + ".") for prefix in LAB_RUNTIME):
                # These packages can load Isaac Sim and prompt for its EULA.
                # Check files without executing them; check_runtime tests the API.
                if module_spec_without_import(name) is None:
                    raise ModuleNotFoundError(f"{name} is not installed", name=name)
                runtime.append(name)
                continue
            module = importlib.import_module(name)
            if name == "simple_raycaster.raycaster":
                cls = getattr(module, "MultiMeshRaycaster")
                for method in ("raycast", "raycast_fused"):
                    if not callable(getattr(cls, method, None)):
                        raise ImportError(f"MultiMeshRaycaster.{method} is unavailable")
            ok.append(name)
        except Exception as error:  # noqa: BLE001
            # Only known simulator bootstrap dependencies can be deferred.
            # Missing gym/numpy, ABI errors, and missing Lab submodules must fail.
            deferred = (root in LAB_PACKAGES and isinstance(error, ModuleNotFoundError)
                        and (error.name or "").split(".")[0] in SIM_RUNTIME
                        and importlib.util.find_spec(root) is not None)
            if deferred:
                runtime.append(name)
            else:
                failed.append((name, f"{type(error).__name__}: {error}", sorted(third_party[name])[:2]))

    print(f"=== {len(third_party)} third-party module imports")
    print(f"  {len(ok)} importable, {len(hardware)} hardware SDK imports skipped, "
          f"{len(runtime)} require runtime verification, {len(failed)} failed")
    if hardware:
        print("\nHardware SDK imports not checked: " + ", ".join(hardware))
    if runtime:
        print("\nIsaac App imports NOT VERIFIED: " + ", ".join(runtime))
    if failed:
        print("\n=== FAILED IMPORTS")
        for name, error, files in failed:
            print(f"  {name}: {error}")
            print(f"     dependency: {PIP_NAME.get(name.split('.')[0], name.split('.')[0])}")
            print(f"     used by: {', '.join(files)}")
        return 1
    print("\nSTATIC IMPORT CHECK PASSED — runtime and hardware checks remain separate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
