"""Portable setup regressions; no Conda, GPU or simulator installation is needed."""
import contextlib
import importlib.util
from importlib.metadata import PackageNotFoundError
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def version_checker():
    spec = importlib.util.spec_from_file_location(
        "dexx_check_versions", ROOT / "tutorial/00_setup/check_versions.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BaselineVersionTests(unittest.TestCase):
    def setUp(self):
        self.checker = version_checker()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.checkout = Path(self.temp.name) / "IsaacLab"
        sim = self.checkout / "_isaac_sim"
        sim.mkdir(parents=True)
        self.sim_version = sim / "VERSION"
        self.sim_version.write_text("4.5.0-rc.36\n")
        self.versions = {
            "isaaclab": "0.45.9", "torch": "2.7.0+cu128",
            "torchvision": "0.22.0+cu128", "gymnasium": "1.2.0",
            "numpy": "1.26.4", "pin": "2.7.0", "opencv-python": "4.11.0.86",
            "setuptools": "80.9.0",
        }
        self.torch = SimpleNamespace(__version__="2.7.0+cu128",
                                     version=SimpleNamespace(cuda="12.8"))
        self.vision = SimpleNamespace(__version__="0.22.0+cu128")
        self.lab = SimpleNamespace(
            __file__=str(self.checkout / "source/isaaclab/isaaclab/__init__.py")
        )

    def run_check(self, *args, sha=None, dirty=False):
        def installed(name):
            if name not in self.versions:
                raise PackageNotFoundError(name)
            return self.versions[name]

        def git_output(command, **kwargs):
            self.assertEqual(Path(command[2]), self.checkout)
            return (sha or self.checker.LAB_SHA) + "\n"

        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(self.checker, "version", installed))
            stack.enter_context(mock.patch.object(self.checker, "distribution", return_value=SimpleNamespace(
                read_text=lambda name: json.dumps({
                    "url": "https://github.com/btx0424/simple-raycaster.git",
                    "vcs_info": {"vcs": "git", "commit_id": "7bab59c56e9a340f20b7af29e4b769108cb697fd"},
                }),
            )))
            stack.enter_context(mock.patch.object(sys, "version_info", (3, 10, 0)))
            stack.enter_context(mock.patch.object(sys, "argv", ["check_versions.py", *args]))
            stack.enter_context(mock.patch.dict(sys.modules, {
                "isaaclab": self.lab, "torch": self.torch, "torchvision": self.vision,
            }))
            stack.enter_context(mock.patch.object(self.checker.subprocess, "check_output", git_output))
            stack.enter_context(mock.patch.object(
                self.checker.subprocess, "run",
                side_effect=subprocess.CalledProcessError(1, "git diff") if dirty else None,
            ))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            return self.checker.main()

    def test_complete_binary_baseline(self):
        self.assertEqual(self.run_check(), 0)

    def test_pre_project_check_does_not_require_project_opencv(self):
        del self.versions["opencv-python"]
        self.assertEqual(self.run_check("--core-only"), 0)
        with self.assertRaises(PackageNotFoundError):
            self.run_check()

    def test_pre_project_check_still_requires_core_packages(self):
        del self.versions["gymnasium"]
        with self.assertRaises(PackageNotFoundError):
            self.run_check("--core-only")

    def test_incompatible_core_and_opencv_versions_fail(self):
        for name, value in [("torch", "2.5.1+cu118"), ("numpy", "2.2.0"),
                            ("gymnasium", "0.29.1"), ("opencv-python", "4.12.0.88")]:
            with self.subTest(name=name), mock.patch.dict(self.versions, {name: value}):
                with self.assertRaisesRegex(RuntimeError, "Expected"):
                    self.run_check()

    def test_wrong_or_modified_lab_checkout_fails(self):
        with self.assertRaisesRegex(RuntimeError, "Isaac Lab v2.2.1"):
            self.run_check(sha="0" * 40)
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_check(dirty=True)

    def test_sim_version_comes_from_binary_checkout(self):
        self.sim_version.write_text("5.0.0\n")
        with self.assertRaisesRegex(RuntimeError, "Isaac Sim 4.5"):
            self.run_check()
        self.sim_version.unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_check()

    def test_sim_version_comes_from_pip_wheels_without_link(self):
        self.sim_version.unlink()
        self.sim_version.parent.rmdir()
        with self.assertRaisesRegex(RuntimeError, "Isaac Sim not found"):
            self.run_check()
        self.versions["isaacsim"] = "4.5.0.0"
        self.assertEqual(self.run_check(), 0)
        self.versions["isaacsim"] = "5.0.0.0"
        with self.assertRaisesRegex(RuntimeError, "Isaac Sim 4.5"):
            self.run_check()

    def test_cpu_and_mixed_cuda_builds_fail(self):
        self.torch.version.cuda = None
        with self.assertRaisesRegex(RuntimeError, "CUDA-enabled"):
            self.run_check()
        self.torch.version.cuda = "12.8"
        self.vision.__version__ = "0.22.0+cu118"
        with self.assertRaisesRegex(RuntimeError, "same CUDA build"):
            self.run_check()


class InstallerBootstrapTests(unittest.TestCase):
    def test_fresh_install_stages_version_checks_before_and_after_project(self):
        """Exercise shell orchestration with unavailable Lab and no OpenCV yet."""
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            bindir = temp / "bin"
            bindir.mkdir()
            lab = temp / "IsaacLab"
            (lab / "_isaac_sim").mkdir(parents=True)
            (lab / "_isaac_sim/VERSION").write_text("4.5.0\n")
            log = temp / "commands.jsonl"
            marker = temp / "project-installed"
            conda_base = temp / "conda"
            hooks = conda_base / "etc/profile.d"
            hooks.mkdir(parents=True)
            (hooks / "conda.sh").write_text('conda () { return 0; }\n')
            executable = "#!" + sys.executable + "\n"
            driver = executable + '''import json, os, pathlib, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
if name == "isaaclab.sh" and args == ["-i", "none"]:
    constraints = os.environ.get("PIP_CONSTRAINT")
    if not constraints or os.environ.get("PIP_BUILD_CONSTRAINT") != constraints:
        sys.exit("Runtime and isolated build constraints must both reach Lab")
    if "setuptools<81" not in pathlib.Path(constraints).read_text():
        sys.exit("flatdict source builds need the setuptools compatibility cap")
with open(os.environ["TEST_SETUP_LOG"], "a") as f:
    f.write(json.dumps([name, *args]) + "\\n")
if name == "conda":
    if args == ["info", "--base"]: print(os.environ["TEST_CONDA_BASE"])
    elif args == ["--version"]: print("conda test")
    elif args == ["env", "list"]: print("dexx /unused/test-env")
elif name == "git":
    if "rev-parse" in args: print("0f00ca2b4b2d54d5f90006a92abb1b00a72b2f20")
elif name == "python":
    if args == ["-c", "import isaaclab"]:
        sys.exit(0 if os.environ.get("TEST_WRONG_LAB") else 1)
    if args and args[0] == "-" and len(args) > 1 and os.environ.get("TEST_WRONG_LAB"):
        import types
        sys.modules["isaaclab"] = types.SimpleNamespace(__file__="/other/IsaacLab/isaaclab/__init__.py")
        sys.argv = ["-", *args[1:]]
        exec(compile(sys.stdin.read(), "<setup source check>", "exec"))
    if args == ["--version"]: print("Python 3.10.0")
    marker = pathlib.Path(os.environ["TEST_PROJECT_MARKER"])
    if "requirements.txt" in args: marker.touch()
    if args and args[0].endswith("check_versions.py"):
        before_project = not marker.exists()
        if before_project != ("--core-only" in args):
            sys.exit("Version check used the wrong bootstrap stage")
'''
            for name in ["conda", "git", "python", "nvidia-smi", "cmake"]:
                command = bindir / name
                command.write_text(driver)
                command.chmod(0o755)
            (lab / "bin").mkdir()
            upstream_driver = lab / "bin/isaaclab.sh"
            upstream_driver.write_text(driver)
            upstream_driver.chmod(0o755)
            # Model the real pinned upstream startup: tabs fails under set -e
            # when a noninteractive shell inherits TERM=dumb.
            (lab / "isaaclab.sh").write_text(
                '#!/usr/bin/env bash\nset -e\ntabs 4 >/dev/null\n'
                'exec "$(dirname "$0")/bin/isaaclab.sh" "$@"\n'
            )
            (lab / "isaaclab.sh").chmod(0o755)
            env = dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ["PATH"],
                       TEST_SETUP_LOG=str(log), TEST_PROJECT_MARKER=str(marker),
                       TEST_CONDA_BASE=str(conda_base), TERM="dumb")
            # A developer's own Isaac paths must not leak into the installer.
            env.pop("ISAACSIM_PATH", None)
            env.pop("ISAACLAB_PATH", None)
            upstream = subprocess.run(
                [str(lab / "isaaclab.sh"), "-i", "none"], env=env, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20,
            )
            self.assertNotEqual(upstream.returncode, 0, upstream.stdout)
            self.assertIn("cannot reset tabs", upstream.stdout)
            result = subprocess.run(
                ["bash", str(ROOT / "tutorial/00_setup/setup_env.sh"),
                 "--isaaclab", str(lab)], env=env, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20,
            )
            self.assertEqual(result.returncode, 0, result.stdout)
            commands = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertIn(["isaaclab.sh", "-i", "none"], commands)
            version_calls = [cmd for cmd in commands
                             if len(cmd) > 1 and cmd[1].endswith("check_versions.py")]
            self.assertEqual(len(version_calls), 2)
            self.assertIn("--core-only", version_calls[0])
            self.assertNotIn("--core-only", version_calls[1])
            self.assertIn("GPU runtime remains unverified", result.stdout)
            # An importable package from a different checkout must not be reused.
            env["TEST_WRONG_LAB"] = "1"
            result = subprocess.run(
                ["bash", str(ROOT / "tutorial/00_setup/setup_env.sh"),
                 "--isaaclab", str(lab)], env=env, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("differs from --isaaclab", result.stdout)


class RuntimeCliTests(unittest.TestCase):
    def test_cpu_and_stale_results_fail_before_starting_simulator(self):
        spec = importlib.util.spec_from_file_location(
            "dexx_check_runtime", ROOT / "tutorial/00_setup/check_runtime.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class Launcher:
            @staticmethod
            def add_app_launcher_args(parser):
                parser.add_argument("--device", default="cuda:0")
                parser.add_argument("--headless", action="store_true")

            def __init__(self, args):
                raise AssertionError("invalid preflight must not launch Isaac Sim")

        with tempfile.TemporaryDirectory() as temp:
            result = Path(temp) / "result.json"
            with mock.patch.dict(sys.modules, {
                "isaaclab": SimpleNamespace(),
                "isaaclab.app": SimpleNamespace(AppLauncher=Launcher),
            }):
                for args, message in [
                    (["--device", "cpu"], "must select CUDA"),
                    ([], "must be a new file"),
                ]:
                    with self.subTest(args=args):
                        stderr = io.StringIO()
                        with mock.patch.object(sys, "argv", [
                            "check_runtime.py", "--headless", "--result", str(result), *args,
                        ]), contextlib.redirect_stderr(stderr):
                            with self.assertRaises(SystemExit) as error:
                                module.main()
                        self.assertEqual(error.exception.code, 2)
                        self.assertIn(message, stderr.getvalue())
                        result.write_text('{"ok": true}')


if __name__ == "__main__":
    unittest.main()
