"""Adversarial checks for the release validators (no simulator/GPU needed).

Run: python -m unittest discover -s tutorial/00_setup -p 'test_*.py'
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]


def load_checker(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tutorial/00_setup" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ImportChecks(unittest.TestCase):
    def run_check(self, files, args=()):
        checker = load_checker("check_imports")
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for name, content in files.items():
                path = base / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            with mock.patch.object(checker, "ROOT", directory), mock.patch.object(sys, "path", [directory, *sys.path]):
                with contextlib.redirect_stdout(io.StringIO()):
                    return checker.main(list(args))

    def test_installed_but_broken_package_fails(self):
        self.assertEqual(self.run_check({"scripts/use.py": "import release_broken_dep",
                                        "release_broken_dep.py": "import release_missing_transitive"}), 1)

    def test_missing_submodule_fails(self):
        self.assertEqual(self.run_check({"scripts/use.py": "import release_root.missing",
                                        "release_root/__init__.py": ""}), 1)

    def test_raycaster_class_is_required(self):
        checker = load_checker("check_imports")
        with mock.patch.object(checker, "source_imports", return_value={"simple_raycaster.raycaster": {"test.py"}}), \
             mock.patch.object(checker.importlib, "import_module", return_value=types.SimpleNamespace()), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(checker.main([]), 1)

    def test_pytorch3d_extension_is_required(self):
        checker = load_checker("check_imports")
        def importing(name):
            if name == "pytorch3d._C":
                raise ImportError("undefined symbol in extension")
            return types.SimpleNamespace()
        with mock.patch.object(checker, "source_imports", return_value={"pytorch3d.ops": {"test.py"}}), \
             mock.patch.object(checker.importlib, "import_module", side_effect=importing), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(checker.main([]), 1)

    def test_only_sim_bootstrap_errors_can_be_deferred(self):
        checker = load_checker("check_imports")
        for missing, expected in (("omni.kit", 0), ("gym", 1), ("isaaclab.missing", 1)):
            with self.subTest(missing=missing), \
                 mock.patch.object(checker, "source_imports", return_value={"isaaclab.utils.warp": {"test.py"}}), \
                 mock.patch.object(checker.importlib, "import_module", side_effect=ModuleNotFoundError(name=missing)), \
                 mock.patch.object(checker.importlib.util, "find_spec", return_value=object()), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(checker.main([]), expected)

    def test_applauncher_is_not_imported_by_static_check(self):
        checker = load_checker("check_imports")
        with mock.patch.object(checker, "source_imports", return_value={"isaaclab.app": {"test.py"}}), \
             mock.patch.object(checker.importlib, "import_module") as importing, \
             mock.patch.object(checker, "module_spec_without_import", return_value=object()), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(checker.main([]), 0)
            importing.assert_not_called()

    def test_missing_applauncher_fails(self):
        checker = load_checker("check_imports")
        with mock.patch.object(checker, "source_imports", return_value={"isaaclab.app": {"test.py"}}), \
             mock.patch.object(checker, "module_spec_without_import", return_value=None), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(checker.main([]), 1)

    def test_runtime_package_parents_are_not_executed(self):
        checker = load_checker("check_imports")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "release_runtime_pkg").mkdir()
            (root / "release_runtime_pkg/__init__.py").write_text("raise RuntimeError('parent executed')")
            (root / "release_runtime_pkg/sub.py").touch()
            with mock.patch.object(sys, "path", [directory, *sys.path]):
                self.assertIsNotNone(checker.module_spec_without_import("release_runtime_pkg.sub"))
                self.assertIsNone(checker.module_spec_without_import("release_runtime_pkg.missing"))

    def test_deploy_dependencies_are_opt_in(self):
        files = {"scripts/use.py": "import json", "src/dexx/scripts/deploy/use.py": "import release_missing_deploy",
                 "tools/calib/capture_multiframe_zmq.py": "import release_missing_deploy"}
        self.assertEqual(self.run_check(files), 0)
        self.assertEqual(self.run_check(files, ["--include-deploy"]), 1)


class PortableChecks(unittest.TestCase):
    def check_metadata(self, metadata):
        checker = load_checker("check_portable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("assets/robot", "assets/generated", "data/robotool_batch/task/demo"):
                (root / name).mkdir(parents=True)
            for name in ("checkpoints/teacher_poseobs.pth", "calib/camera_align/current.npy",
                         "data/retargeting/robotool_batch/mano2sharpa_rh/task/demo@0.pkl",
                         "data/robotool_batch/models/mesh.obj"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            if metadata is not None:
                (root / "data/robotool_batch/task/demo/meta.json").write_text(metadata)
            with mock.patch.object(checker, "ROOT", directory), contextlib.redirect_stdout(io.StringIO()):
                return checker.main()

    def test_missing_malformed_and_empty_metadata_fail(self):
        for metadata in (None, "{", "{}", "[]", '{"obj_mesh_paths": {"obj": "../escape.obj"}}'):
            with self.subTest(metadata=metadata):
                self.assertEqual(self.check_metadata(metadata), 1)

    def test_valid_metadata_passes(self):
        self.assertEqual(self.check_metadata('{"obj_mesh_paths": {"obj": "models/mesh.obj"}}'), 0)


class AcceptanceChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = (ROOT / "tutorial/run_acceptance.sh").read_text()

    def watch(self, command, stale=False):
        functions = self.script.split('active_pid=""', 1)[1].split('say "1/5', 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "artifact"
            if stale:
                artifact.write_text("old")
            runner = 'active_pid=""' + functions
            runner += '\nsleep () { command sleep 0.03; }\n'
            runner += 'watch_for "$1" "$2" 1 -- bash -c "$3" -- "$1"\n'
            return subprocess.run(["bash", "-c", runner, "--", str(artifact), str(Path(directory) / "log"), command],
                                  capture_output=True, timeout=10).returncode

    def test_stale_artifact_cannot_mask_failure(self):
        self.assertNotEqual(self.watch("exit 23", stale=True), 0)

    def test_new_artifact_cannot_mask_failure(self):
        self.assertNotEqual(self.watch('echo complete > "$1"; exit 23'), 0)

    def test_artifact_cannot_mask_delayed_failure(self):
        self.assertNotEqual(self.watch('echo complete > "$1"; sleep 0.1; exit 23'), 0)

    def test_artifact_cannot_mask_shutdown_hang(self):
        self.assertNotEqual(self.watch('echo complete > "$1"; sleep 20'), 0)

    def test_success_without_artifact_fails(self):
        self.assertNotEqual(self.watch("exit 0"), 0)

    def test_success_with_artifact_passes(self):
        self.assertEqual(self.watch('echo complete > "$1"'), 0)

    def test_timeout_fails(self):
        self.assertNotEqual(self.watch("sleep 20"), 0)

    def test_nonempty_output_rejected_before_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "old").touch()
            result = subprocess.run(["bash", str(ROOT / "tutorial/run_acceptance.sh"), directory],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("new or empty output directory", result.stdout)

    def test_failed_precheck_never_starts_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / "python"
            python.write_text('#!/bin/sh\n[ "$1" = "-c" ] && exit 0\necho "$*" >> "$CHECK_CALLS"\nexit 23\n')
            python.chmod(0o755)
            calls = root / "calls"
            result = subprocess.run(["bash", str(ROOT / "tutorial/run_acceptance.sh"), str(root / "out")],
                                    env={**os.environ, "PATH": directory + os.pathsep + os.environ["PATH"],
                                         "CHECK_CALLS": str(calls)}, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(calls.read_text().splitlines(), ["tutorial/00_setup/check_install.py"])

    def test_failed_version_or_pip_check_never_starts_runtime(self):
        for fail_arg in ("tutorial/00_setup/check_versions.py", "-m"):
            with self.subTest(fail_arg=fail_arg), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                python = root / "python"
                python.write_text('#!/bin/sh\n[ "$1" = "-c" ] && exit 0\n'
                                  'echo "$*" >> "$CHECK_CALLS"\n'
                                  '[ "$1" = "$FAIL_ARG" ] && exit 23\nexit 0\n')
                python.chmod(0o755)
                calls = root / "calls"
                result = subprocess.run(["bash", str(ROOT / "tutorial/run_acceptance.sh"), str(root / "out")],
                                        env={**os.environ, "PATH": directory + os.pathsep + os.environ["PATH"],
                                             "CHECK_CALLS": str(calls), "FAIL_ARG": fail_arg},
                                        capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                attempts = calls.read_text().splitlines()
                self.assertEqual(attempts[-1], "-m pip check" if fail_arg == "-m" else fail_arg)
                self.assertFalse(any("check_runtime.py" in line or "train_dagger_pc.py" in line for line in attempts))

    def assert_artifacts(self, demos, count, malformed=False, drift=False, bad_rate=False, bad_init_demos=()):
        code = self.script.split('<<\'PY\' || fail=1\n', 1)[1].split('\nPY\n', 1)[0]
        expected = ["rt/0416_grasp/cube_small_1", "rt/0416_grasp/cube_small_2",
                    "rt/0420_manip/squeegee_1", "rt/0420_manip/squeegee_2"]
        checkpoint = {"cfg": {"proprio_dim": 417, "n_hand": 6, "n_scene": 1024, "n_tactile": 25},
                      "student_drop_slots": ["obj_bps", "tips_distance", "obj_pose_tail"],
                      "student_obs_slots": {"proprioception": (0, 79), "ref_tracking": (79, 390),
                                            "target_obj_pose": (390, 397), "tips_distance": (397, 402),
                                            "obj_bps": (402, 530), "tactile": (530, 550), "obj_pose_tail": (550, 557)},
                      "model": {"mlp.0.weight": types.SimpleNamespace(shape=(1, 481))}}
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "eval"
            out.mkdir()
            valid_count = sum(count for demo in demos if demo not in bad_init_demos)
            (out / "summary.json").write_text(json.dumps({
                "strict": {f"strict{k}": {"rate": 7.0 if bad_rate else 0.0 if drift or not valid_count else 1.0,
                                          "successes": 0 if drift else valid_count,
                                          "episodes": valid_count} for k in (2, 3, 5)},
                "success_rate_per_demo": {d: {"episodes": count,
                                               "strict3": 0.0 if drift or d in bad_init_demos else 1.0,
                                               "strict3_successes": 0 if drift or d in bad_init_demos else count,
                                               "strict3_episodes": 0 if d in bad_init_demos else count} for d in demos},
                "actual_episodes": len(demos) * count, "per_demo_quota": count,
                "bad_init_excluded": len(demos) * count - valid_count}))
            (out / "records.json").write_text(json.dumps({"single": [{"demo_idx": d,
                **({} if malformed else {"survival_len": 5 if d in bad_init_demos else 20,
                                         "end_final_dist": 0.01, "fail_causes": ["obj_pos_drift"] if drift else []})}
                for d in demos for _ in range(count)]}))
            fake_torch = types.SimpleNamespace(load=lambda *a, **kw: checkpoint)
            with mock.patch.dict(sys.modules, {"torch": fake_torch}), \
                 mock.patch.object(sys, "argv", ["assert", directory, json.dumps(expected)]), \
                 contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as result:
                exec(compile(code, "acceptance_assertions", "exec"), {})
            return result.exception.code

    def test_one_demo_one_episode_cannot_pass(self):
        self.assertEqual(self.assert_artifacts(["rt/0416_grasp/cube_small_1"], 1), 1)

    def test_records_without_measurements_fail(self):
        self.assertEqual(self.assert_artifacts(["rt/0416_grasp/cube_small_1", "rt/0416_grasp/cube_small_2",
                                               "rt/0420_manip/squeegee_1", "rt/0420_manip/squeegee_2"],
                                              10, malformed=True), 1)

    def test_impossible_strict_rate_fails(self):
        self.assertEqual(self.assert_artifacts(["rt/0416_grasp/cube_small_1", "rt/0416_grasp/cube_small_2",
                                               "rt/0420_manip/squeegee_1", "rt/0420_manip/squeegee_2"],
                                              10, bad_rate=True), 1)

    def test_honest_zero_success_with_drift_passes(self):
        self.assertEqual(self.assert_artifacts(["rt/0416_grasp/cube_small_1", "rt/0416_grasp/cube_small_2",
                                               "rt/0420_manip/squeegee_1", "rt/0420_manip/squeegee_2"],
                                              10, drift=True), 0)

    def test_all_bad_init_episodes_fail(self):
        demos = ["rt/0416_grasp/cube_small_1", "rt/0416_grasp/cube_small_2",
                 "rt/0420_manip/squeegee_1", "rt/0420_manip/squeegee_2"]
        self.assertEqual(self.assert_artifacts(demos, 10, bad_init_demos=demos), 1)

    def test_one_demo_with_only_bad_init_episodes_fails(self):
        demos = ["rt/0416_grasp/cube_small_1", "rt/0416_grasp/cube_small_2",
                 "rt/0420_manip/squeegee_1", "rt/0420_manip/squeegee_2"]
        self.assertEqual(self.assert_artifacts(demos, 10, bad_init_demos=demos[:1]), 1)

    def test_four_demos_ten_episodes_pass(self):
        self.assertEqual(self.assert_artifacts(["rt/0416_grasp/cube_small_1", "rt/0416_grasp/cube_small_2",
                                               "rt/0420_manip/squeegee_1", "rt/0420_manip/squeegee_2"], 10), 0)


if __name__ == "__main__":
    unittest.main()
