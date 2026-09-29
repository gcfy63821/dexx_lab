"""CPU regressions: python -m unittest discover -s tests -p test_runtime_metrics.py.

Extract the pure helpers so these tests do not launch Isaac Sim or parse its CLI.
"""

import ast
import importlib.util
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


def source_function(path, name):
    tree = ast.parse(path.read_text())
    return next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == name)


class StrictMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = ROOT / "scripts/eval.py"
        main = source_function(source, "main")
        threshold = next(node for node in main.body
                         if isinstance(node, ast.Assign)
                         and any(isinstance(target, ast.Name)
                                 and target.id == "_BAD_INIT_SURVIVAL"
                                 for target in node.targets))
        module = ast.Module(body=[threshold, source_function(source, "_strict")],
                            type_ignores=[])
        namespace = {}
        exec(compile(module, str(source), "exec"), namespace)
        cls.strict = staticmethod(namespace["_strict"])

    def record(self, distance=0.01, survival=20, failures=(), succeeded=True):
        return SimpleNamespace(end_final_dist=distance, survival_len=survival,
                               fail_causes=list(failures), succeeded=succeeded)

    def test_drift_rejected_using_recorded_key(self):
        records = [self.record(), self.record(failures=["obj_pos_drift"])]
        self.assertEqual(self.strict(records, 3), (0.5, 1, 2))

    def test_bad_init_removed_from_denominator(self):
        self.assertEqual(self.strict([self.record(), self.record(survival=5)], 3),
                         (1.0, 1, 1))
        self.assertEqual(self.strict([self.record(survival=0)], 3), (0.0, 0, 0))

    def test_protocol_does_not_require_env_success(self):
        record = self.record(failures=["eef_pos_drift"], succeeded=False)
        self.assertEqual(self.strict([record], 3), (1.0, 1, 1))

    def test_thresholds_and_invalid_distances(self):
        records = [self.record(distance=value) for value in
                   [0.0, 0.02, 0.03, 0.049, 0.05, -1.0, math.nan, math.inf]]
        for cm, successes in [(2, 1), (3, 2), (5, 4)]:
            with self.subTest(cm=cm):
                self.assertEqual(self.strict(records, cm),
                                 (successes / 8, successes, 8))
        self.assertEqual(self.strict([], 3), (0.0, 0, 0))


class EndpointMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("Endpoint tests require PyTorch")
        cls.torch = torch
        source = ROOT / "scripts/eval.py"
        main = source_function(source, "main")
        nodes = list(ast.walk(main))

        def assignment(name):
            return next(node for node in nodes if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == name
                                for target in node.targets))

        diagnostics = next(node for node in nodes
                           if isinstance(node, ast.If)
                           and ast.unparse(node.test) == "rd is not None")
        module = ast.Module(body=[diagnostics, assignment("end_dist"),
                                  assignment("end_rot"),
                                  assignment("_BAD_INIT_SURVIVAL"),
                                  source_function(source, "_strict")], type_ignores=[])
        cls.code = compile(module, str(source), "exec")

    def test_terminal_diagnostics_cross_threshold_in_both_directions(self):
        torch = self.torch
        # The last action can either enter or leave the 3 cm success region.
        # Post-reset observations and the previous step must not determine it.
        for previous, terminal, expected in [(0.02, 0.04, 0), (0.04, 0.02, 1)]:
            with self.subTest(previous=previous, terminal=terminal):
                namespace = {
                    "torch": torch, "env_id": 0, "fail_keys": [],
                    "rd": {"diag/final_pos_dist": torch.tensor([terminal]),
                           "diag/final_rot_angle": torch.tensor([0.2])},
                    "per_env_min_final": torch.tensor([previous]),
                    "per_env_min_final_rot": torch.tensor([0.1]),
                    "per_env_last_final_dist": torch.tensor([previous]),
                    "per_env_last_final_rot": torch.tensor([0.1]),
                    "prev_final_dist": torch.tensor([previous]),
                    "prev_final_rot": torch.tensor([0.1]),
                }
                exec(self.code, namespace)
                self.assertAlmostEqual(namespace["end_dist"], terminal)
                self.assertAlmostEqual(namespace["end_rot"], 0.2)
                record = SimpleNamespace(end_final_dist=namespace["end_dist"],
                                         survival_len=20, fail_causes=[])
                self.assertEqual(namespace["_strict"]([record], 3),
                                 (float(expected), expected, 1))


class QuaternionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("Quaternion tests require PyTorch")
        cls.torch = torch
        source = ROOT / "src/dexx/tasks/franka_sharpa/franka_sharpa_env.py"
        helper = ast.unparse(source_function(source, "quat_to_angle_axis"))
        # Same [-pi, pi] convention used by Isaac Lab. Keep a real source file
        # so TorchScript compiles the production decorator and both outputs.
        prefix = """import torch

def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    wrapped = (angles + torch.pi) % (2 * torch.pi)
    return torch.where((wrapped == 0) & (angles > 0), torch.pi, wrapped - torch.pi)

"""
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        module_path = Path(cls.temp.name) / "quaternion_helper.py"
        module_path.write_text(prefix + helper + "\n")
        spec = importlib.util.spec_from_file_location("quaternion_helper", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.convert = staticmethod(module.quat_to_angle_axis)

    def test_identity_has_finite_unit_axis(self):
        torch = self.torch
        quats = torch.tensor([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]])
        angle, axis = self.convert(quats)
        torch.testing.assert_close(angle, torch.zeros(2))
        torch.testing.assert_close(axis, torch.tensor([[0.0, 0.0, 1.0]]).repeat(2, 1))

    def test_rotations_and_quaternion_sign_reconstruct_same_rotation(self):
        torch = self.torch
        half = math.sqrt(0.5)
        quats = torch.tensor([[half, half, 0, 0], [half, 0, -half, 0],
                              [-half, -half, 0, 0], [0, 0, 0, 1]],
                             dtype=torch.float64).reshape(2, 2, 4)
        angle, axis = self.convert(quats)
        self.assertEqual(axis.shape, (2, 2, 3))
        torch.testing.assert_close(torch.linalg.vector_norm(axis, dim=-1),
                                   torch.ones((2, 2), dtype=torch.float64))
        reconstructed = torch.cat([torch.cos(angle / 2).unsqueeze(-1),
                                   axis * torch.sin(angle / 2).unsqueeze(-1)], dim=-1)
        torch.testing.assert_close((reconstructed * quats).sum(-1).abs(),
                                   torch.ones((2, 2), dtype=torch.float64))


if __name__ == "__main__":
    unittest.main()
