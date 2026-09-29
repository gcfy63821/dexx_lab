"""PPO soft action bound: zero inside [-1.1, 1.1], quadratic outside (ppo.py train_epoch)."""
import ast
import os
import unittest

import torch

SRC = os.path.join(os.path.dirname(__file__), "..", "src", "dexx", "algo", "ppo", "ppo.py")


def bounds_loss(mu):
    """Evaluate the two mu_loss_* lines exactly as written in ppo.py."""
    tree = ast.parse(open(SRC).read())
    lines = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
             and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("mu_loss_high", "mu_loss_low")]
    assert len(lines) == 2, "expected mu_loss_high and mu_loss_low in ppo.py"
    env = {"torch": torch, "mu": mu, "soft_bound": 1.1}
    for n in lines:
        env[n.targets[0].id] = eval(compile(ast.Expression(n.value), SRC, "eval"), env)
    return env["mu_loss_high"] + env["mu_loss_low"]


class BoundsLossTest(unittest.TestCase):
    def test_zero_inside_the_bound(self):
        mu = torch.tensor([-1.1, -1.0, -0.3, 0.0, 0.7, 1.0, 1.1])
        self.assertTrue(torch.equal(bounds_loss(mu), torch.zeros_like(mu)))

    def test_quadratic_outside_on_both_sides(self):
        mu = torch.tensor([-1.6, 1.6])
        torch.testing.assert_close(bounds_loss(mu), torch.tensor([0.25, 0.25]))

    def test_gradient_points_back_inside(self):
        mu = torch.tensor([-2.0, 0.5, 2.0], requires_grad=True)
        bounds_loss(mu).sum().backward()
        self.assertGreater(mu.grad[2].item(), 0)   # descent moves mu=2 down
        self.assertLess(mu.grad[0].item(), 0)      # and mu=-2 up
        self.assertEqual(mu.grad[1].item(), 0.0)   # no pull inside the bound


if __name__ == "__main__":
    unittest.main()
