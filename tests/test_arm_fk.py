"""Wrist FK used by the arm clients: sim EE frame, PhysX quaternion sign, COM velocity."""
import importlib.util
import os
import unittest

import numpy as np

import torch

HAVE_PK = importlib.util.find_spec("pytorch_kinematics") is not None
Q = torch.tensor([0.1, -0.5, 0.2, -2.2, 0.3, 1.8, 0.5])


@unittest.skipUnless(HAVE_PK, "pytorch_kinematics not installed")
class ArmFKTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dexx.tasks.hand_imitation.deploy.arm_fk import ArmFK
        from dexx.tasks.hand_imitation.deploy.polymetis_arm_client import world_angular_velocity
        cls.fk, cls.link7 = ArmFK("right"), ArmFK("right", ee_link="fr3_link7")
        cls.wav = staticmethod(world_angular_velocity)

    def test_velocities_match_finite_differences(self):
        torch.manual_seed(0)
        dq, eps = torch.randn(7) * 0.3, 1e-3
        p, q, v, w = self.fk(Q, dq)
        p2, q2, _, _ = self.fk(Q + dq * eps, dq)

        def com(p_, q_):  # linear velocity is reported at the link's centre of mass
            from pytorch_kinematics.transforms import quaternion_to_matrix
            return p_ + quaternion_to_matrix(q_) @ self.fk.com
        torch.testing.assert_close(v, (com(p2, q2) - com(p, q)) / eps, atol=2e-3, rtol=0)
        torch.testing.assert_close(w, self.wav(q, q2, eps), atol=2e-3, rtol=0)

    def test_sim_ee_is_142mm_and_135deg_from_link7(self):
        # Polymetis' flange (panda_link8) has link7's orientation; the sim EE
        # does not — the offset the clients must not skip.
        p7, q7, _, _ = self.link7(Q, torch.zeros(7))
        p, q, _, _ = self.fk(Q, torch.zeros(7))
        self.assertAlmostEqual(float((p - p7).norm()), 0.142, places=4)
        angle = 2 * torch.arccos(torch.clamp((q * q7).sum().abs(), max=1.0))
        self.assertAlmostEqual(float(torch.rad2deg(angle)), 135.0, places=2)

    def test_matches_recorded_sim_frames(self):
        """Quaternion (with sign), COM linear velocity and angular velocity equal
        what Isaac Lab reported for right_hand_C_MC on recorded sim frames."""
        d = np.load(os.path.join(os.path.dirname(__file__), "data", "sim_wrist_frames.npz"))
        self.assertGreater(int((d["wrist_quat_wxyz"][:, 0] < 0).sum()), 0)  # fixture covers w < 0
        for q, dq, quat, v, w in zip(d["arm_joint_pos"], d["arm_joint_vel"], d["wrist_quat_wxyz"],
                                     d["wrist_lin_vel"], d["wrist_ang_vel"]):
            _, q_fk, v_fk, w_fk = self.fk(torch.from_numpy(q), torch.from_numpy(dq))
            dot = float(np.dot(q_fk.numpy(), quat))
            self.assertGreater(dot, 0.0)  # same sign as PhysX, not just the same rotation
            # a few degrees: the frames were recorded with a slightly different hand
            # mount; the sign is what this test guards
            self.assertLess(np.degrees(2 * np.arccos(min(dot, 1.0))), 5.0)
            np.testing.assert_allclose(v_fk.numpy(), v, atol=5e-3)
            np.testing.assert_allclose(w_fk.numpy(), w, atol=5e-3)


if __name__ == "__main__":
    unittest.main()
