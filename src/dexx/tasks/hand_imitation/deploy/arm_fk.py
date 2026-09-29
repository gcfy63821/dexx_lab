"""Wrist state of the real arm, matching what the simulator reports for sim's EE.

The policy's wrist observations are defined on ``right_hand_C_MC`` (the Sharpa
hand base), in the arm-base frame, exactly as Isaac Lab reports that body:

  * position: the link frame origin;
  * orientation: PhysX's quaternion. PhysX composes each link's rotation joint
    by joint, so its sign is continuous in the joint angles and not
    canonicalised (w may be negative). The observation uses the raw quaternion,
    so the sign has to match, not just the rotation;
  * linear velocity: at the link's centre of mass (`body_lin_vel_w` is
    `body_com_lin_vel_w`), i.e. frame-origin velocity + omega x (R c);
  * angular velocity: world frame.

Arm drivers report their own end effector instead — Polymetis reports
``panda_link8``, the flange, 135° about the tool axis and ~3.5 cm behind
``right_hand_C_MC`` — so the deploy computes the wrist from the measured joints
with the same URDF the simulator uses.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np
import torch

ARM_JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]
SIM_EE_LINK = {"right": "right_hand_C_MC", "left": "left_hand_C_MC"}


def default_urdf_path(side: str = "right") -> str:
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))
    return os.path.join(repo, "assets", "generated", f"fr3_with_{side}_sharpa_wave.urdf")


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _qaxis(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    return np.r_[np.cos(angle / 2), np.sin(angle / 2) * axis]


def _qrpy(r, p, y):  # URDF fixed-axis rpy: R = Rz(y) Ry(p) Rx(r)
    return _qmul(_qmul(_qaxis([0, 0, 1], y), _qaxis([0, 1, 0], p)), _qaxis([1, 0, 0], r))


class ArmFK:
    """FK + geometric Jacobian of the 7-DOF arm, ``fr3_link0`` to ``ee_link``."""

    def __init__(self, side: str = "right", urdf_path: str | None = None, ee_link: str | None = None):
        import pytorch_kinematics as pk

        path = urdf_path or default_urdf_path(side)
        with open(path, "rb") as f:  # bytes: the URDF has an XML encoding declaration
            urdf = f.read()
        self.ee_link = ee_link or SIM_EE_LINK[side]
        chain = pk.build_serial_chain_from_urdf(urdf, end_link_name=self.ee_link, root_link_name="fr3_link0")
        self.chain = chain.to(dtype=torch.float32, device="cpu")
        names = self.chain.get_joint_parameter_names()
        if names != ARM_JOINT_NAMES:
            raise ValueError(f"arm chain to {self.ee_link!r} has joints {names}, expected {ARM_JOINT_NAMES}")

        # Joint-by-joint rotation chain (for the PhysX quaternion sign) and the
        # EE link's centre of mass (for the COM linear velocity), from the URDF.
        root = ET.fromstring(urdf)
        by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
        steps, link = [], self.ee_link
        while link != "fr3_link0":
            j = by_child[link]
            o = j.find("origin")
            rpy = [float(v) for v in o.get("rpy", "0 0 0").split()] if o is not None else [0.0, 0.0, 0.0]
            axis = None
            if j.get("type") in ("revolute", "continuous"):
                axis = [float(v) for v in j.find("axis").get("xyz").split()]
            steps.append((_qrpy(*rpy), axis))
            link = j.find("parent").get("link")
        self._rot_steps = steps[::-1]
        com = np.zeros(3)
        for l in root.findall("link"):
            if l.get("name") == self.ee_link and l.find("inertial") is not None:
                o = l.find("inertial").find("origin")
                if o is not None:
                    com = np.array([float(v) for v in o.get("xyz", "0 0 0").split()])
        self.com = torch.tensor(com, dtype=torch.float32)

    def _chained_quat(self, q: np.ndarray) -> np.ndarray:
        out, i = np.array([1.0, 0.0, 0.0, 0.0]), 0
        for q_origin, axis in self._rot_steps:
            out = _qmul(out, q_origin)
            if axis is not None:
                out = _qmul(out, _qaxis(axis, float(q[i])))
                i += 1
        return out

    def __call__(self, q: torch.Tensor, dq: torch.Tensor):
        """q, dq (7,) -> pos (3,), quat w,x,y,z (4,), COM lin vel (3,), ang vel (3,); arm-base frame."""
        from pytorch_kinematics.transforms import matrix_to_quaternion

        q = q.reshape(1, 7).float()
        m = self.chain.forward_kinematics(q).get_matrix()[0]
        R = m[:3, :3]
        quat = matrix_to_quaternion(R)
        ref = self._chained_quat(q[0].numpy())
        if float(np.dot(quat.numpy(), ref)) < 0:  # take PhysX's sign
            quat = -quat
        twist = self.chain.jacobian(q)[0] @ dq.reshape(7).float()  # (6,): origin linear, angular
        v, w = twist[:3], twist[3:]
        v_com = v + torch.linalg.cross(w, R @ self.com)
        return m[:3, 3].clone(), quat, v_com.clone(), w.clone()
