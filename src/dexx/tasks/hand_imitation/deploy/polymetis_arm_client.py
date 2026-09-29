"""Polymetis arm client (ZMQ) for the Sharpa deploy env.

The deploy env's only arm backend: one ``PolymetisArmClient`` reads arm/wrist
state and sends joint targets.

Because Polymetis (py38 / torch 1.13) cannot coexist with Isaac Lab (py310 /
torch 2.7) in one env, this client does NOT import polymetis. Instead it talks
over ZMQ to ``polymetis_joint_bridge.py`` running on the NUC (in the working
polymetis env). The bridge owns the real Polymetis ``RobotInterface`` and runs
the joint-impedance loop; this client only:
    - SUB (conflate) the bridge's state stream  -> arm/wrist obs
    - PUSH joint targets / control commands to the bridge

    deploy env (this PC, py310)   --ZMQ over Ethernet-->   NUC bridge (py38)
        publish_arm_joint_pos ---- {"cmd":"joint_target"} --> update_desired_joint_positions
        state SUB (conflate)  <---- {joint_pos, joint_vel, ee_*} <-- get_joint_* / get_ee_pose

The wrist is the simulator's end-effector frame (``right_hand_C_MC``), computed
from the measured joints with the sim's URDF (``arm_fk.ArmFK``), with velocities
``J(q) * dq``. Polymetis' own ``get_ee_pose`` is the flange (``panda_link8``),
135° about the tool axis and ~3.5 cm away from it; it is kept as
``flange_position`` / ``flange_quaternion`` for diagnostics only.

Interface exposed (what the deploy env reads/calls):
    read attrs : arm_joint_positions, arm_joint_velocities,
                 wrist_position, wrist_quaternion (w,x,y,z),
                 wrist_linear_velocity, wrist_angular_velocity (arm-base frame),
                 arm_data_received, wrist_msg_count
    methods    : publish_arm_joint_pos(q7), state_age(), get_logger(), shutdown()
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np
from dexx import deploy_config as _dcfg
import torch


def _make_logger(name: str = "PolymetisArmClient") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    if not hasattr(logger, "warn"):
        logger.warn = logger.warning  # type: ignore[attr-defined]
    return logger


def world_angular_velocity(q_prev: torch.Tensor, q_now: torch.Tensor, dt: float) -> torch.Tensor:
    """World-frame angular velocity taking q_prev to q_now (both w,x,y,z) in dt."""
    w1, v1 = q_prev[0], q_prev[1:]
    w2, v2 = q_now[0], q_now[1:]
    # q_now * conj(q_prev): the rotation applied in the world frame.
    w = w2 * w1 + torch.dot(v2, v1)
    v = w1 * v2 - w2 * v1 - torch.linalg.cross(v2, v1)
    if w < 0:  # same rotation, shortest path
        w, v = -w, -v
    n = torch.linalg.norm(v)
    if n < 1e-9:
        return torch.zeros(3, dtype=torch.float32)
    angle = 2.0 * torch.atan2(n, w)
    return (v / n * angle / dt).to(torch.float32)


class PolymetisArmClient:
    """ZMQ client to the NUC polymetis_joint_bridge."""

    def __init__(
        self,
        ip_address: str = "localhost",
        state_port: int = _dcfg.POLYMETIS_STATE_PORT,
        cmd_port: int = _dcfg.POLYMETIS_CMD_PORT,
        kq=None,
        kqd=None,
        start_impedance: bool = True,
        connect_timeout_s: float = 10.0,
        logger: logging.Logger | None = None,
        side: str = "right",
    ):
        self.logger = logger or _make_logger()
        self.ip_address = ip_address
        self.state_port = int(state_port)
        self.cmd_port = int(cmd_port)
        self.kq = list(kq) if kq is not None else None
        self.kqd = list(kqd) if kqd is not None else None

        # ---- state buffers ----
        self.arm_joint_positions = torch.zeros(7, dtype=torch.float32)
        self.arm_joint_velocities = torch.zeros(7, dtype=torch.float32)
        self.flange_position = torch.zeros(3, dtype=torch.float32)
        self.flange_quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)  # w,x,y,z
        self.arm_data_received = False
        self.wrist_msg_count = 0
        # Local receive time (time.monotonic) of the newest state.
        self._last_rx = None
        # Wrist FK runs on demand, once per state message (the env reads the
        # four wrist attributes each step; the bridge streams at 200 Hz).
        from dexx.tasks.hand_imitation.deploy.arm_fk import ArmFK
        self._fk = ArmFK(side=side)
        self._joint_state = (self.arm_joint_positions, self.arm_joint_velocities, 0)
        self._wrist_cache = None

        # ---- ZMQ sockets ----
        try:
            import zmq
            import msgpack
            import msgpack_numpy as _mnp
            _mnp.patch()
        except Exception as exc:  # noqa: BLE001
            raise ImportError(
                "pyzmq + msgpack + msgpack-numpy are required for PolymetisArmClient. "
                "Install with: pip install pyzmq msgpack msgpack-numpy\n  %r" % (exc,)
            )
        self._zmq = zmq
        self._msgpack = msgpack
        self._ctx = zmq.Context.instance()

        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.setsockopt(zmq.CONFLATE, 1)          # keep only newest state
        self._sub.setsockopt(zmq.RCVTIMEO, 200)
        self._sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self._sub.connect(f"tcp://{ip_address}:{self.state_port}")

        self._push = self._ctx.socket(zmq.PUSH)
        self._push.setsockopt(zmq.SNDHWM, 2)
        self._push.connect(f"tcp://{ip_address}:{self.cmd_port}")

        self.logger.info(
            f"connecting to NUC bridge {ip_address} (state:{self.state_port} cmd:{self.cmd_port}) ..."
        )

        # ---- background state poller ----
        self._stop = threading.Event()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

        # Wait for the first state: without it every observation would be built
        # from the zero-initialised buffers above.
        t0 = time.time()
        while time.time() - t0 < connect_timeout_s:
            if self.arm_data_received:
                break
            time.sleep(0.05)
        if not self.arm_data_received:
            self.shutdown(terminate_policy=False)
            raise RuntimeError(
                f"no arm state from the Polymetis bridge at {ip_address}:{self.state_port} "
                f"within {connect_timeout_s}s — is polymetis_joint_bridge.py running on the NUC?"
            )

        if start_impedance:
            self.start_joint_impedance()

    # ------------------------------------------------------------------
    def _send(self, msg: dict):
        try:
            self._push.send(self._msgpack.packb(msg), flags=self._zmq.NOBLOCK)
        except Exception as exc:  # noqa: BLE001
            self.logger.warn(f"cmd send failed: {exc!r}")

    def start_joint_impedance(self):
        self._send({"cmd": "start_impedance", "kq": self.kq, "kqd": self.kqd})
        gains = f"Kq={self.kq}" if self.kq else "Polymetis default gains"
        self.logger.info(f"requested start_joint_impedance ({gains})")

    def go_home(self):
        self._send({"cmd": "go_home"})

    # ------------------------------------------------------------------
    def _poll_loop(self):
        while not self._stop.is_set():
            try:
                raw = self._sub.recv()
            except self._zmq.Again:
                continue
            except Exception as exc:  # noqa: BLE001
                self.logger.warn(f"state recv error: {exc!r}")
                continue
            try:
                s = self._msgpack.unpackb(raw, raw=False)
                q = torch.as_tensor(np.array(s["joint_pos"], dtype=np.float32)).flatten()
                dq = torch.as_tensor(np.array(s["joint_vel"], dtype=np.float32)).flatten()
                if q.numel() != 7 or dq.numel() != 7:
                    raise ValueError(f"expected 7 joints, got {q.numel()} / {dq.numel()}")
                self.arm_joint_positions = q
                self.arm_joint_velocities = dq
                self.wrist_msg_count += 1
                self._joint_state = (q, dq, self.wrist_msg_count)  # one consistent snapshot
                self.arm_data_received = True

                q_xyzw = np.array(s["ee_quat_xyzw"], dtype=np.float32).flatten()
                self.flange_position = torch.as_tensor(np.array(s["ee_pos"], dtype=np.float32)).flatten()
                self.flange_quaternion = torch.tensor(
                    [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=torch.float32)  # -> w,x,y,z
                self._last_rx = time.monotonic()
            except Exception as exc:  # noqa: BLE001
                self.logger.warn(f"state decode error: {exc!r}")

    # ------------------------------------------------------------------
    # Wrist (sim EE frame, from the measured joints)
    # ------------------------------------------------------------------
    def _wrist(self):
        q, dq, n = self._joint_state
        cache = self._wrist_cache
        if cache is None or cache[0] != n:
            cache = (n, *self._fk(q, dq))
            self._wrist_cache = cache
        return cache[1:]

    def wrist_state(self):
        """(pos, quat wxyz, lin vel, ang vel) from one joint snapshot."""
        return self._wrist()

    @property
    def wrist_position(self):
        return self._wrist()[0]

    @property
    def wrist_quaternion(self):
        return self._wrist()[1]

    @property
    def wrist_linear_velocity(self):
        return self._wrist()[2]

    @property
    def wrist_angular_velocity(self):
        return self._wrist()[3]

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------
    def publish_arm_joint_pos(self, arm_joint_pos_des):
        if isinstance(arm_joint_pos_des, torch.Tensor):
            q = arm_joint_pos_des.detach().cpu().flatten().float().numpy()
        else:
            q = np.asarray(arm_joint_pos_des, dtype=np.float32).flatten()
        if q.size != 7:
            self.logger.warn(f"publish_arm_joint_pos expected 7 values, got {q.size}; ignoring")
            return
        if not np.isfinite(q).all():
            self.logger.warn("non-finite arm target; ignoring")
            return
        self._send({"cmd": "joint_target", "q": q})

    # ------------------------------------------------------------------
    def state_age(self) -> float:
        """Seconds since the newest bridge state arrived (inf before the first)."""
        last = self._last_rx
        return float("inf") if last is None else time.monotonic() - last

    def get_logger(self):
        return self.logger

    def shutdown(self, terminate_policy: bool = True):
        """Stop polling and close sockets; by default also ask the bridge to end
        the joint-impedance policy (Polymetis then holds the arm in place)."""
        self._stop.set()
        try:
            if self._poll_thread.is_alive():
                self._poll_thread.join(timeout=1.0)
        except Exception:
            pass
        if terminate_policy:
            try:
                self._send({"cmd": "terminate"})
            except Exception:
                pass
        # Give the queued "terminate" up to 0.5 s to leave; the state socket
        # has nothing worth waiting for.
        for sock, linger in ((getattr(self, "_sub", None), 0), (getattr(self, "_push", None), 500)):
            try:
                if sock is not None:
                    sock.close(linger=linger)
            except Exception:
                pass
