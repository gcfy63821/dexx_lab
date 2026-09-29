#!/usr/bin/env python
"""ZMQ joint-control bridge — runs ON THE NUC (polymetis env).

Bridges the local Polymetis server to a remote deploy client over ZMQ, so the
training PC (py310 / isaaclab, no polymetis) can drive the FR3 with joint
impedance without installing polymetis. Joint-space control.

Run on the NUC:
    conda activate polymetis-local
    python polymetis_joint_bridge.py            # robot=localhost:50051
Sockets (bind on all interfaces so the wired-link client can reach them):
    state : PUB  tcp://*:5560   (msgpack) {joint_pos[7], joint_vel[7],
                                            ee_pos[3], ee_quat_xyzw[4], t}
    cmd   : PULL tcp://*:5561   (msgpack) {"cmd": ...}
              joint_target  {"q":[7]}
              start_impedance {"kq":[7]|None, "kqd":[7]|None}
              terminate / go_home
"""
import argparse
import threading
import time

import numpy as np
import torch
import zmq
import msgpack
import msgpack_numpy as m
m.patch()

from polymetis import RobotInterface

# FR3 joint position limits (rad), from the Franka FR3 datasheet.
FR3_Q_MIN = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159], np.float32)
FR3_Q_MAX = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159], np.float32)


def check_target(q, reference, max_step):
    """Reason to reject a joint target, or None. The bridge is the last hop
    before the robot, so it checks every target whatever client sent it."""
    if q.shape != (7,) or not np.isfinite(q).all():
        return f"not 7 finite values: {q}"
    if (q < FR3_Q_MIN).any() or (q > FR3_Q_MAX).any():
        return f"outside the FR3 joint limits: {np.round(q, 3)}"
    step = float(np.abs(q - reference).max())
    if step > max_step:
        return f"{step:.3f} rad from the previous target (limit {max_step})"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot_ip", default="localhost", help="Polymetis server IP (on the NUC = localhost).")
    ap.add_argument("--robot_port", type=int, default=50051)
    # These two literals are deliberate. This file runs ON THE NUC inside the
    # `polymetis-local` env, where `dexx` is not installed — importing
    # deploy_config here would break the bridge. They must stay in sync with
    # POLYMETIS_STATE_PORT / POLYMETIS_CMD_PORT by hand.
    ap.add_argument("--state_port", type=int, default=5560)
    ap.add_argument("--cmd_port", type=int, default=5561)
    ap.add_argument("--state_hz", type=float, default=200.0)
    ap.add_argument("--max_target_step", type=float, default=0.5,
                    help="Reject a joint target further than this (rad, any joint) from the "
                         "previous one, or from the measured pose for the first one.")
    args = ap.parse_args()

    print(f"[bridge] connecting to Polymetis {args.robot_ip}:{args.robot_port} ...")
    robot = RobotInterface(ip_address=args.robot_ip, port=args.robot_port, enforce_version=False)
    print("[bridge] connected.")

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://*:{args.state_port}")
    pull = ctx.socket(zmq.PULL)
    pull.bind(f"tcp://*:{args.cmd_port}")
    print(f"[bridge] state PUB tcp://*:{args.state_port}   cmd PULL tcp://*:{args.cmd_port}")

    last_target = [None]  # previous accepted target; reset when the arm moves on its own

    def drain():
        """Drop commands queued while a blocking call ran: they are stale."""
        n = 0
        while True:
            try:
                pull.recv(flags=zmq.NOBLOCK)
                n += 1
            except zmq.Again:
                return n

    def cmd_loop():
        while True:
            try:
                msg = msgpack.unpackb(pull.recv(), raw=False)
                cmd = msg.get("cmd")
                if cmd == "joint_target":
                    q = np.asarray(msg["q"], dtype=np.float32).reshape(-1)
                    ref = last_target[0]
                    if ref is None:
                        ref = robot.get_joint_positions().detach().cpu().numpy().astype(np.float32)
                    reason = check_target(q, ref, args.max_target_step)
                    if reason is not None:
                        print(f"[bridge] REJECTED joint target: {reason}")
                        continue
                    robot.update_desired_joint_positions(torch.as_tensor(q))
                    last_target[0] = q
                elif cmd == "start_impedance":
                    kq, kqd = msg.get("kq"), msg.get("kqd")
                    if kq is not None and kqd is not None:
                        robot.start_joint_impedance(
                            Kq=torch.tensor(kq, dtype=torch.float32),
                            Kqd=torch.tensor(kqd, dtype=torch.float32),
                        )
                        print(f"[bridge] started joint impedance (Kq={kq}, Kqd={kqd})")
                    else:
                        robot.start_joint_impedance()
                        print("[bridge] started joint impedance (default gains)")
                    last_target[0] = None
                elif cmd == "terminate":
                    robot.terminate_current_policy()
                    last_target[0] = None
                    print("[bridge] terminated current policy")
                elif cmd == "go_home":
                    robot.go_home()
                    last_target[0] = None
                    print(f"[bridge] go_home done (dropped {drain()} queued commands)")
                else:
                    print(f"[bridge] unknown cmd: {cmd}")
            except Exception as exc:  # noqa: BLE001
                print(f"[bridge] cmd error: {exc!r}")

    threading.Thread(target=cmd_loop, daemon=True).start()

    dt = 1.0 / args.state_hz
    n = 0
    while True:
        t0 = time.time()
        try:
            q = robot.get_joint_positions().detach().cpu().numpy().astype(np.float32)
            qd = robot.get_joint_velocities().detach().cpu().numpy().astype(np.float32)
            ee_pos, ee_quat = robot.get_ee_pose()
            state = {
                "joint_pos": q,
                "joint_vel": qd,
                "ee_pos": ee_pos.detach().cpu().numpy().astype(np.float32),
                "ee_quat_xyzw": ee_quat.detach().cpu().numpy().astype(np.float32),
                "t": t0,
            }
            pub.send(msgpack.packb(state))
            n += 1
            if n % (int(args.state_hz) * 5) == 0:
                print(f"[bridge] streaming... q0={q[0]:.3f} ({n} msgs)")
        except Exception as exc:  # noqa: BLE001
            print(f"[bridge] state error: {exc!r}")
            time.sleep(0.05)
        sleep = dt - (time.time() - t0)
        if sleep > 0:
            time.sleep(sleep)


if __name__ == "__main__":
    main()
