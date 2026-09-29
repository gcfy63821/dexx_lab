#!/usr/bin/env python
"""Staged bring-up test for the Polymetis arm path (client -> ZMQ bridge -> NUC).

Does NOT need Isaac Lab / a policy — just exercises PolymetisArmClient against the
NUC bridge, so we can validate comms and control incrementally and SAFELY.

Run on the training PC (dexmanip env). The NUC must be running:
    conda activate polymetis-local && python polymetis_joint_bridge.py

Stages (each higher stage requires an explicit flag; robot only moves at --nudge):

    (default)   READ-ONLY. Stream state for --seconds; print joint pos/vel + ee
                pose. No commands sent, robot does NOT move. Use this first to
                confirm the joint angles match the real robot.

    --engage    Also send start_joint_impedance (holds the CURRENT pose). The arm
                should NOT visibly move; we print joint-pos drift before/after.

    --nudge J D Also, after engaging, move joint J by D rad (e.g. --nudge 3 0.05),
                hold 2s, then return to the start pose. FIRST real motion —
                keep a hand on the e-stop and the workspace clear.
"""
import argparse
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from dexx import deploy_config as _dcfg
from dexx.tasks.hand_imitation.deploy.polymetis_arm_client import PolymetisArmClient


def fmt(v):
    return "[" + ", ".join(f"{x:+.4f}" for x in np.asarray(v).flatten().tolist()) + "]"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ip", required=True, help="NUC bridge IP (e.g. 192.168.1.10)")
    ap.add_argument("--state_port", type=int, default=_dcfg.POLYMETIS_STATE_PORT)
    ap.add_argument("--cmd_port", type=int, default=_dcfg.POLYMETIS_CMD_PORT)
    ap.add_argument("--seconds", type=float, default=5.0, help="read-only streaming duration")
    ap.add_argument("--engage", action="store_true", help="send start_joint_impedance (holds current pose)")
    ap.add_argument("--nudge", nargs=2, type=float, default=None, metavar=("JOINT", "DELTA"),
                    help="move joint JOINT (0-6) by DELTA rad, hold 2s, return. REAL MOTION.")
    ap.add_argument("--kq", type=str, default=None, help="JSON list of 7 Kq (else Polymetis default)")
    ap.add_argument("--kqd", type=str, default=None, help="JSON list of 7 Kqd")
    args = ap.parse_args()

    import json
    kq = json.loads(args.kq) if args.kq else None
    kqd = json.loads(args.kqd) if args.kqd else None

    print(f"[test] connecting to NUC bridge {args.ip} (state:{args.state_port} cmd:{args.cmd_port})")
    client = PolymetisArmClient(
        ip_address=args.ip, state_port=args.state_port, cmd_port=args.cmd_port,
        kq=kq, kqd=kqd, start_impedance=False,   # never auto-engage in this test
    )

    # ---- Stage 0: read-only ----
    print(f"\n=== Stage 0: READ-ONLY state stream ({args.seconds}s) — robot will NOT move ===")
    t0 = time.time()
    last = 0
    while time.time() - t0 < args.seconds:
        if time.time() - last > 0.5:
            last = time.time()
            print(f"  q  = {fmt(client.arm_joint_positions)}")
            print(f"  qd = {fmt(client.arm_joint_velocities)}")
            print(f"  flange_pos = {fmt(client.flange_position)}  wrist(C_MC)_pos = {fmt(client.wrist_position)}"
                  f"  msgs={client.wrist_msg_count}")
        time.sleep(0.05)
    if not client.arm_data_received:
        print("[test] !! no state received — is the bridge running on the NUC?")
        client.shutdown(); return
    q_start = client.arm_joint_positions.clone()
    print(f"[test] start pose q = {fmt(q_start)}")

    if not (args.engage or args.nudge):
        print("\n[test] read-only done. Re-run with --engage (hold) or --nudge J D (move) to go further.")
        client.shutdown(); return

    # ---- Stage 1: engage impedance (hold current pose) ----
    print("\n=== Stage 1: start_joint_impedance (holds CURRENT pose — should not visibly move) ===")
    input("  press ENTER to engage impedance (Ctrl-C to abort) ... ")
    client.start_joint_impedance()
    time.sleep(1.0)
    # command hold-at-current a few times
    for _ in range(15):
        client.publish_arm_joint_pos(q_start)
        time.sleep(1.0 / 30.0)
    drift = (client.arm_joint_positions - q_start).abs().max().item()
    print(f"  max joint drift after engage+hold: {drift*1000:.2f} mrad ({np.degrees(drift):.3f} deg)")

    # ---- Stage 2: nudge one joint ----
    if args.nudge:
        j, d = int(args.nudge[0]), float(args.nudge[1])
        assert 0 <= j <= 6, "joint index must be 0..6"
        print(f"\n=== Stage 2: NUDGE joint {j} by {d:+.3f} rad ({np.degrees(d):+.1f} deg) — REAL MOTION ===")
        input("  workspace clear? hand on e-stop? press ENTER to move (Ctrl-C to abort) ... ")
        q_target = q_start.clone()
        q_target[j] = q_start[j] + d
        print(f"  target q = {fmt(q_target)}")
        for _ in range(60):   # ~2s @30Hz streaming toward target (impedance interpolates)
            client.publish_arm_joint_pos(q_target)
            time.sleep(1.0 / 30.0)
        reached = client.arm_joint_positions.clone()
        err = (reached[j] - q_target[j]).item()
        print(f"  reached q[{j}] = {reached[j]:+.4f} (target {q_target[j]:+.4f}, err {np.degrees(err):+.2f} deg)")
        print("  returning to start pose ...")
        for _ in range(60):
            client.publish_arm_joint_pos(q_start)
            time.sleep(1.0 / 30.0)
        back_err = (client.arm_joint_positions - q_start).abs().max().item()
        print(f"  back-to-start max err: {np.degrees(back_err):.2f} deg")

    print("\n[test] terminating policy + shutting down.")
    client.shutdown()


if __name__ == "__main__":
    main()
