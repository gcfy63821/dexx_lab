"""Grab N {depth,color} frames from the color ZMQ pub, back-project each to the
CAMERA frame with SIM_INTRINSICS, and accumulate into one dense colored cloud.

Saves accum.npz {points:(M,3) cam-frame, colors:(M,3) uint8}. Feed the calib tool
with --real_pc_npz accum.npz (color-aware). Overlaying N frames of a held-still
scene fills gaps and shows sensor jitter.

  python tools/calib/capture_multiframe_zmq.py \
      --addr tcp://<CAM_HOST>:5562 --n_frames 10 --out_dir logs/real_calib_multi
"""
import argparse
import os
import time

import numpy as np
from dexx import deploy_config as _dcfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--addr", required=True,
                   help="ZMQ address of the camera-host color publisher, "
                        "e.g. tcp://<CAM_HOST>:5562")
    p.add_argument("--n_frames", type=int, default=10)
    p.add_argument("--gap", type=float, default=0.15, help="seconds between grabbed frames")
    p.add_argument("--out_dir", default="logs/real_calib_multi")
    p.add_argument("--wait", type=float, default=10.0)
    args = p.parse_args()

    import torch
    import zmq
    import msgpack
    import msgpack_numpy as mnp
    mnp.patch()
    from dexx.deploy_config import SIM_INTRINSICS
    from dexx.tasks.franka_sharpa.pointcloud.depth_to_pointcloud import DepthToPointCloud

    d2pc = DepthToPointCloud(
        height=_dcfg.DEPTH_H, width=_dcfg.DEPTH_W,
        fx=SIM_INTRINSICS["fx"], fy=SIM_INTRINSICS["fy"],
        cx=SIM_INTRINSICS["cx"], cy=SIM_INTRINSICS["cy"],
        device="cpu", accepts_normalized=False,
    )

    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.CONFLATE, 1)
    sub.setsockopt(zmq.RCVTIMEO, 300)
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    sub.connect(args.addr)
    print(f"[multi] connecting {args.addr}, grabbing {args.n_frames} frames", flush=True)

    all_pts, all_cols = [], []
    grabbed = 0
    t0 = time.time()
    last = 0.0
    while grabbed < args.n_frames and time.time() - t0 < args.wait + args.n_frames * args.gap:
        try:
            raw = sub.recv()
        except zmq.Again:
            continue
        now = time.time()
        if now - last < args.gap:
            continue
        last = now
        msg = msgpack.unpackb(raw, raw=False)
        depth = np.array(msg["depth"], dtype=np.float32)
        color = np.array(msg["color"], dtype=np.uint8) if "color" in msg else None
        pts_cam, valid = d2pc.back_project(torch.from_numpy(depth).unsqueeze(0))
        vm = valid[0].numpy().astype(bool)
        pts = pts_cam[0].numpy()[vm]
        all_pts.append(pts)
        if color is not None:
            all_cols.append(color.reshape(-1, 3)[vm])
        grabbed += 1
        print(f"[multi] frame {grabbed}/{args.n_frames}: +{pts.shape[0]} pts", flush=True)

    if grabbed == 0:
        print("[multi] ERROR: no frames (pub running?)")
        return 1

    P = np.concatenate(all_pts, 0).astype(np.float32)
    os.makedirs(args.out_dir, exist_ok=True)
    if all_cols:
        C = np.concatenate(all_cols, 0).astype(np.uint8)
        np.savez(os.path.join(args.out_dir, "accum.npz"), points=P, colors=C)
        print(f"[multi] accumulated {grabbed} frames -> {P.shape[0]} pts (colored) -> {args.out_dir}/accum.npz", flush=True)
    else:
        np.savez(os.path.join(args.out_dir, "accum.npz"), points=P)
        print(f"[multi] accumulated {grabbed} frames -> {P.shape[0]} pts (NO color) -> {args.out_dir}/accum.npz", flush=True)
    sub.close(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
