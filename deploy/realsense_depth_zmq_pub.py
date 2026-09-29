#!/usr/bin/env python3
"""RealSense depth -> ZMQ publisher. Runs ON THE CAMERA HOST.

Grabs D455 depth at 640x480x30, applies a x2 decimation filter -> 320x240 (whose
intrinsics ~= the sim/deploy SIM_INTRINSICS fx=193.33, cx=160.08, cy=121.05), and
PUBs each frame as fp32 meters over ZMQ. The inference PC subscribes with
RealSenseDepthZmqSubscriber and feeds it to the PointCloud deploy env via
set_depth_source() — no ROS2 anywhere.

Why 640x480 -> decimate x2 (not native 320x240): the D455 has no 320x240 depth
profile; 640x480 depth intrinsics /2 = (193.78,193.78,161.58,119.62) which match
SIM_INTRINSICS to <1.5px, so the deploy's sim-intrinsic back-projection is valid.

Run on the camera host (needs pyrealsense2 + pyzmq + msgpack + msgpack-numpy):
    python realsense_depth_zmq_pub.py --bind_port 5562

Socket: PUB tcp://*:5562  msgpack {"depth": fp32 (240,320) meters, "t": epoch, "seq": int}
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pyrealsense2 as rs
import zmq
import msgpack
import msgpack_numpy as m
m.patch()

# Deploy expects this resolution (sim intrinsics live at 320x240).
OUT_H, OUT_W = 240, 320


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind_port", type=int, default=5562, help="ZMQ PUB port")
    ap.add_argument("--src_width", type=int, default=640)
    ap.add_argument("--src_height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--decimate", type=int, default=2, help="decimation magnitude (2 -> 640x480->320x240)")
    ap.add_argument("--print_every", type=int, default=60)
    args = ap.parse_args()

    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 2)          # drop old frames if subscriber lags
    pub.bind(f"tcp://*:{args.bind_port}")
    print(f"[depth-pub] PUB tcp://*:{args.bind_port}", flush=True)

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, args.src_width, args.src_height, rs.format.z16, args.fps)
    profile = pipe.start(cfg)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    decim = rs.decimation_filter()
    decim.set_option(rs.option.filter_magnitude, float(args.decimate))

    intr0 = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    print(f"[depth-pub] src {args.src_width}x{args.src_height} depth_scale={depth_scale:.5f} "
          f"src_intr fx={intr0.fx:.1f} cx={intr0.ppx:.1f}", flush=True)
    print(f"[depth-pub] decimate x{args.decimate} -> streaming ~{OUT_W}x{OUT_H} fp32 meters", flush=True)

    seq = 0
    try:
        while True:
            frames = pipe.wait_for_frames()
            depth = frames.get_depth_frame()
            if not depth:
                continue
            depth = decim.process(depth)
            d = np.asarray(depth.get_data(), dtype=np.uint16)   # (h,w) raw units
            dm = d.astype(np.float32) * depth_scale             # -> meters

            # decimation may yield 320x240; enforce exact (240,320) for the deploy.
            if dm.shape != (OUT_H, OUT_W):
                # center-crop / pad to (240,320) — decimation of 640x480 gives 320x240 already
                h, w = dm.shape
                out = np.zeros((OUT_H, OUT_W), dtype=np.float32)
                hh, ww = min(h, OUT_H), min(w, OUT_W)
                out[:hh, :ww] = dm[:hh, :ww]
                dm = out

            pub.send(msgpack.packb({"depth": dm, "t": time.time(), "seq": seq}))
            seq += 1
            if args.print_every and seq % args.print_every == 0:
                valid = float((dm > 0.05).mean())
                print(f"[depth-pub] seq={seq} shape={dm.shape} valid={valid:.1%} "
                      f"z[min/med/max]={dm[dm>0.05].min() if (dm>0.05).any() else 0:.2f}/"
                      f"{np.median(dm[dm>0.05]) if (dm>0.05).any() else 0:.2f}/"
                      f"{dm.max():.2f}", flush=True)
    except KeyboardInterrupt:
        print("\n[depth-pub] stopping.", flush=True)
    finally:
        pipe.stop()
        pub.close(0)


if __name__ == "__main__":
    main()
