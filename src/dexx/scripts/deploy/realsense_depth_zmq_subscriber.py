"""ZMQ RealSense depth subscriber — runs ON THE INFERENCE PC.

SUBs depth frames from `deploy/realsense_depth_zmq_pub.py` on the camera host
and keeps the newest one for the PointCloud deploy env: `get_latest()` returns
it as fp32 (H,W) metres on the configured device (or None), `age()` the seconds
since it arrived.

    sub = RealSenseDepthZmqSubscriber(addr="tcp://<CAM_HOST>:5562", device="cuda")
    ...
    depth_m = sub.get_latest()   # fp32 (240,320) meters torch on device, or None

A background thread polls the ZMQ SUB (CONFLATE=1 keeps only the newest frame),
mirroring the PolymetisArmClient state poller.

Frames whose size is not (height, width) are REJECTED (counted in
`n_rejected`, warned about at most once per second): a different resolution
means different intrinsics, and unprojecting it with this size's intrinsics
would put every point in the wrong place.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np
import torch


class RealSenseDepthZmqSubscriber:
    def __init__(
        self,
        addr: str,
        height: int = 240,
        width: int = 320,
        device: str = "cpu",
        rcvtimeo_ms: int = 200,
    ):
        self._H = int(height)
        self._W = int(width)
        self._device = torch.device(device)
        self._latest: Optional[torch.Tensor] = None
        self._latest_stamp: Optional[float] = None
        self._latest_rx: Optional[float] = None   # local time.monotonic() of receipt
        self._n_received = 0
        self._n_rejected = 0
        self._last_reject_warn = float("-inf")
        self._rejected_since_warn = 0
        self._lock = threading.Lock()

        try:
            import zmq
            import msgpack
            import msgpack_numpy as _mnp
            _mnp.patch()
        except Exception as exc:  # noqa: BLE001
            raise ImportError(
                "pyzmq + msgpack + msgpack-numpy are required for "
                "RealSenseDepthZmqSubscriber. pip install pyzmq msgpack msgpack-numpy\n"
                f"  {exc!r}"
            )
        self._zmq = zmq
        self._msgpack = msgpack
        self._ctx = zmq.Context.instance()
        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.setsockopt(zmq.CONFLATE, 1)        # keep only newest depth frame
        self._sub.setsockopt(zmq.RCVTIMEO, int(rcvtimeo_ms))
        self._sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self._sub.connect(addr)
        print(f"[depth-zmq-sub] connecting {addr} -> ({self._H}x{self._W}) on {device}", flush=True)

        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._poll_loop, daemon=True)
        self._thr.start()

    def _poll_loop(self):
        while not self._stop.is_set():
            try:
                raw = self._sub.recv()
            except self._zmq.Again:
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"[depth-zmq-sub] recv error: {exc!r}", flush=True)
                continue
            try:
                msg = self._msgpack.unpackb(raw, raw=False)
                depth = np.array(msg["depth"], dtype=np.float32)  # copy: msgpack buffer is read-only
                if depth.shape != (self._H, self._W):
                    self._reject(depth.shape)
                    continue
                t = torch.from_numpy(depth).to(self._device, dtype=torch.float32)
                with self._lock:
                    self._latest = t
                    self._latest_stamp = float(msg.get("t", time.time()))
                    self._latest_rx = time.monotonic()
                    self._n_received += 1
            except Exception as exc:  # noqa: BLE001
                print(f"[depth-zmq-sub] decode error: {exc!r}", flush=True)

    def _reject(self, shape):
        self._n_rejected += 1
        self._rejected_since_warn += 1
        now = time.monotonic()
        if now - self._last_reject_warn >= 1.0:
            print(f"[depth-zmq-sub] WARNING: rejected {self._rejected_since_warn} depth "
                  f"frame(s) of shape {tuple(shape)}; expected ({self._H}, {self._W}). "
                  f"Publisher and subscriber resolutions (intrinsics) disagree.",
                  flush=True)
            self._last_reject_warn = now
            self._rejected_since_warn = 0

    def get_latest(self) -> Optional[torch.Tensor]:
        with self._lock:
            return self._latest

    def get_latest_with_stamp(self):
        with self._lock:
            return self._latest, self._latest_stamp

    def age(self) -> float:
        """Seconds since the newest frame arrived here (inf before the first).
        Local receive time: the publisher's own stamp is another host's clock."""
        with self._lock:
            rx = self._latest_rx
        return float("inf") if rx is None else time.monotonic() - rx

    @property
    def n_received(self) -> int:
        return self._n_received

    @property
    def n_rejected(self) -> int:
        """Frames dropped because their size did not match (height, width)."""
        return self._n_rejected

    def shutdown(self):
        # Stop the poller before closing: ZMQ sockets are not thread-safe.
        self._stop.set()
        self._thr.join(timeout=1.0)
        try:
            self._sub.close(0)
        except Exception:
            pass
