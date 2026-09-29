"""ROS2 RealSense depth subscriber — runs ON THE INFERENCE PC (optional backend).

Drop-in alternative to ``RealSenseDepthZmqSubscriber`` (same ``get_latest()``,
``age()``, ``n_received``, ``n_rejected``, ``shutdown()``) for a camera driven
by the stock ``realsense2_camera`` ROS2 driver instead of
``deploy/realsense_depth_zmq_pub.py``.

The driver must deliver the same image the ZMQ publisher does: D455 depth at
640x480, x2 decimation -> 320x240, not aligned to color (docs/DEPLOY.md, "ROS2
backend"). Frames of any other size are REJECTED rather than resized: a
different resolution means different intrinsics, and the deploy back-projects
with the 320x240 sim intrinsics.

Accepts 16UC1 (millimetres) and 32FC1 (metres). Spins its own node on a daemon
thread; call ``shutdown()`` when done.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np
import torch

DEFAULT_DEPTH_TOPIC = "/camera/camera/depth/image_rect_raw"


class RealSenseDepthRos2Subscriber:
    def __init__(
        self,
        topic: str = DEFAULT_DEPTH_TOPIC,
        height: int = 240,
        width: int = 320,
        device: str = "cpu",
    ):
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
            from sensor_msgs.msg import Image
        except ImportError as exc:
            raise ImportError(
                "the ROS2 depth backend needs rclpy and sensor_msgs: "
                "`source /opt/ros/humble/setup.bash` before running.\n  %r" % (exc,)
            )
        self._topic = topic
        self._H = int(height)
        self._W = int(width)
        self._device = torch.device(device)
        self._latest: Optional[torch.Tensor] = None
        self._latest_stamp: Optional[float] = None
        self._latest_rx: Optional[float] = None   # local time.monotonic() of receipt
        self._n_received = 0
        self._n_rejected = 0
        self._last_warn = float("-inf")
        self._rejected_since_warn = 0
        self._lock = threading.Lock()

        if not rclpy.ok():
            rclpy.init()
            self._owns_rclpy = True
        else:
            self._owns_rclpy = False
        self._rclpy = rclpy
        self._node = rclpy.create_node("dexx_depth_sub")
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self._node.create_subscription(Image, topic, self._cb, qos)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._spin, daemon=True)
        self._thr.start()
        print(f"[depth-ros2-sub] subscribing {topic} -> ({self._H}x{self._W}) on {device}", flush=True)

    def _spin(self):
        while not self._stop.is_set():
            try:
                self._executor.spin_once(timeout_sec=0.1)
            except Exception as exc:  # noqa: BLE001
                if not self._stop.is_set():
                    print(f"[depth-ros2-sub] executor error: {exc!r}", flush=True)

    def _cb(self, msg):
        try:
            if msg.encoding == "16UC1":
                depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
                depth = depth.astype(np.float32) * 1e-3
            elif msg.encoding == "32FC1":
                depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width).copy()
            else:
                self._warn(f"unsupported encoding {msg.encoding!r}; expected 16UC1 or 32FC1")
                return
            if depth.shape != (self._H, self._W):
                self._n_rejected += 1
                self._rejected_since_warn += 1
                self._warn(f"rejected {self._rejected_since_warn} depth frame(s) of shape "
                           f"{depth.shape}; expected ({self._H}, {self._W}). Configure the driver "
                           f"for 640x480 depth with x2 decimation.", reset=True)
                return
            t = torch.from_numpy(depth).to(self._device, dtype=torch.float32)
            with self._lock:
                self._latest = t
                self._latest_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                self._latest_rx = time.monotonic()
                self._n_received += 1
        except Exception as exc:  # noqa: BLE001
            self._warn(f"decode error: {exc!r}")

    def _warn(self, text: str, reset: bool = False):
        now = time.monotonic()
        if now - self._last_warn >= 1.0:
            print(f"[depth-ros2-sub] WARNING: {text}", flush=True)
            self._last_warn = now
            if reset:
                self._rejected_since_warn = 0

    def get_latest(self) -> Optional[torch.Tensor]:
        with self._lock:
            return self._latest

    def get_latest_with_stamp(self):
        with self._lock:
            return self._latest, self._latest_stamp

    def age(self) -> float:
        """Seconds since the newest frame arrived here (inf before the first)."""
        with self._lock:
            rx = self._latest_rx
        return float("inf") if rx is None else time.monotonic() - rx

    @property
    def n_received(self) -> int:
        return self._n_received

    @property
    def n_rejected(self) -> int:
        return self._n_rejected

    def shutdown(self):
        self._stop.set()
        self._thr.join(timeout=1.0)
        try:
            self._executor.remove_node(self._node)
            self._node.destroy_node()
        except Exception:
            pass
        if self._owns_rclpy:
            try:
                self._rclpy.shutdown()
            except Exception:
                pass
