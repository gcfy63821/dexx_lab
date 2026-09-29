"""Depth ZMQ subscriber against a local stand-in publisher; no camera or simulator."""
import importlib.util
from pathlib import Path
import socket
import sys
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
HAVE_DEPS = all(importlib.util.find_spec(m) for m in ("torch", "zmq", "msgpack", "msgpack_numpy"))


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@unittest.skipUnless(HAVE_DEPS, "needs torch + pyzmq + msgpack + msgpack-numpy")
class DepthSubscriberTests(unittest.TestCase):
    def _run(self, shape):
        """Publish frames of `shape` for ~1 s into a 240x320 subscriber."""
        import msgpack
        import msgpack_numpy
        import numpy as np
        import zmq
        from dexx.scripts.deploy.realsense_depth_zmq_subscriber import RealSenseDepthZmqSubscriber
        msgpack_numpy.patch()
        port = free_port()
        pub = zmq.Context.instance().socket(zmq.PUB)
        pub.bind(f"tcp://127.0.0.1:{port}")
        stop = threading.Event()

        def publish():
            while not stop.is_set():
                pub.send(msgpack.packb({"depth": np.full(shape, 0.5, np.float32),
                                        "t": time.time()}))
                time.sleep(0.01)

        thr = threading.Thread(target=publish, daemon=True)
        thr.start()
        sub = RealSenseDepthZmqSubscriber(f"tcp://127.0.0.1:{port}", height=240, width=320)
        try:
            deadline = time.time() + 3.0
            while sub.n_received + sub.n_rejected < 5 and time.time() < deadline:
                time.sleep(0.02)
            return sub.n_received, sub.n_rejected, sub.get_latest(), sub.age()
        finally:
            sub.shutdown()
            stop.set()
            thr.join(timeout=1.0)
            pub.close(0)

    def test_matching_frames_are_accepted(self):
        n_rx, n_rej, latest, age = self._run((240, 320))
        self.assertGreater(n_rx, 0)
        self.assertEqual(n_rej, 0)
        self.assertEqual(tuple(latest.shape), (240, 320))
        self.assertLess(age, 1.0)

    def test_wrong_size_frames_are_rejected_not_cropped(self):
        n_rx, n_rej, latest, age = self._run((480, 640))
        self.assertEqual(n_rx, 0)
        self.assertGreater(n_rej, 0)
        self.assertIsNone(latest)
        self.assertEqual(age, float("inf"))


if __name__ == "__main__":
    unittest.main()
