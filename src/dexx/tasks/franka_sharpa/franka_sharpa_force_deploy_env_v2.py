# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Deploy environment for force (proprio + tactile) policy on real hardware.

Hardware handling:
- Correct joint ordering (hand_joint_indices; see docs/JOINT_ORDERING.md)
- Arm slow interpolation approach (prevents power_limit_violation)
- Emergency stop
- Proper namespace (empty)
- write_joint_state_to_sim for FK sync
- _is_deploy_env flag (skip observation noise)

Adds tactile observations from Sharpa SDK (F6 force + DEFORM contact position).
"""

from __future__ import annotations

import sys
import os
import time
import threading
import select
import termios
import tty
from datetime import datetime

import cv2
import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from .sim2real.real_hand_limits import clamp_to_real_limits_np

if TYPE_CHECKING:
    from .franka_sharpa_env_cfg import FrankaSharpaEnvCfg

from isaaclab.assets import Articulation, RigidObject
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
import isaaclab.sim as sim_utils

from .franka_sharpa_force_env import FrankaSharpaForceEnv
from dexx.tasks.hand_imitation.deploy.polymetis_arm_client import PolymetisArmClient
from dexx.deploy_config import import_sharpa_sdk


class HandSDKWorker:
    """Background worker that serializes Sharpa SDK LTC calls.

    Why: real_hand.set_joint_position / get_states are blocking LTC requests
    that take ~5–35ms each (latency rises with motor load). Calling them
    inline inside the env step would cap the deploy loop far below 30Hz once
    motors are moving. This worker decouples them:

    * set_target(target): non-blocking, latest-write-wins (older pending
      target is overwritten — we never want a stale command queued behind
      a newer one).
    * cached_angles: most recent reading from get_states(), refreshed by the
      worker as fast as the SDK permits.

    The worker thread alternates: send pending target (if any), then refresh
    state. SDK access is serialized on a single thread which keeps the LTC
    client happy and avoids cross-thread state corruption.
    """

    def __init__(self, real_hand, logger=None, state_poll_hz: float = 200.0):
        self._hand = real_hand
        self._logger = logger
        self._pending_target = None
        self._cached_angles = None
        self._last_state_time = None  # time.monotonic() of the newest get_states()
        self._enabled_at = 0.0        # time.monotonic() when polling last started
        self._warned = {}             # call -> (time of last warning, suppressed count)
        self._lock = threading.Lock()
        self._target_event = threading.Event()
        self._stop = False
        self._enabled = False
        self._dropped_targets = 0
        self._target_count = 0
        self._state_count = 0
        # Throttle state polling — without this the worker spins at MHz rate
        # calling get_states(), starving the SDK's internal locks and hanging
        # set_joint_position. 200Hz is far more than enough for a 30Hz env.
        self._state_poll_period = 1.0 / max(state_poll_hz, 1.0)
        self._thread = threading.Thread(target=self._run, daemon=True, name='HandSDKWorker')
        self._thread.start()

    def enable(self, enabled: bool):
        self._enabled = bool(enabled)
        if self._enabled:
            with self._lock:
                # Age is measured from when polling (re)starts, not from the last
                # read before a pause at the reset prompt.
                self._enabled_at = time.monotonic()
                self._dropped_targets = 0
                self._target_count = 0
                self._state_count = 0

    def is_enabled(self) -> bool:
        return self._enabled

    def set_target(self, target_array):
        # target_array: 1-D ndarray-like of length 22 in Sharpa cfg order
        with self._lock:
            if self._pending_target is not None:
                self._dropped_targets += 1
            self._pending_target = list(target_array)
        self._target_event.set()

    def cached_angles(self):
        with self._lock:
            return None if self._cached_angles is None else self._cached_angles.copy()

    def state_age(self) -> float:
        """Seconds since the newest successful get_states() or since polling
        started, whichever is later (inf before the first read)."""
        with self._lock:
            last, since = self._last_state_time, self._enabled_at
        if last is None:
            return float("inf")
        return time.monotonic() - max(last, since)

    def stats(self):
        with self._lock:
            return {
                'targets_sent': self._target_count,
                'states_read':  self._state_count,
                'dropped':      self._dropped_targets,
            }

    def _warn(self, call: str, exc: Exception):
        """At most one line per call per second: this loop runs at 200 Hz."""
        if self._logger is None:
            return
        now = time.monotonic()
        last, n = self._warned.get(call, (0.0, 0))
        if now - last >= 1.0:
            extra = f' (+{n} more in the last second)' if n else ''
            self._logger.warn(f'[HandSDK] {call} failed: {exc}{extra}')
            self._warned[call] = (now, 0)
        else:
            self._warned[call] = (last, n + 1)

    def shutdown(self):
        self._stop = True
        self._target_event.set()
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass

    def _run(self):
        next_state_deadline = time.perf_counter()
        while not self._stop:
            # 1. Send pending target if any (non-blocking pop).
            t = None
            with self._lock:
                if self._pending_target is not None:
                    t = self._pending_target
                    self._pending_target = None
            if t is not None and self._enabled:
                try:
                    self._hand.set_joint_position(t)
                    with self._lock:
                        self._target_count += 1
                except Exception as e:
                    self._warn('set_joint_position', e)

            # 2. Refresh cached state at a bounded rate. Without throttling
            # this loop would hammer get_states() at MHz rate, starving the
            # SDK's internal locks and hanging set_joint_position.
            now = time.perf_counter()
            if self._enabled and now >= next_state_deadline:
                try:
                    s = self._hand.get_states()
                    angles = np.array(s.angles, dtype=np.float32)
                    with self._lock:
                        self._cached_angles = angles
                        self._last_state_time = time.monotonic()
                        self._state_count += 1
                except Exception as e:
                    self._warn('get_states', e)
                # Re-anchor schedule on "now" so a slow get_states call doesn't
                # cause a burst of catch-up polls.
                next_state_deadline = now + self._state_poll_period

            # 3. Wait. If a new target is queued, wake immediately. Otherwise
            # sleep up to the next state-poll deadline.
            if self._enabled:
                wait_until = next_state_deadline
                wait_for = max(0.0, wait_until - time.perf_counter())
                if wait_for > 0:
                    if self._target_event.wait(timeout=wait_for):
                        self._target_event.clear()
            else:
                self._target_event.wait(timeout=0.05)
                self._target_event.clear()


class FrankaSharpaForceDeployEnvV2(FrankaSharpaForceEnv):
    """Deploy environment: force policy (proprio + tactile) on real hardware.

    Inherits force env obs computation, overrides hardware-specific methods.
    """
    cfg: "FrankaSharpaEnvCfg"

    def __init__(self, cfg: "FrankaSharpaEnvCfg", render_mode: str | None = None, **kwargs):
        self.cfg = cfg
        cfg.scene.num_envs = 1

        # Arm: Polymetis via the NUC joint bridge (ZMQ), or ROS2 (experimental).
        # Both clients expose the same interface and raise if no arm state
        # arrives, so everything below starts from real data.
        backend = getattr(cfg, "arm_backend", "polymetis")
        if backend == "polymetis":
            self.arm_client = PolymetisArmClient(
                ip_address=cfg.polymetis_server_ip,
                state_port=int(cfg.polymetis_state_port),
                cmd_port=int(cfg.polymetis_cmd_port),
                kq=cfg.polymetis_kq,
                kqd=cfg.polymetis_kqd,
                side=cfg.hand_side,
            )
            self._arm_hint = "Check polymetis_joint_bridge.py on the NUC and the wired link."
            self.get_logger().info(
                f'arm backend = Polymetis @ {cfg.polymetis_server_ip}:'
                f'{cfg.polymetis_state_port}/{cfg.polymetis_cmd_port}'
            )
        elif backend == "ros2":
            from dexx.tasks.hand_imitation.deploy.ros2_arm_client import Ros2ArmClient
            self.arm_client = Ros2ArmClient(namespace=cfg.ros2_namespace)
            ns = cfg.ros2_namespace.rstrip("/")
            self._arm_hint = (f"Check the robot-side controller, wrist_state_publisher.py and "
                              f"`ros2 topic hz {ns}/joint_states {ns}/franka_wrist_state`.")
            self.get_logger().info(
                f'arm backend = ROS2 (namespace {cfg.ros2_namespace!r}: /joint_states + '
                f'/franka_wrist_state -> /teleop_joint_commands)'
            )
        else:
            raise ValueError(f"cfg.arm_backend must be 'polymetis' or 'ros2', got {backend!r}")

        # Initialize real hand
        self.real_hand = None
        try:
            self._init_real_hand()
        except Exception:
            # The arm is already engaged; do not leave its client running.
            self.arm_client.shutdown()
            raise

        # Worker thread that owns the (blocking) Sharpa LTC SDK calls. Decouples
        # the env step rate from SDK round-trip latency, which spikes from ~5ms
        # at idle to ~30ms+ once motors are loaded — enough to cap the deploy
        # loop far below 30Hz if called inline. The worker accepts
        # latest-target-wins writes (newest write supersedes any pending one) and
        # continuously polls hand state into a cached numpy array readable by
        # the env without blocking. Disabled (=None) until rollout starts; reset
        # phase keeps using direct sync calls so move-to-init can still wait.
        # The client's logging.Logger is thread-safe, so SDK errors inside the
        # worker are logged; persistent failures also age out its state and
        # trip the stale-sensor e-stop.
        self._hand_io = HandSDKWorker(self.real_hand, logger=self.get_logger())

        try:
            super().__init__(cfg, render_mode, **kwargs)
        except BaseException:
            # The arm and hand are already connected; do not leave them running.
            for stop in (self._hand_io.shutdown, self.arm_client.shutdown):
                try:
                    stop()
                except Exception:  # noqa: BLE001
                    pass
            raise

        # Deploy doesn't need 4-substep PhysX: real_hand / real_arm are commanded
        # via SDK / Polymetis (not simulated), and observations only need a one-shot
        # FK refresh from articulation state. Inherited DirectRLEnv.step loops
        # decimation × {sim.step, scene.update, _apply_action} which cost ~30+ ms
        # per env step at num_envs=1 with the full Franka+Sharpa+Object scene.
        # Action-delay buffer counts env steps (not substeps) so the trained
        # 2-step delay still represents ~67 ms at 30 Hz regardless of decimation.
        self.cfg.decimation = 1
        self._is_deploy_env = True  # skip observation noise and contact randomization
        self.wait_reset_object = True
        self.old_terminal_settings = None
        self.quit_requested = False
        self.start_frame = 0
        self.debug_record_enabled = bool(getattr(self.cfg, "deploy_debug_record", True))
        self.debug_record_steps = int(getattr(self.cfg, "deploy_debug_record_steps", 100))
        self.debug_record_dir = str(getattr(self.cfg, "deploy_debug_record_dir", "logs/deploy_debug"))
        self._debug_run_idx = 0
        self._debug_record_active = False
        self._debug_records = []
        self._latest_policy_action = None
        self._latest_hand_targets_sharpa = None

        # Tactile UV mapping: taxel (u, v) -> 3D contact position on the pad.
        #
        # These maps are a per-hand CALIBRATION and none ships for the Sharpa
        # Wave yet. Rather
        # than fail at construction (which would make the whole deploy path
        # unimportable), degrade: contact POSITION is disabled and contact FORCE
        # still works. Point DEXX_TACTILE_MAP_DIR at a directory holding
        # tactileSensor_map_{4F,TH}_point.npy to enable it.
        from dexx.tasks.franka_sharpa.franka_sharpa_env_cfg import get_workspace_root
        _tac_dir = os.environ.get(
            "DEXX_TACTILE_MAP_DIR",
            os.path.join(get_workspace_root(), "assets", "tactile_map"),
        )
        _f4 = os.path.join(_tac_dir, "tactileSensor_map_4F_point.npy")
        _th = os.path.join(_tac_dir, "tactileSensor_map_TH_point.npy")
        if os.path.isfile(_f4) and os.path.isfile(_th):
            self.tac_uv_map = [np.load(_f4)] * 4 + [np.load(_th)]
        else:
            self.tac_uv_map = None
            print(
                f"[deploy] WARNING: tactile UV maps not found under {_tac_dir}. "
                f"Contact POSITION will be reported as zero (force is unaffected). "
                f"Set DEXX_TACTILE_MAP_DIR if you have a calibration for this hand."
            )

        # EMA buffer for tactile force smoothing (match sim's contact_smooth)
        self._prev_tactile_force = torch.zeros(1, 5, device=self.device)

        # Enable non-blocking keyboard input for manual reset ('r' key)
        self._setup_terminal_for_input()

        self.get_logger().info('FrankaSharpaForceDeployEnvV2 initialized')
        self.get_logger().info('  Press "r" during rollout to trigger manual reset')
        self.get_logger().info('  Press "q" during rollout to quit deploy loop')
        if self.debug_record_enabled:
            self.get_logger().info(
                f'  Debug recording enabled: first {self.debug_record_steps} steps to {self.debug_record_dir}'
            )

    # ---- Hardware helpers ----

    def get_logger(self):
        return self.arm_client.get_logger()

    def _sensor_ages(self) -> dict:
        """Seconds since each live input last updated. The hand is only polled
        by the worker during a rollout; reset reads it synchronously."""
        ages = {"arm": self.arm_client.state_age()}
        if self._hand_io.is_enabled():
            ages["hand"] = self._hand_io.state_age()
        return ages

    def _stale_sensors(self) -> dict:
        limit = float(getattr(self.cfg, "deploy_max_sensor_age_s", 0.25))
        return {k: a for k, a in self._sensor_ages().items() if a > limit}

    def close(self):
        """Stop the hand worker, release the arm (its controller holds position)
        and give the terminal back its echo."""
        try:
            self._restore_terminal()
        except Exception:  # noqa: BLE001
            pass
        for fn in (getattr(getattr(self, "_hand_io", None), "shutdown", None),
                   getattr(getattr(self, "arm_client", None), "shutdown", None)):
            try:
                if fn is not None:
                    fn()
            except Exception as e:  # noqa: BLE001
                print(f"[deploy] shutdown error: {e}")
        return super().close()

    def _init_real_hand(self):
        self.real_hand = self._auto_detect_hand()
        if self.real_hand is None:
            raise RuntimeError("No available Sharpa device found")
        self.get_logger().info("Sharpa Wave - Init Hand Running Mode")
        if not self._initialize_hand():
            raise RuntimeError("Failed to initialize hand")
        self.real_hand.start()
        # One-shot dump of SDK parameter keys so we know which knobs are
        # available (e.g. control rate / tactile rate / response window). The
        # SDK's set_parameter_safe(json) interface accepts any of these keys —
        # discovering them is the prerequisite for tuning the hand-side
        # control loop frequency.
        self._dump_sdk_param_info()
        # Optional: apply user-requested SDK parameters from cfg.sharpa_sdk_params
        # (dict[str, value]). Lets us bump the hand control rate without
        # editing code, e.g. cfg.sharpa_sdk_params = {"control_rate": 200}.
        self._apply_user_sdk_params()

    def _dump_sdk_param_info(self):
        try:
            keys = list(self.real_hand.get_param_info_keys())
        except Exception as e:
            self.get_logger().warn(f'[SDK] get_param_info_keys failed: {e}')
            return
        self.get_logger().info('[SDK] Available param keys (' + str(len(keys)) + '):')
        for k in keys:
            self.get_logger().info(f'  - {k}')
        # Highlight rate / frequency / period related keys for quick scanning.
        rate_kw = ('rate', 'freq', 'hz', 'period', 'interval', 'sample',
                   'response', 'jitter', 'ctrl', 'control')
        rate_keys = [k for k in keys if any(w in k.lower() for w in rate_kw)]
        if rate_keys:
            self.get_logger().info('[SDK] Rate/control-related keys:')
            for k in rate_keys:
                try:
                    err, js = self.real_hand.get_parameter([k])
                    self.get_logger().info(f'  >>> {k} = {js} (err={err.code})')
                except Exception as e:
                    self.get_logger().info(f'  >>> {k} (read failed: {e})')

    def _apply_user_sdk_params(self):
        params = getattr(self.cfg, 'sharpa_sdk_params', None)
        if not params:
            return
        import json as _json
        for k, v in dict(params).items():
            payload = _json.dumps({k: v})
            try:
                err = self.real_hand.set_parameter_safe(payload)
                code = getattr(err, 'code', -1)
                if code == 0:
                    self.get_logger().info(f'[SDK] set {k}={v} OK')
                else:
                    self.get_logger().warn(f'[SDK] set {k}={v} failed: code={code}')
            except Exception as e:
                self.get_logger().warn(f'[SDK] set {k}={v} raised: {e}')

    def _auto_detect_hand(self):
        self.get_logger().info("Searching for devices...")
        try:
            manager = import_sharpa_sdk().SharpaWaveManager.get_instance()
            time.sleep(1)
            deadline = time.time() + float(getattr(self.cfg, "deploy_hand_detect_timeout_s", 30.0))
            while True:
                devices = manager.get_all_device_sn()
                if not devices:
                    if time.time() > deadline:
                        self.get_logger().error("No Sharpa device found before the timeout")
                        return None
                    self.get_logger().warn("No available devices found")
                    time.sleep(1)
                    continue
                self.get_logger().info(f"Device found: {devices[0]}")
                return manager.connect(devices[0])
        except Exception as e:
            self.get_logger().error(f"Failed to connect to device: {e}")
            return None

    def _initialize_hand(self):
        err = self.real_hand.set_control_mode(import_sharpa_sdk().ControlMode.POSITION)
        if err.code != 0:
            return False
        err = self.real_hand.set_speed_coeff(0.1)
        if err.code != 0:
            return False
        err = self.real_hand.set_current_coeff(self.cfg.current_coef)
        if err.code != 0:
            return False
        err = self.real_hand.set_control_source(import_sharpa_sdk().ControlSource.SDK)
        if err.code != 0:
            return False
        return True

    # ---- Keyboard helpers ----

    def _setup_terminal_for_input(self):
        if self.old_terminal_settings is None:
            self.old_terminal_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())

    def _restore_terminal(self):
        if self.old_terminal_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_terminal_settings)
            self.old_terminal_settings = None

    def _check_keyboard_command(self) -> str | None:
        if select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            if key == 'r':
                return "reset"
            if key == 'q':
                return "quit"
        return None

    # ---- Debug recording helpers ----

    def _start_debug_recording(self):
        if not self.debug_record_enabled:
            return
        self._debug_run_idx += 1
        self._debug_record_active = True
        self._debug_records = []
        self.get_logger().info(
            f'[DEBUG-REC] Run {self._debug_run_idx} started at frame {self.start_frame}, '
            f'recording first {self.debug_record_steps} steps.'
        )

    def _flush_debug_recording(self, reason: str):
        if not self.debug_record_enabled or len(self._debug_records) == 0:
            return
        try:
            os.makedirs(self.debug_record_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(
                self.debug_record_dir,
                f"deploy_debug_{timestamp}_run{self._debug_run_idx:03d}_frame{self.start_frame:04d}_{reason}.npz",
            )

            data = {}
            keys = self._debug_records[0].keys()
            for key in keys:
                values = [rec[key] for rec in self._debug_records]
                data[key] = np.stack(values, axis=0)
            np.savez_compressed(out_path, **data)
            self.get_logger().info(f'[DEBUG-REC] Saved {len(self._debug_records)} steps to: {out_path}')
        except Exception as e:
            self.get_logger().error(f'[DEBUG-REC] Failed to save debug record: {e}')
        finally:
            self._debug_records = []
            self._debug_record_active = False

    def _record_debug_step(self, obs_dict):
        """Record one step of deploy data. Records the full episode; saves on reset."""
        if not self.debug_record_enabled or not self._debug_record_active:
            return

        try:
            import time as _time
            policy_key = "obs" if "obs" in obs_dict else "policy"
            obs_policy = obs_dict[policy_key][0].detach().cpu().numpy().astype(np.float32)
            obs_priv = obs_dict["priv_info"][0].detach().cpu().numpy().astype(np.float32) if "priv_info" in obs_dict else np.zeros(0, dtype=np.float32)
            obs_hist = obs_dict["proprio_hist"][0].detach().cpu().numpy().astype(np.float32) if "proprio_hist" in obs_dict else np.zeros((0,), dtype=np.float32)

            policy_action = (
                self._latest_policy_action.astype(np.float32)
                if self._latest_policy_action is not None
                else np.zeros((self.cfg.action_space,), dtype=np.float32)
            )
            applied_action = self.actions[0].detach().cpu().numpy().astype(np.float32) if hasattr(self, "actions") else np.zeros((self.cfg.action_space,), dtype=np.float32)
            arm_joint_pos = self.arm_joint_pos[0].detach().cpu().numpy().astype(np.float32)
            arm_joint_vel = self.arm_joint_vel[0].detach().cpu().numpy().astype(np.float32) if hasattr(self, "arm_joint_vel") else np.zeros((7,), dtype=np.float32)
            arm_joint_pos_des = self.arm_joint_pos_des[0].detach().cpu().numpy().astype(np.float32) if hasattr(self, "arm_joint_pos_des") else np.zeros((7,), dtype=np.float32)
            hand_joint_pos_sim_order = self.hand_dof_pos[0].detach().cpu().numpy().astype(np.float32)
            hand_joint_pos_sharpa_real = (
                self._last_hand_angles_sharpa.detach().cpu().numpy().astype(np.float32)
                if hasattr(self, "_last_hand_angles_sharpa")
                else np.zeros((22,), dtype=np.float32)
            )
            hand_target_sharpa = (
                self._latest_hand_targets_sharpa.astype(np.float32)
                if self._latest_hand_targets_sharpa is not None
                else np.zeros((22,), dtype=np.float32)
            )
            contact_force = self.last_contacts[0].detach().cpu().numpy().astype(np.float32)
            contact_pos = (
                self._last_tactile_pos[0].detach().cpu().numpy().astype(np.float32)
                if hasattr(self, "_last_tactile_pos")
                else np.zeros((15,), dtype=np.float32)
            )

            # Per-step timing & profiling snapshot. We accumulate cumulative
            # counters in the env (via [STEP-PROF] / [STATE]) and also stash
            # per-step deltas here so post-hoc analysis doesn't depend on the
            # terminal log.
            now = _time.perf_counter()
            prev_ts = self._debug_records[-1]["timestamp"][0] if self._debug_records else now
            dt_step = float(now - prev_ts)

            io_stats = self._hand_io.stats()
            prev_io = getattr(self, '_debug_prev_io_stats', {'targets_sent': 0, 'states_read': 0, 'dropped': 0})
            self._debug_prev_io_stats = io_stats

            wrist_msgs = int(self.arm_client.wrist_msg_count)
            prev_wrist = getattr(self, '_debug_prev_wrist_msg_count', wrist_msgs)
            self._debug_prev_wrist_msg_count = wrist_msgs

            # E-stop limits for offline analysis (so we don't have to look up
            # the cfg defaults when reviewing a run).
            estop = np.array([
                float(getattr(self.cfg, 'arm_joint_vel_limit',  0.0)),
                float(getattr(self.cfg, 'arm_joint_delta_limit', 0.0)),
                float(getattr(self.cfg, 'hand_joint_delta_limit', 0.0)),
            ], dtype=np.float32)

            self._debug_records.append({
                "timestamp": np.array([now], dtype=np.float64),
                "step_idx": np.array([len(self._debug_records)], dtype=np.int32),
                "progress_buf": np.array([int(self.progress_buf[0].item())], dtype=np.int32),
                # Actions
                "policy_action": policy_action,
                "applied_action": applied_action,
                "arm_joint_pos_des": arm_joint_pos_des,
                "hand_target_sharpa": hand_target_sharpa,
                # Arm state
                "arm_joint_pos": arm_joint_pos,
                "arm_joint_vel": arm_joint_vel,
                # Hand state
                "hand_joint_pos_sim_order": hand_joint_pos_sim_order,
                "hand_joint_pos_sharpa_real": hand_joint_pos_sharpa_real,
                # Wrist state
                "wrist_pos": self.base_pos[0].detach().cpu().numpy().astype(np.float32),
                "wrist_quat": self.base_quat[0].detach().cpu().numpy().astype(np.float32),
                "wrist_lin_vel": self.base_lin_vel[0].detach().cpu().numpy().astype(np.float32),
                "wrist_ang_vel": self.base_ang_vel[0].detach().cpu().numpy().astype(np.float32),
                # Fingertips (from FK via write_joint_state_to_sim)
                "fingertip_pos": self.fingertip_pos[0].detach().cpu().numpy().astype(np.float32),
                # Tactile
                "contact_force": contact_force,
                "contact_pos": contact_pos,
                # Full obs
                "obs_policy": obs_policy,
                "obs_priv_info": obs_priv,
                "obs_proprio_hist": obs_hist.reshape(-1),
                # ---- Diagnostics for sim2real debugging. All are per-step
                # deltas / latest snapshots so a single npz tells the full story without the log.
                "dt_step": np.array([dt_step], dtype=np.float32),
                "loop_hz": np.array([1.0 / dt_step if dt_step > 1e-6 else 0.0], dtype=np.float32),
                "hand_io_targets_sent_step": np.array([io_stats['targets_sent'] - prev_io['targets_sent']], dtype=np.int32),
                "hand_io_states_read_step":  np.array([io_stats['states_read']  - prev_io['states_read']],  dtype=np.int32),
                "hand_io_dropped_step":      np.array([io_stats['dropped']      - prev_io['dropped']],      dtype=np.int32),
                "hand_io_targets_sent_total": np.array([io_stats['targets_sent']], dtype=np.int32),
                "hand_io_states_read_total":  np.array([io_stats['states_read']],  dtype=np.int32),
                "hand_io_dropped_total":      np.array([io_stats['dropped']],      dtype=np.int32),
                "hand_io_enabled":  np.array([1 if self._hand_io.is_enabled() else 0], dtype=np.int32),
                "wrist_msg_step":   np.array([wrist_msgs - prev_wrist], dtype=np.int32),
                "wrist_msg_total":  np.array([wrist_msgs], dtype=np.int32),
                "estop_limits": estop,  # [arm_vel, arm_delta, hand_delta]
                "decim_substeps": np.array([int(self.cfg.decimation)], dtype=np.int32),
                "control_freq":   np.array([float(self.cfg.control_freq)], dtype=np.float32),
            })
        except Exception as e:
            self.get_logger().warn(f'[DEBUG-REC] Failed to record step: {e}')

    def _prompt_start_frame(self, env_ids: torch.Tensor):
        seq_len = int(self.demo_data["seq_len"][env_ids[0]].item())
        max_frame = max(0, seq_len - 1)
        prompt = f'Input start frame [0-{max_frame}] (Enter for 0): '
        while True:
            text = input(prompt).strip()
            if text == "":
                frame = 0
                break
            try:
                frame = int(text)
            except ValueError:
                print(f"Invalid input '{text}'. Please enter an integer.")
                continue
            if 0 <= frame <= max_frame:
                break
            print(f"Frame out of range: {frame}. Expected 0..{max_frame}.")

        self.start_frame = frame
        self.progress_buf[env_ids] = frame
        self.running_progress_buf[env_ids] = 0
        self.get_logger().info(f'Using start frame: {frame}')

    # ---- Scene setup (minimal sim for compatibility) ----

    def _setup_scene(self):
        # Training _setup_scene loads the demo batch and sets has_aux + aux_*
        # bookkeeping for MultiAssetSpawner. Deploy doesn't load demos, so the
        # base _build_data() (which runs right after this) would AttributeError
        # on has_aux. Default to no-aux for deploy.
        self.has_aux = False
        self.aux_object = None
        self.aux_urdf_path_list = []
        self.aux_scale_list = []
        self.aux_present_mask_list = []

        self.hand = Articulation(self.cfg.robot_cfg)
        if getattr(self, '_env0_obj_urdf', None):
            # Branch on asset type, as the training env does: robotool objects
            # prefer a hand-authored tool.usd, which must spawn via UsdFileCfg.
            # Setting it as asset_path on the default UrdfFileCfg yields a prim
            # without RigidBodyAPI ("Failed to find a rigid body ... /object").
            _ext = os.path.splitext(self._env0_obj_urdf)[1].lower()
            if _ext in (".usd", ".usda", ".usdc"):
                _orig = self.cfg.object_cfg.spawn
                self.cfg.object_cfg.spawn = sim_utils.UsdFileCfg(
                    usd_path=self._env0_obj_urdf,
                    rigid_props=getattr(_orig, "rigid_props", None),
                    collision_props=getattr(_orig, "collision_props", None),
                    mass_props=getattr(_orig, "mass_props", None),
                )
                self.get_logger().info(f'[deploy] object asset = USD: {self._env0_obj_urdf}')
            else:
                self.cfg.object_cfg.spawn.asset_path = self._env0_obj_urdf
                self.get_logger().info(f'[deploy] object asset = URDF: {self._env0_obj_urdf}')
        self.object = RigidObject(self.cfg.object_cfg)
        self.table = RigidObject(self.cfg.table_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(physics_material=None))
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions()
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        self.scene.rigid_objects["table"] = self.table

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ---- Tactile ----

    def get_tactile_info(self):
        """Get tactile information from real Sharpa hand sensors.

        Returns:
            force: [1, 5] smoothed contact force per finger
            contact_pos: [1, 15] contact position per finger (3d each)
        """
        force = torch.zeros(5, dtype=torch.float32, device=self.device)
        contact_pos = torch.zeros((5, 3), dtype=torch.float32, device=self.device)
        fill_ch = [None] * 5
        while True:
            if None not in fill_ch:
                break
            for ch in range(5):
                ret = self.real_hand.fetch_tactile_frame(ch, timeout=0.1)
                if ret is None:
                    continue
                fill_ch[ch] = True
                deform_data = ret["content"].get("DEFORM")
                deform = deform_data.reshape(240, 240).astype(np.uint8)
                f6_data = torch.tensor(ret["content"].get("F6"))
                force[ch] = torch.norm(f6_data[:3])
                _, binary = cv2.threshold(deform, 30, 255, cv2.THRESH_BINARY)
                center = self._largest_component_centroid(binary.astype(np.uint8))
                if self.tac_uv_map is not None and center[0] is not None and center[1] is not None:
                    center_pos_ch = self.tac_uv_map[ch][int(center[0]), int(center[1])]
                    contact_pos[ch] = torch.tensor(center_pos_ch[:3]) / 1000.0

        # Config-based processing
        if not self.cfg.enable_contact_pos:
            contact_pos[:] = 0.0
        if not self.cfg.enable_tactile:
            force[:] = 0.0
            contact_pos[:] = 0.0

        # Reorder and mask
        force = torch.flip(force, dims=[0])
        force[self.cfg.disable_tactile_ids] = 0.0
        force = force.reshape(1, -1)
        force *= getattr(self.cfg, 'force_scale', 1.0)
        force[force < self.cfg.contact_threshold] = 0.0
        if self.cfg.binary_contact:
            force = torch.where(force > self.cfg.contact_threshold, 1.0, 0.0)

        # EMA smoothing (match sim's contact_smooth)
        force = self.cfg.contact_smooth * force + (1 - self.cfg.contact_smooth) * self._prev_tactile_force
        self._prev_tactile_force = force.clone()

        contact_pos = torch.flip(contact_pos, dims=[0])
        contact_pos[self.cfg.disable_tactile_ids, :] = 0.0
        contact_pos = contact_pos.reshape(1, -1)

        return force, contact_pos

    def _largest_component_centroid(self, binary_image):
        img = (binary_image > 0).astype(np.uint8)
        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
        if num_labels <= 1:
            return None, None
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = np.argmax(areas) + 1
        return tuple(centroids[largest_idx])

    # ---- Real hardware data ----

    def _refresh_lab(self):
        super()._refresh_lab()

        # Real hand joint positions. When the async hand worker is enabled,
        # read the cached state it polls in the background (non-blocking).
        # During reset (worker disabled) fall back to a direct SDK call.
        if self._hand_io.is_enabled():
            cached = self._hand_io.cached_angles()
            if cached is None:
                # Worker not warmed up yet — do one sync read.
                cached = np.array(self.real_hand.get_states().angles, dtype=np.float32)
        else:
            cached = np.array(self.real_hand.get_states().angles, dtype=np.float32)
        self._cached_hand_angles_np = cached
        hand_angles_sharpa = torch.from_numpy(self._cached_hand_angles_np).to(self.device)
        self._last_hand_angles_sharpa = hand_angles_sharpa.clone()

        full_joints = torch.zeros(self.hand.num_joints, device=self.device)
        full_joints[self.hand_joint_indices] = hand_angles_sharpa
        arm_pos_real = self.arm_client.arm_joint_positions.to(self.device)
        full_joints[self.arm_joint_indices] = arm_pos_real

        # Write to sim for FK
        full_joints_batch = full_joints.unsqueeze(0)
        self.hand.write_joint_state_to_sim(
            full_joints_batch,
            torch.zeros_like(full_joints_batch),
        )

        self.hand_dof_pos = full_joints[self.actuated_dof_indices].unsqueeze(0)
        self.arm_joint_pos = arm_pos_real.unsqueeze(0)

        # Real wrist state in sim's EE frame (right_hand_C_MC, the Sharpa base),
        # in the arm-base frame: both arm clients compute it by FK of the
        # measured joints on the sim URDF. cfg.wrist_pos_offset moves it to
        # env-local; cfg.wrist_quat_offset stays identity.
        wrist_pos, wrist_quat, wrist_lin_vel, wrist_ang_vel = self.arm_client.wrist_state()
        arm_wrist_pos = wrist_pos.to(self.device)
        arm_wrist_quat = wrist_quat.to(self.device)  # [w,x,y,z]

        pos_offset  = torch.tensor(self.cfg.wrist_pos_offset,  dtype=torch.float32, device=self.device)
        quat_offset = torch.tensor(self.cfg.wrist_quat_offset, dtype=torch.float32, device=self.device)

        # Apply position offset (pure translation in world frame)
        self.base_pos = (arm_wrist_pos + pos_offset).unsqueeze(0)

        # Apply quaternion pre-rotation: R_corrected = R_offset ⊗ R_arm
        from isaaclab.utils.math import quat_apply, quat_mul
        self.base_quat = quat_mul(quat_offset.unsqueeze(0), arm_wrist_quat.unsqueeze(0))

        # The constant pre-rotation maps the angular velocity by R_offset;
        # translation leaves the linear velocity unchanged.
        self.base_lin_vel = wrist_lin_vel.unsqueeze(0).to(self.device)
        self.base_ang_vel = quat_apply(
            quat_offset.unsqueeze(0),
            wrist_ang_vel.unsqueeze(0).to(self.device),
        )

        # Use actual elapsed wall-clock time (not nominal 1/control_freq) so a
        # stalled step (e.g. a blocking SDK call) does not produce phantom
        # velocities: the arm keeps tracking the previous target during the
        # stall, and dividing by the nominal period would inflate the velocity
        # and trip the e-stop on a benign motion.
        now_t = time.perf_counter()
        nominal_dt = 1.0 / self.cfg.control_freq
        prev_t = getattr(self, '_prev_refresh_time', None)
        if prev_t is None:
            dt = nominal_dt
        else:
            # Floor at 0.5*nominal so single-frame jitter doesn't explode the
            # FD; longer stalls use the actual dt to keep velocities physical.
            dt = max(nominal_dt * 0.5, now_t - prev_t)
        self._prev_refresh_time = now_t

        # Hand joint velocity from finite differences
        if not hasattr(self, '_prev_hand_dof_pos_deploy'):
            self._prev_hand_dof_pos_deploy = self.hand_dof_pos.clone()
        self.hand_dof_vel = (self.hand_dof_pos - self._prev_hand_dof_pos_deploy) / dt
        self._prev_hand_dof_pos_deploy = self.hand_dof_pos.clone()

        # Arm joint velocity from finite differences
        if not hasattr(self, '_prev_arm_joint_pos_deploy'):
            self._prev_arm_joint_pos_deploy = self.arm_joint_pos.clone()
        self.arm_joint_vel = (self.arm_joint_pos - self._prev_arm_joint_pos_deploy) / dt
        self._prev_arm_joint_pos_deploy = self.arm_joint_pos.clone()

        # Real tactile data → override last_contacts for compute_observations
        tactile_force, tactile_pos = self.get_tactile_info()
        self.last_contacts = tactile_force
        self._last_tactile_pos = tactile_pos  # store for debug

        # Debug: periodic state logging
        if not hasattr(self, '_refresh_debug_count'):
            self._refresh_debug_count = 0
        self._refresh_debug_count += 1
        if self._refresh_debug_count % 30 == 1:  # every 1s @ 30Hz
            force_vals = tactile_force[0].cpu().numpy()
            pos_vals = tactile_pos[0].cpu().numpy().reshape(5, 3)
            wrist = self.base_pos[0].cpu().numpy()
            hand_q = self.hand_dof_pos[0, :5].cpu().numpy()  # first 5 hand joints
            contact_str = ' '.join([f'{v:.2f}' for v in force_vals])
            wrist_msgs = self.arm_client.wrist_msg_count
            wrist_delta = wrist_msgs - getattr(self, '_prev_wrist_msg_count', 0)
            self._prev_wrist_msg_count = wrist_msgs
            io_stats = self._hand_io.stats() if self._hand_io.is_enabled() else None
            io_str = ''
            if io_stats is not None:
                prev = getattr(self, '_prev_io_stats', {'targets_sent': 0, 'states_read': 0, 'dropped': 0})
                io_str = (f' io_send={io_stats["targets_sent"]-prev["targets_sent"]}/s '
                          f'io_read={io_stats["states_read"]-prev["states_read"]}/s '
                          f'io_drop={io_stats["dropped"]-prev["dropped"]}/s')
                self._prev_io_stats = io_stats
            self.get_logger().info(
                f'[STATE #{self._refresh_debug_count}] '
                f'force=[{contact_str}] '
                f'wrist=[{wrist[0]:.3f},{wrist[1]:.3f},{wrist[2]:.3f}] '
                f'arm_msgs/s={wrist_delta}{io_str} '
                f'hand_q0-4=[{" ".join(f"{v:.2f}" for v in hand_q)}]'
            )
            if wrist_delta == 0:
                self.get_logger().error(
                    f'ARM STATE STALE: no arm state in the last 1s. {self._arm_hint}'
                )
            # Print contact positions for fingers that have contact
            for i in range(5):
                if force_vals[i] > 0.01:
                    p = pos_vals[i]
                    self.get_logger().info(
                        f'  finger[{i}] force={force_vals[i]:.3f} pos=[{p[0]:.4f},{p[1]:.4f},{p[2]:.4f}]'
                    )

    # ---- Rewards: not meaningful in deploy ----

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    # ---- Dones: inject manual reset from keyboard ----

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated, time_out = super()._get_dones()
        command = self._check_keyboard_command()
        if command == "reset":
            self.get_logger().info('Manual reset triggered by "r" key')
            terminated[:] = True
        elif command == "quit":
            self.get_logger().info('Quit requested by "q" key')
            self.quit_requested = True
            terminated[:] = True
        return terminated, time_out

    # ---- Emergency stop ----

    def _check_emergency_stop(self, arm_targets, hand_targets_sharpa):
        if not getattr(self.cfg, 'emergency_stop_enabled', True):
            return False
        # The policy acts on its latest observation; if any input stopped
        # updating it is acting blind, so stop before sending the command.
        stale = self._stale_sensors()
        if stale:
            ages = ', '.join(f'{k}={v:.3f}s' for k, v in stale.items())
            self.get_logger().error(
                f'E-STOP: stale sensor data ({ages}) > '
                f'deploy_max_sensor_age_s={getattr(self.cfg, "deploy_max_sensor_age_s", 0.25)}'
            )
            return True
        # NaN compares False against every limit below and np.clip passes it
        # through, so a non-finite target would reach the hardware unchecked.
        arm_ok = bool(torch.isfinite(arm_targets).all())
        hand_ok = bool(np.isfinite(np.asarray(hand_targets_sharpa, dtype=np.float64)).all())
        if not (arm_ok and hand_ok):
            self.get_logger().error(
                f'E-STOP: non-finite target (arm finite={arm_ok}, hand finite={hand_ok})')
            return True
        # DEBUG: track how many times we've run the check, so we can dump
        # per-joint breakdowns for the first few steps after reset.
        self._estop_check_count = int(getattr(self, '_estop_check_count', 0)) + 1
        debug_print_estop = self._estop_check_count <= 3
        if hasattr(self, '_prev_arm_targets_deploy'):
            arm_delta_per_joint = torch.abs(arm_targets - self._prev_arm_targets_deploy)
            arm_delta = arm_delta_per_joint.max().item()
            if debug_print_estop:
                pj = arm_delta_per_joint[0].detach().cpu().numpy()
                tgt = arm_targets[0].detach().cpu().numpy()
                prev = self._prev_arm_targets_deploy[0].detach().cpu().numpy()
                j = int(arm_delta_per_joint[0].argmax().item())
                self.get_logger().info(
                    f'[ESTOP-DBG #{self._estop_check_count}] '
                    f'arm_delta_max={arm_delta:.4f} @joint{j} '
                    f'(target={tgt[j]:+.4f}, prev={prev[j]:+.4f}, delta={tgt[j]-prev[j]:+.4f}) '
                    f'per_joint=[{" ".join(f"{v:.3f}" for v in pj)}]'
                )
            if arm_delta > self.cfg.arm_joint_delta_limit:
                j = int(arm_delta_per_joint[0].argmax().item())
                tgt = arm_targets[0].detach().cpu().numpy()
                prev = self._prev_arm_targets_deploy[0].detach().cpu().numpy()
                self.get_logger().error(
                    f'E-STOP: arm joint delta {arm_delta:.4f} > limit {self.cfg.arm_joint_delta_limit} '
                    f'(joint{j} target={tgt[j]:+.4f} prev={prev[j]:+.4f})'
                )
                return True
        arm_vel = torch.abs(self.arm_joint_vel).max().item() if hasattr(self, 'arm_joint_vel') else 0
        if debug_print_estop:
            self.get_logger().info(
                f'[ESTOP-DBG #{self._estop_check_count}] arm_vel_max={arm_vel:.4f}'
            )
        if arm_vel > self.cfg.arm_joint_vel_limit:
            self.get_logger().error(
                f'E-STOP: arm joint vel {arm_vel:.4f} > limit {self.cfg.arm_joint_vel_limit}')
            return True
        # Hand e-stop should compare command against real measured hand angles,
        # not previous command target, to avoid false positives when hardware lags.
        # Reuse the cached angles read in _refresh_lab earlier this env step
        # (saving one ~25ms blocking SDK round-trip).
        try:
            hand_cur = getattr(self, '_cached_hand_angles_np', None)
            if hand_cur is None:
                hand_cur = np.array(self.real_hand.get_states().angles, dtype=np.float32)
            hand_delta_arr = np.abs(hand_targets_sharpa - hand_cur)
            hand_delta = float(hand_delta_arr.max())
            if hand_delta > self.cfg.hand_joint_delta_limit:
                max_idx = int(np.argmax(hand_delta_arr))
                self.get_logger().error(
                    f'E-STOP: hand joint delta {hand_delta:.4f} > limit {self.cfg.hand_joint_delta_limit} '
                    f'(joint_idx={max_idx}, target={hand_targets_sharpa[max_idx]:.4f}, current={hand_cur[max_idx]:.4f})'
                )
                return True
        except Exception as e:  # noqa: BLE001
            # Without a hand reading the delta check cannot run; acting blind
            # is exactly what the e-stop is for.
            self.get_logger().error(f'E-STOP: failed to read hand states for the e-stop check: {e}')
            return True
        return False

    def _emergency_stop(self):
        self.get_logger().error('EMERGENCY STOP - holding current position')
        stale = self._stale_sensors()
        # Disable async worker so hold-position takes effect immediately
        # (and isn't superseded by a stale pending target).
        self._hand_io.enable(False)
        # Hand: hold at the measured angles; if the SDK is what failed, at the
        # last reading we have.
        try:
            hold = list(self.real_hand.get_states().angles)
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f'E-STOP: hand get_states failed ({e}); holding last reading')
            cached = getattr(self, '_cached_hand_angles_np', None)
            hold = None if cached is None else cached.tolist()
        if hold is not None:
            try:
                self.real_hand.set_joint_position(hold)
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f'E-STOP: hand hold failed: {e}')
        # Arm: hold at the measured pose — unless that measurement is what went
        # stale. Then send nothing: the arm controller keeps tracking the last target,
        # which already passed the checks above, rather than an outdated pose.
        if "arm" in stale:
            self.get_logger().error('E-STOP: arm state stale; not commanding the arm '
                                    '(the arm controller holds its last target)')
        else:
            self.arm_client.publish_arm_joint_pos(self.arm_joint_pos)
        self.reset_buf[:] = 1

    # ---- Actions ----

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Reset the decimation substep counter at the start of each env step so
        # _apply_action knows when it is on the LAST substep (only that substep
        # touches real hardware to avoid 4× Sharpa SDK blocking per env step).
        self._decim_substep = 0

        # First-step baseline refresh. Between _reset_idx (which seeds
        # `arm_joint_pos_des_prev` from `franka_joint_pos` in the base env, and
        # `_prev_arm_targets_deploy` from cur_arm_pos here) and the very first
        # env step, the arm continues impedance-settling toward target_arm_pos
        # (~0.2 rad on j5). That drift would produce a fake `arm_delta` on
        # ACTION #1 (base env's MA blends stale
        # `arm_joint_pos_des_prev` with fresh `arm_joint_pos` → arm_des shifts
        # off measured pose; e-stop check then sees arm_des vs stale
        # `_prev_arm_targets_deploy` and reports big delta).
        #
        # So at the start of the FIRST `_pre_physics_step` after a reset,
        # refresh the measured arm pose ONE FINAL TIME and snapshot it into
        # BOTH MA history slots, so super()'s MA blend has a consistent
        # baseline and the e-stop check sees a clean delta.
        first_step = bool(getattr(self, '_first_action_after_reset', False))
        if first_step:
            try:
                # Snapshot the pre-refresh stale values for the debug print.
                arm_pos_des_prev_before = (
                    self.arm_joint_pos_des_prev[0].detach().cpu().numpy().copy()
                    if hasattr(self, 'arm_joint_pos_des_prev') else None
                )
                prev_targets_before = (
                    self._prev_arm_targets_deploy[0].detach().cpu().numpy().copy()
                    if hasattr(self, '_prev_arm_targets_deploy') else None
                )
                arm_pos_before = (
                    self.arm_joint_pos[0].detach().cpu().numpy().copy()
                    if hasattr(self, 'arm_joint_pos') else None
                )
                self._refresh_lab()
                if hasattr(self, 'arm_joint_pos') and self.arm_joint_pos is not None:
                    arm_now = self.arm_joint_pos.clone()
                    self.arm_joint_pos_des_prev = arm_now.clone()
                    self.arm_joint_pos_des = arm_now.clone()
                    self._prev_arm_targets_deploy = arm_now.clone()
                    # DEBUG: log the snapshot so we can verify the first-step
                    # baseline refresh is doing what we think it is.
                    arm_now_np = arm_now[0].detach().cpu().numpy()
                    import numpy as _np
                    drift_arm_pos = (
                        _np.abs(arm_now_np - arm_pos_before).max()
                        if arm_pos_before is not None else float('nan')
                    )
                    drift_des_prev = (
                        _np.abs(arm_now_np - arm_pos_des_prev_before).max()
                        if arm_pos_des_prev_before is not None else float('nan')
                    )
                    drift_estop = (
                        _np.abs(arm_now_np - prev_targets_before).max()
                        if prev_targets_before is not None else float('nan')
                    )
                    self.get_logger().info(
                        f'[FIRST-STEP-FIX] refreshed arm_now=[{" ".join(f"{v:+.3f}" for v in arm_now_np)}] '
                        f'drift(arm_pos pre→post refresh)={drift_arm_pos:.4f} '
                        f'drift(MA prev → arm_now)={drift_des_prev:.4f} '
                        f'drift(estop prev → arm_now)={drift_estop:.4f}'
                    )
            except Exception as e:
                self.get_logger().warn(f'[first-step baseline refresh] {e}')
            self._first_action_after_reset = False

        super()._pre_physics_step(actions)

        # DEBUG: on the first step, also log what super() actually produced so
        # we can confirm arm_joint_pos_des ≈ arm_now (action=0 in ramp).
        if first_step:
            try:
                arm_des_post = self.arm_joint_pos_des[0].detach().cpu().numpy()
                arm_pos_after = self.arm_joint_pos[0].detach().cpu().numpy()
                import numpy as _np
                self.get_logger().info(
                    f'[FIRST-STEP-FIX] after super() '
                    f'arm_joint_pos=[{" ".join(f"{v:+.3f}" for v in arm_pos_after)}] '
                    f'arm_joint_pos_des=[{" ".join(f"{v:+.3f}" for v in arm_des_post)}] '
                    f'|des - pos|={_np.abs(arm_des_post - arm_pos_after).max():.4f} '
                    f'|des - prev_target|={_np.abs(arm_des_post - self._prev_arm_targets_deploy[0].detach().cpu().numpy()).max():.4f}'
                )
            except Exception as e:
                self.get_logger().warn(f'[first-step-fix post-super log] {e}')

    def _apply_action(self) -> None:
        all_joint_targets = torch.zeros((self.num_envs, self.hand.num_joints), device=self.device)
        all_joint_targets[:, self.actuated_dof_indices] = self.cur_targets
        all_joint_targets[:, self.arm_joint_indices] = self.arm_joint_pos_des

        # Always update sim every substep so PhysX sees up-to-date targets.
        self.hand.set_joint_position_target(all_joint_targets)

        # Decide whether this is the final decimation substep. Real-hand SDK,
        # Polymetis command, and emergency check all happen ONLY here. Real hand
        # cannot react faster than the env control rate (30Hz), so issuing the
        # same target 4× per env step just multiplies LTC blocking latency
        # without any control benefit.
        substep = getattr(self, '_decim_substep', 0)
        self._decim_substep = substep + 1
        is_last_substep = (self._decim_substep >= int(self.cfg.decimation))
        if not is_last_substep:
            return

        hand_targets_sharpa = all_joint_targets[0, self.hand_joint_indices].cpu().numpy()
        # Clamp to real-hand reachable range (physical limits measured on the
        # Sharpa Wave hand). Avoids commanding motors to targets they cannot reach —
        # which otherwise triggers limit_protection / overheat, and diverges sim vs real.
        hand_targets_sharpa = clamp_to_real_limits_np(hand_targets_sharpa)
        self._latest_hand_targets_sharpa = hand_targets_sharpa.copy()

        # Debug: log first few actions and periodically
        if not hasattr(self, '_apply_action_count'):
            self._apply_action_count = 0
        self._apply_action_count += 1
        if self._apply_action_count <= 3 or self._apply_action_count % 60 == 0:
            arm_des = self.arm_joint_pos_des[0].cpu().detach().numpy()
            prev_arm = self._prev_arm_targets_deploy[0].cpu().numpy() if hasattr(self, '_prev_arm_targets_deploy') else None
            delta = np.abs(arm_des - prev_arm).max() if prev_arm is not None else 0
            self.get_logger().info(
                f'[ACTION #{self._apply_action_count}] '
                f'arm_delta={delta:.4f} '
                f'arm_des=[{" ".join(f"{v:.3f}" for v in arm_des)}] '
                f'hand_first5=[{" ".join(f"{v:.3f}" for v in hand_targets_sharpa[:5])}]'
            )

        if self._check_emergency_stop(self.arm_joint_pos_des, hand_targets_sharpa):
            self._emergency_stop()
            return

        # Hand command goes through the async worker (non-blocking, latest-wins)
        # to keep this step from being gated by Sharpa LTC round-trip latency.
        if self._hand_io.is_enabled():
            self._hand_io.set_target(hand_targets_sharpa)
        else:
            self.real_hand.set_joint_position(hand_targets_sharpa.tolist())
        self.prev_targets = self.cur_targets.clone()
        self.arm_client.publish_arm_joint_pos(self.arm_joint_pos_des)

        self._prev_arm_targets_deploy = self.arm_joint_pos_des.clone()
        self._prev_hand_targets_deploy = hand_targets_sharpa.copy()

    # ---- Reset ----

    def _move_to_initial_pose(self, env_ids):
        # Disable async hand worker during reset: ramp loop wants synchronous
        # control to ensure each interpolated target lands in order, and the
        # 2s settle-pause needs blocking confirmation.
        self._hand_io.enable(False)

        # The ramps start from the measured arm pose; an outdated one would make
        # the first target a jump. Wait briefly for fresh state, else refuse.
        limit = float(getattr(self.cfg, "deploy_max_sensor_age_s", 0.25))
        deadline = time.time() + 2.0
        while self.arm_client.state_age() > limit and time.time() < deadline:
            time.sleep(0.05)
        if self.arm_client.state_age() > limit:
            raise RuntimeError(
                f"arm state is {self.arm_client.state_age():.1f}s old — not moving the arm. "
                f"{self._arm_hint}")

        seq_idx = self.progress_buf[env_ids]

        dof_pos = self.demo_data["opt_dof_pos"][env_ids, seq_idx]
        hand_joint_pos_sharpa = dof_pos[0].cpu().numpy()
        # Clamp init frame to real-hand reachable range before sending.
        hand_joint_pos_sharpa_raw = hand_joint_pos_sharpa.copy()
        hand_joint_pos_sharpa = clamp_to_real_limits_np(hand_joint_pos_sharpa)
        _clip_delta = np.abs(hand_joint_pos_sharpa - hand_joint_pos_sharpa_raw)
        if _clip_delta.max() > 1e-4:
            self.get_logger().info(
                f'  [init] clamped {int((_clip_delta > 1e-4).sum())} hand joints to real-hand limits '
                f'(max_clip={_clip_delta.max():.4f} rad @ idx={int(_clip_delta.argmax())})'
            )

        arm_joint_pos = self.demo_data["opt_arm_joint_pos"][env_ids, seq_idx]
        arm_joint_pos_np = arm_joint_pos[0].cpu().numpy()

        obj_transf = self.demo_data["obj_trajectory"][env_ids, seq_idx]
        obj_pos = obj_transf[0, :3, 3].cpu().numpy()

        self.get_logger().info(f'Init frame: {seq_idx[0].item()}')
        self.get_logger().info(f'  Hand joints (first 5): {hand_joint_pos_sharpa[:5]}')
        self.get_logger().info(f'  Arm joints: {arm_joint_pos_np}')
        self.get_logger().info(f'  Object target pos: [{obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}]')

        # Move hand slowly
        self.real_hand.set_speed_coeff(0.1)
        self.real_hand.set_current_coeff(self.cfg.current_coef)
        self.real_hand.set_joint_position(hand_joint_pos_sharpa.tolist())

        # Arm: two-stage interpolation to avoid stabbing the table.
        #
        # Either straight to the demo start, or (operator's choice, below) in
        # two stages via home:
        # Stage A: current pose -> Franka FR3 factory home (q_ready) pose.
        #   q_ready = [0, -pi/4, 0, -3pi/4, 0, pi/2, pi/4] — same pose used by
        #   franka_ros2's "go home" service / desk app's home button. EE is
        #   well above the table (joint4 = -3pi/4) and away from singularities.
        #   We deliberately do NOT use cfg.arm_init_joint_pos here: that's the
        #   sim training init (close to demo start), which on some demos sits
        #   too low and would defeat the purpose of going via home.
        #
        # Stage B: factory home -> demo start pose.
        #
        # Each stage is paced by max joint delta so total time scales with
        # how far we have to travel; both stages run at the same 30Hz publish
        # rate as the rollout for consistency.
        FRANKA_FR3_HOME = np.array(
            [0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4],
            dtype=np.float32,
        )
        start_arm_pos = self.arm_client.arm_joint_positions.cpu().numpy()
        home_arm_pos = FRANKA_FR3_HOME
        target_arm_pos = arm_joint_pos_np

        def _ramp_arm(p_from, p_to, label: str):
            dist = float(np.abs(p_from - p_to).max())
            if dist < 1e-3:
                self.get_logger().info(f'  Arm {label}: already at target (max_dist={dist:.4f}); skipping.')
                self.arm_client.publish_arm_joint_pos(p_to)
                return
            approach_time = max(3.0, dist * 5.0)
            n_steps = int(approach_time * 30)
            self.get_logger().info(
                f'  Arm {label}: ramping {approach_time:.1f}s (max_dist={dist:.3f} rad)...'
            )
            for i in range(n_steps):
                alpha = (i + 1) / n_steps
                interp = p_from * (1 - alpha) + p_to * alpha
                self.arm_client.publish_arm_joint_pos(interp)
                time.sleep(1.0 / 30.0)

        # Route choice. After a rollout the arm is already on the demo
        # trajectory, so going straight to the next init frame is both faster
        # and the normal case; the via-home detour exists for when the arm is
        # somewhere awkward and a straight joint interpolation would sweep the
        # hand across the table. That judgement is the operator's — they can
        # see the arm, this code cannot.
        # Ctrl+C here aborts the reset (propagates); only a closed stdin falls
        # back to the configured route.
        via_home = not bool(getattr(self.cfg, 'deploy_direct_init_move', True))
        if bool(getattr(self.cfg, 'deploy_prompt_home_route', True)):
            try:
                ans = input('  Route via home first? [y/N] (Enter = direct): ').strip().lower()
                via_home = ans.startswith('y')
            except EOFError:
                pass

        if via_home:
            _ramp_arm(start_arm_pos, home_arm_pos, label='-> home (lift)')
            # Stage B starts from the measured pose: the arm lags the last
            # commanded interp (impedance), so this avoids a jump at the seam.
            stage_b_start = self.arm_client.arm_joint_positions.cpu().numpy()
            _ramp_arm(stage_b_start, target_arm_pos, label='-> demo start')
        else:
            _ramp_arm(start_arm_pos, target_arm_pos, label='-> demo start (direct)')
        self.get_logger().info('  Arm at init pose.')

        self.real_hand.set_joint_position(hand_joint_pos_sharpa.tolist())
        time.sleep(2)

        self._debug_dump_init_hand_state(hand_joint_pos_sharpa)

        self.real_hand.set_speed_coeff(self.cfg.speed_coef)
        self.real_hand.set_current_coeff(self.cfg.current_coef)

        # Reset finite-diff history so the first refresh after reset doesn't
        # produce a phantom velocity from the (large) jump in arm/hand state
        # caused by reset itself, which would trip the e-stop (arm_vel >> 2 rad/s)
        # on the very first env step.
        cur_arm_pos = self.arm_client.arm_joint_positions.to(self.device).unsqueeze(0)
        self._prev_arm_joint_pos_deploy = cur_arm_pos.clone()

        # Initialize e-stop prev targets to the MEASURED arm pose, NOT the
        # demo target_arm_pos. After ramp_arm, the real arm settles to
        # target_arm_pos ± impedance steady-state error (~0.1-0.2 rad on j5).
        # If we baseline e-stop on target_arm_pos, ACTION #1 shows a fake
        # ~0.18 rad arm_delta (60% of the 0.3 e-stop margin) even when
        # --action_ramp_steps fully suppresses policy contribution — because
        # the base env's MA smoother seeds arm_joint_pos_des from measured
        # arm_joint_pos (_reset_idx in franka_sharpa_env.py), not target.
        # Using cur_arm_pos here aligns the baseline so first-step delta ≈ 0.
        # NOTE: cur_arm_pos here is still stale by the time the FIRST env step
        # runs (impedance keeps settling during the ~100ms+ between this read
        # and step). The DEFINITIVE refresh happens in _pre_physics_step on
        # the first action — `_first_action_after_reset` flag below triggers it.
        self._prev_arm_targets_deploy = cur_arm_pos.clone()
        self._prev_hand_targets_deploy = hand_joint_pos_sharpa.copy()
        # Trigger a final arm-pose snapshot inside the first _pre_physics_step
        # call AFTER reset (see that override for rationale).
        self._first_action_after_reset = True
        # Reset e-stop debug counter so we get fresh per-joint logs for the
        # first 3 _apply_action calls after this reset.
        self._estop_check_count = 0
        try:
            cur_hand_states = self.real_hand.get_states()
            cur_hand_sharpa = torch.tensor(
                cur_hand_states.angles, dtype=torch.float32, device=self.device
            )
            full_joints = torch.zeros(self.hand.num_joints, device=self.device)
            full_joints[self.hand_joint_indices] = cur_hand_sharpa
            self._prev_hand_dof_pos_deploy = (
                full_joints[self.actuated_dof_indices].unsqueeze(0).clone()
            )
            self._cached_hand_angles_np = np.array(cur_hand_states.angles, dtype=np.float32)
        except Exception as e:
            self.get_logger().warn(f'[init] failed to seed hand FD history: {e}')
        # Drop base velocity FD history; it will be re-seeded on next _refresh_lab
        # from the actual base_pos/base_quat at that time.
        for _attr in ('_prev_base_pos_deploy', '_prev_base_quat_deploy'):
            if hasattr(self, _attr):
                delattr(self, _attr)
        # Drop FD timestamp so the first post-reset velocity is computed against
        # the nominal control period (not the multi-second wallclock gap that
        # the reset prompt + ramp + settle adds, which would otherwise yield
        # near-zero vel and mask real motion).
        if hasattr(self, '_prev_refresh_time'):
            delattr(self, '_prev_refresh_time')
        # Reset rate-limiter deadline so the first post-reset cycle isn't forced
        # to "catch up" multiple periods.
        if hasattr(self, '_step_next_deadline'):
            self._step_next_deadline = time.perf_counter()

        # Re-enable the async hand SDK worker for the rollout phase. From now on
        # _apply_action enqueues hand targets via the worker (non-blocking) and
        # _refresh_lab reads cached angles polled in the background.
        self._hand_io.enable(True)

    def _debug_dump_init_hand_state(self, hand_target_sharpa):
        """Dump per-joint target/actual/limits (all in cfg=Sharpa order) for debugging
        sim-vs-real mismatch at init frame. Call after real hand has settled."""
        try:
            import numpy as _np
            target = _np.asarray(hand_target_sharpa, dtype=_np.float32)

            states = self.real_hand.get_states()
            actual = _np.asarray(states.angles, dtype=_np.float32)
            tracking_err = actual - target

            full_limits = self.hand.root_physx_view.get_dof_limits().cpu().numpy()
            if full_limits.ndim == 3:
                full_limits = full_limits[0]
            hji = _np.asarray(self.hand_joint_indices, dtype=_np.int64)
            lower = full_limits[hji, 0] * self.cfg.dof_limits_scale
            upper = full_limits[hji, 1] * self.cfg.dof_limits_scale
            over_upper = target - upper
            under_lower = lower - target

            names = list(self.cfg.actuated_joint_names)
            n = min(len(names), target.shape[0])

            self.get_logger().info('=' * 78)
            self.get_logger().info('[DEBUG] init-frame hand state (cfg / Sharpa order, radians)')
            self.get_logger().info(
                f"{'idx':>3} {'joint_name':<18} {'target':>8} {'actual':>8} "
                f"{'err':>8} {'lower':>8} {'upper':>8} {'over_up':>8} {'und_lo':>8}"
            )
            for i in range(n):
                self.get_logger().info(
                    f"{i:>3} {names[i]:<18} "
                    f"{target[i]:>8.3f} {actual[i]:>8.3f} {tracking_err[i]:>8.3f} "
                    f"{lower[i]:>8.3f} {upper[i]:>8.3f} "
                    f"{over_upper[i]:>8.3f} {under_lower[i]:>8.3f}"
                )
            self.get_logger().info(
                f"summary: max|err|={_np.abs(tracking_err).max():.4f} rad "
                f"({_np.degrees(_np.abs(tracking_err).max()):.2f} deg), "
                f"mean_err={tracking_err.mean():+.4f}, "
                f"joints_over_upper={(over_upper > 0).sum()}, "
                f"joints_under_lower={(under_lower > 0).sum()}"
            )
            self.get_logger().info('=' * 78)
        except Exception as e:
            self.get_logger().warn(f"[DEBUG] _debug_dump_init_hand_state failed: {e}")

    def _reset_idx(self, env_ids: Sequence[int] | None):
        print(f"Resetting env, {env_ids}")
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Flush any active debug recording from previous rollout before reset.
        if self._debug_record_active and len(self._debug_records) > 0:
            self._flush_debug_recording(reason="reset")

        super()._reset_idx(env_ids)

        if self.wait_reset_object:
            self._restore_terminal()
            try:
                self._prompt_start_frame(env_ids)
                self._move_to_initial_pose(env_ids)

                obj_transf = self.demo_data["obj_trajectory"][env_ids, self.progress_buf[env_ids]]
                obj_pos = obj_transf[0, :3, 3].cpu().numpy()
                self.get_logger().info('=' * 60)
                self.get_logger().info(f'Robot at init frame {self.progress_buf[env_ids[0]].item()}.')
                self.get_logger().info(f'Place object at [{obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}]')
                self.get_logger().info('Press Enter to continue, or type q then Enter to quit...')
                self.get_logger().info('=' * 60)
                line = input().strip().lower()
                if line == "q":
                    self.get_logger().info('Quit requested during reset prompt')
                    self.quit_requested = True
                else:
                    self._start_debug_recording()
            finally:
                self._setup_terminal_for_input()
        else:
            self.real_hand.set_speed_coeff(0.1)
            self.real_hand.set_current_coeff(self.cfg.current_coef)
            self.real_hand.set_joint_position([0] * 22)
            time.sleep(3)
            self.real_hand.set_speed_coeff(self.cfg.speed_coef)
            self.real_hand.set_current_coeff(self.cfg.current_coef)

    # ---- Step with control frequency ----

    def step(self, action: torch.Tensor):
        """Custom deploy step that bypasses DirectRLEnv.step.

        The real arm/hand are commanded via SDK / Polymetis, not simulated, so we do not
        need DirectRLEnv's full physics loop (decimation × {sim.step,
        scene.update, _apply_action} + event_manager + auto-reset wrapper).
        Sim is kept around only for articulation FK — body_pos_w drives the
        fingertip-tracking observations — and we trigger that with a single
        no-render sim.step per env step instead of 4 in the inherited loop.

        Saved overhead (vs super().step() at decimation=1):
          - event_manager.apply (DR — not used at deploy)
          - second _get_observations after auto-reset
          - per-substep scene.write_data_to_sim / scene.update churn
          - any GUI/RTX render path inside DirectRLEnv.step
        """
        period = 1.0 / self.cfg.control_freq
        if not hasattr(self, '_step_next_deadline'):
            self._step_next_deadline = time.perf_counter()

        t0 = time.perf_counter()
        action = action.to(self.device)
        self._latest_policy_action = action[0].detach().cpu().numpy().copy()

        # 1. Action -> targets (sets self.cur_targets, self.arm_joint_pos_des).
        self._pre_physics_step(action)
        t1 = time.perf_counter()

        # 2. Push targets to sim articulation, send to real hardware. Real-side
        # IO goes through the HandSDKWorker (non-blocking) and the Polymetis client.
        self._apply_action()
        t2 = time.perf_counter()

        # 3. Sync sim joint state with the latest real measurements so the FK
        # pass below produces body poses that match reality. Without this the
        # sim would drift away under PD control toward stale targets.
        self._sync_sim_joint_state_from_real()
        t3 = time.perf_counter()

        # 4. One physics step purely as an FK driver (render=False, no GUI).
        # We need this because Articulation.body_pos_w is only refreshed on a
        # PhysX step. With num_envs=1 and joint state already at the real
        # measurement (zero PD error), this is the cheapest possible step.
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=1.0 / self.cfg.control_freq)
        t4 = time.perf_counter()

        # 5. Episode bookkeeping. progress_buf / running_progress_buf are
        # normally advanced by FrankaSharpaEnv.step, which we bypass — so we
        # MUST increment them manually here. Without this the demo target index
        # stays frozen at start_frame for the entire rollout. See compute_observations
        # which reads `cur_idx = self.progress_buf + 1` to pull the demo
        # target for the next frame.
        self.episode_length_buf += 1

        # 6. Observation (calls _refresh_lab + compute_observations internally).
        # Uses progress_buf at its CURRENT value to read demo[progress+1].
        obs_dict = self._get_observations()
        t5 = time.perf_counter()

        # 7. Reward (zero at deploy) and dones (keyboard r/q + progress check).
        reward = self._get_rewards()
        terminated, truncated = self._get_dones()

        # Advance progress AFTER reward/done computation, matching the order
        # in FrankaSharpaEnv.step (rewards at frame t, then bump to t+1).
        self.progress_buf += 1
        if hasattr(self, 'running_progress_buf'):
            self.running_progress_buf += 1

        # 8. Auto-reset on done. Mirrors DirectRLEnv but skips the post-reset
        # _compute_intermediate_values pass (we re-fetch obs which already runs
        # _refresh_lab against the real hardware).
        self.reset_buf = terminated | truncated
        if self.reset_buf.any():
            reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            if reset_env_ids.numel() > 0:
                self._reset_idx(reset_env_ids)
                # Resync sim FK after reset (real hand/arm just moved a lot)
                self._sync_sim_joint_state_from_real()
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                self.scene.update(dt=1.0 / self.cfg.control_freq)
                obs_dict = self._get_observations()

        extras = {}
        t6 = time.perf_counter()

        # 9. Debug recording + rate limiter.
        self._record_debug_step(obs_dict)
        if self.quit_requested and self._debug_record_active and len(self._debug_records) > 0:
            self._flush_debug_recording(reason="quit")

        self._step_next_deadline += period
        sleep_for = self._step_next_deadline - time.perf_counter()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            self._step_next_deadline = time.perf_counter()
        t7 = time.perf_counter()

        # Per-stage timing (1/s). Buckets show exactly where the cycle time
        # goes — if t4-t3 (sim.step FK) dominates, use V3, which replaces it
        # with a pytorch_kinematics FK call.
        if not hasattr(self, '_step_prof'):
            self._step_prof = {k: 0.0 for k in ('pre', 'apply', 'sync', 'fk',
                                                'obs_done_reset', 'rec_sleep', 'cycle')}
            self._step_prof['n'] = 0
            self._step_prof['last_log'] = time.perf_counter()
        self._step_prof['pre']             += t1 - t0
        self._step_prof['apply']           += t2 - t1
        self._step_prof['sync']            += t3 - t2
        self._step_prof['fk']              += t4 - t3
        self._step_prof['obs_done_reset']  += t6 - t4
        self._step_prof['rec_sleep']       += t7 - t6
        self._step_prof['cycle']           += t7 - t0
        self._step_prof['n'] += 1
        if time.perf_counter() - self._step_prof['last_log'] >= 1.0:
            n = max(self._step_prof['n'], 1)
            avg_cycle = self._step_prof['cycle'] / n
            self.get_logger().info(
                f'[STEP-PROF] {n} steps in 1s | '
                f'cycle={avg_cycle*1000:.1f}ms ({1.0/avg_cycle:.1f}Hz) | '
                f'pre={self._step_prof["pre"]/n*1000:.1f} '
                f'apply={self._step_prof["apply"]/n*1000:.1f} '
                f'sync={self._step_prof["sync"]/n*1000:.1f} '
                f'fk={self._step_prof["fk"]/n*1000:.1f} '
                f'obs+done={self._step_prof["obs_done_reset"]/n*1000:.1f} '
                f'rec+sleep={self._step_prof["rec_sleep"]/n*1000:.1f} ms '
                f'(target {period*1000:.1f}ms)'
            )
            for k in ('pre', 'apply', 'sync', 'fk', 'obs_done_reset', 'rec_sleep', 'cycle'):
                self._step_prof[k] = 0.0
            self._step_prof['n'] = 0
            self._step_prof['last_log'] = time.perf_counter()

        return obs_dict, reward, terminated, truncated, extras

    # ---- Sim FK refresh helper ----

    def _sync_sim_joint_state_from_real(self):
        """Write the latest real arm + hand measurements into the sim
        articulation so the next sim.step is a pure FK pass (zero PD error).

        Uses cached angles from the HandSDKWorker when available; falls back
        to a synchronous SDK read only on the first call. Arm comes from the
        Polymetis client (already populated in _refresh_lab, but read here
        directly to keep step() self-contained).
        """
        # Arm joint pos from Polymetis
        try:
            arm_pos = self.arm_client.arm_joint_positions.to(
                self.device, dtype=torch.float32
            )
        except Exception:
            arm_pos = None

        # Hand joint pos from worker cache (preferred) or SDK fallback.
        hand_angles = None
        if hasattr(self, '_cached_hand_angles_np') and self._cached_hand_angles_np is not None:
            hand_angles = self._cached_hand_angles_np
        elif self._hand_io is not None:
            cached = self._hand_io.cached_angles()
            if cached is not None:
                hand_angles = cached
                self._cached_hand_angles_np = cached
        if hand_angles is None:
            try:
                hand_angles = np.array(self.real_hand.get_states().angles, dtype=np.float32)
                self._cached_hand_angles_np = hand_angles
            except Exception:
                hand_angles = None

        # Build full joint vector in sim (USD) order and write.
        full_joints = self.hand.data.joint_pos[0].clone()
        if arm_pos is not None and arm_pos.numel() == len(self.arm_joint_indices):
            full_joints[self.arm_joint_indices] = arm_pos
        if hand_angles is not None:
            hand_t = torch.from_numpy(hand_angles).to(self.device, dtype=torch.float32)
            full_joints[self.hand_joint_indices] = hand_t
        self.hand.write_joint_state_to_sim(
            position=full_joints.unsqueeze(0),
            velocity=torch.zeros_like(full_joints).unsqueeze(0),
        )
