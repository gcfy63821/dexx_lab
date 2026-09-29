"""Adaptive initialization via rollout state buffer.

During training, high-reward states discovered in rollouts are saved into a
ring buffer. On environment reset, the buffer is sampled (with configurable
probability) as an alternative to the retarget-based demo initialization.

This lets the policy "bootstrap" from empirically better grasp/manipulation
states rather than always re-approaching from (imperfect) retarget frames.

Usage in FrankaSharpaEnv:
    # In __init__, after _build_data():
    if cfg.adaptive_init_enabled:
        from .adaptive_init import AdaptiveStateBuffer
        self._state_buffer = AdaptiveStateBuffer(cfg, self.num_envs, self.device)

    # In _get_rewards(), after computing reward:
    if self._state_buffer is not None:
        self._state_buffer.maybe_capture(env, reward)

    # In _reset_idx(), replace seq_idx logic for selected envs:
    if self._state_buffer is not None:
        seq_idx, use_buffer = self._state_buffer.sample_init(env_ids, seq_idx)
"""

from __future__ import annotations

import torch


class AdaptiveStateBuffer:
    """Ring buffer of high-reward sim states for adaptive reset.

    Each entry stores the minimal state needed to reconstruct the full sim:
      - all_joint_pos:   [num_joints]    robot joint positions (arm + hand)
      - all_joint_vel:   [num_joints]    robot joint velocities
      - obj_root_state:  [13]            object [pos(3), quat(4), lin_vel(3), ang_vel(3)]
      - progress_idx:    scalar          demo trajectory index (for future-frame targets)
      - env_origin:      [3]             per-env scene origin offset
      - reward:          scalar          reward at capture time (used for ranking)
      - demo_env_id:     scalar          which demo this state belongs to (for multi-demo)

    Config fields (all on env.cfg):
      adaptive_init_enabled:    bool   master switch (default False)
      adaptive_init_prob:       float  probability of sampling from buffer vs. demo (default 0.3)
      adaptive_init_buffer_size: int   max entries in the buffer (default 8192)
      adaptive_init_warmup:     int    training steps before buffer is used (default 500)
      adaptive_init_capture_top_k: float  top fraction of per-step rewards to capture (default 0.1)
      adaptive_init_min_progress: int  minimum running_progress_buf before capture (default 10)
    """

    def __init__(self, cfg, num_envs: int, device: str | torch.device):
        self.device = device
        self.num_envs = num_envs
        self.buffer_size = int(getattr(cfg, "adaptive_init_buffer_size", 8192))
        self.prob = float(getattr(cfg, "adaptive_init_prob", 0.3))
        self.warmup = int(getattr(cfg, "adaptive_init_warmup", 500))
        self.top_k_frac = float(getattr(cfg, "adaptive_init_capture_top_k", 0.1))
        self.min_progress = int(getattr(cfg, "adaptive_init_min_progress", 10))

        self._ptr = 0      # write pointer into the ring
        self._size = 0     # valid entries (≤ buffer_size)
        self._step = 0     # global step counter

        # Pre-allocate storage (populated lazily on first capture because
        # num_joints isn't known until the env is fully initialized).
        self._num_joints: int | None = None
        self._all_joint_pos: torch.Tensor | None = None
        self._all_joint_vel: torch.Tensor | None = None
        self._obj_root_state: torch.Tensor | None = None
        self._progress_idx: torch.Tensor | None = None
        self._env_origin: torch.Tensor | None = None
        self._reward: torch.Tensor | None = None
        self._demo_env_id: torch.Tensor | None = None

    # ------------------------------------------------------------------ init
    def _lazy_init(self, num_joints: int):
        B = self.buffer_size
        d = self.device
        self._num_joints = num_joints
        self._all_joint_pos = torch.zeros(B, num_joints, device=d)
        self._all_joint_vel = torch.zeros(B, num_joints, device=d)
        self._obj_root_state = torch.zeros(B, 13, device=d)
        self._progress_idx = torch.zeros(B, dtype=torch.long, device=d)
        self._env_origin = torch.zeros(B, 3, device=d)
        self._reward = torch.full((B,), -1e9, device=d)
        self._demo_env_id = torch.zeros(B, dtype=torch.long, device=d)

    # ------------------------------------------------------------------ capture
    def maybe_capture(self, env, reward: torch.Tensor):
        """Called each step from _get_rewards. Captures top-k reward states.

        Args:
            env: the FrankaSharpaEnv instance (self in the env class).
            reward: per-env reward tensor [num_envs].
        """
        self._step += 1

        # Don't capture too-early states (policy hasn't done anything yet)
        eligible = env.running_progress_buf >= self.min_progress
        if not eligible.any():
            return

        # Lazy init on first call
        if self._num_joints is None:
            self._lazy_init(env.hand.num_joints)

        # Select top_k_frac of eligible envs by reward
        eligible_ids = eligible.nonzero(as_tuple=False).squeeze(-1)
        k = max(1, int(len(eligible_ids) * self.top_k_frac))
        topk_vals, topk_local = torch.topk(reward[eligible_ids], k)
        capture_ids = eligible_ids[topk_local]

        n = len(capture_ids)
        if n == 0:
            return

        # Write into ring buffer
        write_idx = torch.arange(self._ptr, self._ptr + n, device=self.device) % self.buffer_size

        self._all_joint_pos[write_idx] = env.hand.data.joint_pos[capture_ids].detach()
        self._all_joint_vel[write_idx] = env.hand.data.joint_vel[capture_ids].detach()
        if hasattr(env, "object") and env.object is not None:
            self._obj_root_state[write_idx] = env.object.data.root_state_w[capture_ids].detach()
        self._progress_idx[write_idx] = env.progress_buf[capture_ids].detach()
        self._env_origin[write_idx] = env.scene.env_origins[capture_ids].detach()
        self._reward[write_idx] = reward[capture_ids].detach()
        # Track which demo trajectory this env was using (for multi-demo consistency).
        # env_id itself encodes the demo assignment (env i always uses demo i % num_demos).
        self._demo_env_id[write_idx] = capture_ids.detach()

        self._ptr = (self._ptr + n) % self.buffer_size
        self._size = min(self._size + n, self.buffer_size)

    # ------------------------------------------------------------------ sample
    @torch.no_grad()
    def sample_init(
        self,
        env,
        env_ids: torch.Tensor,
        seq_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Decide per-env whether to use buffer or demo init.

        Returns:
            seq_idx:        updated demo frame index (unchanged for buffer envs,
                            but buffer envs ignore it and use the returned state).
            joint_pos:      [len(env_ids), num_joints] or None if no buffer envs.
            joint_vel:      [len(env_ids), num_joints] or None.
            obj_root_state: [len(env_ids), 13] or None.
            use_buffer_mask: [len(env_ids)] bool tensor — True for envs that use buffer.

        The caller should:
          1. Apply demo-based init for ~use_buffer_mask envs (using seq_idx).
          2. For use_buffer_mask envs, write the returned joint_pos/vel/obj state
             to sim and set progress_buf from the returned seq_idx.
        """
        n = len(env_ids)

        # Not ready yet
        if self._size == 0 or self._step < self.warmup:
            mask = torch.zeros(n, dtype=torch.bool, device=self.device)
            return seq_idx, None, None, None, mask

        # Coin flip per env
        use_buffer = torch.rand(n, device=self.device) < self.prob
        buf_count = use_buffer.sum().item()
        if buf_count == 0:
            return seq_idx, None, None, None, use_buffer

        # Sample from buffer, weighted by reward (softmax temperature).
        # Higher-reward states are sampled more frequently.
        valid_rewards = self._reward[: self._size]
        probs = torch.softmax(valid_rewards * 2.0, dim=0)  # temperature=0.5
        sample_idx = torch.multinomial(probs, buf_count, replacement=True)

        # Build output tensors (full-size, only use_buffer positions matter)
        joint_pos = torch.zeros(n, self._num_joints, device=self.device)
        joint_vel = torch.zeros(n, self._num_joints, device=self.device)
        obj_state = torch.zeros(n, 13, device=self.device)

        buf_positions = use_buffer.nonzero(as_tuple=False).squeeze(-1)
        joint_pos[buf_positions] = self._all_joint_pos[sample_idx]
        joint_vel[buf_positions] = self._all_joint_vel[sample_idx]
        obj_state[buf_positions] = self._obj_root_state[sample_idx]

        # Re-base object position: buffer stores world-frame obj state with the
        # original env's scene origin baked in. We need to re-base it to the
        # target env's origin.
        buf_origin = self._env_origin[sample_idx]           # origin of source env
        target_origin = env.scene.env_origins[env_ids[buf_positions]]  # origin of target env
        obj_state[buf_positions, :3] += (target_origin - buf_origin)

        # Set seq_idx for buffer envs so reward targets are correct
        seq_idx = seq_idx.clone()
        seq_idx[buf_positions] = self._progress_idx[sample_idx]

        return seq_idx, joint_pos, joint_vel, obj_state, use_buffer

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict:
        """Return buffer statistics for logging."""
        if self._size == 0:
            return {"buffer_size": 0, "buffer_reward_mean": 0.0}
        valid = self._reward[: self._size]
        return {
            "buffer_size": self._size,
            "buffer_reward_mean": valid.mean().item(),
            "buffer_reward_max": valid.max().item(),
            "buffer_reward_min": valid.min().item(),
            "buffer_step": self._step,
        }
