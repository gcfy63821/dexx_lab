# Hand joint ordering (critical for sim2real)

There are **two joint orderings** in the codebase. Mixing them up maps
joints to the wrong motor — on the real hand this is a safety hazard. Read this
before any deploy or retarget work.

## The two orderings

| Name | How it's built | Where it's used |
|------|---------------|-----------------|
| **cfg order** (= Sharpa order = real hand order) | `cfg.actuated_joint_names` list in `franka_sharpa_env_cfg.py` | retarget `opt_dof_pos`, `hand_joint_indices`, real Sharpa `set_joint_position()` |
| **USD order** (= sorted order) | `actuated_dof_indices` after `.sort()` | `cur_targets`, `hand_dof_pos`, policy action space |

## cfg order (= Sharpa order)

This is the canonical order, matching the real Sharpa hand:

```
[0]  thumb_CMC_FE    [5]  index_MCP_FE    [9]  middle_MCP_FE   [13] ring_MCP_FE    [17] pinky_CMC
[1]  thumb_CMC_AA    [6]  index_MCP_AA    [10] middle_MCP_AA   [14] ring_MCP_AA    [18] pinky_MCP_FE
[2]  thumb_MCP_FE    [7]  index_PIP       [11] middle_PIP      [15] ring_PIP       [19] pinky_MCP_AA
[3]  thumb_MCP_AA    [8]  index_DIP       [12] middle_DIP      [16] ring_DIP       [20] pinky_PIP
[4]  thumb_IP                                                                       [21] pinky_DIP
```

## Key variables and their ordering

| Variable | Ordering | Defined in |
|----------|----------|-----------|
| `hand_joint_indices` | cfg order (NOT sorted) | `franka_sharpa_env.py` |
| `actuated_dof_indices` | sorted by USD index | `franka_sharpa_env.py` |
| `cur_targets` | `actuated_dof_indices` (sorted) | `_pre_physics_step` |
| `hand_dof_pos` | `actuated_dof_indices` (sorted) | `_refresh_lab` |
| `real_hand_dof_pos` | `hand_joint_indices` (cfg = Sharpa) | `_refresh_lab` |
| `opt_dof_pos` (retarget pkl) | cfg order (= Sharpa) | retarget output |

## Data flow through the pipeline

```
retarget pkl: opt_dof_pos (cfg order)
    ↓ written via hand_joint_indices
sim articulation: all joints (USD order)
    ↓ read via actuated_dof_indices (sorted)
cur_targets / hand_dof_pos (sorted order) ← policy trains & infers with this
    ↓ written via actuated_dof_indices
sim articulation: all joints (USD order)
    ↓ extracted via hand_joint_indices
real_hand_dof_pos (cfg order = Sharpa order) → send directly to real hand
```

## Deploy: correct way to send commands to the real hand

```python
# In _apply_action:
all_joint_targets[:, actuated_dof_indices] = cur_targets   # sorted → full array
hand_sharpa = all_joint_targets[0, hand_joint_indices]      # full array → cfg order
real_hand.set_joint_position(hand_sharpa)                   # cfg order = Sharpa order
```

There is no third ordering: the old `dof_isaaclab2sharpa()` helper assumed a legacy
order that matches no current variable and has been removed. Always route through the `all_joint_targets`
intermediate array to convert between sorted (policy) order and cfg (Sharpa) order:
scatter the policy's sorted-order targets into a full joint array via
`actuated_dof_indices`, then gather back out in cfg order via `hand_joint_indices`.
