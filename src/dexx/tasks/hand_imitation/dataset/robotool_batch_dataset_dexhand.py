"""
Dataset loader for batch-processed RoboTool data.

Data index format: "rt/{task}/{exp}" e.g. "rt/blue_cup/blue_cup_1"
Supports glob patterns: "rt/blue_cup/*" expands to all experiments under blue_cup/
Legacy format "rt_{exp}" (e.g. "rt_blue_cup_1") is also accepted.

Data lives in: data/robotool_batch/{task}/{exp}/
  - mano_joints.pkl (or mano_joints_corrected.pkl, mano_joints_optimized.pkl)
  - meta.json
  - models/{object_id}/cleaned_mesh_10000.obj

Retarget results saved at: data/retargeting/robotool_batch/mano2{dexhand}/{task}/{exp}@0.pkl
"""

import glob
import json
import os
import pickle
from functools import lru_cache

import numpy as np
import torch
import trimesh
from pytorch3d.structures import Meshes
from termcolor import cprint

from dexx.tasks.hand_imitation.dataset.transform import (
    aa_to_rotmat,
    rotmat_to_aa,
    quat_to_rotmat,
)

from .base import ManipData
from .factory import register_manipdata


def _is_failed_retarget(opt_params: dict) -> bool:
    """Return True if a retargeted pkl is a failure/partial sentinel.

    scripts/retarget.py writes a "partial" pkl (via make_partial_result)
    when a variant fails the reachability gate: reachable=False and the opt_*
    trajectory fields are stored as `False` instead of arrays. Such a pkl must be
    treated like a missing result (fall back to placeholders); using it directly
    torch-ifies `False` into a 0-d tensor and crashes downstream indexing.
    """
    if not isinstance(opt_params, dict):
        return True
    if opt_params.get("reachable", True) is False:
        return True
    # Defensive: even without the flag, a non-array trajectory is unusable.
    owp = opt_params.get("opt_wrist_pos", None)
    if owp is None or owp is False:
        return True
    return getattr(np.asarray(owp), "ndim", 0) < 2


def _parse_rt_index(index: str):
    """Parse robotool_batch index -> (task, exp).

    Formats:
        "rt/blue_cup/blue_cup_1"  -> ("blue_cup", "blue_cup_1")
        "rt_blue_cup_1"           -> ("blue_cup", "blue_cup_1")  [legacy]
    """
    if index.startswith("rt/"):
        # New format: rt/{task}/{exp}
        parts = index[3:].split("/", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        else:
            raise ValueError(
                f"Invalid rt index '{index}'. Expected 'rt/{{task}}/{{exp}}', "
                f"e.g. 'rt/blue_cup/blue_cup_1'"
            )
    elif index.startswith("rt_"):
        # Legacy format: rt_{exp} — guess task by splitting last _digit
        name = index[3:]
        parts = name.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0], name
        else:
            return name, name
    else:
        raise ValueError(f"Invalid rt index '{index}'. Must start with 'rt/' or 'rt_'")


def _normalize_or(vec: torch.Tensor, fallback: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize batched 3D vectors, using fallback when a vector is degenerate."""
    norm = torch.linalg.norm(vec, dim=-1, keepdim=True)
    normalized = vec / torch.clamp(norm, min=eps)
    return torch.where(norm > eps, normalized, fallback)


def _derive_sharpa_wrist_rot_from_mano(
    wrist_pos: torch.Tensor,
    mano_joints: dict[str, torch.Tensor],
    hand_side: str,
    fallback_rot: torch.Tensor,
) -> torch.Tensor:
    """Derive a Sharpa C_MC-like wrist frame from MANO MCP geometry.

    RoboTool's stored ``wrist_orientation`` is not always in the same local
    frame as the exported MANO joints.  The MCP layout is more reliable for
    retargeting: z follows the fingers, y spans pinky/index across the palm,
    and x completes a right-handed frame.
    """
    required = ("index_proximal", "middle_proximal", "ring_proximal", "pinky_proximal")
    if any(k not in mano_joints for k in required):
        return fallback_rot

    rel = {k: mano_joints[k] - wrist_pos for k in required}
    fallback_x = fallback_rot[..., :, 0]
    fallback_y = fallback_rot[..., :, 1]
    fallback_z = fallback_rot[..., :, 2]

    z_raw = rel["index_proximal"] + rel["middle_proximal"] + rel["ring_proximal"] + rel["pinky_proximal"]
    z_axis = _normalize_or(z_raw, fallback_z)

    if hand_side == "left":
        y_raw = rel["pinky_proximal"] - rel["index_proximal"]
    else:
        y_raw = rel["index_proximal"] - rel["pinky_proximal"]
    y_raw = y_raw - torch.sum(y_raw * z_axis, dim=-1, keepdim=True) * z_axis
    y_axis = _normalize_or(y_raw, fallback_y)

    x_axis = _normalize_or(torch.cross(y_axis, z_axis, dim=-1), fallback_x)
    y_axis = _normalize_or(torch.cross(z_axis, x_axis, dim=-1), fallback_y)

    wrist_rot = torch.stack((x_axis, y_axis, z_axis), dim=-1)
    valid = torch.isfinite(wrist_rot).all(dim=(-2, -1)) & (torch.det(wrist_rot) > 0.5)
    return torch.where(valid[:, None, None], wrist_rot, fallback_rot)


def _clamp_rotation_steps(rot_seq: torch.Tensor, max_step_deg: float = 25.0) -> torch.Tensor:
    """Limit frame-to-frame wrist rotation jumps in noisy RoboTool tracks."""
    if rot_seq.shape[0] <= 1:
        return rot_seq

    max_step = max_step_deg * np.pi / 180.0
    smoothed = [rot_seq[0]]
    for i in range(1, rot_seq.shape[0]):
        prev = smoothed[-1]
        target = rot_seq[i]
        delta = prev.transpose(-1, -2) @ target
        delta_aa = rotmat_to_aa(delta.unsqueeze(0))[0]
        angle = torch.linalg.norm(delta_aa)
        if angle.item() > max_step:
            delta_aa = delta_aa * (max_step / torch.clamp(angle, min=1e-8))
            target = prev @ aa_to_rotmat(delta_aa)
        smoothed.append(target)
    return torch.stack(smoothed, dim=0)


def expand_rt_indices(indices: list, data_dir: str = "data/robotool_batch",
                      dataset=None) -> list:
    """Expand rt/ indices into per-experiment keys.

    If a dataset instance is given, each key is further expanded into all of its
    retarget-augmentation entries.
    """
    # Expand task / wildcard / legacy indices into per-experiment keys.
    expanded = []
    for idx in indices:
        if not (idx.startswith("rt/") or idx.startswith("rt_")):
            expanded.append(idx)
            continue
        if "*" not in idx:
            if idx.startswith("rt_"):
                task, exp = _parse_rt_index(idx)
                expanded.append(f"rt/{task}/{exp}")
            elif idx.startswith("rt/") and idx.count("/") == 1:
                task = idx[3:]
                task_dir = os.path.join(data_dir, task)
                if os.path.isdir(task_dir):
                    for exp in sorted(os.listdir(task_dir)):
                        exp_dir = os.path.join(task_dir, exp)
                        if os.path.isdir(exp_dir) and os.path.exists(
                            os.path.join(exp_dir, "mano_joints.pkl")
                        ):
                            expanded.append(f"rt/{task}/{exp}")
                else:
                    cprint(f"[WARN] Task dir not found: {task_dir}", "yellow")
            else:
                expanded.append(idx)
            continue

        if idx.startswith("rt/"):
            rest = idx[3:]
            task = rest.split("/")[0]
        else:
            name = idx[3:]
            task = name.rsplit("_", 1)[0]

        task_dir = os.path.join(data_dir, task)
        if os.path.isdir(task_dir):
            for exp in sorted(os.listdir(task_dir)):
                exp_dir = os.path.join(task_dir, exp)
                if os.path.isdir(exp_dir) and os.path.exists(
                    os.path.join(exp_dir, "mano_joints.pkl")
                ):
                    expanded.append(f"rt/{task}/{exp}")
        else:
            cprint(f"[WARN] Task dir not found: {task_dir}", "yellow")

    # With a dataset instance, also expand into augmentation entries.
    if dataset is not None:
        expanded = dataset.expand_with_all_aug(expanded)

    return expanded

@register_manipdata("robotool_batch_rh")
class RobotoolBatchDatasetRH(ManipData):
    """Right-hand dataset loader for batch-processed RoboTool data."""

    def __init__(
        self,
        *,
        data_dir: str = "data/robotool_batch",
        split: str = "all",
        skip: int = 1,
        device="cuda:0",
        mujoco2gym_transf=None,
        max_seq_len=int(1e10),
        dexhand=None,
        source_fps: float = 30.0,  # RoboTool camera capture rate
        retarget_root: str | None = None,
        verbose=True,
        **kwargs,
    ):
        super().__init__(
            data_dir=data_dir,
            split=split,
            skip=skip,
            device=device,
            mujoco2gym_transf=mujoco2gym_transf,
            max_seq_len=max_seq_len,
            dexhand=dexhand,
            verbose=verbose,
            **kwargs,
        )
        self.hand_side = "right"
        self.source_fps = source_fps
        self.retarget_root = retarget_root or os.path.join(
            "data", "retargeting", "robotool_batch", f"mano2{str(self.dexhand)}"
        )
        self._cache = {}

        # Discover all available sequences
        self.data_pathes = []
        self._seq_map = {}  # Maps "rt/{task}/{exp}" -> (task, exp, dir_path)

        for task in sorted(os.listdir(data_dir)):
            task_dir = os.path.join(data_dir, task)
            if not os.path.isdir(task_dir) or task in ("TODO.md", "batch_process.py"):
                continue
            for exp in sorted(os.listdir(task_dir)):
                exp_dir = os.path.join(task_dir, exp)
                mano_path = os.path.join(exp_dir, "mano_joints.pkl")
                if os.path.isdir(exp_dir) and os.path.exists(mano_path):
                    key = f"rt/{task}/{exp}"
                    self.data_pathes.append(mano_path)
                    self._seq_map[key] = (task, exp, exp_dir)

        # Enumerate all augmentation pkls and build augmentation indices.
        # Key format: "rt/{task}/{exp}@{aug_tag}", where aug_tag is everything
        # after "@" in the pkl name, e.g. "0" or "0_dxp3.9cm_dyn8.1cm".
        self._aug_map = {}  # "rt/{task}/{exp}@{aug_tag}" -> (task, exp, exp_dir, pkl_path)
        self._base_to_aug_map = {}
        if self.dexhand is not None:
            retarget_base = self.retarget_root
            if os.path.isdir(retarget_base):
                for task in sorted(os.listdir(retarget_base)):
                    task_retarget_dir = os.path.join(retarget_base, task)
                    if not os.path.isdir(task_retarget_dir):
                        continue
                    for pkl_file in sorted(glob.glob(os.path.join(task_retarget_dir, "*.pkl"))):
                        if pkl_file.endswith("_hand.pkl"):
                            continue
                        basename = os.path.basename(pkl_file)  # e.g. "blue_cup_1@0.pkl" or "blue_cup_1@0_dxp3.9cm.pkl"
                        # Parse exp and aug_tag; basename format: {exp}@0{suffix}.pkl
                        name_no_ext = basename[:-4]  # strip .pkl
                        if "@" not in name_no_ext:
                            continue
                        at_idx = name_no_ext.index("@")
                        exp = name_no_ext[:at_idx]
                        after_at = name_no_ext[at_idx+1:]  # "0" or "0_dxp3.9cm_dyn8.1cm"
                        aug_tag = after_at  # kept in full, e.g. "0" or "0_dxp3.9cm"

                        base_key = f"rt/{task}/{exp}"
                        if base_key not in self._seq_map:
                            continue

                        _, _, exp_dir = self._seq_map[base_key]
                        aug_key = f"rt/{task}/{exp}@{aug_tag}"
                        self._aug_map[aug_key] = (task, exp, exp_dir, pkl_file)

                        if base_key not in self._base_to_aug_map:
                            self._base_to_aug_map[base_key] = []
                        self._base_to_aug_map[base_key].append(aug_key)

                        if aug_key not in self.data_pathes:
                            self.data_pathes.append(aug_key)

        if self.verbose:
            cprint(
                f"[INFO] RobotoolBatch: found {len(self._seq_map)} sequences, "
                f"{len(self._aug_map)} augmented entries in {data_dir}",
                "green",
            )
    def __len__(self):
        return len(self.data_pathes)
    def expand_with_all_aug(self, indices):
        expanded = []
        for idx in indices:
            # legacy -> new
            if isinstance(idx, str) and idx.startswith("rt_"):
                task, exp = _parse_rt_index(idx)
                idx = f"rt/{task}/{exp}"

            # Already an augmentation entry: keep as is.
            if isinstance(idx, str) and idx in self._aug_map:
                expanded.append(idx)
                continue

            # Base key: expand into every augmentation pkl of that experiment.
            if isinstance(idx, str) and idx in self._base_to_aug_map:
                expanded.extend(sorted(self._base_to_aug_map[idx]))
                continue

            # Otherwise keep unchanged.
            expanded.append(idx)

        return expanded
    @lru_cache(maxsize=None)
    def __getitem__(self, index):
        """Load data by index string like 'rt/blue_cup/blue_cup_1'."""
        assert self.mujoco2gym_transf is not None, "mujoco2gym_transf must be provided"
        aug_pkl_path = None
        if isinstance(index, str) and "@" in index and index in self._aug_map:
            task, exp, seq_dir, aug_pkl_path = self._aug_map[index]
            # Load MANO data as usual; only the retarget pkl comes from aug_pkl_path.
        elif isinstance(index, str) and (index.startswith("rt/") or index.startswith("rt_")):
            # Normalize legacy "rt_blue_cup_1" -> "rt/blue_cup/blue_cup_1"
            if index.startswith("rt_"):
                task, exp = _parse_rt_index(index)
                index = f"rt/{task}/{exp}"
            if index not in self._seq_map:
                raise KeyError(
                    f"Sequence '{index}' not found. Available: {list(self._seq_map.keys())[:10]}..."
                )
            task, exp, seq_dir = self._seq_map[index]
        elif isinstance(index, int):
            seq_dir = os.path.dirname(self.data_pathes[index])
            exp = os.path.basename(seq_dir)
            task = os.path.basename(os.path.dirname(seq_dir))
            index = f"rt/{task}/{exp}"
        else:
            raise ValueError(f"Invalid index type: {type(index)}, value: {index}")

        # Load mano_joints.pkl (prefer corrected > optimized > original)
        mano_path = os.path.join(seq_dir, "mano_joints.pkl")
        for candidate in ("mano_joints_corrected.pkl", "mano_joints_optimized.pkl"):
            candidate_path = os.path.join(seq_dir, candidate)
            if os.path.exists(candidate_path):
                mano_path = candidate_path
                if self.verbose:
                    cprint(f"[INFO] Using {candidate}: {mano_path}", "cyan")
                break
        with open(mano_path, "rb") as f:
            mano_data = pickle.load(f)

        # Load meta.json
        meta_path = os.path.join(seq_dir, "meta.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)

        hand_data = mano_data[self.hand_side]
        original_data = mano_data["original_data"]

        num_frames = len(hand_data["wrist"])

        # Object pose
        tool_object_pose = original_data.get("tool_object_pose")
        if tool_object_pose is None:
            raise KeyError(f"tool_object_pose not found in {mano_path}")
        if not isinstance(tool_object_pose, np.ndarray):
            tool_object_pose = np.array(tool_object_pose)

        max_valid_frame = min(num_frames, tool_object_pose.shape[0]) - 1
        frame_indices = list(range(0, max_valid_frame + 1))[:: self.skip]

        if len(frame_indices) == 0:
            raise ValueError(f"No valid frames for {index}")

        # Extract mano joints
        joint_keys = [
            "index_proximal", "index_intermediate", "index_distal", "index_tip",
            "middle_proximal", "middle_intermediate", "middle_distal", "middle_tip",
            "pinky_proximal", "pinky_intermediate", "pinky_distal", "pinky_tip",
            "ring_proximal", "ring_intermediate", "ring_distal", "ring_tip",
            "thumb_proximal", "thumb_intermediate", "thumb_distal", "thumb_tip",
        ]

        mano_joints = {}
        for key in joint_keys:
            if key in hand_data:
                mano_joints[key] = torch.tensor(
                    np.array(hand_data[key])[frame_indices],
                    device=self.device,
                    dtype=torch.float32,
                )
            else:
                mano_joints[key] = torch.zeros(
                    (len(frame_indices), 3), device=self.device, dtype=torch.float32
                )

        # Wrist position and orientation
        wrist_translation = torch.tensor(
            np.array(hand_data["wrist_translation"])[frame_indices],
            device=self.device,
            dtype=torch.float32,
        )

        wrist_orientation_quat = np.array(hand_data["wrist_orientation"])[frame_indices]
        wrist_orientation_quat_wxyz = np.zeros_like(wrist_orientation_quat)
        wrist_orientation_quat_wxyz[:, 0] = wrist_orientation_quat[:, 3]
        wrist_orientation_quat_wxyz[:, 1:] = wrist_orientation_quat[:, :3]

        wrist_rot_from_quat = quat_to_rotmat(
            torch.tensor(wrist_orientation_quat_wxyz, device=self.device, dtype=torch.float32)
        )
        wrist_pos = wrist_translation.clone()
        if "middle_proximal" in mano_joints:
            middle_pos = mano_joints["middle_proximal"]
            wrist_pos = wrist_pos - (middle_pos - wrist_pos) * 0.25

        wrist_rot = _derive_sharpa_wrist_rot_from_mano(
            wrist_pos=wrist_pos,
            mano_joints=mano_joints,
            hand_side=self.hand_side,
            fallback_rot=wrist_rot_from_quat,
        )
        wrist_rot = _clamp_rotation_steps(wrist_rot)
        if self.dexhand is not None:
            wrist_pos += torch.tensor(self.dexhand.relative_translation, device=self.device)
        # Object trajectory
        if tool_object_pose.ndim == 3 and tool_object_pose.shape[1] == 4 and tool_object_pose.shape[2] == 4:
            obj_trajectory = torch.tensor(
                tool_object_pose[frame_indices],
                device=self.device,
                dtype=torch.float32,
            )
        elif tool_object_pose.ndim == 2 and tool_object_pose.shape[1] == 7:
            tool_pos = tool_object_pose[frame_indices, 4:7]
            tool_quat = tool_object_pose[frame_indices, :4]
            quat_norm = np.linalg.norm(tool_quat, axis=1, keepdims=True)
            tool_quat = tool_quat / (quat_norm + 1e-8)

            tool_quat_wxyz = np.zeros_like(tool_quat)
            tool_quat_wxyz[:, 0] = tool_quat[:, 3]
            tool_quat_wxyz[:, 1:] = tool_quat[:, :3]

            tool_rot = quat_to_rotmat(
                torch.tensor(tool_quat_wxyz, device=self.device, dtype=torch.float32)
            )
            obj_trajectory = torch.zeros(
                (len(frame_indices), 4, 4), device=self.device, dtype=torch.float32
            )
            obj_trajectory[:, :3, :3] = tool_rot
            obj_trajectory[:, :3, 3] = torch.tensor(
                tool_pos, device=self.device, dtype=torch.float32
            )
            obj_trajectory[:, 3, 3] = 1.0
        else:
            raise ValueError(f"Unexpected tool_object_pose shape: {tool_object_pose.shape}")

        # Coordinate rotation: convert from camera frame to z-up frame
        # Rx(+90°) maps camera z-forward to +y (up); Rx(-90°) would flip the object upside-down.
        # After Rx(+90°), new_y = -raw_z, so to zero frame-0 origin:
        # offset_z = raw_z_0 (positive, since new_y = -raw_z + offset_z = 0)
        offset_z = obj_trajectory[0, 2, 3].item()
        # offset_z = 0.0
        # If z_bottom_offset is saved (mesh bottom distance from origin in raw z),
        # add it so the mesh bottom (not origin) sits at the table surface.
        z_bottom_offset = original_data.get("z_bottom_offset", 0.0)
        offset_z += z_bottom_offset
        trans = torch.tensor([0.0, offset_z, 0.0], device=self.device, dtype=torch.float32)
        rot_x_pos90 = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            device=self.device,
            dtype=torch.float32,
        )
        rot_z_90 = torch.tensor(
            [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            device=self.device,
            dtype=torch.float32,
        )
        rot = rot_x_pos90 @ rot_z_90

        obj_trajectory[:, :3, :3] = rot @ obj_trajectory[:, :3, :3]
        obj_trajectory[:, :3, 3] = (rot @ obj_trajectory[:, :3, 3].T).T + trans
        wrist_pos = (rot @ wrist_pos.T).T + trans
        wrist_rot = rot @ wrist_rot
        for k in mano_joints.keys():
            mano_joints[k] = (rot @ mano_joints[k].T).T + trans

        # Load object mesh from centralized models dir
        object_id = meta["object_ids"][0] if meta["object_ids"] else None
        models_dir = os.path.join(self.data_dir, "models")
        obj_mesh_path = os.path.join(models_dir, object_id, "cleaned_mesh_10000.obj") if object_id else None

        if obj_mesh_path and os.path.exists(obj_mesh_path):
            obj_mesh = trimesh.load(obj_mesh_path, process=False)
            mesh = Meshes(
                verts=torch.from_numpy(obj_mesh.vertices[None, ...].astype(np.float32)),
                faces=torch.from_numpy(obj_mesh.faces[None, ...].astype(np.float32)),
            )
            rs_verts_obj = self.random_sampling_pc(mesh)
        else:
            cprint(f"[WARN] Object mesh not found: {obj_mesh_path}", "yellow")
            rs_verts_obj = torch.zeros((1000, 3), device=self.device, dtype=torch.float32)

        # Asset path: prefer hand-authored tool.usd next to the mesh (better
        # collision approximation, e.g. SDF / convex decomposition baked into
        # USD). Fall back to the auto-generated .urdf wrapper around the .obj.
        # Key name stays "obj_urdf_path" for downstream back-compat; the env
        # branches on file extension when spawning.
        if obj_mesh_path:
            usd_candidate = os.path.join(os.path.dirname(obj_mesh_path), "tool.usd")
            if os.path.exists(usd_candidate):
                obj_urdf_path = usd_candidate
            else:
                obj_urdf_path = obj_mesh_path.replace(".obj", ".urdf")
        else:
            obj_urdf_path = ""

        data = {
            "data_path": mano_path,
            "obj_id": object_id or "-1",
            "obj_mesh_path": obj_mesh_path or "",
            "obj_verts": rs_verts_obj,
            "obj_trajectory": obj_trajectory,
            "obj_urdf_path": obj_urdf_path,
            "scene_objs": [],
            "wrist_pos": wrist_pos,
            "wrist_rot": wrist_rot,
            "mano_joints": mano_joints,
        }

        # Process data (velocities, distances, etc.)
        self._process_data_with_fps(data, 0, rs_verts_obj)

        # Load retargeted data if available

        self._load_retargeted(data, index, frame_indices, aug_pkl_path=aug_pkl_path)

        # -------- Aux object sidecar (optional) --------
        # If `aux_object.json` exists next to the pkl, load it and convert the
        # aux pose from visualizer (z-up world, anchored at main frame-0) to
        # env-local frame.
        #   Visualizer ↔ env: xy identical; z differs by env_z_correction.
        #   env_z_correction = offset_z + table_surface_z, where
        #     offset_z = cz0_frame0 + z_bottom_offset   (computed above)
        #     table_surface_z = self.mujoco2gym_transf[2, 3]
        aux_sidecar_path = os.path.join(os.path.dirname(mano_path), "aux_object.json")
        if os.path.exists(aux_sidecar_path):
            import json as _json
            with open(aux_sidecar_path, "r") as f:
                aux_info = _json.load(f)
            aux_obj_id = aux_info.get("aux_obj_id")

            # Resolve aux asset path: prefer tool.usd, fall back to .urdf
            aux_asset_path = ""
            if aux_obj_id:
                aux_dir = os.path.join(self.data_dir, "models", aux_obj_id)
                usd_p = os.path.join(aux_dir, "tool.usd")
                urdf_p = os.path.join(aux_dir, "cleaned_mesh_10000.urdf")
                if os.path.exists(usd_p):
                    aux_asset_path = usd_p
                elif os.path.exists(urdf_p):
                    aux_asset_path = urdf_p
                else:
                    cprint(f"[WARN] aux asset not found for {aux_obj_id}", "yellow")

            table_surface_z = float(self.mujoco2gym_transf[2, 3].item())
            aux_z_correction = float(offset_z) + table_surface_z

            aux_pos_world = torch.tensor(
                aux_info.get("aux_pos_world", [0.0, 0.0, 0.0]),
                dtype=torch.float32, device=self.device,
            )
            aux_pos_env = aux_pos_world.clone()
            aux_pos_env[2] = aux_pos_env[2] + aux_z_correction

            aux_quat = torch.tensor(
                aux_info.get("aux_quat_wxyz_world", [1.0, 0.0, 0.0, 0.0]),
                dtype=torch.float32, device=self.device,
            )

            data["has_aux"] = True
            data["aux_obj_id"] = aux_obj_id or "-1"
            data["aux_obj_urdf_path"] = aux_asset_path
            data["aux_obj_pos"] = aux_pos_env       # [3], env-local frame
            data["aux_obj_quat"] = aux_quat         # [4] wxyz
            data["aux_obj_scale"] = float(aux_info.get("aux_scale", 1.0))
        else:
            data["has_aux"] = False
            data["aux_obj_id"] = ""
            data["aux_obj_urdf_path"] = ""
            data["aux_obj_pos"] = torch.zeros(3, dtype=torch.float32, device=self.device)
            data["aux_obj_quat"] = torch.tensor([1.0, 0.0, 0.0, 0.0],
                                                  dtype=torch.float32, device=self.device)
            data["aux_obj_scale"] = 1.0

        return data

    def _process_data_with_fps(self, data, idx, rs_verts_obj):
        """Process data with correct FPS for time_delta calculation."""
        data["obj_trajectory"] = self.mujoco2gym_transf @ data["obj_trajectory"]
        data["wrist_pos"] = (
            self.mujoco2gym_transf[:3, :3] @ data["wrist_pos"].T
        ).T + self.mujoco2gym_transf[:3, 3]
        data["wrist_rot"] = rotmat_to_aa(self.mujoco2gym_transf[:3, :3] @ data["wrist_rot"])
        for k in data["mano_joints"].keys():
            data["mano_joints"][k] = (
                self.mujoco2gym_transf[:3, :3] @ data["mano_joints"][k].T
            ).T + self.mujoco2gym_transf[:3, 3]

        obj_verts_transf = (
            data["obj_trajectory"][:, :3, :3] @ rs_verts_obj.T[None]
        ).transpose(-1, -2) + data["obj_trajectory"][:, :3, 3][:, None]

        tip_list = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
        tips = torch.cat([data["mano_joints"][t_k][:, None] for t_k in tip_list], dim=1)
        tips_near, _, _, _ = self.ch_dist(tips, obj_verts_transf)
        data["tips_distance"] = torch.sqrt(tips_near)

        time_delta = 1 / (self.source_fps / self.skip)

        data["obj_velocity"] = self.compute_velocity(
            data["obj_trajectory"][:, None, :3, 3], time_delta, guassian_filter=True
        ).squeeze(1)
        data["obj_angular_velocity"] = self.compute_angular_velocity(
            data["obj_trajectory"][:, None, :3, :3], time_delta, guassian_filter=True
        ).squeeze(1)
        data["wrist_velocity"] = self.compute_velocity(
            data["wrist_pos"][:, None], time_delta, guassian_filter=True
        ).squeeze(1)
        data["wrist_angular_velocity"] = self.compute_angular_velocity(
            aa_to_rotmat(data["wrist_rot"][:, None]), time_delta, guassian_filter=True
        ).squeeze(1)
        data["mano_joints_velocity"] = {}
        for k in data["mano_joints"].keys():
            data["mano_joints_velocity"][k] = self.compute_velocity(
                data["mano_joints"][k], time_delta, guassian_filter=True
            )

        if len(data["obj_trajectory"]) > self.max_seq_len:
            for key in [
                "obj_trajectory", "obj_velocity", "obj_angular_velocity",
                "wrist_pos", "wrist_rot", "wrist_velocity", "wrist_angular_velocity",
                "tips_distance",
            ]:
                data[key] = data[key][: self.max_seq_len]
            for k in data["mano_joints"].keys():
                data["mano_joints"][k] = data["mano_joints"][k][: self.max_seq_len]
            for k in data["mano_joints_velocity"].keys():
                data["mano_joints_velocity"][k] = data["mano_joints_velocity"][k][: self.max_seq_len]


    def _recompute_tips_distance_from_opt(self, data, opt_obj_trajectory=None, obj_scale=1.0):
        """Recompute fingertip-to-object distances from retargeted keypoints."""
        opt_joints = data.get("opt_joints_pos", None)
        if opt_joints is None:
            return False

        body_names = data.get(
            "opt_joints_body_names",
            list(self.dexhand.body_names[: opt_joints.shape[1]]),
        )
        if body_names and isinstance(body_names[0], list):
            body_names = body_names[0]

        tip_names = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
        tip_indices = []
        for tip_name in tip_names:
            candidates = []
            for i, body_name in enumerate(body_names):
                try:
                    mano_name = self.dexhand.to_hand(body_name)[0]
                except Exception:
                    continue
                if mano_name == tip_name:
                    candidates.append(i)
            if not candidates:
                if self.verbose:
                    cprint(f"[WARN] Cannot recompute retarget tips_distance: missing {tip_name}", "yellow")
                return False
            fingertip_candidates = [i for i in candidates if "fingertip" in str(body_names[i]).lower()]
            tip_indices.append(fingertip_candidates[-1] if fingertip_candidates else candidates[-1])

        tips = opt_joints[:, tip_indices, :]
        if opt_obj_trajectory is not None:
            obj_trajectory = opt_obj_trajectory
        else:
            obj_trajectory = data["obj_trajectory"].clone()
            if "xy_offset" in data:
                xy_offset = data["xy_offset"].to(device=self.device, dtype=obj_trajectory.dtype)
                xy_offset = xy_offset.reshape(-1)
                obj_trajectory[:, :2, 3] += xy_offset[:2]
            if "z_offset" in data:
                z_offset = data["z_offset"]
                if not isinstance(z_offset, torch.Tensor):
                    z_offset = torch.tensor(z_offset, device=self.device, dtype=obj_trajectory.dtype)
                z_offset = z_offset.to(device=self.device, dtype=obj_trajectory.dtype)
                z_offset = z_offset.reshape(-1)[0]
                obj_trajectory[:, 2, 3] += z_offset
        obj_trajectory = obj_trajectory.to(device=self.device, dtype=tips.dtype)
        obj_verts = data["obj_verts"].to(device=self.device, dtype=tips.dtype)
        if obj_scale != 1.0:
            obj_verts = obj_verts * float(obj_scale)

        if obj_trajectory.shape[0] != tips.shape[0]:
            if self.verbose:
                cprint(
                    f"[WARN] Cannot recompute retarget tips_distance: "
                    f"obj_traj T={obj_trajectory.shape[0]} vs opt_joints T={tips.shape[0]}",
                    "yellow",
                )
            return False

        obj_verts_transf = (
            obj_trajectory[:, :3, :3] @ obj_verts.T[None]
        ).transpose(-1, -2) + obj_trajectory[:, :3, 3][:, None]
        tips_near, _, _, _ = self.ch_dist(tips, obj_verts_transf)
        data["tips_distance"] = torch.sqrt(torch.clamp(tips_near, min=0.0))
        if self.verbose:
            cprint("[RT LOAD] tips_distance recomputed from retargeted opt_joints_pos", "cyan")
        return True

    def _load_retargeted(self, data, index, frame_indices, aug_pkl_path=None):
        if self.dexhand is None:
            return

        # Base index (strip the @aug_tag part).
        base_index = index.split("@")[0] if "@" in index else index
        task, exp = _parse_rt_index(base_index)

        if getattr(self, "use_bimanual_retarget", False):
            dexhand_base = str(self.dexhand).replace("_rh", "").replace("_lh", "")
            retargeted_dir = os.path.join(
                "data", "retargeting", "robotool_batch",
                f"mano2{dexhand_base}_bm",
                task,
            )
        else:
            retargeted_dir = os.path.join(self.retarget_root, task)

        # Prefer the given aug_pkl_path.
        if aug_pkl_path and os.path.exists(aug_pkl_path):
            retargeted_path = aug_pkl_path
            hand_retargeted_path = aug_pkl_path.replace(".pkl", "_hand.pkl")
        else:
            retargeted_path = os.path.join(retargeted_dir, f"{exp}@0.pkl")
            if not os.path.exists(retargeted_path):
                candidates = sorted(glob.glob(os.path.join(retargeted_dir, f"{exp}@0*.pkl")))
                candidates = [c for c in candidates if not c.endswith("_hand.pkl")]
                retargeted_path = candidates[0] if candidates else retargeted_path
            hand_retargeted_path = retargeted_path.replace(".pkl", "_hand.pkl")
        print(f"\n[RT LOAD][{self.hand_side}] index={index}")
        print(f"[RT LOAD][{self.hand_side}] use_bimanual={getattr(self, 'use_bimanual_retarget', False)}")
        print(f"[RT LOAD][{self.hand_side}] retargeted_path={retargeted_path}")
        print(f"[RT LOAD][{self.hand_side}] exists={os.path.exists(retargeted_path)}")
        retarget_obj_traj_for_tips = None
        retarget_obj_scale_for_tips = 1.0
        loaded_valid_retarget = False
        if os.path.exists(retargeted_path):
            opt_params = pickle.load(open(retargeted_path, "rb"))
            print(f"[PKL TOP][{self.hand_side}] keys={list(opt_params.keys())[:20]}")
            # Bimanual retarget pkls are {"meta": ..., "right": ..., "left": ...};
            # single-hand pkls skip this branch.
            if (
                getattr(self, "use_bimanual_retarget", False)
                and isinstance(opt_params, dict)
                and "meta" in opt_params
                and ("right" in opt_params or "left" in opt_params)
            ):
                side_key = "right" if self.hand_side == "right" else "left"
                side_params = opt_params.get(side_key, None)

                if side_params is None:
                    raise KeyError(f"Bimanual retarget pkl missing side '{side_key}': {retargeted_path}")

                # reachable=False is still usable if the core fields are present.
                if (not side_params.get("reachable", True)) and ("opt_arm_joint_pos" not in side_params):
                    raise KeyError(f"Bimanual retarget side '{side_key}' unreachable and no opt_arm_joint_pos: {retargeted_path}")

                opt_params = side_params
                print(f"[RT BM][{self.hand_side}] side_key={side_key}")
                print(f"[RT BM][{self.hand_side}] keys={list(opt_params.keys())[:30]}")
                print(f"[RT BM][{self.hand_side}] opt_dof_pos={np.array(opt_params['opt_dof_pos']).shape}")
                print(f"[RT BM][{self.hand_side}] opt_arm_joint_pos={np.array(opt_params['opt_arm_joint_pos']).shape}")
                print(f"[RT BM][{self.hand_side}] has opt_joints_pos={'opt_joints_pos' in opt_params}")
            # Failure-sentinel guard: scripts/retarget.py writes a "partial"
            # pkl when a variant fails the reachability gate. In it the opt_*
            # trajectories are stored as `False` (a 0-d value) and reachable=False.
            # torch.tensor(False) yields a 0-d tensor and later `[:, None]` indexing
            # crashes with "too many indices for tensor of dimension 0". Treat such a
            # sentinel like a missing file and fall back to zero placeholders below.
            if _is_failed_retarget(opt_params):
                cprint(
                    f"[WARN] Retargeted result unreachable/failed "
                    f"(reachable={opt_params.get('reachable')}); using placeholders: {retargeted_path}",
                    "yellow",
                )
            else:
                data.update({
                    "opt_wrist_pos": torch.tensor(opt_params["opt_wrist_pos"], device=self.device),
                    "opt_wrist_rot": torch.tensor(opt_params["opt_wrist_rot"], device=self.device),
                    "opt_dof_pos": torch.tensor(opt_params["opt_dof_pos"], device=self.device),
                    "opt_arm_joint_pos": torch.tensor(opt_params["opt_arm_joint_pos"], device=self.device),
                    "xy_offset": torch.tensor(opt_params["xy_offset"], device=self.device),
                })
                opt_joints_pos = opt_params.get("opt_joints_pos", None)
                if opt_joints_pos is not None and opt_joints_pos is not False:
                    opt_joints_pos = torch.tensor(opt_joints_pos, device=self.device, dtype=torch.float32)
                    data["opt_joints_pos"] = opt_joints_pos
                    # Retarget saves these positions in dexhand.body_names order.
                    # Keep the names so envs can reorder to Isaac body order.
                    data["opt_joints_body_names"] = list(self.dexhand.body_names[: opt_joints_pos.shape[1]])
                    if opt_params.get("obj_traj", None) is not None and opt_params.get("obj_traj", None) is not False:
                        retarget_obj_traj_for_tips = torch.tensor(
                            opt_params["obj_traj"], device=self.device, dtype=torch.float32
                        )

                if "z_offset" in opt_params:
                    data["z_offset"] = torch.tensor(opt_params["z_offset"], device=self.device)
                if "obj_scale" in opt_params:
                    data["obj_scale"] = float(opt_params["obj_scale"])
                    retarget_obj_scale_for_tips = float(opt_params["obj_scale"])
                if "aug_yaw_deg" in opt_params:
                    data["aug_yaw_deg"] = torch.tensor(opt_params["aug_yaw_deg"], device=self.device)
                loaded_valid_retarget = True

        if not loaded_valid_retarget:
            if self.verbose:
                cprint(f"[WARN] Retargeted data not found: {retargeted_path}", "yellow")
            T = data["wrist_pos"].shape[0]
            data.update({
                "opt_wrist_pos": data["wrist_pos"],
                "opt_wrist_rot": data["wrist_rot"],
                "opt_dof_pos": torch.zeros(
                    [T, self.dexhand.n_dofs], device=self.device
                ),
                "opt_arm_joint_pos": torch.zeros(
                    [T, 7], device=self.device
                ),
                "xy_offset": torch.zeros(2, device=self.device),
                "z_offset": torch.tensor(0.0, device=self.device),
            })

            if os.path.exists(hand_retargeted_path):
                hand_params = pickle.load(open(hand_retargeted_path, "rb"))
                gym_dofs = [
                    'right_index_MCP_FE', 'right_index_MCP_AA', 'right_index_PIP', 'right_index_DIP',
                    'right_middle_MCP_FE', 'right_middle_MCP_AA', 'right_middle_PIP', 'right_middle_DIP',
                    'right_pinky_CMC', 'right_pinky_MCP_FE', 'right_pinky_MCP_AA', 'right_pinky_PIP', 'right_pinky_DIP',
                    'right_ring_MCP_FE', 'right_ring_MCP_AA', 'right_ring_PIP', 'right_ring_DIP',
                    'right_thumb_CMC_FE', 'right_thumb_CMC_AA', 'right_thumb_MCP_FE', 'right_thumb_MCP_AA', 'right_thumb_IP'
                ]
                sim_dofs = [
                    'right_index_MCP_FE', 'right_middle_MCP_FE', 'right_pinky_CMC', 'right_ring_MCP_FE', 'right_thumb_CMC_FE',
                    'right_index_MCP_AA', 'right_middle_MCP_AA', 'right_pinky_MCP_FE', 'right_ring_MCP_AA', 'right_thumb_CMC_AA',
                    'right_index_PIP', 'right_middle_PIP', 'right_pinky_MCP_AA', 'right_ring_PIP', 'right_thumb_MCP_FE',
                    'right_index_DIP', 'right_middle_DIP', 'right_pinky_PIP', 'right_ring_DIP', 'right_thumb_MCP_AA', 'right_pinky_DIP',
                    'right_thumb_IP'
                ]
                gym_to_sim_idx = [gym_dofs.index(name) for name in sim_dofs]
                opt_dofs_gym = torch.tensor(hand_params["opt_dof_pos"], device=self.device)
                data["opt_dof_pos"] = opt_dofs_gym[:, gym_to_sim_idx]
                if "opt_wrist_pos" in hand_params:
                    data["opt_wrist_pos"] = torch.tensor(hand_params["opt_wrist_pos"], device=self.device)
                    data["opt_wrist_rot"] = torch.tensor(hand_params["opt_wrist_rot"], device=self.device)

        time_delta = 1 / (self.source_fps / self.skip)
        data["opt_wrist_velocity"] = self.compute_velocity(
            data["opt_wrist_pos"][:, None], time_delta, guassian_filter=True
        ).squeeze(1)
        data["opt_wrist_angular_velocity"] = self.compute_angular_velocity(
            aa_to_rotmat(data["opt_wrist_rot"][:, None]), time_delta, guassian_filter=True
        ).squeeze(1)
        data["opt_dof_velocity"] = self.compute_dof_velocity(
            data["opt_dof_pos"], time_delta, guassian_filter=True
        )
        if "opt_joints_pos" in data:
            data["opt_joints_velocity"] = self.compute_velocity(
                data["opt_joints_pos"], time_delta, guassian_filter=True
            )
            self._recompute_tips_distance_from_opt(
                data,
                opt_obj_trajectory=retarget_obj_traj_for_tips,
                obj_scale=retarget_obj_scale_for_tips,
            )

        if len(data["opt_wrist_pos"]) > self.max_seq_len:
            for key in [
                "opt_wrist_pos", "opt_wrist_rot", "opt_wrist_velocity",
                "opt_wrist_angular_velocity", "opt_dof_pos", "opt_dof_velocity",
                "opt_joints_pos", "opt_joints_velocity", "tips_distance",
            ]:
                if key in data:
                    data[key] = data[key][: self.max_seq_len]
            if "opt_arm_joint_pos" in data:
                data["opt_arm_joint_pos"] = data["opt_arm_joint_pos"][: self.max_seq_len]

        assert len(data["opt_wrist_pos"]) == len(data["obj_trajectory"])


@register_manipdata("robotool_batch_lh")
class RobotoolBatchDatasetLH(RobotoolBatchDatasetRH):
    """Left hand version."""

    def __init__(self, **kwargs):
        kwargs["hand_side"] = "left"
        super().__init__(**kwargs)
        self.hand_side = "left"
