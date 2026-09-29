#!/usr/bin/env python3
"""Convert teacher rollouts (collect_reference_rollouts.py) into retarget-schema
reference pkls, selecting the best top-K per demo.

Self-imitation reference refinement: each selected rollout is written in the SAME
schema as the original retarget pkl (so the existing loader reads it 1:1), with the
trajectory fields overwritten by the rollout's ACHIEVED states:

    opt_wrist_pos       <- wrist_pos            (T,3)  env-local
    opt_wrist_rot       <- quat_to_aa(wrist_quat_wxyz)  (T,3) axis-angle
    opt_dof_pos         <- hand_dof_pos_cfg     (T,22) Sharpa/cfg order
    opt_arm_joint_pos   <- arm_joint_pos        (T,7)
    opt_joints_pos      <- opt_joints_pos       (T,K,3) hand keypoints
    opt_joints_body_names <- meta hand_body_names
    obj_traj            <- SE3(object_pos, object_quat_wxyz)  (T,4,4) env-local
                           (rollout object trajectory -> used for reference AND
                            object init once the loader override is on)

Meta fields (xy_offset, z_offset, obj_scale, arm_base_pos, side) are kept from the
original pkl; reachable is forced True; per-frame loss arrays are zeroed to the new T.

Usage:
    python scripts/build_rollout_references.py \
        --rollout_dir data/recordings/ref_rollouts_YYYYMMDD \
        --orig_root  data/retargeting/robotool_batch/mano2sharpa_rh \
        --out_root   data/retargeting/robotool_batch/mano2sharpa_rh_rollout \
        --top_k 3 --min_length 80

Output: {out_root}/{task}/{exp}@0.pkl (rank 0), {exp}@0_r1.pkl, ... + data_idx.json + summary.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from dexx.tasks.hand_imitation.dataset.transform import quat_to_aa, quat_to_rotmat  # noqa: E402


def _np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def base_index(data_idx: str) -> str:
    """rt/0416_grasp/cube_small_2@0_dxn0.5cm... -> rt/0416_grasp/cube_small_2"""
    return data_idx.split("@", 1)[0]


def smoothness_cost(wrist_pos: np.ndarray, arm_joint_pos: np.ndarray) -> float:
    """Lower = smoother. Mean |jerk| of wrist pos + arm joints (3rd difference)."""
    def jerk(x):
        if x.shape[0] < 4:
            return 0.0
        return float(np.abs(np.diff(x, n=3, axis=0)).mean())
    return jerk(wrist_pos) + jerk(arm_joint_pos)


PER_FRAME_KEYS = ("wrist_pos", "wrist_quat_wxyz", "hand_dof_pos_cfg", "arm_joint_pos",
                  "opt_joints_pos", "object_pos", "object_quat_wxyz")


def build_obj_traj(object_pos: np.ndarray, object_quat_wxyz: np.ndarray) -> np.ndarray:
    """(T,3)+(T,4 wxyz) -> (T,4,4) SE3 (env-local)."""
    T = object_pos.shape[0]
    R = quat_to_rotmat(torch.from_numpy(object_quat_wxyz)).numpy().astype(np.float32)  # (T,3,3)
    M = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))
    M[:, :3, :3] = R
    M[:, :3, 3] = object_pos
    return M


def demo_frame_index(progress: np.ndarray, n_demo: int) -> np.ndarray:
    """For each demo frame t, the rollout frame recorded at (or last before) t.

    The loader requires a reference to have exactly the demo's length, and a
    rollout is shorter (it ends on success or timeout) and may repeat a
    progress value at its start. Frames before the first recorded one take the
    first, frames after the last take the last."""
    progress = np.maximum.accumulate(np.asarray(progress).reshape(-1))
    idx = np.searchsorted(progress, np.arange(n_demo), side="right") - 1
    return np.clip(idx, 0, len(progress) - 1)


def convert_one(rollout: dict, orig_pkl: dict, kp_body_names) -> dict:
    ref = dict(orig_pkl)  # shallow copy of the original retarget reference
    T = len(orig_pkl["opt_wrist_pos"])             # the demo's length, which the loader requires
    sel = demo_frame_index(_np(rollout["progress_buf"]), T)
    rollout = {k: (_np(v)[sel] if k in PER_FRAME_KEYS else v) for k, v in rollout.items()}
    wp = rollout["wrist_pos"]                      # (T,3)
    wq = rollout["wrist_quat_wxyz"]                # (T,4) wxyz

    ref["opt_wrist_pos"] = wp
    ref["opt_wrist_rot"] = quat_to_aa(torch.from_numpy(wq)).numpy().astype(np.float32)  # (T,3) aa
    ref["opt_dof_pos"] = _np(rollout["hand_dof_pos_cfg"])          # (T,22)
    ref["opt_arm_joint_pos"] = _np(rollout["arm_joint_pos"])       # (T,7)
    if "opt_joints_pos" in rollout:
        ref["opt_joints_pos"] = _np(rollout["opt_joints_pos"])     # (T,K,3)
        if kp_body_names is not None:
            ref["opt_joints_body_names"] = list(kp_body_names)
    ref["obj_traj"] = build_obj_traj(_np(rollout["object_pos"]), _np(rollout["object_quat_wxyz"]))

    # unclamped copy some loaders expect
    ref["opt_dof_pos_unclamped"] = ref["opt_dof_pos"].copy()
    ref["reachable"] = True
    # zero per-frame loss arrays to the new length (original ones have the old T)
    for lk in ("opt_final_loss_per_frame", "opt_final_hand_loss_per_frame",
               "opt_final_arm_pos_loss_per_frame", "opt_final_arm_rot_loss_per_frame"):
        if lk in ref:
            ref[lk] = np.zeros((T,), dtype=np.float32)
    if "opt_final_loss_mean" in ref:
        ref["opt_final_loss_mean"] = np.float32(0.0)
    # provenance
    ref["source"] = "rollout"
    ref["rollout_data_idx"] = rollout.get("data_idx", "")
    ref["rollout_succeeded"] = float(rollout.get("succeeded", 0.0))
    return ref


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollout_dir", type=str, nargs="+", required=True,
                   help="dir(s) containing rollout_*.pkl + meta.json")
    p.add_argument("--orig_root", type=str,
                   default="data/retargeting/robotool_batch/mano2sharpa_rh")
    p.add_argument("--out_root", type=str,
                   default="data/retargeting/robotool_batch/mano2sharpa_rh_rollout")
    p.add_argument("--top_k", type=int, default=3, help="successful rollouts to keep for a GOOD demo")
    p.add_argument("--min_success", type=int, default=3,
                   help="a demo is GOOD (rollout-only) if it has >= this many successful rollouts; "
                        "otherwise POOR: keep best-reward rollout(s) + the original retarget ref.")
    p.add_argument("--fallback_k", type=int, default=1,
                   help="best-reward rollouts to keep for a POOR demo (alongside the original).")
    p.add_argument("--no_orig_for_good", action="store_true", default=False,
                   help="GOOD demos: rollout-only (do NOT also mix in the original ref).")
    p.add_argument("--min_length", type=int, default=80)
    p.add_argument("--base_demos", type=str, default=None,
                   help="JSON file OR inline JSON list of ALL demos to cover. Demos with no "
                        "rollout get the original retarget ref copied in (so training covers all).")
    args = p.parse_args()

    # ---- Load all rollouts, group by base demo ----
    kp_body_names = None
    by_demo: dict[str, list[dict]] = {}
    n_loaded = 0
    for rdir in args.rollout_dir:
        mpath = os.path.join(rdir, "meta.json")
        if kp_body_names is None and os.path.exists(mpath):
            meta = json.load(open(mpath))
            kp_body_names = meta.get("opt_joints_body_names", None)
        for fp in sorted(glob.glob(os.path.join(rdir, "rollout_*.pkl"))):
            r = pickle.load(open(fp, "rb"))
            r["_file"] = fp
            by_demo.setdefault(base_index(r["data_idx"]), []).append(r)
            n_loaded += 1
    print(f"[build] loaded {n_loaded} rollouts across {len(by_demo)} base demos; "
          f"kp_body_names={'set' if kp_body_names else 'MISSING'}")

    # Full demo list to cover (union of rollout demos + requested base_demos).
    base_list = set(by_demo.keys())
    if args.base_demos:
        if os.path.exists(args.base_demos):
            bd = json.load(open(args.base_demos))
        else:
            bd = json.loads(args.base_demos)
        base_list |= {base_index(x) for x in bd}

    produced = []
    summary = []
    for base in sorted(base_list):
        _, task, exp = base.split("/")
        orig_path = os.path.join(args.orig_root, task, f"{exp}@0.pkl")
        if not os.path.exists(orig_path):
            print(f"[build][SKIP] no original pkl: {orig_path}")
            continue
        orig = pickle.load(open(orig_path, "rb"))
        out_dir = os.path.join(args.out_root, task)
        os.makedirs(out_dir, exist_ok=True)

        rolls = by_demo.get(base, [])
        cand = [r for r in rolls if int(r.get("num_frames", 0)) >= args.min_length]
        succ = [r for r in cand if float(r.get("succeeded", 0.0)) >= 0.5]

        def _write(obj, suffix):
            fp = os.path.join(out_dir, f"{exp}{suffix}.pkl")
            with open(fp, "wb") as f:
                pickle.dump(obj, f)
            produced.append(f"rt/{task}/{exp}" if suffix == "@0" else f"rt/{task}/{exp}{suffix}")

        if len(succ) >= args.min_success:
            # GOOD demo: top-K successful rollouts (by smoothness) MIXED WITH the
            # original retarget ref (so every demo trains on both its own achieved
            # trajectory and the human reference).
            succ.sort(key=lambda r: (smoothness_cost(_np(r["wrist_pos"]), _np(r["arm_joint_pos"])),
                                     -int(r.get("num_frames", 0))))
            keep = succ[: args.top_k]
            for rank, r in enumerate(keep):
                _write(convert_one(r, orig, kp_body_names),
                       "@0" if rank == 0 else f"@0_r{rank}")
            if not args.no_orig_for_good:
                _write(orig, "@0_orig")                         # original ref, mixed in
            kind = f"GOOD rollout×{len(keep)}+orig"
            summary.append({"demo": base, "kind": "good", "n_success": len(succ),
                            "n_cand": len(cand), "kept_rollout": len(keep),
                            "kept_orig": 0 if args.no_orig_for_good else 1})
        elif len(cand) > 0:
            # POOR demo: original ref (@0, full-length) + best-reward rollout(s) as supplements.
            cand.sort(key=lambda r: -float(r.get("total_reward", -1e9)))
            keep = cand[: args.fallback_k]
            _write(orig, "@0")                                  # original as the base ref
            for rank, r in enumerate(keep):
                ref = convert_one(r, orig, kp_body_names)
                ref["rollout_low_confidence"] = True
                _write(ref, f"@0_rollout{rank+1}")
            kind = f"POOR orig+rollout×{len(keep)} (rew top)"
            summary.append({"demo": base, "kind": "poor", "n_success": len(succ),
                            "n_cand": len(cand), "kept_rollout": len(keep), "kept_orig": 1,
                            "best_reward": float(keep[0].get("total_reward", 0.0))})
        else:
            # UNCOVERED demo: original ref only.
            _write(orig, "@0")
            kind = "UNCOVERED orig-only"
            summary.append({"demo": base, "kind": "uncovered", "n_success": 0,
                            "n_cand": 0, "kept_rollout": 0, "kept_orig": 1})
        print(f"[build] {base}: {kind}  (succ={len(succ)}/{len(cand)}) -> {out_dir}")

    os.makedirs(args.out_root, exist_ok=True)
    json.dump(produced, open(os.path.join(args.out_root, "data_idx.json"), "w"), indent=2)
    json.dump(summary, open(os.path.join(args.out_root, "summary.json"), "w"), indent=2)
    print(f"\n[build] DONE. produced {len(produced)} reference pkls in {args.out_root}")
    print(f"[build] data_idx list -> {args.out_root}/data_idx.json")


if __name__ == "__main__":
    main()
