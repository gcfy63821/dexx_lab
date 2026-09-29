#!/usr/bin/env python3
"""Show a RETARGETED demo the way the environment will load it (viser).

The robot (arm + hand, FK of the retargeted joints on the merged URDF, root at
the arm base), the object mesh, the table, the human MANO keypoints and the
target wrist frame — all in env-local coordinates, with the same loader,
`mujoco2gym` transform, augmentation yaw and `xy_offset` / `z_offset` the env
applies. Use it after every retarget, and whenever a new hand or new data is
involved: retargeting is sensitive to where the hand and the object start, and
a new hand brings its own base rotation and offsets.

    python tools/dataset/view_retarget.py --data_idx rt/0416_grasp/cube_small_2@0
    python tools/dataset/view_retarget.py --data_idx rt/0416_grasp/cube_small_2@0 \\
        --retarget_root logs/retarget          # a scratch --dump_root
    python tools/dataset/view_retarget.py --data_idx rt/0416_grasp/cube_small_2@0 --summary_only

Colours: robot meshes as in the URDF; object grey; table brown; MANO keypoints
green, robot keypoints (retarget output) red, yellow lines robot tip -> MANO tip;
axes: robot end effector (FK) vs MANO wrist target.

The frame-0 summary (also printed with --summary_only, no browser) is the set of
numbers to check before training: object bottom vs table, wrist height, EE
position/rotation error, per-finger tip error, fingers within 8 mm of the object.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_idx", required=True, help="rt/<task>/<seq>@<aug>, e.g. rt/0416_grasp/cube_small_2@0")
    ap.add_argument("--side", default="right", choices=["right", "left"])
    ap.add_argument("--dexhand", default="sharpa", help="registered DexHand type (hand_imitation/envs)")
    ap.add_argument("--urdf", default=None, help="merged arm+hand URDF (default assets/generated/fr3_with_<side>_sharpa_wave.urdf)")
    ap.add_argument("--ee_link", default=None, help="end-effector link (default <side>_hand_C_MC)")
    ap.add_argument("--retarget_root", default=None, help="read retargeted pkls from here (a scratch --dump_root)")
    ap.add_argument("--frame", type=int, default=0, help="initial frame")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--summary_only", action="store_true", help="print the frame-0 checks and exit")
    return ap.parse_args(argv)


def _yaw_about(center_xy, yaw_deg):
    a = np.deg2rad(yaw_deg)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return R, np.asarray(center_xy, dtype=float)


def load_demo(args):
    """Loader output in env-local coordinates, exactly as the env builds it."""
    import torch

    import dexx.tasks.hand_imitation.dataset.robotool_batch_dataset_dexhand  # noqa: F401  (registers the loader)
    import dexx.tasks.hand_imitation.envs.sharpa  # noqa: F401  (registers the Sharpa hand)
    from dexx import deploy_config as dc
    from dexx.tasks.hand_imitation.dataset.factory import ManipDataFactory
    from dexx.tasks.hand_imitation.dataset.transform import aa_to_rotmat
    from dexx.tasks.hand_imitation.envs.factory import DexHandFactory

    dexhand = DexHandFactory.create_hand(args.dexhand, args.side)
    m2g = torch.eye(4)
    m2g[:3, :3] = aa_to_rotmat(torch.tensor([0.0, 0.0, -np.pi / 2])) @ aa_to_rotmat(torch.tensor([np.pi / 2, 0.0, 0.0]))
    m2g[:3, 3] = torch.tensor([0.0, 0.0, dc.TABLE_SURFACE_Z])
    kw = dict(manipdata_type=ManipDataFactory.dataset_type(args.data_idx), side=args.side, device="cpu",
              mujoco2gym_transf=m2g, max_seq_len=100000, dexhand=dexhand, embodiment="sharpa")
    if args.retarget_root:
        kw["retarget_root"] = args.retarget_root
    ds = ManipDataFactory.create_data(**kw)
    d = dict(ds[args.data_idx])
    np_ = lambda x: x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    # The loader replaces an unreachable result with placeholders (the env would
    # not train on it). Show the pkl's own solution anyway, so you can see why.
    import glob
    import pickle

    task, rest = args.data_idx.split("/", 1)[1].split("/", 1)
    if "@" in rest:
        pkl = os.path.join(ds.retarget_root, task, rest + ".pkl")
    else:  # base index: same resolution as the loader
        pkl = os.path.join(ds.retarget_root, task, f"{rest}@0.pkl")
        if not os.path.exists(pkl):
            cands = [c for c in sorted(glob.glob(os.path.join(ds.retarget_root, task, f"{rest}@0*.pkl")))
                     if not c.endswith("_hand.pkl")]
            pkl = cands[0] if cands else pkl
    if not os.path.exists(pkl):
        raise SystemExit(f"{args.data_idx}: no retargeted result at {pkl} (run scripts/retarget.py first)")
    with open(pkl, "rb") as fh:
        raw = pickle.load(fh)
    reachable = bool(raw.get("reachable", True))
    if not reachable:
        diag = {k: raw[k] for k in ("ik_mean_err", "ik_max_err", "aug_arm_ee_loss") if k in raw}
        print(f"[view_retarget] WARNING: {pkl} is UNREACHABLE {diag}; the env would replace it with "
              f"placeholders. Showing the demo at the pkl's offsets for debugging.", flush=True)
        T0 = len(np_(d["obj_trajectory"]))
        for k in ("xy_offset", "z_offset", "aug_yaw_deg"):
            if k in raw:
                d[k] = np_(raw[k])
        for k in ("opt_arm_joint_pos", "opt_dof_pos"):  # a partial result may store a solution
            if isinstance(raw.get(k), np.ndarray) and raw[k].ndim == 2:
                d[k] = raw[k][:T0]
            else:
                print(f"[view_retarget]   no {k} stored; the robot is shown in the loader's placeholder pose",
                      flush=True)
        d.pop("opt_joints_pos", None)

    obj = np_(d["obj_trajectory"]).astype(float).copy()                 # (T,4,4)
    wrist_pos = np_(d["wrist_pos"]).astype(float).copy()                 # (T,3)
    wrist_R = np_(aa_to_rotmat(d["wrist_rot"])).astype(float)            # (T,3,3)
    mano = {k: np_(v).astype(float).copy() for k, v in d["mano_joints"].items()}

    # Same order as the env: augmentation yaw about the arm-base xy, then offsets.
    yaw = float(np_(d.get("aug_yaw_deg", 0.0)))
    if abs(yaw) > 1e-6:
        R, c = _yaw_about(dc.ARM_BASE_POS[:2], yaw)
        rot_xy = lambda p: np.concatenate([(p[..., :2] - c) @ R[:2, :2].T + c, p[..., 2:]], axis=-1)
        wrist_pos, obj[:, :3, 3] = rot_xy(wrist_pos), rot_xy(obj[:, :3, 3])
        wrist_R = R @ wrist_R
        obj[:, :3, :3] = R @ obj[:, :3, :3]
        mano = {k: rot_xy(v) for k, v in mano.items()}
    off = np.r_[np_(d["xy_offset"]).reshape(-1)[:2], np_(d.get("z_offset", [0.0])).reshape(-1)[:1]]
    obj[:, :3, 3] += off
    wrist_pos += off
    mano = {k: v + off for k, v in mano.items()}

    return dict(
        dexhand=dexhand, dc=dc, obj=obj, wrist_pos=wrist_pos, wrist_R=wrist_R, mano=mano,
        arm_q=np_(d["opt_arm_joint_pos"]).astype(float), hand_q=np_(d["opt_dof_pos"]).astype(float),
        stored_ee=np_(d["opt_joints_pos"])[:, 0].astype(float) if "opt_joints_pos" in d else None,
        reachable=reachable,
        obj_mesh_path=os.path.join(REPO, d["obj_mesh_path"]) if not os.path.isabs(d["obj_mesh_path"]) else d["obj_mesh_path"],
        obj_scale=float(d.get("obj_scale", 1.0)),
        loss=np_(d["opt_final_loss_per_frame"]) if "opt_final_loss_per_frame" in d else None,
        offsets=off, yaw=yaw,
    )


class Robot:
    """FK of the merged URDF with joints set by name (never by position)."""

    def __init__(self, urdf_path, arm_names, hand_names, base_pos):
        import yourdfpy

        self.urdf = yourdfpy.URDF.load(urdf_path, load_meshes=True, build_scene_graph=True)
        self.names = list(arm_names) + list(hand_names)
        missing = [n for n in self.names if n not in self.urdf.joint_map]
        if missing:
            raise SystemExit(f"joints not in {urdf_path}: {missing}")
        self.base = np.asarray(base_pos, dtype=float)

    def set(self, arm_q, hand_q):
        self.cfg = {n: float(v) for n, v in zip(self.names, np.r_[arm_q, hand_q])}
        self.urdf.update_cfg(self.cfg)

    def link(self, name):
        T = self.urdf.get_transform(name, self.urdf.base_link).copy()
        T[:3, 3] += self.base
        return T


def tip_links(dm, robot):
    """MANO tip name -> robot fingertip link, via the DexHand mapping (to_dex)."""
    out = {}
    for mano_name in dm["mano"]:
        if not mano_name.endswith("_tip"):
            continue
        links = [l for l in dm["dexhand"].to_dex(mano_name) if l in robot.urdf.link_map]
        if links:  # prefer an explicit fingertip link, else the last mapped body
            out[mano_name] = next((l for l in links if "fingertip" in l), links[-1])
    return out


def robot_keypoints(dm, robot):
    """Env-local positions of every DexHand body present in the URDF (current cfg)."""
    names = [n for n in dm["dexhand"].body_names if n in robot.urdf.link_map]
    return names, np.stack([robot.link(n)[:3, 3] for n in names])


def rot_err_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def frame_checks(dm, robot, ee_link, mesh_v, f):
    dc = dm["dc"]
    robot.set(dm["arm_q"][f], dm["hand_q"][f])
    T_ee = robot.link(ee_link)
    To = dm["obj"][f]
    v = mesh_v @ To[:3, :3].T + To[:3, 3]
    out = {
        "object bottom - table (mm)": 1000 * (v[:, 2].min() - dc.TABLE_SURFACE_Z),
        "object centre env-local": np.round(To[:3, 3], 4).tolist(),
        "wrist target height above table (mm)": 1000 * (dm["wrist_pos"][f, 2] - dc.TABLE_SURFACE_Z),
        "EE position error (mm)": 1000 * float(np.linalg.norm(T_ee[:3, 3] - dm["wrist_pos"][f])),
        "EE rotation error (deg)": rot_err_deg(T_ee[:3, :3], dm["wrist_R"][f]),
    }
    if dm["stored_ee"] is not None:
        out["FK vs retarget's stored EE (mm)"] = 1000 * float(np.linalg.norm(T_ee[:3, 3] - dm["stored_ee"][f]))
    if not dm["reachable"]:
        out["reachable"] = False
    tips, near = {}, 0
    for mano_name, link in tip_links(dm, robot).items():
        p = robot.link(link)[:3, 3]
        tips[mano_name.replace("_tip", "")] = round(1000 * float(np.linalg.norm(p - dm["mano"][mano_name][f])), 1)
        near += int(np.linalg.norm(v - p, axis=1).min() < 0.008)
    out["tip error per finger (mm)"] = tips
    out["robot fingertips within 8 mm of the object"] = near
    if dm["loss"] is not None:
        out["retarget loss"] = float(dm["loss"][f])
    return out


def main(argv=None):
    args = parse_args(argv)
    # The loader resolves data/ paths against the working directory: run from the
    # repo root, with user-given paths resolved against the caller's directory.
    for k in ("retarget_root", "urdf"):
        if getattr(args, k):
            setattr(args, k, os.path.abspath(getattr(args, k)))
    os.chdir(REPO)
    import trimesh

    dm = load_demo(args)
    urdf = args.urdf or os.path.join(REPO, "assets", "generated", f"fr3_with_{args.side}_sharpa_wave.urdf")
    ee_link = args.ee_link or f"{args.side}_hand_C_MC"
    robot = Robot(urdf, [f"fr3_joint{i}" for i in range(1, 8)], dm["dexhand"].dof_names, dm["dc"].ARM_BASE_POS)
    mesh = trimesh.load(dm["obj_mesh_path"], force="mesh")
    mesh.apply_scale(dm["obj_scale"])
    T = len(dm["obj"])

    print(f"[view_retarget] {args.data_idx}: {T} frames, offsets xyz {np.round(dm['offsets'], 4).tolist()}, "
          f"aug yaw {dm['yaw']:.1f} deg")
    for k, v in frame_checks(dm, robot, ee_link, np.asarray(mesh.vertices), args.frame).items():
        print(f"  frame {args.frame}: {k}: {v if not isinstance(v, float) else round(v, 2)}")
    if args.summary_only:
        return 0

    import viser
    from viser.extras import ViserUrdf

    srv = viser.ViserServer(port=args.port)
    srv.scene.add_box("/table", dimensions=(1.5, 2.4, 0.03), position=(0.1, 0.0, dm["dc"].TABLE_SURFACE_Z - 0.015),
                      color=(160, 120, 80), opacity=0.5)
    srv.scene.add_frame("/world", axes_length=0.15, axes_radius=0.004)
    base = srv.scene.add_frame("/robot", position=tuple(robot.base), axes_length=0.1, axes_radius=0.003)
    vurdf = ViserUrdf(srv, robot.urdf, root_node_name="/robot")
    obj_node = srv.scene.add_mesh_simple("/object", vertices=np.asarray(mesh.vertices, np.float32),
                                         faces=np.asarray(mesh.faces, np.uint32), color=(180, 180, 180))
    gui_f = srv.gui.add_slider("frame", min=0, max=T - 1, step=1, initial_value=args.frame)
    gui_play = srv.gui.add_checkbox("play", initial_value=False)
    gui_info = srv.gui.add_markdown("")
    tips = tip_links(dm, robot)
    del base

    def draw(f):
        robot.set(dm["arm_q"][f], dm["hand_q"][f])
        vurdf.update_cfg(np.array([robot.cfg.get(n, 0.0) for n in vurdf.get_actuated_joint_names()]))
        To = dm["obj"][f]
        obj_node.position = tuple(To[:3, 3])
        q = trimesh.transformations.quaternion_from_matrix(To)  # w, x, y, z
        obj_node.wxyz = tuple(q)
        mano_pts = np.stack([v[f] for v in dm["mano"].values()])
        srv.scene.add_point_cloud("/mano", points=mano_pts.astype(np.float32),
                                  colors=np.tile([0, 200, 0], (len(mano_pts), 1)).astype(np.uint8), point_size=0.008)
        _, kp = robot_keypoints(dm, robot)
        srv.scene.add_point_cloud("/robot_kp", points=kp.astype(np.float32),
                                  colors=np.tile([220, 0, 0], (len(kp), 1)).astype(np.uint8), point_size=0.006)
        segs = np.array([[robot.link(l)[:3, 3], dm["mano"][m][f]] for m, l in tips.items()], np.float32)
        if len(segs):
            srv.scene.add_line_segments("/tip_err", points=segs, colors=(255, 220, 0), line_width=3.0)
        T_ee = robot.link(ee_link)
        for name, R, p in (("/ee_fk", T_ee[:3, :3], T_ee[:3, 3]), ("/wrist_target", dm["wrist_R"][f], dm["wrist_pos"][f])):
            M = np.eye(4)
            M[:3, :3] = R
            srv.scene.add_frame(name, wxyz=tuple(trimesh.transformations.quaternion_from_matrix(M)),
                                position=tuple(p), axes_length=0.08, axes_radius=0.003)
        checks = frame_checks(dm, robot, ee_link, np.asarray(mesh.vertices), f)
        gui_info.content = "\n".join(f"- **{k}**: {v if not isinstance(v, float) else round(v, 1)}" for k, v in checks.items())

    gui_f.on_update(lambda _: draw(int(gui_f.value)))
    draw(args.frame)
    print(f"[view_retarget] open http://localhost:{args.port}", flush=True)
    while True:
        if gui_play.value:
            gui_f.value = (int(gui_f.value) + 1) % T
        time.sleep(1 / 30)


if __name__ == "__main__":
    sys.exit(main())
