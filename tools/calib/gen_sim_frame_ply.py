"""Render one robotool_batch retarget frame (arm+hand meshes via URDF FK) to a
PLY point cloud in ENV-LOCAL frame, for camera-extrinsic alignment.

  points = FK(link meshes at opt_arm_joint_pos[frame] + opt_dof_pos[frame])
           expressed in fr3_link0 frame, then + arm_base_pos  -> env-local.
Plus a sampled table-top plane at z=table_z (env-local).
"""
import argparse, pickle, numpy as np, trimesh, yourdfpy

from dexx import deploy_config as _dcfg

# The merged arm+hand URDF this release builds (scripts/build_merged_urdf.py).
URDF = "assets/generated/fr3_with_right_sharpa_wave.urdf"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pkl", default="data/retargeting/robotool_batch/mano2sharpa_rh/0416_grasp/cube_small_2@0.pkl")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--out", default="calib/camera_align/sim_frame.ply")
    p.add_argument("--pts_per_m2", type=float, default=40000.0, help="table sampling density")
    p.add_argument("--arm_pts", type=int, default=120000)
    p.add_argument("--table_z", type=float, default=_dcfg.TABLE_SURFACE_Z,
                   help="table top z in env-local")
    p.add_argument("--no_table", action="store_true")
    p.add_argument("--out_frame", choices=["env_local", "link0"], default="link0",
                   help="link0: express points in fr3_link0 frame (arm base at origin), "
                        "matches live_calibrate_extrinsic --sim_ply_frame link0. "
                        "env_local: world/table frame.")
    args = p.parse_args()

    d = pickle.load(open(args.pkl, "rb"))
    arm = np.asarray(d["opt_arm_joint_pos"])[args.frame]      # (7,)
    hand = np.asarray(d["opt_dof_pos"])[args.frame]            # (22,)
    base = np.asarray(d["arm_base_pos"], dtype=np.float64)     # env-local pos of fr3_link0
    print(f"[gen] pkl={args.pkl} frame={args.frame}/{np.asarray(d['opt_arm_joint_pos']).shape[0]}")
    print(f"[gen] arm={arm.round(3)}  base={base}")

    u = yourdfpy.URDF.load(URDF, load_meshes=True, build_scene_graph=True)
    names = list(u.actuated_joint_names)                      # 7 arm + 22 hand (cfg order)
    cfg = dict(zip(names, np.concatenate([arm, hand]).tolist()))
    u.update_cfg(cfg)

    # Sample points from all posed visual meshes (fr3_link0 frame == URDF base).
    scene = u.scene
    verts = []
    for gname, geom in scene.geometry.items():
        # world transform of this geometry within the scene graph
        T = scene.graph.get(gname)[0]
        m = geom.copy().apply_transform(T)
        n = max(200, int(args.arm_pts * (m.area / max(scene.area, 1e-6))))
        pts, _ = trimesh.sample.sample_surface(m, n)
        verts.append(np.asarray(pts))
    arm_pc = np.concatenate(verts, 0)
    # FK is in fr3_link0 frame (URDF root at origin).
    # env_local: shift by arm_base_pos. link0: keep as-is.
    origin_shift = base if args.out_frame == "env_local" else np.zeros(3)
    table_z = args.table_z if args.out_frame == "env_local" else (args.table_z - base[2])
    arm_pc = arm_pc + origin_shift[None]
    arm_col = np.tile(np.array([255, 140, 0], np.uint8), (arm_pc.shape[0], 1))  # orange

    clouds = [(arm_pc, arm_col)]
    if not args.no_table:
        # table-top plane in env-local, spanning a reasonable workspace
        xs = np.arange(-0.2, 0.85, 1.0/np.sqrt(args.pts_per_m2))
        ys = np.arange(-0.55, 0.55, 1.0/np.sqrt(args.pts_per_m2))
        gx, gy = np.meshgrid(xs, ys)
        if args.out_frame == "env_local":
            gx = gx + base[0]; gy = gy + base[1]   # table spans around arm base in env-local
        tab = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, table_z)], 1)
        tab_col = np.tile(np.array([120, 120, 120], np.uint8), (tab.shape[0], 1))  # grey
        clouds.append((tab, tab_col))

    P = np.concatenate([c[0] for c in clouds], 0)
    C = np.concatenate([c[1] for c in clouds], 0)
    print(f"[gen] arm_pc={arm_pc.shape[0]}  total={P.shape[0]}  bounds min={P.min(0).round(3)} max={P.max(0).round(3)}")

    with open(args.out, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {P.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for q, c in zip(P, C):
            f.write(f"{q[0]:.5f} {q[1]:.5f} {q[2]:.5f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
    print(f"[gen] saved -> {args.out}")

if __name__ == "__main__":
    main()
