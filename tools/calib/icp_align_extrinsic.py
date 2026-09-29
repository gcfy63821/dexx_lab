"""ICP-align the real arm/hand point cloud onto the sim robot mesh to refine the
camera extrinsic — no open3d, just numpy + scipy cKDTree (Kabsch/SVD ICP).

Pipeline (all in fr3_link0 / arm-base frame):
  real: accum.npz cam-frame pts -> apply init extrinsic -> crop box
  sim : URDF FK at pkl[frame] -> sample mesh surfaces -> crop SAME box
  ICP : find rigid T s.t. T @ real ≈ sim, reject matches > --max_corr
  out : new extrinsic = T @ init_extrinsic

  python tools/calib/icp_align_extrinsic.py \
    --real_npz logs/real_calib_multi2/accum.npz \
    --init_extrinsic calib/camera_align/refined_extrinsic_v4.npy \
    --pkl data/retargeting/robotool_batch/mano2sharpa_rh/0416_grasp/cube_small_1@0.pkl \
    --frame 0 --box -0.2 0.6 -0.2 0.2 0.0 0.6 \
    --out calib/camera_align/refined_extrinsic_v5.npy
"""
import argparse
import numpy as np
from scipy.spatial import cKDTree

from dexx import deploy_config as _dcfg


def crop(P, box):
    m = ((P[:, 0] >= box[0]) & (P[:, 0] <= box[1]) &
         (P[:, 1] >= box[2]) & (P[:, 1] <= box[3]) &
         (P[:, 2] >= box[4]) & (P[:, 2] <= box[5]))
    return P[m], m


def kabsch(A, B):
    """Best rigid R,t mapping A -> B (both (N,3), paired)."""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    t = cb - R @ ca
    return R, t


def icp(src, tgt, max_corr, iters=60, tol=1e-6):
    tree = cKDTree(tgt)
    T = np.eye(4)
    cur = src.copy()
    prev = None
    for it in range(iters):
        dist, idx = tree.query(cur, k=1)
        keep = dist < max_corr
        if keep.sum() < 50:
            print(f"[icp] it{it}: too few inliers ({keep.sum()}) — stop")
            break
        R, t = kabsch(cur[keep], tgt[idx[keep]])
        cur = cur @ R.T + t
        step = np.eye(4); step[:3, :3] = R; step[:3, 3] = t
        T = step @ T
        rmse = np.sqrt((dist[keep] ** 2).mean())
        if prev is not None and abs(prev - rmse) < tol:
            print(f"[icp] converged it{it} rmse={rmse*1000:.2f}mm inliers={keep.sum()}/{len(cur)}")
            break
        prev = rmse
    else:
        print(f"[icp] max iters, rmse={rmse*1000:.2f}mm inliers={keep.sum()}/{len(cur)}")
    return T, rmse, keep.sum(), len(cur)


def write_ply(path, P, C=None):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(P)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if C is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        if C is None:
            for p in P:
                f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")
        else:
            for p, c in zip(P, C):
                f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real_npz", required=True)
    p.add_argument("--init_extrinsic", required=True)
    p.add_argument("--pkl", required=True)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--box", type=float, nargs=6,
                   default=[-0.2, 0.6, -0.2, 0.2, 0.0, 0.6],
                   help="xmin xmax ymin ymax zmin zmax in fr3_link0 frame")
    p.add_argument("--sim_pts", type=int, default=200000, help="sim mesh sample count")
    p.add_argument("--voxel", type=float, default=0.006, help="voxel downsample (m) before ICP")
    p.add_argument("--max_corr", type=float, default=0.05, help="ICP correspondence reject dist (m)")
    p.add_argument("--sim_table", action="store_true",
                   help="add a sim table plane to the target so ICP also fixes vertical/tilt")
    p.add_argument("--table_z", type=float, default=_dcfg.TABLE_SURFACE_Z, help="table z in env-local")
    p.add_argument("--table_density", type=float, default=40000.0, help="table pts/m^2")
    p.add_argument("--out", required=True)
    p.add_argument("--dump_dir", default="logs/icp_align")
    args = p.parse_args()

    box = args.box
    # ---- real -> arm-base -> crop ----
    z = np.load(args.real_npz, allow_pickle=True)
    pc_cam = np.asarray(z["points"], dtype=np.float64).reshape(-1, 3)
    T0 = np.load(args.init_extrinsic).astype(np.float64)
    real_arm = pc_cam @ T0[:3, :3].T + T0[:3, 3]
    real_c, _ = crop(real_arm, box)
    print(f"[real] {len(pc_cam)} -> box {len(real_c)} pts")

    # ---- sim robot mesh sample -> crop ----
    import pickle, trimesh, yourdfpy
    URDF = "assets/generated/fr3_with_right_sharpa_wave.urdf"
    d = pickle.load(open(args.pkl, "rb"))
    arm = np.asarray(d["opt_arm_joint_pos"])[args.frame]
    hand = np.asarray(d["opt_dof_pos"])[args.frame]
    u = yourdfpy.URDF.load(URDF, load_meshes=True, build_scene_graph=True)
    u.update_cfg(dict(zip(list(u.actuated_joint_names),
                          np.concatenate([arm, hand]).tolist())))
    sc = u.scene
    # density in points per m^2 of the POSED (meter-scale) mesh surface.
    density = float(args.sim_pts)   # reuse --sim_pts as pts/m^2 (default 200000/m^2)
    verts = []
    for gname, geom in sc.geometry.items():
        Tg = sc.graph.get(gname)[0]
        m = geom.copy().apply_transform(Tg)   # meters
        n = max(300, int(density * m.area))   # m.area now in m^2
        pts, _ = trimesh.sample.sample_surface(m, n)
        verts.append(np.asarray(pts))
    sim_all = np.concatenate(verts, 0)
    # optional: add a sim table plane (link0 z = table_z - arm_base_z) so ICP
    # constrains the vertical + 2 tilts too, not just the arm.
    if args.sim_table:
        base_z = float(np.asarray(d["arm_base_pos"], dtype=np.float64)[2])
        tz = args.table_z - base_z
        xs = np.arange(box[0], box[1], 1.0 / np.sqrt(args.table_density))
        ys = np.arange(box[2], box[3], 1.0 / np.sqrt(args.table_density))
        gx, gy = np.meshgrid(xs, ys)
        tab = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, tz)], 1)
        sim_all = np.concatenate([sim_all, tab], 0)
        print(f"[sim ] added table plane at link0 z={tz:+.3f} ({len(tab)} pts)")
    sim_c, _ = crop(sim_all, box)
    print(f"[sim ] {len(sim_all)} -> box {len(sim_c)} pts")

    def voxel_ds(P, v):
        key = np.floor(P / v).astype(np.int64)
        _, idx = np.unique(key, axis=0, return_index=True)
        return P[idx]
    real_ds = voxel_ds(real_c, args.voxel)
    sim_ds = voxel_ds(sim_c, args.voxel)
    print(f"[ds  ] real {len(real_c)}->{len(real_ds)}  sim {len(sim_c)}->{len(sim_ds)}")

    d0, _ = cKDTree(sim_ds).query(real_ds, k=1)
    print(f"[pre ] real->sim NN before ICP: mean={d0.mean()*1000:.1f}mm median={np.median(d0)*1000:.1f}mm")

    T_corr, rmse, ninl, ntot = icp(real_ds, sim_ds, args.max_corr)
    dt = T_corr[:3, 3]; dR = T_corr[:3, :3]
    ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
    print(f"[icp ] correction: |t|={np.linalg.norm(dt)*1000:.1f}mm  angle={ang:.2f}deg  "
          f"t={dt.round(4).tolist()}")

    T_new = T_corr @ T0
    np.save(args.out, T_new.astype(np.float32))
    print(f"[out ] new extrinsic = T_corr @ init -> {args.out}")

    import os
    os.makedirs(args.dump_dir, exist_ok=True)
    real_aligned = real_c @ T_corr[:3, :3].T + T_corr[:3, 3]
    write_ply(os.path.join(args.dump_dir, "sim_box.ply"), sim_c,
              np.tile([255, 140, 0], (len(sim_c), 1)))
    write_ply(os.path.join(args.dump_dir, "real_box_before.ply"), real_c,
              np.tile([80, 160, 255], (len(real_c), 1)))
    write_ply(os.path.join(args.dump_dir, "real_box_after.ply"), real_aligned,
              np.tile([80, 255, 120], (len(real_aligned), 1)))
    print(f"[dump] sim_box / real_box_before / real_box_after PLY -> {args.dump_dir}")
    print(f"[done] rmse={rmse*1000:.2f}mm inliers={ninl}/{ntot}")


if __name__ == "__main__":
    main()
