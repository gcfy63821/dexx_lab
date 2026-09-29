"""Headless API and CUDA smoke check; complete training is tested by acceptance.

Run with --headless --result <new-file.json>. Require the JSON success marker:
some simulator startup failures terminate the process with exit code zero.
"""
import argparse
import json
from pathlib import Path


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.result.exists():
        parser.error("--result must be a new file; remove the previous result explicitly")
    if not args.device.startswith("cuda"):
        parser.error("--device must select CUDA for this GPU preflight")
    app = AppLauncher(args).app
    try:
        import torch
        import trimesh
        from pytorch3d.ops import knn_points
        from simple_raycaster.raycaster import MultiMeshRaycaster
        from isaaclab.assets import ArticulationData
        from isaaclab.sensors import ContactSensorData
        from dexx.tasks.franka_sharpa.franka_sharpa_pointcloud_env_cfg import FrankaSharpaPointCloudEnvCfg
        from dexx.tasks.franka_sharpa.franka_sharpa_env_cfg import update_cfg_for_hand_side

        if not torch.cuda.is_available():
            raise RuntimeError("Torch cannot use CUDA; check the driver and CUDA build")
        if not hasattr(ArticulationData, "joint_vel_limits"):
            raise RuntimeError("Isaac Lab lacks ArticulationData.joint_vel_limits")
        if "contact_pos_w" not in ContactSensorData.__dataclass_fields__:
            raise RuntimeError("Isaac Lab lacks contact-point data")
        cfg = FrankaSharpaPointCloudEnvCfg()
        update_cfg_for_hand_side(cfg, "right")

        device = args.device
        points = torch.tensor([[[0., 0., 0.], [1., 0., 0.]]], device=device)
        torch.testing.assert_close(knn_points(points, points, K=1).dists, torch.zeros((1, 2, 1), device=device))
        meshes = [trimesh.creation.box(extents=(s, s, s)) for s in (1, 2)]
        raycaster = MultiMeshRaycaster(meshes, device=device)
        raycaster.initialize()
        positions = torch.zeros((2, 1, 3), device=device)
        kwargs = dict(
            mesh_pos_w=positions,
            mesh_quat_w=torch.tensor([[[1., 0., 0., 0.]]] * 2, device=device),
            mesh_indices=torch.tensor([[0], [1]], device=device),
            ray_starts_w=torch.tensor([[[0., 0., 3.]]] * 2, device=device),
            ray_dirs_w=torch.tensor([[[0., 0., -1.]]] * 2, device=device),
            min_dist=0.01, max_dist=10.,
        )
        _, depth = raycaster.raycast_fused(**kwargs)
        torch.testing.assert_close(depth.cpu(), torch.tensor([[2.5], [2.0]]))
        positions[1, 0, 2] = 1.
        _, depth = raycaster.raycast_fused(**kwargs)
        torch.testing.assert_close(depth.cpu(), torch.tensor([[2.5], [1.0]]))
        torch.cuda.synchronize()
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(json.dumps({"ok": True, "torch": torch.__version__, "device": device}))
        print("API AND CUDA CHECKS PASSED; run acceptance for task physics/training", flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
