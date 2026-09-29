# Camera extrinsics

4x4 **camera-in-armbase** transforms in the ROS optical convention
(x-right, y-down, z-forward). Pass one to training, evaluation, `play.py` and
deploy with `--camera_extrinsic calib/camera_align/<file>.npy`. **Train and
deploy must use the same file**: the scene point cloud is rendered from this
viewpoint in sim.

| file | drift from `current.npy` | use |
|---|---|---|
| `current.npy` | — | the default, and the extrinsic the shipped student expects. ICP on the arm for yaw and in-plane translation, then height and tilt levelled against the table. |
| `refined_extrinsic_last_it3.npy` | 3.0 cm / 2.44° | a separate ICP calibration of the same mount |
| `refined_extrinsic_manual.npy` | 2.5 cm / 3.03° | a manual refinement |

Drifts are as printed by `tutorial/06_camera_calibration/inspect_extrinsic.py`.
They are different calibrations, not noise: a student trained on one is out of
distribution on another.

Without `--camera_extrinsic`, the sim raycaster loads `current.npy`
(`dexx.deploy_config.default_camera_extrinsic`); a missing file is an error,
never a guess. `deploy/deploy_pc.py` requires the flag. Pass the file anyway for any run you intend to reproduce.

Write new calibrations to a new, dated file rather than over `current.npy`. The
procedure is [tutorial/06](../../tutorial/06_camera_calibration/).
