import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# ---- Teacher: critic-horizon force env + observed object pose (privileged). ----
gym.register(
    id="franka-sharpa-force-poseobs",
    entry_point="dexx.tasks.franka_sharpa.franka_sharpa_force_poseobs_env:FrankaSharpaForcePoseObsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_sharpa_force_poseobs_cfg:FrankaSharpaPoseObsCfg",
        "gym_style_cfg_entry_point": f"{agents.__name__}:dexmanip_critic_horizon_ppo_cfg.yaml",
    },
)

# ---- Student: the same env with scene / hand / tactile point clouds. ----
gym.register(
    id="franka-sharpa-pointcloud",
    entry_point="dexx.tasks.franka_sharpa.franka_sharpa_pointcloud_env:FrankaSharpaPointCloudEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_sharpa_pointcloud_env_cfg:FrankaSharpaPointCloudEnvCfg",
        "gym_style_cfg_entry_point": f"{agents.__name__}:dexmanip_ppo_cfg.yaml",
    },
)

# ---- Real robot: the student's obs from hardware. Arm via Polymetis (NUC
# joint bridge over ZMQ), depth over ZMQ from the camera host, hand and tactile
# from the Sharpa SDK. Same obs dict schema as the sim student env. ----
gym.register(
    id="franka-sharpa-pointcloud-polymetis-deploy",
    entry_point="dexx.tasks.franka_sharpa.franka_sharpa_pointcloud_deploy_env:FrankaSharpaPointCloudDeployEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_sharpa_pointcloud_env_cfg:FrankaSharpaPointCloudEnvCfg",
        "gym_style_cfg_entry_point": f"{agents.__name__}:dexmanip_ppo_cfg.yaml",
    },
)

# ---- Offline recording: the student env + a third-person RGB+depth camera.
# Requires `--enable_cameras` and is GPU-heavy: `--num_envs ≤ 16` viz runs only. ----
gym.register(
    id="franka-sharpa-pointcloud-record",
    entry_point="dexx.tasks.franka_sharpa.franka_sharpa_pointcloud_record_env:FrankaSharpaPointCloudRecordEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_sharpa_pointcloud_record_env_cfg:FrankaSharpaPointCloudRecordEnvCfg",
        "gym_style_cfg_entry_point": f"{agents.__name__}:dexmanip_ppo_cfg.yaml",
    },
)
