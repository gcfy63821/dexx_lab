from abc import ABC, abstractmethod
from scipy.ndimage import gaussian_filter1d
import numpy as np
import torch
from dexx.tasks.hand_imitation.dataset.transform import rotmat_to_aa
from torch.utils.data import Dataset
from pytorch3d.ops import sample_points_from_meshes


class ManipData(Dataset, ABC):
    def __init__(
        self,
        *,
        data_dir: str,
        split: str = "all",
        skip: int = 4,
        device="cuda:0",
        mujoco2gym_transf=None,
        max_seq_len=int(1e10),
        dexhand=None,
        verbose=True,
        use_bimanual_retarget: bool = False,
        **kwargs,
    ):
        self.data_dir = data_dir
        self.split = split
        self.skip = skip
        self.data_pathes = None

        self.dexhand = dexhand
        self.device = device

        self.verbose = verbose

        # ? modify this depending on the origin point
        self.transf_offset = torch.eye(4, dtype=torch.float32, device=mujoco2gym_transf.device)

        self.mujoco2gym_transf = mujoco2gym_transf
        self.max_seq_len = max_seq_len
        self.use_bimanual_retarget = use_bimanual_retarget
        # caculate contact

        import chamfer_distance as chd

        self.ch_dist = chd.ChamferDistance()

    def __len__(self):
        return len(self.data_pathes)

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @staticmethod
    def compute_velocity(p, time_delta, guassian_filter=True):
        # [T, K, 3]
        velocity = np.gradient(p.cpu().numpy(), axis=0) / time_delta
        if guassian_filter:
            velocity = gaussian_filter1d(velocity, 2, axis=0, mode="nearest")
        return torch.from_numpy(velocity).to(p)

    @staticmethod
    def compute_angular_velocity(rot, time_delta: float, guassian_filter=True):
        # [T, K, 3, 3]
        diff_r = rot[1:] @ rot[:-1].transpose(-1, -2)  # [T-1, K, 3, 3]
        diff_aa = rotmat_to_aa(diff_r).cpu().numpy()  # [T-1, K, 3]
        diff_angle = np.linalg.norm(diff_aa, axis=-1)  # [T-1, K]
        diff_axis = diff_aa / (diff_angle[:, :, None] + 1e-8)  # [T-1, K, 3]
        angular_velocity = diff_axis * diff_angle[:, :, None] / time_delta  # [T-1, K, 3]
        angular_velocity = np.concatenate([angular_velocity, angular_velocity[-1:]], axis=0)  # [T, K, 3]
        if guassian_filter:
            angular_velocity = gaussian_filter1d(angular_velocity, 2, axis=0, mode="nearest")
        return torch.from_numpy(angular_velocity).to(rot)

    @staticmethod
    def compute_dof_velocity(dof, time_delta, guassian_filter=True):
        # [T, K]
        velocity = np.gradient(dof.cpu().numpy(), axis=0) / time_delta
        if guassian_filter:
            velocity = gaussian_filter1d(velocity, 2, axis=0, mode="nearest")
        return torch.from_numpy(velocity).to(dof)

    def random_sampling_pc(self, mesh):
        numpy_random_state = np.random.get_state()
        torch_random_state = torch.random.get_rng_state()
        torch_random_state_cuda = torch.cuda.get_rng_state()
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        rs_verts_obj = sample_points_from_meshes(mesh, 1000, return_normals=False).to(self.device).squeeze(0)

        # reset seed
        np.random.set_state(numpy_random_state)
        torch.random.set_rng_state(torch_random_state)
        torch.cuda.set_rng_state(torch_random_state_cuda)

        return rs_verts_obj
