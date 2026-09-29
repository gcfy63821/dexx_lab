from typing import Union

import numpy as np
import torch
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_to_axis_angle,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
)


class Compose:
    def __init__(self, transforms: list):
        """Composes several transforms together. This transform does not
        support torchscript.

        Args:
            transforms (list): (list of transform functions)
        """
        self.transforms = transforms

    def __call__(self, rotation: Union[torch.Tensor, np.ndarray], convention: str = "xyz", **kwargs):
        convention = convention.lower()
        if not (set(convention) == set("xyz") and len(convention) == 3):
            raise ValueError(f"Invalid convention {convention}.")
        if isinstance(rotation, np.ndarray):
            data_type = "numpy"
            rotation = torch.FloatTensor(rotation)
        elif isinstance(rotation, torch.Tensor):
            data_type = "tensor"
        else:
            raise TypeError("Type of rotation should be torch.Tensor or numpy.ndarray")
        for t in self.transforms:
            if "convention" in t.__code__.co_varnames:
                rotation = t(rotation, convention.upper(), **kwargs)
            else:
                rotation = t(rotation, **kwargs)
        if data_type == "numpy":
            rotation = rotation.detach().cpu().numpy()
        return rotation


def aa_to_rotmat(axis_angle: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert axis_angle to rotation matrixs.
    Args:
        axis_angle (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3, 3).
    """
    if axis_angle.shape[-1] != 3:
        raise ValueError(f"Invalid input axis angles shape f{axis_angle.shape}.")
    t = Compose([axis_angle_to_matrix])
    return t(axis_angle)


def rotmat_to_aa(matrix: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert rotation matrixs to axis angles.

    Args:
        matrix (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3, 3). ndim of input is unlimited.
        convention (str, optional): Convention string of three letters
                from {“x”, “y”, and “z”}. Defaults to 'xyz'.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3).
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    t = Compose([matrix_to_quaternion, quaternion_to_axis_angle])
    return t(matrix)


def aa_to_quat(axis_angle: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert axis_angle to quaternions.
    Args:
        axis_angle (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 4).
    """
    if axis_angle.shape[-1] != 3:
        raise ValueError(f"Invalid input axis angles f{axis_angle.shape}.")
    t = Compose([axis_angle_to_quaternion])
    return t(axis_angle)


def rot6d_to_aa(rotation_6d: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert rotation 6d representations to axis angles.

    Args:
        rotation_6d (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 6). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f"Invalid input rotation_6d f{rotation_6d.shape}.")
    t = Compose([rotation_6d_to_matrix, matrix_to_quaternion, quaternion_to_axis_angle])
    return t(rotation_6d)


def quat_to_aa(quaternions: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert quaternions to axis angles.

    Args:
        quaternions (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.
    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3).
    """
    if quaternions.shape[-1] != 4:
        raise ValueError(f"Invalid input quaternions f{quaternions.shape}.")
    t = Compose([quaternion_to_axis_angle])
    return t(quaternions)


def rot6d_to_rotmat(rotation_6d: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert rotation 6d representations to rotation matrixs.

    Args:
        rotation_6d (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 6). ndim of input is unlimited.
    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3, 3).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f"Invalid input rotation_6d f{rotation_6d.shape}.")
    t = Compose([rotation_6d_to_matrix])
    return t(rotation_6d)


def rotmat_to_rot6d(matrix: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert rotation matrixs to rotation 6d representations.

    Args:
        matrix (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3, 3). ndim of input is unlimited.
    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 6).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    t = Compose([matrix_to_rotation_6d])
    return t(matrix)


def rotmat_to_quat(matrix: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert rotation matrixs to quaternions.

    Args:
        matrix (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3, 3). ndim of input is unlimited.
    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 4).
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    t = Compose([matrix_to_quaternion])
    return t(matrix)


def quat_to_rotmat(quaternions: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert quaternions to rotation matrixs.

    Args:
        quaternions (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.
    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3, 3).
    """
    if quaternions.shape[-1] != 4:
        raise ValueError(f"Invalid input quaternions shape f{quaternions.shape}.")
    t = Compose([quaternion_to_matrix])
    return t(quaternions)


def rot6d_to_quat(rotation_6d: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert rotation 6d representations to quaternions.

    Args:
        rotation (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 6). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 4).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f"Invalid input rotation_6d shape f{rotation_6d.shape}.")
    t = Compose([rotation_6d_to_matrix, matrix_to_quaternion])
    return t(rotation_6d)
