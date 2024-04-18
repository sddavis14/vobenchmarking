import numpy as np
import torch


def generate_transformation(translation, rotation):
    rotation_matrix = rotation_matrix_from_angles(rotation)  # current transformation matrix
    translation = translation.unsqueeze(dim=2)
    transform_matrix = torch.cat((rotation_matrix, translation), dim=2)
    tmp = torch.tensor([[0, 0, 0, 1]] * translation.shape[0], dtype=torch.float).to(translation.device)
    tmp = tmp.unsqueeze(dim=1)
    transform_matrix = torch.cat((transform_matrix, tmp), dim=1)
    return transform_matrix

def rotation_matrix_from_angles(angles):
    """Convert euler angles to rotation matrix.
     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        angles: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    batch_size = angles.size(0)
    x, y, z = angles[:, 0], angles[:, 1], angles[:, 2]

    cos_z = torch.cos(z)
    sin_z = torch.sin(z)

    zeros = z.detach() * 0
    ones = zeros.detach() + 1
    z_part = torch.stack([cos_z, -sin_z, zeros,
                          sin_z, cos_z, zeros,
                          zeros, zeros, ones], dim=1).reshape(batch_size, 3, 3)

    cos_y = torch.cos(y)
    sin_y = torch.sin(y)

    y_part = torch.stack([cos_y, zeros, sin_y,
                          zeros, ones, zeros,
                          -sin_y, zeros, cos_y], dim=1).reshape(batch_size, 3, 3)

    cos_x = torch.cos(x)
    sin_x = torch.sin(x)

    x_part = torch.stack([ones, zeros, zeros,
                          zeros, cos_x, -sin_x,
                          zeros, sin_x, cos_x], dim=1).reshape(batch_size, 3, 3)

    rotation_matrix = x_part @ y_part @ z_part
    return rotation_matrix


def translate_pose(position, angles, translation):
    rotation_matrix = rotation_matrix_from_angles(angles)
    translated_position = torch.matmul(rotation_matrix, translation[:, :, None]) + position[:, :, None]
    return translated_position[:, :, 0]
