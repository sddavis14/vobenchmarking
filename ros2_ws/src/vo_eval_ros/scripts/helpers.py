import numpy as np
import torch
import transforms3d as tr

from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Transform
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Vector3

import numpy as np


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


def pose_to_pq(msg):
    """Convert a C{geometry_msgs/Pose} into position/quaternion np arrays

    @param msg: ROS message to be converted
    @return:
      - p: position as a np.array
      - q: quaternion as a numpy array (order = [x,y,z,w])
    """
    p = np.array([msg.position.x, msg.position.y, msg.position.z])
    q = np.array([msg.orientation.x, msg.orientation.y,
                  msg.orientation.z, msg.orientation.w])
    return p, q


def pose_stamped_to_pq(msg):
    """Convert a C{geometry_msgs/PoseStamped} into position/quaternion np arrays

    @param msg: ROS message to be converted
    @return:
      - p: position as a np.array
      - q: quaternion as a numpy array (order = [x,y,z,w])
    """
    return pose_to_pq(msg.pose)


def transform_to_pq(msg):
    """Convert a C{geometry_msgs/Transform} into position/quaternion np arrays

    @param msg: ROS message to be converted
    @return:
      - p: position as a np.array
      - q: quaternion as a numpy array (order = [x,y,z,w])
    """
    p = np.array([msg.translation.x, msg.translation.y, msg.translation.z])
    q = np.array([msg.rotation.x, msg.rotation.y,
                  msg.rotation.z, msg.rotation.w])
    return p, q


def transform_stamped_to_pq(msg):
    """Convert a C{geometry_msgs/TransformStamped} into position/quaternion np arrays

    @param msg: ROS message to be converted
    @return:
      - p: position as a np.array
      - q: quaternion as a numpy array (order = [x,y,z,w])
    """
    return transform_to_pq(msg.transform)


def msg_to_se3(msg):
    """Conversion from geometric ROS messages into SE(3)

    @param msg: Message to transform. Acceptable types - C{geometry_msgs/Pose}, C{geometry_msgs/PoseStamped},
    C{geometry_msgs/Transform}, or C{geometry_msgs/TransformStamped}
    @return: a 4x4 SE(3) matrix as a numpy array
    @note: Throws TypeError if we receive an incorrect type.
    """
    if isinstance(msg, Pose):
        p, q = pose_to_pq(msg)
    elif isinstance(msg, PoseStamped):
        p, q = pose_stamped_to_pq(msg)
    elif isinstance(msg, Transform):
        p, q = transform_to_pq(msg)
    elif isinstance(msg, TransformStamped):
        p, q = transform_stamped_to_pq(msg)
    else:
        raise TypeError("Invalid type for conversion to SE(3)")
    norm = np.linalg.norm(q)
    if np.abs(norm - 1.0) > 1e-3:
        raise ValueError(
            "Received un-normalized quaternion (q = {0:s} ||q|| = {1:3.6f})".format(
                str(q), np.linalg.norm(q)))
    elif np.abs(norm - 1.0) > 1e-6:
        q = q / norm
    g = tr.quaternions.quat2mat(q)
    g[0:3, -1] = p
    return g