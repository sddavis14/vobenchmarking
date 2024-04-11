import numpy as np


def to_euler_angles(quaternion: list) -> tuple:
    """
    Generates the euler angles from quaternion.
    """
    q0 = quaternion[0]
    q1 = quaternion[1]
    q2 = quaternion[2]
    q3 = quaternion[3]

    t0 = +2.0 * (q3 * q0 + q1 * q2)
    t1 = +1.0 - 2.0 * (q0 * q0 + q1 * q1)
    roll_x = np.arctan2(t0, t1)

    t2 = +2.0 * (q3 * q1 - q2 * q0)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = np.arcsin(t2)

    t3 = +2.0 * (q3 * q2 + q0 * q1)
    t4 = +1.0 - 2.0 * (q1 * q1 + q2 * q2)
    yaw_z = np.arctan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians


def extract_pose(single_pose_data: list) -> tuple:
    """
    Converts data of a single pose such as the translation vector and quaternion
    into Euler angles.
    """
    translation = single_pose_data[1:4]
    quaternion = single_pose_data[4:8]
    euler_angles = to_euler_angles(quaternion)
    pose = ((translation[0], translation[1], translation[2]), euler_angles)
    return pose


def preprocess():
    # Input: the dataset directory that contains data for single day.
    path = '/Users/devadattamandaogane/NCSU_Subjects/Robotics/vobenchmarking/'
    dataset_dir = path+'recording/'
    poses_dir = dataset_dir+'poses/'
    image_dir = dataset_dir+'undistorted_images/'
    camera_choice = image_dir+'cam0/'

    # Get all the timestamps in a common timestamp array.
    timestamps = np.loadtxt(dataset_dir + 'times.txt', delimiter=' ')

    # Get all the poses in a common pose array
    poses = np.loadtxt(poses_dir + 'GNSSPoses.txt', delimiter=',')

    # Iterate over each pose and convert the quaternion to Rotation matrix
    #
    processed_poses = []
    for pose in poses:
        processed_pose = extract_pose(pose)
        processed_poses.append(processed_pose)

    processed_poses = np.array(processed_poses)
    print(processed_poses)
    np.save('poses.npy', processed_poses)

    # Convert quaternions to rotation matrices.

    '''
    Convert Pose Quaternion to Rotation Matrix and Translation Vector.
    Monocular image sequence:
    At each image, subtract RGB of Image by mean RGB values of the training set.
    Resize to a multiple of 64, i.e. 64a*64b*3
    Stack two consecutive images -> tensor
    '''

if __name__ == '__main__':
    preprocess()
