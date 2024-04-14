# import rclpy
from rclpy.serialization import serialize_message
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image as RosImage
from PIL import Image
from nav_msgs.msg import Odometry
import io
import rosbag2_py
from rclpy.time import Time
import numpy as np

folder = 'recording/'
seq = 'P001'
image_min = 1585068331330778880
image_max = 1585068756635811584
framerate = 20
maxlimit = 1000

import os


def get_full_image_path(cam_key: str, image_name: str) -> str:
    name: str = folder + 'undistorted_images/' + cam_key + image_name + '.png'
    return name

def main(args=None):
    writer = rosbag2_py.SequentialWriter()

    storage_options = rosbag2_py._storage.StorageOptions(
        uri='4seasonsbag',
        storage_id='mcap')
    converter_options = rosbag2_py._storage.ConverterOptions('', '')
    writer.open(storage_options, converter_options)

    left_topic = rosbag2_py._storage.TopicMetadata(
        name='camera_left/image_rect/compressed',
        type='sensor_msgs/msg/CompressedImage',
        serialization_format='cdr')
    writer.create_topic(left_topic)

    left_info_topic = rosbag2_py._storage.TopicMetadata(
        name='camera_left/camera_info',
        type='sensor_msgs/msg/CameraInfo',
        serialization_format='cdr')
    writer.create_topic(left_info_topic)

    right_topic = rosbag2_py._storage.TopicMetadata(
        name='camera_right/image_rect/compressed',
        type='sensor_msgs/msg/CompressedImage',
        serialization_format='cdr')
    writer.create_topic(right_topic)

    right_info_topic = rosbag2_py._storage.TopicMetadata(
        name='camera_right/camera_info',
        type='sensor_msgs/msg/CameraInfo',
        serialization_format='cdr')
    writer.create_topic(right_info_topic)

    odom_gnss_topic = rosbag2_py._storage.TopicMetadata(
        name='odom_gnss',
        type='nav_msgs/msg/Odometry',
        serialization_format='cdr')
    writer.create_topic(odom_gnss_topic)

    camera_gt = np.loadtxt(folder + 'poses/' + 'GNSSPoses.txt', delimiter=',')
    timestamps = np.loadtxt(folder + 'times.txt', delimiter=' ')

    for i in range(0, maxlimit):
        filename = get_full_image_path('cam0/', str(int(timestamps[i][0])))
        img = Image.open(filename, mode='r', formats=None)

        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='png')
        img_byte_arr = img_byte_arr.getvalue()

        msg = CompressedImage()
        msg.format = 'png'
        msg.data = np.array(img_byte_arr).tobytes()

        ns = int(timestamps[i][0])
        t = Time(seconds=0, nanoseconds=ns)
        msg.header.stamp = t.to_msg()
        msg.header.frame_id = 'left_camera'

        info_msg = CameraInfo()
        info_msg.header = msg.header
        info_msg.height = 400
        info_msg.width = 800
        info_msg.k = np.array([527.9990706330082, 0, 399.18451401412665,
                               0, 527.963495807245, 172.8193108347693,
                               0, 0, 1])
        info_msg.r = np.array([1, 0, 0,
                               0, 1, 0,
                               0, 0, 1])
        info_msg.p = np.array([527.9990706330082, 0, 399.18451401412665, 0,
                               0, 527.963495807245, 172.8193108347693, 0,
                               0, 0, 1, 0])

        writer.write('camera_left/camera_info',
                     serialize_message(info_msg),
                     ns)

        writer.write(
            'camera_left/image_rect/compressed',
            serialize_message(msg),
            ns)

    for i in range(0, maxlimit):
        filename = get_full_image_path('cam1/', str(int(timestamps[i][0])))
        img = Image.open(filename, mode='r', formats=None)

        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='png')
        img_byte_arr = img_byte_arr.getvalue()

        msg = CompressedImage()
        msg.format = 'png'
        msg.data = np.array(img_byte_arr).tobytes()

        ns = int(timestamps[i][0])
        t = Time(seconds=0, nanoseconds=ns)
        msg.header.stamp = t.to_msg()
        msg.header.frame_id = 'right_camera'

        info_msg = CameraInfo()
        info_msg.header = msg.header
        info_msg.height = 400
        info_msg.width = 800
        info_msg.k = np.array([529.2496538273606, 0, 412.4733148308946,
                               0, 529.4013679656194, 172.1405434152354,
                               0, 0, 1])
        info_msg.r = np.array([1, 0, 0,
                               0, 1, 0,
                               0, 0, 1])
        info_msg.p = np.array([529.2496538273606, 0, 412.4733148308946, -159.03748965956,
                               0, 529.4013679656194, 172.1405434152354, 0,
                               0, 0, 1, 0])

        writer.write('camera_right/camera_info',
                     serialize_message(info_msg),
                     ns)

        writer.write(
            'camera_right/image_rect/compressed',
            serialize_message(msg),
            ns)

    for i in range(0, maxlimit):
        msg = Odometry()
        ns = int(timestamps[i][0])
        t = Time(seconds=0, nanoseconds=ns)
        msg.header.stamp = t.to_msg()
        msg.header.frame_id = 'map'
        msg.child_frame_id = 'base_link'

        x = camera_gt[i][1]
        y = camera_gt[i][2]
        z = camera_gt[i][3]

        qx = camera_gt[i][4]
        qy = camera_gt[i][5]
        qz = camera_gt[i][6]
        qw = camera_gt[i][7]

        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = z

        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        writer.write(
            'odom_gnss',
            serialize_message(msg),
            ns)


if __name__ == '__main__':
    main()
