#!/usr/bin/env python3

#import rclpy
from rclpy.serialization import serialize_message
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image as RosImage
from nav_msgs.msg import Odometry

from PIL import Image
import io
import rosbag2_py
from rclpy.time import Time
import numpy as np
import os

folder = 'abandonedfactory/Easy'
seq = 'P000'
framerate = 2

def get_left_image_filename(idx):
    image_name = '{:06d}_left.png'.format(idx)
    name = folder + '/' + seq + '/image_left/' + image_name
    return name

def get_right_image_filename(idx):
    image_name = '{:06d}_right.png'.format(idx)
    name = folder + '/' + seq + '/image_right/' + image_name
    return name


def main(args=None):
    img_count = len(os.listdir(folder + '/' + seq + '/image_right/'))
    print(f'{img_count} images found in sequence ' + seq + ' of ' + folder)

    writer = rosbag2_py.SequentialWriter()

    storage_options = rosbag2_py._storage.StorageOptions(
        uri='tartan_air_bag',
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

    odom_left_topic = rosbag2_py._storage.TopicMetadata(
        name='odom_left',
        type='nav_msgs/msg/Odometry',
        serialization_format='cdr')
    writer.create_topic(odom_left_topic)

    odom_right_topic = rosbag2_py._storage.TopicMetadata(
        name='odom_right',
        type='nav_msgs/msg/Odometry',
        serialization_format='cdr')
    writer.create_topic(odom_right_topic)

    for i in range(0, img_count):
        filename = get_left_image_filename(i)
        print(filename)
        img = Image.open(filename, mode='r', formats=None)

        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='jpeg')
        img_byte_arr = img_byte_arr.getvalue()

        msg = CompressedImage()
        msg.format = 'jpeg'
        msg.data = np.array(img_byte_arr).tobytes()

        ns = (int) (i * 1000000 * (1/framerate) * 1000)
        t = Time(seconds=0, nanoseconds=ns)
        msg.header.stamp = t.to_msg()
        msg.header.frame_id = 'left_camera'

        info_msg = CameraInfo()
        info_msg.header = msg.header
        info_msg.height = 480
        info_msg.width = 640
        info_msg.k = np.array([320, 0, 320,
                               0, 320, 240,
                               0, 0, 1])
        info_msg.r = np.array([1, 0, 0,
                               0, 1, 0,
                               0, 0, 1])
        info_msg.p = np.array([320, 0, 320, 0,
                               0, 320, 240, 0,
                               0, 0, 1, 0])

        writer.write('camera_left/camera_info',
                serialize_message(info_msg),
                ns)
        
        writer.write(
                'camera_left/image_rect/compressed',
                serialize_message(msg),
                ns)

    for i in range(0, img_count):
        filename = get_right_image_filename(i)
        print(filename)
        img = Image.open(filename, mode='r', formats=None)

        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='jpeg')
        img_byte_arr = img_byte_arr.getvalue()

        msg = CompressedImage()
        msg.format = 'jpeg'
        msg.data = np.array(img_byte_arr).tobytes()

        ns = (int) (i * 1000000 * (1/framerate) * 1000)
        t = Time(seconds=0, nanoseconds=ns)
        msg.header.stamp = t.to_msg()
        msg.header.frame_id = 'right_camera'

        info_msg = CameraInfo()
        info_msg.header = msg.header
        info_msg.height = 480
        info_msg.width = 640
        info_msg.k = np.array([320, 0, 320,
                               0, 320, 240,
                               0, 0, 1])
        info_msg.r = np.array([1, 0, 0,
                               0, 1, 0,
                               0, 0, 1])
        info_msg.p = np.array([320, 0, 320, 0,
                               0, 320, 240, 0,
                               0, 0, 1, 0])

        writer.write('camera_right/camera_info',
                serialize_message(info_msg),
                ns)

        writer.write(
                'camera_right/image_rect/compressed',
                serialize_message(msg),
                ns)

    left_camera_gt = np.loadtxt(folder + '/' + seq + '/pose_left.txt')
    right_camera_gt = np.loadtxt(folder + '/' + seq + '/pose_right.txt')
    
    print('Writing left camera ground truth')
    for i in range(0, img_count):
        msg = Odometry()
        ns = (int) (i * 1000000 * (1/framerate) * 1000)
        t = Time(seconds=0, nanoseconds=ns)
        msg.header.stamp = t.to_msg()
        msg.header.frame_id = 'map'
        msg.child_frame_id = 'left_camera'

        x = left_camera_gt[i][0]
        y = left_camera_gt[i][1]
        z = left_camera_gt[i][2]

        qx = left_camera_gt[i][3]
        qy = left_camera_gt[i][4]
        qz = left_camera_gt[i][5]
        qw = left_camera_gt[i][6]

        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = z

        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        writer.write(
                'odom_left',
                serialize_message(msg),
                ns)
        
    print('Writing right camera ground truth')
    for i in range(0, img_count):
        msg = Odometry()
        ns = (int) (i * 1000000 * (1/framerate) * 1000)
        t = Time(seconds=0, nanoseconds=ns)
        msg.header.stamp = t.to_msg()
        msg.header.frame_id = 'map'
        msg.child_frame_id = 'right_camera'

        x = right_camera_gt[i][0]
        y = right_camera_gt[i][1]
        z = right_camera_gt[i][2]

        qx = right_camera_gt[i][3]
        qy = right_camera_gt[i][4]
        qz = right_camera_gt[i][5]
        qw = right_camera_gt[i][6]

        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = z

        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        writer.write(
                'odom_right',
                serialize_message(msg),
                ns)


if __name__ == '__main__':
    main()