#import rclpy
from rclpy.serialization import serialize_message
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
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
framerate = 20

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
        name='image_left',
        type='sensor_msgs/msg/Image',
        serialization_format='cdr')
    writer.create_topic(left_topic)

    right_topic = rosbag2_py._storage.TopicMetadata(
        name='image_right',
        type='sensor_msgs/msg/Image',
        serialization_format='cdr')
    writer.create_topic(right_topic)

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
        rgb_img = img.convert('RGB')

        msg = RosImage()
        msg.height = rgb_img.height
        msg.width = rgb_img.width
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = 3 * rgb_img.width
        msg.data = np.array(rgb_img).tobytes()

        ns = (int) (i * 1000000 * (1/framerate) * 1000)
        t = Time(seconds=0, nanoseconds=ns)
        msg.header.stamp = t.to_msg()

        writer.write(
                'image_left',
                serialize_message(msg),
                ns)


    for i in range(0, img_count):
        filename = get_right_image_filename(i)
        print(filename)
        img = Image.open(filename, mode='r', formats=None)
        rgb_img = img.convert('RGB')

        msg = RosImage()
        msg.height = rgb_img.height
        msg.width = rgb_img.width
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = 3 * rgb_img.width
        msg.data = np.array(rgb_img).tobytes()

        ns = (int) (i * 1000000 * (1/framerate) * 1000)
        t = Time(seconds=0, nanoseconds=ns)
        msg.header.stamp = t.to_msg()

        writer.write(
                'image_right',
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