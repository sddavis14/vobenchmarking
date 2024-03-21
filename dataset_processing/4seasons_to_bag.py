# import rclpy
from rclpy.serialization import serialize_message
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image as RosImage
from PIL import Image
import io
import rosbag2_py
from rclpy.time import Time
import numpy as np

folder = 'recording/undistorted_images'
seq = 'P001'
image_min = 1585068331330778880
image_max = 1585068756635811584

import os


def main(args=None):
    writer = rosbag2_py.SequentialWriter()

    storage_options = rosbag2_py._storage.StorageOptions(
        uri='4seasonsbag',
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

    directory = os.fsencode(folder+'/cam0')
    limit: int = 20

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        print(filename)
        if filename.endswith(".png"):
            img = Image.open(folder + '/' + 'cam0/' + filename, mode='r', formats=None)

            rgb_img = img.convert('RGB')

            msg = RosImage()
            msg.height = rgb_img.height
            msg.width = rgb_img.width
            msg.encoding = "rgb8"
            msg.is_bigendian = False
            msg.step = 3 * rgb_img.width
            msg.data = np.array(rgb_img).tobytes()

            ns = (int)(filename.strip('.')[0])
            t = Time(seconds=0, nanoseconds=ns)
            msg.header.stamp = t.to_msg()

            writer.write(
                'image_left',
                serialize_message(msg),
                ns)
            limit -= 1
            if limit == 0:
                break
        else:
            continue

    directory = os.fsencode(folder+'/cam1')

    limit = 20

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
            img = Image.open(folder + '/' + 'cam1/' + filename, mode='r', formats=None)

            rgb_img = img.convert('RGB')

            msg = RosImage()
            msg.height = rgb_img.height
            msg.width = rgb_img.width
            msg.encoding = "rgb8"
            msg.is_bigendian = False
            msg.step = 3 * rgb_img.width
            msg.data = np.array(rgb_img).tobytes()

            ns = (int)(filename.strip('.')[0])
            t = Time(seconds=0, nanoseconds=ns)
            msg.header.stamp = t.to_msg()

            writer.write(
                'image_right',
                serialize_message(msg),
                ns)
            limit -= 1
            if limit == 0:
                break
        else:
            continue


if __name__ == '__main__':
    main()
