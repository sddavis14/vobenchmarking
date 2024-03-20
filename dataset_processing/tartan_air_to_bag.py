#import rclpy
from rclpy.serialization import serialize_message
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image as RosImage
from PIL import Image
import io
import rosbag2_py
from rclpy.time import Time
import numpy as np

folder = 'abandonedfactory_sample_P001'
seq = 'P001'
image_min = 0
image_max = 433

def main(args=None):
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


    for i in range(image_min, image_max + 1):
        image_name = '{:06d}_left.png'.format(i)
        print(image_name)
        img = Image.open(folder + '/' + seq + '/' +  'image_left/' + image_name, mode='r', formats=None)

        rgb_img = img.convert('RGB')

        msg = RosImage()
        msg.height = rgb_img.height
        msg.width = rgb_img.width
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = 3 * rgb_img.width
        msg.data = np.array(rgb_img).tobytes()

        ns = (int) (i * 1000000 * 33.33333)
        t = Time(seconds=0, nanoseconds=ns)
        msg.header.stamp = t.to_msg()

        writer.write(
                'image_left',
                serialize_message(msg),
                ns)


    for i in range(image_min, image_max + 1):
        image_name = '{:06d}_right.png'.format(i)
        print(image_name)
        img = Image.open(folder + '/' + seq + '/' +  'image_right/' + image_name, mode='r', formats=None)

        rgb_img = img.convert('RGB')

        msg = RosImage()
        msg.height = rgb_img.height
        msg.width = rgb_img.width
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = 3 * rgb_img.width
        msg.data = np.array(rgb_img).tobytes()

        ns = (int) (i * 1000000 * 33.33333)
        t = Time(seconds=0, nanoseconds=ns)
        msg.header.stamp = t.to_msg()

        writer.write(
                'image_right',
                serialize_message(msg),
                ns)



if __name__ == '__main__':
    main()