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


def main(args=None):
    writer = rosbag2_py.SequentialWriter()

    storage_options = rosbag2_py._storage.StorageOptions(
        uri='tartan_air_bag',
        storage_id='mcap')
    converter_options = rosbag2_py._storage.ConverterOptions('', '')
    writer.open(storage_options, converter_options)

    topic_info = rosbag2_py._storage.TopicMetadata(
        name='image',
        type='sensor_msgs/msg/CompressedImage',
        serialization_format='cdr')
    writer.create_topic(topic_info)

    img = Image.open('sample.png', mode='r', formats=None)

    # rgb_img = img.convert('RGB')

    output = io.BytesIO()
    img.save(output, format='png')
    hex_data = output.getvalue()

    msg = CompressedImage()
    msg.format = 'png'
    msg.data = hex_data
    msg.header.frame_id = "camera"
    

    # msg = RosImage()
    # #msg.header.stamp = rospy.Time.now()
    # msg.height = rgb_img.height
    # msg.width = rgb_img.width
    # msg.encoding = "rgb8"
    # msg.is_bigendian = False
    # msg.step = 3 * rgb_img.width
    # msg.data = np.array(rgb_img).tobytes()

    for i in range(0, 1000):
        ns = (int) (i * 1000000 * 16.6)
        t = Time(seconds=0, nanoseconds=ns)
        msg.header.stamp = t.to_msg()

        writer.write(
                'image',
                serialize_message(msg),
                ns)



if __name__ == '__main__':
    main()