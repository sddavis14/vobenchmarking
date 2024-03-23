#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import String
from PIL import Image
import io
import numpy as np


def decompress(msg):
    img = RosImage()
    img.header.stamp = msg.header.stamp
    img.header.frame_id = msg.header.frame_id

    image = Image.open(io.BytesIO(msg.data), formats=['jpeg'])
    rgb_img = image.convert('RGB')

    img.height = rgb_img.height
    img.width = rgb_img.width
    img.encoding = "rgb8"
    img.is_bigendian = False
    img.step = 3 * rgb_img.width
    img.data = np.array(rgb_img).tobytes()

    return img


class ImageDecompressor(Node):
    def __init__(self):
        super().__init__('image_decompressor_left')
        self.left_sub = self.create_subscription(
            CompressedImage,
            'camera_left/image_rect/compressed',
            self.left_cam_sub,
            10)
        self.left_sub  # prevent unused variable warning

        self.left_pub = self.create_publisher(RosImage, 'camera_left/image_rect', 10)

    def left_cam_sub(self, msg):
        print('publish left')
        self.left_pub.publish(decompress(msg))


def main(args=None):
    rclpy.init(args=args)

    image_decompressor = ImageDecompressor()

    rclpy.spin(image_decompressor)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    image_decompressor.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
