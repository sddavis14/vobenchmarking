#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from rclpy import qos
from nav_msgs.msg import Odometry
import transforms3d as tfs
from model.full_model import UnDeepVO
from PIL import Image
import io
import numpy as np
import torch
from helpers import generate_transformation
from ament_index_python.packages import get_package_share_directory
from matplotlib import pyplot as plt
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from helpers import msg_to_se3
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy

class UnDeepVOModelRunner(Node):
    def __init__(self):
        super().__init__('undeepvo_model_runner')

        self.tf_buffer = Buffer(rclpy.time.Duration(seconds=100.0, nanoseconds=0))
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        self.figure = plt.figure()
        plt.title('Left depth')
        self.depth_img = plt.imshow(np.zeros(shape=(192, 512)), interpolation='nearest', cmap='inferno', vmin=0, vmax=2)

        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        self.prev_img = None
        self.img_resize_height = 256
        self.img_resize_width = 512
        self.bottom_crop_pixels = 64

        self.device = torch.device('cpu')
        self.model = UnDeepVO().to(self.device)

        state_dict = torch.load(get_package_share_directory('vo_eval_ros') + '/four_seasons_weights.tar.gz', map_location=self.device)

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove 'module.' of DataParallel/DistributedDataParallel
            new_state_dict[name] = v

        self.model.load_state_dict(new_state_dict)
        self.model.eval()

        self.odom_to_camera = None

        self.left_sub = self.create_subscription(
            CompressedImage,
            'camera_left/image_rect/compressed',
            self.right_cam_sub,
            10)

    def handle_init(self, stamp):
        if self.odom_to_camera is not None:
            return True

        self.get_logger().info('Attempting to initialize pose')
        try:
            t = self.tf_buffer.lookup_transform(
                'odom',
                'base_link_gt',
                stamp,
                rclpy.time.Duration(seconds=0.5, nanoseconds=0))

            mat = tfs.quaternions.quat2mat([t.transform.rotation.w,
                                            t.transform.rotation.x,
                                            t.transform.rotation.y,
                                            t.transform.rotation.z,])
            transform = np.identity(4, dtype=np.float32)
            transform[:3, :3] = mat
            transform[0, 3] = t.transform.translation.x
            transform[1, 3] = t.transform.translation.y
            transform[2, 3] = t.transform.translation.z
            transform[3, 0] = 0
            transform[3, 1] = 0
            transform[3, 2] = 0
            transform[3, 3] = 1
            self.odom_to_camera = transform
            print(self.odom_to_camera)
            self.get_logger().info('Initialized pose with coordinate transform')
            return True
        except TransformException as ex:
            self.get_logger().error(
                f'Could not transform odom to base_link_gt: {ex}')
            return False

    def msg_to_numpy(self, msg):
        image_data = msg.data
        image = Image.open(io.BytesIO(image_data))
        resized = image.resize((self.img_resize_width, self.img_resize_height))
        new_height = self.img_resize_height - self.bottom_crop_pixels

        img_numpy = (np.asarray(resized, dtype=np.float32)[0:new_height] / 127.5) - 1
        expanded = np.expand_dims(np.repeat(np.expand_dims(img_numpy, axis=0), 3, axis=0), axis=0)
        return expanded

    def mat_to_odom(self, mat):
        odom = Odometry()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        quat = tfs.quaternions.mat2quat(self.odom_to_camera[0:3, 0:3])

        odom.pose.pose.orientation.w = quat[0]
        odom.pose.pose.orientation.x = quat[1]
        odom.pose.pose.orientation.y = quat[2]
        odom.pose.pose.orientation.z = quat[3]

        odom.pose.pose.position.x = float(self.odom_to_camera[0, 3])
        odom.pose.pose.position.y = float(self.odom_to_camera[1, 3])
        odom.pose.pose.position.z = float(self.odom_to_camera[2, 3])

        return odom

    def right_cam_sub(self, msg):
        if not self.handle_init(msg.header.stamp):
            return

        current_img = self.msg_to_numpy(msg)

        if self.prev_img is None:
            self.prev_img = self.msg_to_numpy(msg)
            return

        with torch.no_grad():
            current_torch = torch.from_numpy(current_img).to(self.device)
            prev_torch = torch.from_numpy(self.prev_img).to(self.device)
            depth, (rotation, translation) = self.model(prev_torch, current_torch)
            transform = generate_transformation(translation, rotation)
            current_to_next = transform.cpu().numpy()[0]
            self.odom_to_camera = self.odom_to_camera @ current_to_next
            self.depth_img.set_data(np.log10(depth.cpu().numpy()[0, 0]))
            odom = self.mat_to_odom(self.odom_to_camera)
            odom.header.stamp = msg.header.stamp
            self.odom_pub.publish(odom)

        self.figure.canvas.draw()
        plt.pause(0.001)
        self.prev_img = self.msg_to_numpy(msg)


def main(args=None):
    rclpy.init(args=args)

    model_runner = UnDeepVOModelRunner()

    rclpy.spin(model_runner)

    model_runner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
