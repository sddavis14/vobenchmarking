import rclpy
from rclpy.node import Node


class Evaluator(Node):
    def __init__(self):
        super().__init__('vo_evaluator')
        self.get_logger().info("Hello, I am evaluator")


def main(args=None):
    rclpy.init(args=args)
    eval_node = Evaluator()
    rclpy.spin(eval_node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()