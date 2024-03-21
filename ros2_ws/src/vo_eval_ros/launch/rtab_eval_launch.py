from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='rtabmap_odom',
            executable='stereo_odometry',
            name='stereo_odometry_node',
            remappings=[
            ]
        )
    ])
