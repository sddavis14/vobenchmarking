from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='rtabmap_odom',
            executable='stereo_odometry',
            name='stereo_odometry_node',
            remappings=[
                ('left/image_rect', 'camera_left/image_rect'),
                ('left/camera_info', 'camera_left/camera_info'),
                ('right/image_rect', 'camera_right/image_rect'),
                ('right/camera_info', 'camera_right/camera_info'),
            ],
            parameters=['/home/spencer/vobenchmarking/ros2_ws/src/vo_eval_ros/config/odometer_params.yml'],
        ),
        Node(
            package='image_transport',
            executable='republish',
            name='republish_left',
            remappings=[
            ],
            arguments=['compressed', 'raw', 
                       '--ros-args', 
                       '--remap', 'in/compressed:=/camera_left/image_rect/compressed', 
                       '--remap', 'out:=/camera_left/image_rect' ]
        ),
        Node(
            package='image_transport',
            executable='republish',
            name='republish_right',
            remappings=[
            ],
            arguments=['compressed', 'raw', 
                       '--ros-args', 
                       '--remap', 'in/compressed:=/camera_right/image_rect/compressed', 
                       '--remap', 'out:=/camera_right/image_rect' ]
        ),
        Node(
            package='tf2_ros', 
            executable='static_transform_publisher', 
            arguments=['-0.125', '0', '0', '0', '0', '0', 'base_link', 'camera_left'], 
            output='screen'
        ),
        Node(
            package='tf2_ros', 
            executable='static_transform_publisher', 
            arguments=['0.125', '0', '0', '0', '0', '0', 'base_link', 'camera_right'], 
            output='screen'
        ),
        # Node(
        #     package='tf2_ros', 
        #     executable='static_transform_publisher', 
        #     arguments=['0', '0', '0', '0', '0', '0', 'odom_left', 'map'], 
        #     output='screen'
        # ),
    ])
