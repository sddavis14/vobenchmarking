from launch import LaunchDescription
from launch_ros.actions import Node
import launch_ros
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    return LaunchDescription([
        launch_ros.actions.SetParameter(name='use_sim_time', value=True),
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
            package='rtabmap_odom',
            executable='stereo_odometry',
            name='stereo_odometry_node',
            remappings=[
                ('left/image_rect', '/camera_left/image_rect'),
                ('left/camera_info', '/camera_left/camera_info'),
                ('right/image_rect', '/camera_right/image_rect'),
                ('right/camera_info', '/camera_right/camera_info'),
            ],

            parameters=[get_package_share_directory('vo_eval_ros') + '/config/odometer_params.yml'],
            output='screen'
        ),
        Node(
            package='tf2_ros', 
            executable='static_transform_publisher', 
            arguments=['0', '0.125', '0', '1.57', '0', '1.57', 'base_link', 'left_camera',  ], 
            output='screen'
        ),
        Node(
            package='tf2_ros', 
            executable='static_transform_publisher', 
            arguments=['0', '-0.125', '0', '1.57', '0', '1.57', 'base_link', 'right_camera',  ], 
            output='screen'
        ),
        Node(
            package='tf2_ros', 
            executable='static_transform_publisher', 
            arguments=['0', '0', '0', '-1.57', '-1.57', '0', 'left_camera', 'left_camera_ned',  ], 
            output='screen'
        ),
        Node(
            package='tf2_ros', 
            executable='static_transform_publisher', 
            arguments=['0', '0', '0', '-1.57', '-1.57', '0', 'right_camera', 'right_camera_ned',  ], 
            output='screen'
        ),
        Node(
            package='tf2_ros', 
            executable='static_transform_publisher', 
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom',  ], 
            output='screen'
        ),
        Node(
            package='rviz2', 
            executable='rviz2', 
            arguments=[], 
            output='screen'
        ),
        # Node(
        #     package='tf2_ros', 
        #     executable='static_transform_publisher', 
        #     arguments=['0', '0', '0', '0', '0', '0', 'odom_left', 'map'], 
        #     output='screen'
        # ),
    ])
