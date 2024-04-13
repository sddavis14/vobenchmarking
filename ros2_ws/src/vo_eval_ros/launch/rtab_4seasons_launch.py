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
            parameters=[get_package_share_directory('vo_eval_ros') + '/config/odom_gnss_params.yml'],
            output='screen'
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0.17541216744862287', '0.0036894333751345677', '-0.05810612695941222', '1.5751938', '0.0068314', '3.128069', 'base_link', 'left_camera',  ],
            output='screen'
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['-0.12432919293024627', '0.0023471917960152505', '-0.05662052461099226', '1.576158', '-0.0031757', '3.1411816', 'base_link', 'right_camera',  ],
            output='screen'
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
            output='screen'
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=[],
            output='screen'
        ),
        Node(
            package='vo_eval_ros',
            executable='rtab_odom_eval_supervisor',
            arguments=[],
            output='screen'
        ),
    ])
