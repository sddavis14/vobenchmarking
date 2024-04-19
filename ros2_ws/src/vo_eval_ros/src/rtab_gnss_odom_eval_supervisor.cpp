#include "rclcpp/rclcpp.hpp"
#include <memory>
#include <chrono>
#include <Eigen/Dense>
#include <rtabmap_msgs/rtabmap_msgs/srv/reset_pose.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_eigen/tf2_eigen/tf2_eigen.hpp>

#include <geometry_msgs/msg/transform_stamped.hpp>

using namespace std::chrono_literals;

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);

    auto node = rclcpp::Node::make_shared("rtab_gnss_odom_eval_supervisor");

    auto client = node->create_client<rtabmap_msgs::srv::ResetPose>("reset_odom_to_pose");

    auto request = std::make_shared<rtabmap_msgs::srv::ResetPose::Request>();
    auto tf_broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(node);
    auto tf_buffer = std::make_shared<tf2_ros::Buffer>(node->get_clock());
    auto tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

    auto odom_publisher = node->create_publisher<nav_msgs::msg::Odometry>("gt_odom", 10);

    auto odom_cb = [&](const nav_msgs::msg::Odometry& odom_msg) {
        auto request = std::make_shared<rtabmap_msgs::srv::ResetPose::Request>();

        auto ros_quat = odom_msg.pose.pose.orientation;
        auto ros_pos = odom_msg.pose.pose.position;

        auto gt_odom = odom_msg;
        gt_odom.header.frame_id = "odom";
        gt_odom.child_frame_id = "base_link_gt";
        gt_odom.pose.pose.orientation.w = ros_quat.w;
        gt_odom.pose.pose.orientation.x = ros_quat.x;
        gt_odom.pose.pose.orientation.y = ros_quat.y;
        gt_odom.pose.pose.orientation.z = ros_quat.z;

        gt_odom.pose.pose.position.x = ros_pos.x;
        gt_odom.pose.pose.position.y = ros_pos.y;
        gt_odom.pose.pose.position.z = ros_pos.z;

        odom_publisher->publish(gt_odom);

        geometry_msgs::msg::TransformStamped t;

        t.header = gt_odom.header;
        t.child_frame_id = "base_link_gt";

        t.transform.translation.x = gt_odom.pose.pose.position.x;
        t.transform.translation.y = gt_odom.pose.pose.position.y;
        t.transform.translation.z = gt_odom.pose.pose.position.z;

        t.transform.rotation.w = ros_quat.w;
        t.transform.rotation.x = ros_quat.x;
        t.transform.rotation.y = ros_quat.y;
        t.transform.rotation.z = ros_quat.z;

        tf_broadcaster->sendTransform(t);
    };

    auto odom_sub =
            node->create_subscription<nav_msgs::msg::Odometry>("odom_gnss", 10,  odom_cb);

    rclcpp::spin(node);
    rclcpp::shutdown();
}