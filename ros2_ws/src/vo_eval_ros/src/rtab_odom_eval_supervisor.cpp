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

    auto node = rclcpp::Node::make_shared("rtab_odom_eval_supervisor");

    auto client =
            node->create_client<rtabmap_msgs::srv::ResetPose>("reset_odom_to_pose");

    auto request = std::make_shared<rtabmap_msgs::srv::ResetPose::Request>();
    auto tf_broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(node);
    auto tf_buffer = std::make_shared<tf2_ros::Buffer>(node->get_clock());
    auto tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

    while (!client->wait_for_service(1s)) {
        if (!rclcpp::ok()) {
            RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Interrupted while waiting for the service. Exiting.");
            return 0;
        }
        RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "service not available, waiting again...");
    }

    auto odom_publisher = node->create_publisher<nav_msgs::msg::Odometry>("gt_odom", 10);

    bool initialized = false;
    auto odom_cb = [&](const nav_msgs::msg::Odometry& odom_msg) {
        auto request = std::make_shared<rtabmap_msgs::srv::ResetPose::Request>();

        auto ros_quat = odom_msg.pose.pose.orientation;
        auto ros_pos = odom_msg.pose.pose.position;

        Eigen::Isometry3d odom_to_left_camera_ned;
        odom_to_left_camera_ned.translation() = Eigen::Vector3d(ros_pos.x, ros_pos.y, ros_pos.z);
        odom_to_left_camera_ned.linear() = Eigen::Quaterniond(ros_quat.w, ros_quat.x, ros_quat.y, ros_quat.z).toRotationMatrix();
        odom_to_left_camera_ned.makeAffine();

        auto transform =
                tf_buffer->lookupTransform("base_link", "left_camera_ned", tf2::TimePointZero);
        Eigen::Isometry3d left_camera_ned_to_base_link;
        left_camera_ned_to_base_link = tf2::transformToEigen(transform);

        Eigen::Isometry3d odom_to_base_link = odom_to_left_camera_ned * left_camera_ned_to_base_link;

        Eigen::Quaterniond quat(odom_to_base_link.linear());

        auto gt_odom = odom_msg;
        gt_odom.child_frame_id = "base_link";
        gt_odom.pose.pose.orientation.w = quat.w();
        gt_odom.pose.pose.orientation.x = quat.x();
        gt_odom.pose.pose.orientation.y = quat.y();
        gt_odom.pose.pose.orientation.z = quat.z();

        gt_odom.pose.pose.position.x = odom_to_base_link.translation().x();
        gt_odom.pose.pose.position.y = odom_to_base_link.translation().y();
        gt_odom.pose.pose.position.z = odom_to_base_link.translation().z();

        odom_publisher->publish(gt_odom);

        geometry_msgs::msg::TransformStamped t;

        t.header = gt_odom.header;
        t.child_frame_id = "base_link_gt";

        t.transform.translation.x = gt_odom.pose.pose.position.x;
        t.transform.translation.y = gt_odom.pose.pose.position.y;
        t.transform.translation.z = gt_odom.pose.pose.position.z;

        t.transform.rotation.w = quat.w();
        t.transform.rotation.x = quat.x();
        t.transform.rotation.y = quat.y();
        t.transform.rotation.z = quat.z();

        tf_broadcaster->sendTransform(t);
    };

    auto odom_sub =
            node->create_subscription<nav_msgs::msg::Odometry>("odom_left", 10, odom_cb);

    rclcpp::spin(node);
    rclcpp::shutdown();
}