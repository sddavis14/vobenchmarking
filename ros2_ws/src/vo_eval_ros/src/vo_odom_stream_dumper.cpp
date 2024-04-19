#include "rclcpp/rclcpp.hpp"
#include <memory>
#include <chrono>
#include <rtabmap_msgs/rtabmap_msgs/srv/reset_pose.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_eigen/tf2_eigen/tf2_eigen.hpp>
#include <fstream>

#include <geometry_msgs/msg/transform_stamped.hpp>

using namespace std::chrono_literals;

int main(int argc, char **argv) {
    std::ofstream gtData("./evaluator/results/gt.csv");
    std::ofstream predictedData("./evaluator/results/predicted.csv");

    if (!gtData.is_open() || !predictedData.is_open()) {
        std::cerr << "Failed to open file for writing.\n";
        return 1;
    }
    rclcpp::init(argc, argv);

    auto node = rclcpp::Node::make_shared("vo_odom_stream_dumper");

    auto gt_odom_dump = [&](const nav_msgs::msg::Odometry& odom_msg) {
        auto ros_quat = odom_msg.pose.pose.orientation;
        auto ros_pos = odom_msg.pose.pose.position;
        auto gt_odom = odom_msg;
        gtData << ros_pos.x << ","<< ros_pos.y << ","<< ros_pos.z << ","<< ros_quat.w << "," << ros_quat.x << "," << ros_quat.y<< "," << ros_quat.z << "\n";
    };

    auto odom_dump = [&](const nav_msgs::msg::Odometry& odom_msg) {
        auto ros_quat = odom_msg.pose.pose.orientation;
        auto ros_pos = odom_msg.pose.pose.position;
        auto gt_odom = odom_msg;
        predictedData << ros_pos.x << ","<< ros_pos.y << ","<< ros_pos.z << ","<< ros_quat.w << "," << ros_quat.x << "," << ros_quat.y<< "," << ros_quat.z << "\n";
    };

    auto odom_sub_1 =
            node->create_subscription<nav_msgs::msg::Odometry>("gt_odom", 10, gt_odom_dump);

    auto odom_sub_2 =
            node->create_subscription<nav_msgs::msg::Odometry>("odom", 10, odom_dump);

    rclcpp::spin(node);
    rclcpp::shutdown();
}