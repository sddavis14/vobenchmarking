cmake_minimum_required(VERSION 3.8)
project(vo_eval_ros)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package (Eigen3 3.3 REQUIRED)
find_package(rtabmap_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)

ament_python_install_package(${PROJECT_NAME})

add_executable(rtab_odom_eval_supervisor src/rtab_odom_eval_supervisor.cpp)
target_link_libraries (rtab_odom_eval_supervisor Eigen3::Eigen)

add_executable(rtab_gnss_odom_eval_supervisor src/rtab_gnss_odom_eval_supervisor.cpp)
target_link_libraries (rtab_gnss_odom_eval_supervisor Eigen3::Eigen)

ament_target_dependencies(rtab_odom_eval_supervisor rclcpp rtabmap_msgs nav_msgs tf2_ros tf2_eigen)
ament_target_dependencies(rtab_gnss_odom_eval_supervisor rclcpp rtabmap_msgs nav_msgs tf2_ros tf2_eigen)

install(TARGETS
  rtab_odom_eval_supervisor
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS
  rtab_gnss_odom_eval_supervisor
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
