cmake_minimum_required(VERSION 3.8)
project(bcr_arm_demo)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(control_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

add_executable(physics_pick_place_red_block_demo
  src/physics_pick_place_red_block_demo.cpp
)

ament_target_dependencies(physics_pick_place_red_block_demo
  rclcpp
  rclcpp_action
  control_msgs
  trajectory_msgs
  geometry_msgs
  moveit_ros_planning_interface
  tf2
  tf2_geometry_msgs
)

install(TARGETS
  physics_pick_place_red_block_demo
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

