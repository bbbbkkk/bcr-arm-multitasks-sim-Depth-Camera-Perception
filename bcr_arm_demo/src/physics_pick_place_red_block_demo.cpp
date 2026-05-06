#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

using namespace std::chrono_literals;

using FollowJointTrajectory = control_msgs::action::FollowJointTrajectory;

static geometry_msgs::msg::Pose make_pose(
  double x,
  double y,
  double z,
  double roll,
  double pitch,
  double yaw)
{
  geometry_msgs::msg::Pose pose;

  pose.position.x = x;
  pose.position.y = y;
  pose.position.z = z;

  tf2::Quaternion q;
  q.setRPY(roll, pitch, yaw);
  q.normalize();

  pose.orientation = tf2::toMsg(q);

  return pose;
}

static bool send_gripper_goal(
  const rclcpp::Node::SharedPtr & node,
  double position,
  double duration_sec)
{
  auto client = rclcpp_action::create_client<FollowJointTrajectory>(
    node,
    "/simple_gripper_controller/follow_joint_trajectory"
  );

  RCLCPP_INFO(node->get_logger(), "Waiting for simple gripper action server...");

  if (!client->wait_for_action_server(5s)) {
    RCLCPP_ERROR(node->get_logger(), "simple_gripper_controller action server not available.");
    return false;
  }

  FollowJointTrajectory::Goal goal;
  goal.trajectory.joint_names = {
    "simple_left_finger_joint",
    "simple_right_finger_joint",
  };

  trajectory_msgs::msg::JointTrajectoryPoint point;
  point.positions = {position, position};

  int sec = static_cast<int>(duration_sec);
  int nanosec = static_cast<int>((duration_sec - sec) * 1e9);

  point.time_from_start.sec = sec;
  point.time_from_start.nanosec = nanosec;

  goal.trajectory.points.push_back(point);

  RCLCPP_INFO(node->get_logger(), "Sending gripper command: %.3f", position);

  auto goal_future = client->async_send_goal(goal);

  if (rclcpp::spin_until_future_complete(node, goal_future, 10s) !=
      rclcpp::FutureReturnCode::SUCCESS)
  {
    RCLCPP_ERROR(node->get_logger(), "Failed to send gripper goal.");
    return false;
  }

  auto goal_handle = goal_future.get();

  if (!goal_handle) {
    RCLCPP_ERROR(node->get_logger(), "Gripper goal rejected.");
    return false;
  }

  auto result_future = client->async_get_result(goal_handle);

  if (rclcpp::spin_until_future_complete(node, result_future, 10s) !=
      rclcpp::FutureReturnCode::SUCCESS)
  {
    RCLCPP_ERROR(node->get_logger(), "Failed to get gripper result.");
    return false;
  }

  auto result = result_future.get().result;

  if (result->error_code != 0) {
    RCLCPP_ERROR(
      node->get_logger(),
      "Gripper command failed. error_code=%d, error_string=%s",
      result->error_code,
      result->error_string.c_str()
    );
    return false;
  }

  RCLCPP_INFO(node->get_logger(), "Gripper command succeeded.");
  return true;
}

static bool execute_cartesian_to_pose(
  const rclcpp::Node::SharedPtr & node,
  moveit::planning_interface::MoveGroupInterface & move_group,
  const std::string & name,
  const geometry_msgs::msg::Pose & target_pose,
  double required_fraction = 0.80,
  bool avoid_collisions = true)
{
  RCLCPP_INFO(
    node->get_logger(),
    "Cartesian move [%s] to x=%.3f y=%.3f z=%.3f",
    name.c_str(),
    target_pose.position.x,
    target_pose.position.y,
    target_pose.position.z
  );

  move_group.setStartStateToCurrentState();

  std::vector<geometry_msgs::msg::Pose> waypoints;
  waypoints.push_back(target_pose);

  moveit_msgs::msg::RobotTrajectory trajectory;

  const double eef_step = 0.005;
  const double jump_threshold = 0.0;

  double fraction = move_group.computeCartesianPath(
    waypoints,
    eef_step,
    jump_threshold,
    trajectory,
    avoid_collisions
  );

  RCLCPP_INFO(
    node->get_logger(),
    "Cartesian path [%s] fraction = %.3f",
    name.c_str(),
    fraction
  );

  if (fraction < required_fraction) {
    RCLCPP_ERROR(
      node->get_logger(),
      "Cartesian path failed: %s, fraction %.3f < %.3f",
      name.c_str(),
      fraction,
      required_fraction
    );
    return false;
  }

  auto result = move_group.execute(trajectory);

  if (result != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(node->get_logger(), "Cartesian execution failed: %s", name.c_str());
    return false;
  }

  RCLCPP_INFO(node->get_logger(), "Cartesian motion succeeded: %s", name.c_str());
  return true;
}

static bool plan_and_execute_pose(
  const rclcpp::Node::SharedPtr & node,
  moveit::planning_interface::MoveGroupInterface & move_group,
  const std::string & name,
  const geometry_msgs::msg::Pose & target_pose)
{
  RCLCPP_INFO(
    node->get_logger(),
    "Plan to [%s]: x=%.3f y=%.3f z=%.3f",
    name.c_str(),
    target_pose.position.x,
    target_pose.position.y,
    target_pose.position.z
  );

  move_group.setStartStateToCurrentState();
  move_group.setPoseTarget(target_pose);

  moveit::planning_interface::MoveGroupInterface::Plan plan;
  const bool planned = static_cast<bool>(move_group.plan(plan));

  move_group.clearPoseTargets();

  if (!planned) {
    RCLCPP_ERROR(node->get_logger(), "Pose planning failed: %s", name.c_str());
    return false;
  }

  const auto result = move_group.execute(plan);

  if (result != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(node->get_logger(), "Pose execution failed: %s", name.c_str());
    return false;
  }

  RCLCPP_INFO(node->get_logger(), "Pose motion succeeded: %s", name.c_str());
  return true;
}

static void set_motion_profile(
  moveit::planning_interface::MoveGroupInterface & move_group,
  double velocity_scale,
  double acceleration_scale)
{
  move_group.setMaxVelocityScalingFactor(velocity_scale);
  move_group.setMaxAccelerationScalingFactor(acceleration_scale);
}

static bool execute_cartesian_waypoints(
  const rclcpp::Node::SharedPtr & node,
  moveit::planning_interface::MoveGroupInterface & move_group,
  const std::string & name,
  const std::vector<geometry_msgs::msg::Pose> & waypoints,
  double required_fraction = 0.95,
  bool avoid_collisions = true)
{
  if (waypoints.empty()) {
    RCLCPP_ERROR(node->get_logger(), "Cartesian waypoints are empty: %s", name.c_str());
    return false;
  }

  const auto & first_pose = waypoints.front();
  const auto & last_pose = waypoints.back();

  RCLCPP_INFO(
    node->get_logger(),
    "Cartesian waypoints [%s]: count=%zu, start=(%.3f, %.3f, %.3f), end=(%.3f, %.3f, %.3f)",
    name.c_str(),
    waypoints.size(),
    first_pose.position.x,
    first_pose.position.y,
    first_pose.position.z,
    last_pose.position.x,
    last_pose.position.y,
    last_pose.position.z
  );

  move_group.setStartStateToCurrentState();

  moveit_msgs::msg::RobotTrajectory trajectory;

  const double eef_step = 0.005;
  const double jump_threshold = 0.0;

  const double fraction = move_group.computeCartesianPath(
    waypoints,
    eef_step,
    jump_threshold,
    trajectory,
    avoid_collisions
  );

  RCLCPP_INFO(
    node->get_logger(),
    "Cartesian waypoints [%s] fraction = %.3f",
    name.c_str(),
    fraction
  );

  if (fraction < required_fraction) {
    RCLCPP_ERROR(
      node->get_logger(),
      "Cartesian waypoints failed: %s, fraction %.3f < %.3f",
      name.c_str(),
      fraction,
      required_fraction
    );
    return false;
  }

  auto result = move_group.execute(trajectory);

  if (result != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(node->get_logger(), "Cartesian waypoints execution failed: %s", name.c_str());
    return false;
  }

  RCLCPP_INFO(node->get_logger(), "Cartesian waypoints succeeded: %s", name.c_str());
  return true;
}

static bool execute_cartesian_waypoints_with_segment_fallback(
  const rclcpp::Node::SharedPtr & node,
  moveit::planning_interface::MoveGroupInterface & move_group,
  const std::string & name,
  const std::vector<geometry_msgs::msg::Pose> & waypoints,
  double required_fraction = 0.95,
  double segment_required_fraction = 0.85,
  bool avoid_collisions = true)
{
  if (execute_cartesian_waypoints(
        node,
        move_group,
        name,
        waypoints,
        required_fraction,
        avoid_collisions))
  {
    return true;
  }

  RCLCPP_WARN(
    node->get_logger(),
    "Cartesian waypoints [%s] fallback to segmented constant-z execution.",
    name.c_str()
  );

  for (size_t i = 0; i < waypoints.size(); ++i) {
    const auto segment_name =
      name + "_segment_" + std::to_string(static_cast<unsigned long long>(i + 1));

    if (!execute_cartesian_to_pose(
          node,
          move_group,
          segment_name,
          waypoints[i],
          segment_required_fraction,
          avoid_collisions))
    {
      return false;
    }

    std::this_thread::sleep_for(150ms);
  }

  return true;
}

struct CartesianExecuteResult
{
  bool success;
  double fraction;
  geometry_msgs::msg::Pose reference_target_pose;
};

static CartesianExecuteResult execute_cartesian_to_pose_with_result(
  const rclcpp::Node::SharedPtr & node,
  moveit::planning_interface::MoveGroupInterface & move_group,
  const std::string & name,
  const geometry_msgs::msg::Pose & target_pose,
  double required_fraction = 0.80,
  bool avoid_collisions = true)
{
  CartesianExecuteResult output;
  output.success = false;
  output.fraction = 0.0;
  output.reference_target_pose = target_pose;

  RCLCPP_INFO(
    node->get_logger(),
    "Cartesian move with result [%s] to x=%.3f y=%.3f z=%.3f",
    name.c_str(),
    target_pose.position.x,
    target_pose.position.y,
    target_pose.position.z
  );

  move_group.setStartStateToCurrentState();

  std::vector<geometry_msgs::msg::Pose> waypoints;
  waypoints.push_back(target_pose);

  moveit_msgs::msg::RobotTrajectory trajectory;

  const double eef_step = 0.005;
  const double jump_threshold = 0.0;

  const double fraction = move_group.computeCartesianPath(
    waypoints,
    eef_step,
    jump_threshold,
    trajectory,
    avoid_collisions
  );

  output.fraction = fraction;

  RCLCPP_INFO(
    node->get_logger(),
    "Cartesian path [%s] fraction = %.3f",
    name.c_str(),
    fraction
  );

  if (fraction < required_fraction) {
    RCLCPP_ERROR(
      node->get_logger(),
      "Purple Cartesian path failed: %s, fraction %.3f < %.3f",
      name.c_str(),
      fraction,
      required_fraction
    );
    return output;
  }

  auto result = move_group.execute(trajectory);

  if (result != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(node->get_logger(), "Cartesian execution failed: %s", name.c_str());
    return output;
  }

  std::this_thread::sleep_for(300ms);

  output.success = true;

  RCLCPP_INFO(
    node->get_logger(),
    "Cartesian motion succeeded: %s, fraction=%.3f",
    name.c_str(),
    output.fraction
  );

  return output;
}

static bool move_named_target(
  const rclcpp::Node::SharedPtr & node,
  moveit::planning_interface::MoveGroupInterface & move_group,
  const std::string & target_name)
{
  auto named_targets = move_group.getNamedTargets();

  if (std::find(named_targets.begin(), named_targets.end(), target_name) == named_targets.end()) {
    RCLCPP_ERROR(
      node->get_logger(),
      "Named target [%s] not found.",
      target_name.c_str()
    );
    return false;
  }

  RCLCPP_INFO(node->get_logger(), "Moving to named target: %s", target_name.c_str());

  move_group.setStartStateToCurrentState();
  move_group.setNamedTarget(target_name);

  moveit::planning_interface::MoveGroupInterface::Plan plan;
  const bool planned = static_cast<bool>(move_group.plan(plan));

  if (!planned) {
    RCLCPP_ERROR(node->get_logger(), "Named target planning failed: %s", target_name.c_str());
    return false;
  }

  const auto result = move_group.execute(plan);

  if (result != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(node->get_logger(), "Named target execution failed: %s", target_name.c_str());
    return false;
  }

  RCLCPP_INFO(node->get_logger(), "Named target motion succeeded: %s", target_name.c_str());
  return true;
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = rclcpp::Node::make_shared(
    "physics_pick_place_red_block_demo",
    rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true)
  );

  RCLCPP_INFO(node->get_logger(), "Starting red + blue + purple stacking demo.");
  RCLCPP_INFO(node->get_logger(), "No attach. No fake teleport. Pure Gazebo physics grasp.");
  RCLCPP_INFO(node->get_logger(), "Horizontal transfer motions use Cartesian paths with layered z heights.");

  const std::string planning_group =
    node->get_parameter_or<std::string>("planning_group", "bcr_arm");

  const std::string eef_link =
    node->get_parameter_or<std::string>("eef_link", "flange");

  moveit::planning_interface::MoveGroupInterface move_group(node, planning_group);

  move_group.setEndEffectorLink(eef_link);
  move_group.setPlanningTime(10.0);
  move_group.setNumPlanningAttempts(10);

  set_motion_profile(move_group, 0.08, 0.06);

  RCLCPP_INFO(node->get_logger(), "Planning group: %s", planning_group.c_str());
  RCLCPP_INFO(node->get_logger(), "Planning frame: %s", move_group.getPlanningFrame().c_str());
  RCLCPP_INFO(node->get_logger(), "End effector link: %s", move_group.getEndEffectorLink().c_str());

  const double red_x = 0.42;
  const double red_y = 0.48;
  const double red_yaw = 0.25;

  const double blue_x = 0.38;
  const double blue_y = -0.32;
  const double blue_yaw = -0.40;

  // 把紫色木块放到红色木块前方，便于从堆叠区安全横移过去抓取。
  const double purple_x = 0.42;
  const double purple_y = 0.26;
  const double purple_yaw = 0.65;

  const double place_x = 0.52;
  const double place_y = 0.00;

  const double cube_size = 0.09;

  const double red_pick_high_z = 0.45;

  const double red_transfer_z = 0.52;
  const double blue_transfer_z = 0.55;
  const double purple_transfer_z = 0.60;

  const double pick_low_z = 0.34;
  const double purple_pick_low_z = 0.36;

  const double place_low_z = 0.37;

  const double red_lift_z = red_transfer_z;
  const double red_place_high_z = red_transfer_z;

  const double blue_safe_high_z = red_transfer_z;
  const double blue_mid_lift_z = red_transfer_z;
  const double blue_lift_z = blue_transfer_z;
  const double blue_place_high_z = blue_transfer_z;

  const double blue_place_low_z = place_low_z + cube_size - 0.005;

  const double purple_safe_high_z = purple_transfer_z;

  // 紫色放置时只保留很小的释放余量，避免悬空过高后自由掉落偏出蓝色木块顶面。
  const double purple_place_low_z = place_low_z + 2.0 * cube_size + 0.005;

  const double purple_lift_safety_margin = 0.015;

  // 补偿上限降低，避免紫色搬运高度太高导致机械臂扭曲。
  const double purple_max_lift_extra = 0.030;

  const double red_gripper_center_offset_x = 0.00;
  const double red_gripper_center_offset_y = 0.00;

  const double blue_gripper_center_offset_x = 0.00;
  const double blue_gripper_center_offset_y = 0.00;

  const double purple_gripper_center_offset_x = 0.00;
  const double purple_gripper_center_offset_y = 0.00;

  const double gripper_open = 0.040;
  const double gripper_pre_close = 0.026;
  const double gripper_close = 0.020;

  // 紫色木块单独夹紧一点，防止滑落。
  const double purple_gripper_pre_close = 0.018;
  const double purple_gripper_close = 0.010;

  const double roll = M_PI;
  const double pitch = 0.0;

  const double gripper_yaw_offset = 1.5708;

  const double red_pick_yaw = red_yaw + gripper_yaw_offset;
  const double blue_pick_yaw = blue_yaw + gripper_yaw_offset;

  // 紫色不用 purple_yaw + pi/2，避免低位 IK 不稳定。
  const double purple_pick_yaw = purple_yaw;

  const std::string final_target = "upright";


  // =========================================================
  // Red block poses
  // =========================================================

  const double red_pick_flange_x =
    red_x
    - std::cos(red_pick_yaw) * red_gripper_center_offset_x
    + std::sin(red_pick_yaw) * red_gripper_center_offset_y;

  const double red_pick_flange_y =
    red_y
    - std::sin(red_pick_yaw) * red_gripper_center_offset_x
    - std::cos(red_pick_yaw) * red_gripper_center_offset_y;

  RCLCPP_INFO(
    node->get_logger(),
    "Red block center: x=%.3f y=%.3f red_yaw=%.3f",
    red_x,
    red_y,
    red_yaw
  );

  RCLCPP_INFO(
    node->get_logger(),
    "Red flange pick target: x=%.3f y=%.3f pick_yaw=%.3f",
    red_pick_flange_x,
    red_pick_flange_y,
    red_pick_yaw
  );

  const auto red_high_pose = make_pose(
    red_pick_flange_x,
    red_pick_flange_y,
    red_pick_high_z,
    roll,
    pitch,
    red_pick_yaw
  );

  const auto red_pick_entry_pose = make_pose(
    place_x,
    place_y,
    red_pick_high_z,
    roll,
    pitch,
    red_pick_yaw
  );

  const auto red_low_pose = make_pose(
    red_pick_flange_x,
    red_pick_flange_y,
    pick_low_z,
    roll,
    pitch,
    red_pick_yaw
  );

  const auto red_lift_pose = make_pose(
    red_pick_flange_x,
    red_pick_flange_y,
    red_lift_z,
    roll,
    pitch,
    red_pick_yaw
  );

  const auto red_place_high_pose = make_pose(
    place_x,
    place_y,
    red_place_high_z,
    roll,
    pitch,
    red_pick_yaw
  );

  const auto red_place_low_pose = make_pose(
    place_x,
    place_y,
    place_low_z,
    roll,
    pitch,
    red_pick_yaw
  );

  // =========================================================
  // Red block task
  // =========================================================

  if (!send_gripper_goal(node, gripper_open, 2.0)) {
    rclcpp::shutdown();
    return 1;
  }

  std::this_thread::sleep_for(600ms);

  if (!plan_and_execute_pose(
        node,
        move_group,
        "move_to_red_pick_entry_high",
        red_pick_entry_pose))
  {
    rclcpp::shutdown();
    return 1;
  }

  std::this_thread::sleep_for(150ms);

  if (!execute_cartesian_to_pose(
        node,
        move_group,
        "cartesian_transfer_entry_to_red_block_high_keep_z",
        red_high_pose,
        0.90))
  {
    rclcpp::shutdown();
    return 1;
  }

  std::this_thread::sleep_for(150ms);

  if (!send_gripper_goal(node, gripper_open, 2.0)) {
    rclcpp::shutdown();
    return 1;
  }

  std::this_thread::sleep_for(900ms);

  if (!execute_cartesian_to_pose(node, move_group, "cartesian_down_to_red_block_sides", red_low_pose, 0.90)) {
    rclcpp::shutdown();
    return 1;
  }

  std::this_thread::sleep_for(150ms);

  RCLCPP_INFO(node->get_logger(), "Pre-close gripper for red block.");

  if (!send_gripper_goal(node, gripper_pre_close, 2.0)) {
    rclcpp::shutdown();
    return 1;
  }

  std::this_thread::sleep_for(400ms);

  RCLCPP_INFO(node->get_logger(), "Final close gripper for red block.");

  if (!send_gripper_goal(node, gripper_close, 2.0)) {
    rclcpp::shutdown();
    return 1;
  }

  std::this_thread::sleep_for(900ms);

  if (!execute_cartesian_to_pose(node, move_group, "cartesian_lift_red_block_to_transfer", red_lift_pose, 0.90)) {
    rclcpp::shutdown();
    return 1;
  }

  std::this_thread::sleep_for(200ms);

  if (!execute_cartesian_to_pose(node, move_group, "cartesian_transfer_red_to_stack_high", red_place_high_pose, 0.80)) {
    rclcpp::shutdown();
    return 1;
  }

  std::this_thread::sleep_for(150ms);

  if (!execute_cartesian_to_pose(node, move_group, "cartesian_down_to_stacking_area", red_place_low_pose, 0.90)) {
    rclcpp::shutdown();
    return 1;
  }

  std::this_thread::sleep_for(150ms);

  if (!send_gripper_goal(node, gripper_open, 2.0)) {
    rclcpp::shutdown();
    return 1;
  }

  std::this_thread::sleep_for(500ms);

  if (!execute_cartesian_to_pose(node, move_group, "cartesian_retreat_from_stacking_area", red_place_high_pose, 0.90)) {
    rclcpp::shutdown();
    return 1;
  }

  std::this_thread::sleep_for(600ms);

  // =========================================================
  // Blue block poses
  // =========================================================

  const double blue_pick_flange_x =
    blue_x
    - std::cos(blue_pick_yaw) * blue_gripper_center_offset_x
    + std::sin(blue_pick_yaw) * blue_gripper_center_offset_y;

  const double blue_pick_flange_y =
    blue_y
    - std::sin(blue_pick_yaw) * blue_gripper_center_offset_x
    - std::cos(blue_pick_yaw) * blue_gripper_center_offset_y;

  RCLCPP_INFO(
    node->get_logger(),
    "Blue block center: x=%.3f y=%.3f blue_yaw=%.3f",
    blue_x,
    blue_y,
    blue_yaw
  );

  RCLCPP_INFO(
    node->get_logger(),
    "Blue flange pick target: x=%.3f y=%.3f pick_yaw=%.3f",
    blue_pick_flange_x,
    blue_pick_flange_y,
    blue_pick_yaw
  );

  const auto blue_high_pose = make_pose(
    blue_pick_flange_x,
    blue_pick_flange_y,
    blue_safe_high_z,
    roll,
    pitch,
    blue_pick_yaw
  );

  const auto blue_low_pose = make_pose(
    blue_pick_flange_x,
    blue_pick_flange_y,
    pick_low_z,
    roll,
    pitch,
    blue_pick_yaw
  );

  const auto blue_mid_lift_pose = make_pose(
    blue_pick_flange_x,
    blue_pick_flange_y,
    blue_mid_lift_z,
    roll,
    pitch,
    blue_pick_yaw
  );

  const auto blue_lift_pose = make_pose(
    blue_pick_flange_x,
    blue_pick_flange_y,
    blue_lift_z,
    roll,
    pitch,
    blue_pick_yaw
  );

  const auto blue_place_high_pose = make_pose(
    place_x,
    place_y,
    blue_place_high_z,
    roll,
    pitch,
    blue_pick_yaw
  );

  const auto blue_place_low_pose = make_pose(
    place_x,
    place_y,
    blue_place_low_z,
    roll,
    pitch,
    blue_pick_yaw
  );

  if (!execute_cartesian_to_pose(node, move_group, "cartesian_transfer_to_blue_block_high", blue_high_pose, 0.80)) {
    rclcpp::shutdown();
    return 1;
  }

  std::this_thread::sleep_for(150ms);

  if (!send_gripper_goal(node, gripper_open, 2.0)) {
    rclcpp::shutdown();
    return 1;
  }

  std::this_thread::sleep_for(900ms);

  if (!execute_cartesian_to_pose(node, move_group, "cartesian_down_to_blue_block_sides", blue_low_pose, 0.90)) {
    rclcpp::shutdown();
    return 1;
  }

  std::this_thread::sleep_for(150ms);

  RCLCPP_INFO(node->get_logger(), "Pre-close gripper for blue block.");

  if (!send_gripper_goal(node, gripper_pre_close, 2.0)) {
    rclcpp::shutdown();
    return 1;
  }

  std::this_thread::sleep_for(400ms);

  RCLCPP_INFO(node->get_logger(), "Final close gripper for blue block.");

  if (!send_gripper_goal(node, gripper_close, 2.0)) {
    rclcpp::shutdown();
    return 1;
  }

  std::this_thread::sleep_for(900ms);

  if (!execute_cartesian_to_pose(node, move_group, "cartesian_lift_blue_block_mid", blue_mid_lift_pose, 0.90)) {
    rclcpp::shutdown();
    return 1;
  }

  std::this_thread::sleep_for(150ms);

  if (!execute_cartesian_to_pose(node, move_group, "cartesian_lift_blue_block_to_transfer", blue_lift_pose, 0.85)) {
    rclcpp::shutdown();
    return 1;
  }

  std::this_thread::sleep_for(200ms);

  if (!execute_cartesian_to_pose(node, move_group, "cartesian_transfer_blue_to_red_block_top_high", blue_place_high_pose, 0.80)) {
    rclcpp::shutdown();
    return 1;
  }

  std::this_thread::sleep_for(150ms);

  if (!execute_cartesian_to_pose(node, move_group, "cartesian_down_to_red_block_top", blue_place_low_pose, 0.90)) {
    rclcpp::shutdown();
    return 1;
  }

  std::this_thread::sleep_for(150ms);

  if (!send_gripper_goal(node, gripper_open, 2.0)) {
    rclcpp::shutdown();
    return 1;
  }

  std::this_thread::sleep_for(500ms);

  if (!execute_cartesian_to_pose(node, move_group, "cartesian_retreat_from_blue_stack", blue_place_high_pose, 0.90)) {
    rclcpp::shutdown();
    return 1;
  }

  std::this_thread::sleep_for(600ms);

  // =========================================================
  // Purple block poses
  // =========================================================

  const double purple_pick_flange_x =
    purple_x
    - std::cos(purple_pick_yaw) * purple_gripper_center_offset_x
    + std::sin(purple_pick_yaw) * purple_gripper_center_offset_y;

  const double purple_pick_flange_y =
    purple_y
    - std::sin(purple_pick_yaw) * purple_gripper_center_offset_x
    - std::cos(purple_pick_yaw) * purple_gripper_center_offset_y;

  RCLCPP_INFO(
    node->get_logger(),
    "Purple block center: x=%.3f y=%.3f purple_yaw=%.3f",
    purple_x,
    purple_y,
    purple_yaw
  );

  RCLCPP_INFO(
    node->get_logger(),
    "Purple flange pick target: x=%.3f y=%.3f pick_yaw=%.3f",
    purple_pick_flange_x,
    purple_pick_flange_y,
    purple_pick_yaw
  );

  const auto purple_high_pose = make_pose(
    purple_pick_flange_x,
    purple_pick_flange_y,
    purple_safe_high_z,
    roll,
    pitch,
    purple_pick_yaw
  );

  const auto purple_low_pose = make_pose(
    purple_pick_flange_x,
    purple_pick_flange_y,
    purple_pick_low_z,
    roll,
    pitch,
    purple_pick_yaw
  );

  const auto purple_place_low_pose = make_pose(
    place_x,
    place_y,
    purple_place_low_z,
    roll,
    pitch,
    purple_pick_yaw
  );

  const auto purple_reorient_pose = make_pose(
    place_x,
    place_y,
    purple_safe_high_z,
    roll,
    pitch,
    purple_pick_yaw
  );

  const auto purple_transfer_x_align_pose = make_pose(
    purple_pick_flange_x,
    place_y,
    purple_safe_high_z,
    roll,
    pitch,
    purple_pick_yaw
  );

  std::vector<geometry_msgs::msg::Pose> purple_pick_transfer_waypoints = {
    purple_transfer_x_align_pose,
    purple_high_pose,
  };

  if (!execute_cartesian_to_pose(node, move_group, "cartesian_reorient_for_purple_pick", purple_reorient_pose, 0.80)) {
    rclcpp::shutdown();
    return 1;
  }

  std::this_thread::sleep_for(150ms);

  if (!execute_cartesian_waypoints_with_segment_fallback(
        node,
        move_group,
        "cartesian_transfer_to_purple_block_high_constant_z",
        purple_pick_transfer_waypoints,
        0.95,
        0.80))
  {
    rclcpp::shutdown();
    return 1;
  }

  std::this_thread::sleep_for(150ms);

  if (!send_gripper_goal(node, gripper_open, 2.0)) {
    rclcpp::shutdown();
    return 1;
  }

  std::this_thread::sleep_for(900ms);

  // 紫色下降抓取：关闭 collision checking。
  auto purple_down_result = execute_cartesian_to_pose_with_result(
    node,
    move_group,
    "cartesian_down_to_purple_block_sides",
    purple_low_pose,
    0.80,
    false
  );

  if (!purple_down_result.success) {
    rclcpp::shutdown();
    return 1;
  }

  const double purple_target_low_z = purple_low_pose.position.z;
  const double purple_start_high_z = purple_safe_high_z;
  const double purple_total_down_z = std::max(
    0.0,
    purple_start_high_z - purple_target_low_z
  );

  double purple_missing_down_z =
    purple_total_down_z * (1.0 - purple_down_result.fraction);

  if (purple_missing_down_z < 0.0) {
    purple_missing_down_z = 0.0;
  }

  const double purple_lift_extra_z = std::min(
    purple_missing_down_z + purple_lift_safety_margin,
    purple_max_lift_extra
  );

  RCLCPP_INFO(
    node->get_logger(),
    "Purple descend compensation by fraction: start_z=%.3f target_z=%.3f fraction=%.3f missing_down_z=%.3f extra_lift_z=%.3f",
    purple_start_high_z,
    purple_target_low_z,
    purple_down_result.fraction,
    purple_missing_down_z,
    purple_lift_extra_z
  );

  std::this_thread::sleep_for(350ms);

  RCLCPP_INFO(node->get_logger(), "Pre-close gripper for purple block.");

  if (!send_gripper_goal(node, purple_gripper_pre_close, 2.0)) {
    rclcpp::shutdown();
    return 1;
  }

  std::this_thread::sleep_for(450ms);

  RCLCPP_INFO(node->get_logger(), "Final close gripper for purple block.");

  if (!send_gripper_goal(node, purple_gripper_close, 2.0)) {
    rclcpp::shutdown();
    return 1;
  }

  std::this_thread::sleep_for(450ms);

  // 补夹一次，减少紫色木块滑落。
  if (!send_gripper_goal(node, purple_gripper_close, 1.5)) {
    rclcpp::shutdown();
    return 1;
  }

  std::this_thread::sleep_for(1000ms);

  // =========================================================
  // Purple lift and reverse-transfer after grasp
  //
  // 逻辑：
  //   1. 夹住紫色木块后，原地垂直上升到 carry_z
  //   2. 在高位保持 z 不变，沿 L 形路径回到堆叠区
  //   3. 抓取和放置只在目标正上方做垂直上下，避免扫到红蓝木块
  //   4. 到蓝色木块上方后垂直下降 soft place
  // =========================================================

  double purple_carry_z = purple_safe_high_z + purple_lift_extra_z;

  if (purple_carry_z < 0.620) {
    purple_carry_z = 0.620;
  }

  if (purple_carry_z > 0.630) {
    purple_carry_z = 0.630;
  }

  RCLCPP_INFO(
    node->get_logger(),
    "Purple carry height selected: z=%.3f",
    purple_carry_z
  );

  const auto purple_lift_to_carry_pose = make_pose(
    purple_pick_flange_x,
    purple_pick_flange_y,
    purple_carry_z,
    roll,
    pitch,
    purple_pick_yaw
  );

  if (!execute_cartesian_to_pose(
        node,
        move_group,
        "cartesian_lift_purple_to_carry_z",
        purple_lift_to_carry_pose,
        0.90))
  {
    rclcpp::shutdown();
    return 1;
  }

  std::this_thread::sleep_for(200ms);

  const auto purple_return_y_align_pose = make_pose(
    purple_pick_flange_x,
    place_y,
    purple_carry_z,
    roll,
    pitch,
    purple_pick_yaw
  );

  const auto purple_stack_high_pose = make_pose(
    place_x,
    place_y,
    purple_carry_z,
    roll,
    pitch,
    purple_pick_yaw
  );

  std::vector<geometry_msgs::msg::Pose> purple_reverse_waypoints = {
    purple_return_y_align_pose,
    purple_stack_high_pose,
  };

  if (!execute_cartesian_waypoints_with_segment_fallback(
        node,
        move_group,
        "cartesian_purple_reverse_constant_z",
        purple_reverse_waypoints,
        0.95,
        0.85))
  {
    rclcpp::shutdown();
    return 1;
  }

  std::this_thread::sleep_for(200ms);

  // 紫色 soft place：不要压进蓝色木块，稍高释放。
  if (!execute_cartesian_to_pose(
        node,
        move_group,
        "cartesian_down_to_blue_block_top_soft_place",
        purple_place_low_pose,
        0.85))
  {
    rclcpp::shutdown();
    return 1;
  }

  std::this_thread::sleep_for(200ms);

  if (!send_gripper_goal(node, gripper_open, 2.0)) {
    rclcpp::shutdown();
    return 1;
  }

  std::this_thread::sleep_for(700ms);

  if (!execute_cartesian_to_pose(
        node,
        move_group,
        "cartesian_retreat_from_purple_stack_keep_z",
        purple_stack_high_pose,
        0.90))
  {
    rclcpp::shutdown();
    return 1;
  }

  std::this_thread::sleep_for(200ms);

  set_motion_profile(move_group, 0.10, 0.08);

  if (!move_named_target(node, move_group, final_target)) {
    rclcpp::shutdown();
    return 1;
  }

  RCLCPP_INFO(node->get_logger(), "Red + blue + purple stacking demo finished.");

  rclcpp::shutdown();
  return 0;
}
