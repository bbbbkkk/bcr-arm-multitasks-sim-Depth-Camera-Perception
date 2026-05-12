#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

#include <tf2/LinearMath/Matrix3x3.h>
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

struct DetectedBlockPose
{
  bool received = false;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double yaw = 0.0;
};

static double get_yaw_from_quaternion(const geometry_msgs::msg::Quaternion & orientation)
{
  tf2::Quaternion q;
  tf2::fromMsg(orientation, q);

  double roll = 0.0;
  double pitch = 0.0;
  double yaw = 0.0;
  tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
  return yaw;
}

static bool wait_for_detected_block_poses(
  const rclcpp::Node::SharedPtr & node,
  DetectedBlockPose & red_pose,
  DetectedBlockPose & blue_pose,
  DetectedBlockPose & purple_pose,
  DetectedBlockPose & place_pose,
  std::chrono::seconds timeout)
{
  auto red_sub = node->create_subscription<geometry_msgs::msg::PoseStamped>(
    "/detected_blocks/red_pose",
    10,
    [&](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
      red_pose.received = true;
      red_pose.x = msg->pose.position.x;
      red_pose.y = msg->pose.position.y;
      red_pose.z = msg->pose.position.z;
      red_pose.yaw = get_yaw_from_quaternion(msg->pose.orientation);
    });

  auto blue_sub = node->create_subscription<geometry_msgs::msg::PoseStamped>(
    "/detected_blocks/blue_pose",
    10,
    [&](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
      blue_pose.received = true;
      blue_pose.x = msg->pose.position.x;
      blue_pose.y = msg->pose.position.y;
      blue_pose.z = msg->pose.position.z;
      blue_pose.yaw = get_yaw_from_quaternion(msg->pose.orientation);
    });

  auto purple_sub = node->create_subscription<geometry_msgs::msg::PoseStamped>(
    "/detected_blocks/purple_pose",
    10,
    [&](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
      purple_pose.received = true;
      purple_pose.x = msg->pose.position.x;
      purple_pose.y = msg->pose.position.y;
      purple_pose.z = msg->pose.position.z;
      purple_pose.yaw = get_yaw_from_quaternion(msg->pose.orientation);
    });

  auto place_sub = node->create_subscription<geometry_msgs::msg::PoseStamped>(
    "/detected_workspace/place_pose",
    10,
    [&](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
      place_pose.received = true;
      place_pose.x = msg->pose.position.x;
      place_pose.y = msg->pose.position.y;
      place_pose.z = msg->pose.position.z;
      place_pose.yaw = get_yaw_from_quaternion(msg->pose.orientation);
    });

  RCLCPP_INFO(
    node->get_logger(),
    "Waiting for detected poses: /detected_blocks/red_pose, /detected_blocks/blue_pose, /detected_blocks/purple_pose, /detected_workspace/place_pose"
  );

  const auto start = std::chrono::steady_clock::now();
  while (rclcpp::ok()) {
    rclcpp::spin_some(node);

    if (red_pose.received && blue_pose.received && purple_pose.received && place_pose.received) {
      RCLCPP_INFO(node->get_logger(), "Detected block and place poses received.");
      return true;
    }

    if ((std::chrono::steady_clock::now() - start) >= timeout) {
      RCLCPP_ERROR(
        node->get_logger(),
        "Timeout waiting for detected poses. red=%s blue=%s purple=%s place=%s",
        red_pose.received ? "ok" : "missing",
        blue_pose.received ? "ok" : "missing",
        purple_pose.received ? "ok" : "missing",
        place_pose.received ? "ok" : "missing"
      );
      return false;
    }

    std::this_thread::sleep_for(100ms);
  }

  return false;
}

static bool wait_for_detected_pose_on_topic(
  const rclcpp::Node::SharedPtr & node,
  const std::string & topic_name,
  DetectedBlockPose & pose,
  std::chrono::seconds timeout,
  bool log_timeout_as_error = true)
{
  pose.received = false;

  auto sub = node->create_subscription<geometry_msgs::msg::PoseStamped>(
    topic_name,
    10,
    [&](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
      pose.received = true;
      pose.x = msg->pose.position.x;
      pose.y = msg->pose.position.y;
      pose.z = msg->pose.position.z;
      pose.yaw = get_yaw_from_quaternion(msg->pose.orientation);
    });

  const auto start = std::chrono::steady_clock::now();
  while (rclcpp::ok()) {
    rclcpp::spin_some(node);

    if (pose.received) {
      return true;
    }

    if ((std::chrono::steady_clock::now() - start) >= timeout) {
      if (log_timeout_as_error) {
        RCLCPP_ERROR(
          node->get_logger(),
          "Timeout waiting for detected pose on topic: %s",
          topic_name.c_str()
        );
      } else {
        RCLCPP_WARN(
          node->get_logger(),
          "No detected pose received on topic during check window: %s",
          topic_name.c_str()
        );
      }
      return false;
    }

    std::this_thread::sleep_for(100ms);
  }

  return false;
}

static bool wait_for_current_gripper_opening(
  const rclcpp::Node::SharedPtr & node,
  double & opening,
  std::chrono::seconds timeout)
{
  bool received = false;

  auto sub = node->create_subscription<sensor_msgs::msg::JointState>(
    "/joint_states",
    20,
    [&](const sensor_msgs::msg::JointState::SharedPtr msg) {
      bool has_left = false;
      bool has_right = false;
      double left = 0.0;
      double right = 0.0;

      for (size_t i = 0; i < msg->name.size() && i < msg->position.size(); ++i) {
        if (msg->name[i] == "simple_left_finger_joint") {
          left = msg->position[i];
          has_left = true;
        } else if (msg->name[i] == "simple_right_finger_joint") {
          right = msg->position[i];
          has_right = true;
        }
      }

      if (has_left && has_right) {
        opening = 0.5 * (left + right);
        received = true;
      }
    });

  const auto start = std::chrono::steady_clock::now();
  while (rclcpp::ok()) {
    rclcpp::spin_some(node);

    if (received) {
      return true;
    }

    if ((std::chrono::steady_clock::now() - start) >= timeout) {
      RCLCPP_ERROR(node->get_logger(), "Timeout waiting for /joint_states gripper opening.");
      return false;
    }

    std::this_thread::sleep_for(50ms);
  }

  return false;
}

static bool check_pick_success(
  const rclcpp::Node::SharedPtr & node,
  const std::string & color_name,
  const std::string & topic_name,
  const DetectedBlockPose & before_pose,
  const geometry_msgs::msg::Pose & check_pose,
  double original_position_xy_threshold,
  double min_gripper_opening_with_object,
  std::chrono::seconds timeout)
{
  double gripper_opening = 0.0;
  const bool got_gripper_opening =
    wait_for_current_gripper_opening(node, gripper_opening, 3s);
  const bool gripper_has_object =
    got_gripper_opening && gripper_opening > min_gripper_opening_with_object;

  DetectedBlockPose after_pose;
  const bool got_after_pose =
    wait_for_detected_pose_on_topic(node, topic_name, after_pose, timeout, false);

  double dx = 0.0;
  double dy = 0.0;
  double dz = 0.0;
  double xy_shift = 0.0;
  bool original_position_cleared = false;

  if (got_after_pose) {
    dx = after_pose.x - before_pose.x;
    dy = after_pose.y - before_pose.y;
    dz = after_pose.z - before_pose.z;
    xy_shift = std::hypot(dx, dy);
    original_position_cleared = xy_shift > original_position_xy_threshold;
  } else {
    // At the check pose the grasped block can be fully occluded by the gripper
    // from the top-down camera. In that case, no red pose being published is
    // still useful evidence that the block is no longer sitting at the
    // original tabletop location.
    original_position_cleared = true;
  }

  const auto & eef_pose = check_pose;
  double eef_xy = -1.0;
  double eef_dz = -1.0;
  bool block_near_gripper = false;

  if (got_after_pose) {
    const double eef_dx = after_pose.x - eef_pose.position.x;
    const double eef_dy = after_pose.y - eef_pose.position.y;
    eef_xy = std::hypot(eef_dx, eef_dy);
    eef_dz = eef_pose.position.z - after_pose.z;
    block_near_gripper = eef_xy < 0.120 && eef_dz > 0.030 && eef_dz < 0.200;
  }

  int score = 0;
  score += gripper_has_object ? 1 : 0;
  score += original_position_cleared ? 1 : 0;
  score += block_near_gripper ? 1 : 0;

  RCLCPP_INFO(
    node->get_logger(),
    "%s pick check: gripper_opening=%.3f object_by_gripper=%s | before=(%.3f, %.3f, %.3f) after=(%.3f, %.3f, %.3f) delta=(%+.3f, %+.3f, %+.3f) xy_shift=%.3f cleared=%s | eef=(%.3f, %.3f, %.3f) eef_xy=%.3f eef_dz=%.3f near_gripper=%s | score=%d/3",
    color_name.c_str(),
    gripper_opening,
    gripper_has_object ? "yes" : "no",
    before_pose.x,
    before_pose.y,
    before_pose.z,
    got_after_pose ? after_pose.x : std::numeric_limits<double>::quiet_NaN(),
    got_after_pose ? after_pose.y : std::numeric_limits<double>::quiet_NaN(),
    got_after_pose ? after_pose.z : std::numeric_limits<double>::quiet_NaN(),
    dx,
    dy,
    dz,
    xy_shift,
    original_position_cleared ? "yes" : "no",
    eef_pose.position.x,
    eef_pose.position.y,
    eef_pose.position.z,
    eef_xy,
    eef_dz,
    block_near_gripper ? "yes" : "no",
    score
  );

  if (score < 2) {
    RCLCPP_ERROR(
      node->get_logger(),
      "%s pick check failed: score %d/3 < 2",
      color_name.c_str(),
      score
    );
    return false;
  }

  RCLCPP_INFO(node->get_logger(), "%s pick check passed.", color_name.c_str());
  return true;
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

  DetectedBlockPose red_block_pose;
  DetectedBlockPose blue_block_pose;
  DetectedBlockPose purple_block_pose;
  DetectedBlockPose place_pose;

  if (!wait_for_detected_block_poses(
        node,
        red_block_pose,
        blue_block_pose,
        purple_block_pose,
        place_pose,
        15s))
  {
    rclcpp::shutdown();
    return 1;
  }

  // Former fixed block poses kept here for checking:
  // red=(0.42, 0.48, yaw=0.25)
  // blue=(0.38, -0.32, yaw=-0.40)
  // purple=(0.42, 0.26, yaw=0.65)
  const double red_x = red_block_pose.x;
  const double red_y = red_block_pose.y;
  const double red_yaw = red_block_pose.yaw;

  const double blue_x = blue_block_pose.x;
  const double blue_y = blue_block_pose.y;
  const double blue_yaw = blue_block_pose.yaw;

  const double purple_x = purple_block_pose.x;
  const double purple_y = purple_block_pose.y;
  const double purple_yaw = purple_block_pose.yaw;

  // Former fixed place center kept here for checking:
  // place=(0.52, 0.00)
  const double place_x = place_pose.x;
  const double place_y = place_pose.y;
  // Former fixed place surface / yaw kept here for checking:
  // place_surface_z=0.010
  // place_yaw=0.0
  const double place_surface_z = place_pose.z;
  const double place_yaw = place_pose.yaw;

  const double cube_size = 0.09;

  // Former fixed z values kept here for checking:
  // red_pick_high_z=0.45
  // red_transfer_z=0.52
  // blue_transfer_z=0.55
  // purple_transfer_z=0.60
  // red_pick_low_z=0.34
  // blue_pick_low_z=0.34
  // purple_pick_low_z=0.36
  // Former fixed place low z values kept here for checking:
  // place_low_z=0.37
  // blue_place_low_z=0.455
  // purple_place_low_z=0.555
  const double place_low_offset_z = 0.37 - 0.010;
  const double place_low_z = place_surface_z + place_low_offset_z;

  const double red_pick_high_offset_z = 0.45 - 0.045;
  const double red_transfer_offset_z = 0.52 - 0.045;
  const double blue_safe_high_offset_z = 0.52 - 0.045;
  const double blue_transfer_offset_z = 0.55 - 0.045;
  const double purple_safe_high_offset_z = 0.60 - 0.045;
  const double red_pick_low_offset_z = 0.34 - 0.045;
  const double blue_pick_low_offset_z = 0.34 - 0.045;
  const double purple_pick_low_offset_z = 0.36 - 0.045;

  const double red_pick_high_z = red_block_pose.z + red_pick_high_offset_z;
  const double red_transfer_z = red_block_pose.z + red_transfer_offset_z;
  const double blue_safe_high_z = blue_block_pose.z + blue_safe_high_offset_z;
  const double blue_transfer_z = blue_block_pose.z + blue_transfer_offset_z;
  const double purple_safe_high_z = purple_block_pose.z + purple_safe_high_offset_z;
  const double red_pick_low_z = red_block_pose.z + red_pick_low_offset_z;
  const double blue_pick_low_z = blue_block_pose.z + blue_pick_low_offset_z;
  const double purple_pick_low_z = purple_block_pose.z + purple_pick_low_offset_z;

  const double red_lift_z = red_transfer_z;
  const double red_place_high_z = red_transfer_z;

  const double blue_mid_lift_z = blue_safe_high_z;
  const double blue_lift_z = blue_transfer_z;
  const double blue_place_high_z = blue_transfer_z;

  const double blue_place_low_z = place_low_z + cube_size - 0.005;

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

  const double pick_check_original_position_xy_threshold = 0.060;
  const double red_pick_check_min_gripper_opening = 0.024;
  const double blue_pick_check_min_gripper_opening = 0.024;
  const double purple_pick_check_min_gripper_opening = 0.018;

  const double roll = M_PI;
  const double pitch = 0.0;

  const double gripper_yaw_offset = 1.5708;

  const double red_pick_yaw = red_yaw + gripper_yaw_offset;
  const double blue_pick_yaw = blue_yaw + gripper_yaw_offset;
  const double red_place_yaw = place_yaw + gripper_yaw_offset;
  const double blue_place_yaw = place_yaw + gripper_yaw_offset;
  const double purple_place_yaw = place_yaw;

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
    "Detected place center: x=%.3f y=%.3f z=%.3f place_yaw=%.3f",
    place_x,
    place_y,
    place_pose.z,
    place_pose.yaw
  );

  RCLCPP_INFO(
    node->get_logger(),
    "Derived place profile: place_low_z=%.3f blue_place_low_z=%.3f purple_place_low_z=%.3f",
    place_low_z,
    blue_place_low_z,
    purple_place_low_z
  );

  RCLCPP_INFO(
    node->get_logger(),
    "Red detected block center: x=%.3f y=%.3f z=%.3f red_yaw=%.3f",
    red_x,
    red_y,
    red_block_pose.z,
    red_yaw
  );

  RCLCPP_INFO(
    node->get_logger(),
    "Red derived z profile: pick_high_z=%.3f pick_low_z=%.3f transfer_z=%.3f",
    red_pick_high_z,
    red_pick_low_z,
    red_transfer_z
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
    red_pick_low_z,
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
    red_place_yaw
  );

  const auto red_place_low_pose = make_pose(
    place_x,
    place_y,
    place_low_z,
    roll,
    pitch,
    red_place_yaw
  );

  // Camera check pose: keep the lifted z, and move the grasped red block only
  // a moderate distance away from the original red pickup site so the top-down
  // camera can verify that:
  // 1. the original red position is cleared
  // 2. the block stays near the gripper
  //
  // This point intentionally stays much closer than the previous lower-right
  // corner check pose, because that farther Cartesian move was not reliably
  // reachable at constant z.
  const double red_check_x = red_pick_flange_x + 0.04;
  const double red_check_y = place_y + 0.05;
  const auto red_check_pose = make_pose(
    red_check_x,
    red_check_y,
    red_lift_z,
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

  RCLCPP_INFO(
    node->get_logger(),
    "================ RED PICK CHECK START ================"
  );
  RCLCPP_INFO(
    node->get_logger(),
    "Red check step 1/2: move grasped red block to camera check pose while keeping z fixed."
  );

  if (!execute_cartesian_to_pose(
        node,
        move_group,
        "cartesian_move_to_red_pick_check_pose_keep_z",
        red_check_pose,
        0.90))
  {
    rclcpp::shutdown();
    return 1;
  }

  std::this_thread::sleep_for(300ms);

  RCLCPP_INFO(
    node->get_logger(),
    "Red check step 2/2: verify gripper opening, original red position, and red block near gripper."
  );

  if (!check_pick_success(
        node,
        "Red",
        "/detected_blocks/red_pose",
        red_block_pose,
        red_check_pose,
        pick_check_original_position_xy_threshold,
        red_pick_check_min_gripper_opening,
        5s))
  {
    RCLCPP_WARN(
      node->get_logger(),
      "================ RED PICK CHECK FAILED ================"
    );
    RCLCPP_WARN(
      node->get_logger(),
      "Red pick check failed or was inconclusive."
    );
    rclcpp::shutdown();
    return 1;
  }

  RCLCPP_INFO(
    node->get_logger(),
    "================ RED PICK CHECK PASSED ================"
  );

  RCLCPP_INFO(
    node->get_logger(),
    "Red check passed. Continue directly from check pose to red place transfer."
  );

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
    "Blue detected block center: x=%.3f y=%.3f z=%.3f blue_yaw=%.3f",
    blue_x,
    blue_y,
    blue_block_pose.z,
    blue_yaw
  );

  RCLCPP_INFO(
    node->get_logger(),
    "Blue derived z profile: pick_high_z=%.3f pick_low_z=%.3f transfer_z=%.3f",
    blue_safe_high_z,
    blue_pick_low_z,
    blue_transfer_z
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
    blue_pick_low_z,
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
    blue_place_yaw
  );

  const auto blue_place_low_pose = make_pose(
    place_x,
    place_y,
    blue_place_low_z,
    roll,
    pitch,
    blue_place_yaw
  );

  // Camera check pose for blue: move the grasped block to a nearby lower-middle
  // area at the same lifted z, so the top-down camera can verify the original
  // blue position is cleared without sending the arm too far away.
  const double blue_check_x = blue_pick_flange_x + 0.06;
  const double blue_check_y = place_y - 0.08;
  const auto blue_check_pose = make_pose(
    blue_check_x,
    blue_check_y,
    blue_lift_z,
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

  RCLCPP_INFO(
    node->get_logger(),
    "================ BLUE PICK CHECK START ================"
  );
  RCLCPP_INFO(
    node->get_logger(),
    "Blue check step 1/2: move grasped blue block to camera check pose while keeping z fixed."
  );

  if (!execute_cartesian_to_pose(
        node,
        move_group,
        "cartesian_move_to_blue_pick_check_pose_keep_z",
        blue_check_pose,
        0.90))
  {
    rclcpp::shutdown();
    return 1;
  }

  std::this_thread::sleep_for(300ms);

  RCLCPP_INFO(
    node->get_logger(),
    "Blue check step 2/2: verify gripper opening, original blue position, and blue block near gripper."
  );

  if (!check_pick_success(
        node,
        "Blue",
        "/detected_blocks/blue_pose",
        blue_block_pose,
        blue_check_pose,
        pick_check_original_position_xy_threshold,
        blue_pick_check_min_gripper_opening,
        5s))
  {
    RCLCPP_WARN(
      node->get_logger(),
      "================ BLUE PICK CHECK FAILED ================"
    );
    RCLCPP_WARN(
      node->get_logger(),
      "Blue pick check failed or was inconclusive."
    );
    rclcpp::shutdown();
    return 1;
  }

  RCLCPP_INFO(
    node->get_logger(),
    "================ BLUE PICK CHECK PASSED ================"
  );
  RCLCPP_INFO(
    node->get_logger(),
    "Blue check passed. Continue directly from check pose to blue place transfer."
  );

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
    "Purple detected block center: x=%.3f y=%.3f z=%.3f purple_yaw=%.3f",
    purple_x,
    purple_y,
    purple_block_pose.z,
    purple_yaw
  );

  RCLCPP_INFO(
    node->get_logger(),
    "Purple derived z profile: pick_high_z=%.3f pick_low_z=%.3f transfer_z=%.3f",
    purple_safe_high_z,
    purple_pick_low_z,
    purple_safe_high_z
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
    purple_place_yaw
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

  // Camera check pose for purple: move the grasped block a moderate distance
  // toward the workspace middle at the same carry z so the top-down camera can
  // re-check the original purple position.
  const double purple_check_x = purple_pick_flange_x + 0.04;
  const double purple_check_y = place_y + 0.10;
  const auto purple_check_pose = make_pose(
    purple_check_x,
    purple_check_y,
    purple_carry_z,
    roll,
    pitch,
    purple_pick_yaw
  );

  RCLCPP_INFO(
    node->get_logger(),
    "================ PURPLE PICK CHECK START ================"
  );
  RCLCPP_INFO(
    node->get_logger(),
    "Purple check step 1/2: move grasped purple block to camera check pose while keeping z fixed."
  );

  if (!execute_cartesian_to_pose(
        node,
        move_group,
        "cartesian_move_to_purple_pick_check_pose_keep_z",
        purple_check_pose,
        0.90))
  {
    rclcpp::shutdown();
    return 1;
  }

  std::this_thread::sleep_for(300ms);

  RCLCPP_INFO(
    node->get_logger(),
    "Purple check step 2/2: verify gripper opening, original purple position, and purple block near gripper."
  );

  if (!check_pick_success(
        node,
        "Purple",
        "/detected_blocks/purple_pose",
        purple_block_pose,
        purple_check_pose,
        pick_check_original_position_xy_threshold,
        purple_pick_check_min_gripper_opening,
        5s))
  {
    RCLCPP_WARN(
      node->get_logger(),
      "================ PURPLE PICK CHECK FAILED ================"
    );
    RCLCPP_WARN(
      node->get_logger(),
      "Purple pick check failed or was inconclusive."
    );
    rclcpp::shutdown();
    return 1;
  }

  RCLCPP_INFO(
    node->get_logger(),
    "================ PURPLE PICK CHECK PASSED ================"
  );
  RCLCPP_INFO(
    node->get_logger(),
    "Purple check passed. Continue directly from check pose to purple place transfer."
  );

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
    purple_place_yaw
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
