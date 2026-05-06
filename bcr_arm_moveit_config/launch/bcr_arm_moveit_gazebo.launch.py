#!/usr/bin/env python3

# Still under development, this launch file sets up the BCR arm with MoveIt! and Gazebo simulation.
# There are some issues with MoveIt and using Gazebo's physics engine for executing planned trajectory.

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
    ExecuteProcess,
)
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # -------------------------------------------------------------------------
    # Gazebo resource path
    # Make sure Gazebo can find meshes from bcr_arm_description and robotiq_description.
    # -------------------------------------------------------------------------
    bcr_arm_description_share = get_package_share_directory("bcr_arm_description")
    bcr_arm_description_parent = os.path.dirname(bcr_arm_description_share)

    current_gz_resource_path = os.environ.get("GZ_SIM_RESOURCE_PATH", "")
    resource_paths = [
        bcr_arm_description_parent,
    ]

    for path in resource_paths:
        if path and path not in current_gz_resource_path.split(os.pathsep):
            current_gz_resource_path = path + os.pathsep + current_gz_resource_path

    if current_gz_resource_path.endswith(os.pathsep):
        current_gz_resource_path = current_gz_resource_path[:-1]

    os.environ["GZ_SIM_RESOURCE_PATH"] = current_gz_resource_path
    os.environ["IGN_GAZEBO_RESOURCE_PATH"] = current_gz_resource_path

    # -------------------------------------------------------------------------
    # Launch arguments
    # -------------------------------------------------------------------------
    declared_arguments = []

    declared_arguments.append(
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use simulation (Gazebo) clock if true",
        )
    )

    use_sim_time = LaunchConfiguration("use_sim_time")

    # -------------------------------------------------------------------------
    # Gazebo launch
    # -------------------------------------------------------------------------
    bcr_arm_gazebo_pkg = FindPackageShare("bcr_arm_gazebo")

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    bcr_arm_gazebo_pkg,
                    "launch",
                    "bcr_arm.gazebo.launch.py",
                ]
            )
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
        }.items(),
    )

    # -------------------------------------------------------------------------
    # MoveIt configuration
    # -------------------------------------------------------------------------
    moveit_config = (
        MoveItConfigsBuilder("bcr_arm", package_name="bcr_arm_moveit_config")
        .robot_description(
            os.path.join(
                get_package_share_directory("bcr_arm_moveit_config"),
                "config",
                "bcr_arm.urdf.xacro",
            )
        )
        .robot_description_semantic(
            os.path.join(
                get_package_share_directory("bcr_arm_moveit_config"),
                "config",
                "bcr_arm.srdf",
            )
        )
        .trajectory_execution(
            os.path.join(
                get_package_share_directory("bcr_arm_moveit_config"),
                "config",
                "moveit_controllers.yaml",
            )
        )
        .to_moveit_configs()
    )

    # -------------------------------------------------------------------------
    # Move group
    # -------------------------------------------------------------------------
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"use_sim_time": use_sim_time},
        ],
        arguments=["--ros-args", "--log-level", "info"],
    )

    # -------------------------------------------------------------------------
    # RViz
    # -------------------------------------------------------------------------
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=[
            "-d",
            os.path.join(
                get_package_share_directory("bcr_arm_moveit_config"),
                "config",
                "moveit.rviz",
            ),
        ],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            {"use_sim_time": use_sim_time},
        ],
    )

    # -------------------------------------------------------------------------
    # Controllers
    # These are currently not used here because bcr_arm.gazebo.launch.py /
    # moveit_config already handles controllers in this setup.
    # -------------------------------------------------------------------------
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager",
        ],
        output="screen",
    )

    joint_trajectory_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_trajectory_controller",
            "--controller-manager",
            "/controller_manager",
        ],
        output="screen",
    )

    # -------------------------------------------------------------------------
    # Robotiq joint state merger
    #
    # Purpose:
    #   joint_state_broadcaster publishes only joint1~joint7.
    #   Robotiq gripper joints are not controlled yet in V1.
    #   This node merges fixed Robotiq joint positions into /joint_states,
    #   so robot_state_publisher can publish Robotiq finger TF.
    # -------------------------------------------------------------------------


    # -------------------------------------------------------------------------
    # Set robot_state_publisher ignore_timestamp
    #
    # robot_state_publisher is launched inside bcr_arm.gazebo.launch.py.
    # So here we delay a ros2 param set command until the node exists.
    # -------------------------------------------------------------------------
    set_robot_state_publisher_ignore_timestamp = TimerAction(
        period=6.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    "ros2",
                    "param",
                    "set",
                    "/robot_state_publisher",
                    "ignore_timestamp",
                    "true",
                ],
                output="screen",
            )
        ],
    )

    simple_gripper_controller_spawner = TimerAction(
        period=10.0,
        actions=[
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=[
                    "simple_gripper_controller",
                    "--controller-manager",
                    "/controller_manager",
                ],
                output="screen",
            )
        ],
    )

    add_world_collision_objects_node = TimerAction(
        period=14.0,
        actions=[
            Node(
                package="bcr_arm_moveit_config",
                executable="add_world_collision_objects.py",
                name="add_world_collision_objects",
                output="screen",
                parameters=[
                    {"use_sim_time": use_sim_time},
                    {"planning_frame": "world"},
                ],
            )
        ],
    )
    
    

    # -------------------------------------------------------------------------
    # Launch description
    # -------------------------------------------------------------------------
    return LaunchDescription(
        declared_arguments
        + [
            gazebo_launch,
            move_group_node,
            rviz_node,

            # controller gets loaded by default by moveit_config / gazebo launch
            # joint_state_broadcaster_spawner,
            # joint_trajectory_controller_spawner,

            # Robotiq V1 helper nodes
            set_robot_state_publisher_ignore_timestamp,
            simple_gripper_controller_spawner,
            add_world_collision_objects_node,
            
        ]
    )
