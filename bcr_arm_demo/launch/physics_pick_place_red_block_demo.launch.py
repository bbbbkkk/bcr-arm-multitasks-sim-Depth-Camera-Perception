#!/usr/bin/env python3

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    declared_arguments = []

    declared_arguments.append(
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use Gazebo simulation clock",
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "planning_group",
            default_value="bcr_arm",
            description="MoveIt planning group name",
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "eef_link",
            default_value="flange",
            description="End effector link used by MoveIt",
        )
    )

    use_sim_time = LaunchConfiguration("use_sim_time")
    planning_group = LaunchConfiguration("planning_group")
    eef_link = LaunchConfiguration("eef_link")

    # -------------------------------------------------------------------------
    # Load MoveIt robot_description / robot_description_semantic / kinematics
    # for this standalone demo node.
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

    demo_node = Node(
        package="bcr_arm_demo",
        executable="physics_pick_place_red_block_demo",
        name="physics_pick_place_red_block_demo",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {
                "use_sim_time": ParameterValue(use_sim_time, value_type=bool),
                "planning_group": planning_group,
                "eef_link": eef_link,
            },
        ],
    )

    return LaunchDescription(
        declared_arguments
        + [
            demo_node,
        ]
    )
