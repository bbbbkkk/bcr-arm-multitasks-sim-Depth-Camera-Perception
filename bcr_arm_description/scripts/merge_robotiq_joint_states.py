#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class RobotiqJointStateMerger(Node):
    def __init__(self):
        super().__init__("robotiq_joint_state_merger")

        self.pub = self.create_publisher(JointState, "/joint_states", 10)
        self.sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_cb,
            10,
        )

        self.robotiq_names = [
            "robotiq_85_left_knuckle_joint",
            "robotiq_85_right_knuckle_joint",
            "robotiq_85_left_inner_knuckle_joint",
            "robotiq_85_right_inner_knuckle_joint",
            "robotiq_85_left_finger_tip_joint",
            "robotiq_85_right_finger_tip_joint",
        ]

        # open pose for visualization
        self.robotiq_positions = [
            0.4,
            -0.4,
            0.4,
            -0.4,
            -0.4,
            0.4,
        ]

        self.get_logger().info("Robotiq joint state merger started.")
        self.get_logger().info("It will merge arm joint_states with Robotiq gripper joints.")

    def joint_state_cb(self, msg: JointState):
        # Avoid re-processing our own merged message.
        if any(name.startswith("robotiq_") for name in msg.name):
            return

        merged = JointState()

        # Critical: reuse the original arm joint_state timestamp.
        # This keeps gripper TF in the same time tree as the Gazebo arm.
        merged.header = msg.header

        merged.name = list(msg.name) + self.robotiq_names
        merged.position = list(msg.position) + self.robotiq_positions

        if len(msg.velocity) == len(msg.name):
            merged.velocity = list(msg.velocity) + [0.0] * len(self.robotiq_names)
        else:
            merged.velocity = []

        if len(msg.effort) == len(msg.name):
            merged.effort = list(msg.effort) + [0.0] * len(self.robotiq_names)
        else:
            merged.effort = []

        self.pub.publish(merged)


def main():
    rclpy.init()
    node = RobotiqJointStateMerger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
