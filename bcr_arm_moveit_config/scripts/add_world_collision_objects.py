#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node

from moveit_msgs.srv import ApplyPlanningScene
from moveit_msgs.msg import PlanningScene, CollisionObject, ObjectColor

from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose


def quaternion_from_rpy(roll: float, pitch: float, yaw: float):
    """
    SDF pose format:
      x y z roll pitch yaw

    MoveIt CollisionObject needs quaternion orientation.
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy

    return qx, qy, qz, qw


def make_box_collision_object(
    object_id,
    frame_id,
    size_xyz,
    pose_xyz,
    rpy_xyz=(0.0, 0.0, 0.0),
):
    obj = CollisionObject()
    obj.header.frame_id = frame_id
    obj.id = object_id
    obj.operation = CollisionObject.ADD

    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.BOX
    primitive.dimensions = [
        float(size_xyz[0]),
        float(size_xyz[1]),
        float(size_xyz[2]),
    ]

    pose = Pose()
    pose.position.x = float(pose_xyz[0])
    pose.position.y = float(pose_xyz[1])
    pose.position.z = float(pose_xyz[2])

    qx, qy, qz, qw = quaternion_from_rpy(
        float(rpy_xyz[0]),
        float(rpy_xyz[1]),
        float(rpy_xyz[2]),
    )

    pose.orientation.x = qx
    pose.orientation.y = qy
    pose.orientation.z = qz
    pose.orientation.w = qw

    obj.primitives.append(primitive)
    obj.primitive_poses.append(pose)
    return obj


def make_remove_object(object_id, frame_id):
    obj = CollisionObject()
    obj.header.frame_id = frame_id
    obj.id = object_id
    obj.operation = CollisionObject.REMOVE
    return obj


def make_object_color(object_id, rgba):
    color = ObjectColor()
    color.id = object_id
    color.color.r = float(rgba[0])
    color.color.g = float(rgba[1])
    color.color.b = float(rgba[2])
    color.color.a = float(rgba[3])
    return color


class AddWorldCollisionObjects(Node):
    def __init__(self):
        super().__init__("add_world_collision_objects")

        self.declare_parameter("planning_frame", "world")

        self.planning_frame = (
            self.get_parameter("planning_frame")
            .get_parameter_value()
            .string_value
        )

        self.client = self.create_client(
            ApplyPlanningScene,
            "/apply_planning_scene",
        )

    def wait_for_service(self):
        self.get_logger().info("Waiting for /apply_planning_scene service...")

        if not self.client.wait_for_service(timeout_sec=15.0):
            self.get_logger().error("/apply_planning_scene service not available.")
            return False

        self.get_logger().info("/apply_planning_scene service is available.")
        return True

    def apply_scene(self, collision_objects, object_colors):
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = collision_objects
        scene.object_colors = object_colors

        request = ApplyPlanningScene.Request()
        request.scene = scene

        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

        if not future.done():
            self.get_logger().error("ApplyPlanningScene timeout.")
            return False

        result = future.result()

        if result is None:
            self.get_logger().error("ApplyPlanningScene result is None.")
            return False

        if not result.success:
            self.get_logger().error("ApplyPlanningScene failed.")
            return False

        return True

    def run(self):
        if not self.wait_for_service():
            return False

        frame = self.planning_frame

        object_ids = [
            "dirt_floor",
            "stacking_area",
            "red_block",
            "blue_block",
            "purple_block",
        ]

        remove_objects = [
            make_remove_object(object_id, frame)
            for object_id in object_ids
        ]

        self.get_logger().info("Removing old collision objects...")
        self.apply_scene(remove_objects, [])

        collision_objects = []
        object_colors = []

        # =========================================================
        # Match current empty.world
        #
        # SDF pose format:
        #   x y z roll pitch yaw
        #
        # dirt_floor:
        #   pose = 0.45 0.00 -0.010 0 0 0
        #   size = 1.50 1.50 0.020
        #
        # In MoveIt, put floor slightly lower to avoid base collision.
        # =========================================================

        collision_objects.append(
            make_box_collision_object(
                "dirt_floor",
                frame,
                size_xyz=[1.50, 1.50, 0.010],
                pose_xyz=[0.45, 0.00, -0.030],
                rpy_xyz=[0.0, 0.0, 0.0],
            )
        )
        object_colors.append(
            make_object_color("dirt_floor", [0.55, 0.39, 0.22, 1.0])
        )

        # stacking_area:
        #   pose = 0.72 0.00 0.006 0 0 0
        #   size = 0.30 0.30 0.008

        collision_objects.append(
            make_box_collision_object(
                "stacking_area",
                frame,
                size_xyz=[0.30, 0.30, 0.008],
                pose_xyz=[0.52, 0.00, 0.006],
                rpy_xyz=[0.0, 0.0, 0.0],
            )
        )
        object_colors.append(
            make_object_color("stacking_area", [0.0, 0.85, 0.0, 0.85])
        )

        # red_block:
        #   empty.world pose = 0.55 0.48 0.045 0 0 0.25
        #   size = 0.09 0.09 0.09

        collision_objects.append(
            make_box_collision_object(
                "red_block",
                frame,
                size_xyz=[0.09, 0.09, 0.09],
                pose_xyz=[0.42, 0.48, 0.045],
                rpy_xyz=[0.0, 0.0, 0.25],
            )
        )
        object_colors.append(
            make_object_color("red_block", [1.0, 0.0, 0.0, 1.0])
        )

        # blue_block:
        #   empty.world pose = 0.68 -0.55 0.045 0 0 -0.40
        #   size = 0.09 0.09 0.09

        collision_objects.append(
            make_box_collision_object(
                "blue_block",
                frame,
                size_xyz=[0.09, 0.09, 0.09],
                pose_xyz=[0.38, -0.32, 0.045],
                rpy_xyz=[0.0, 0.0, -0.40],
            )
        )
        object_colors.append(
            make_object_color("blue_block", [0.0, 0.1, 1.0, 1.0])
        )

        # purple_block:
        #   empty.world pose = 0.42 0.26 0.045 0 0 0.65
        #   size = 0.09 0.09 0.09

        collision_objects.append(
            make_box_collision_object(
                "purple_block",
                frame,
                size_xyz=[0.09, 0.09, 0.09],
                pose_xyz=[0.42, 0.26, 0.045],
                rpy_xyz=[0.0, 0.0, 0.65],
            )
        )
        object_colors.append(
            make_object_color("purple_block", [0.55, 0.0, 0.85, 1.0])
        )

        self.get_logger().info("Applying collision objects to MoveIt PlanningScene...")

        ok = self.apply_scene(collision_objects, object_colors)

        if ok:
            self.get_logger().info(
                f"Applied {len(collision_objects)} collision objects in frame [{frame}]."
            )
            for obj in collision_objects:
                self.get_logger().info(f"  - {obj.id}")
        else:
            self.get_logger().error("Failed to apply collision objects.")

        return ok


def main():
    rclpy.init()
    node = AddWorldCollisionObjects()

    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
