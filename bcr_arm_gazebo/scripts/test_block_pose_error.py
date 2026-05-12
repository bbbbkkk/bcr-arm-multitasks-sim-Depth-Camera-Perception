#!/usr/bin/env python3

from collections import deque
import math
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener


class BlockPoseErrorTester(Node):
    def __init__(self) -> None:
        super().__init__("block_pose_error_tester")

        self.declare_parameter("world_frame", "world")
        self.declare_parameter("color_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("min_contour_area", 800.0)
        self.declare_parameter("optical_frame", "camera_depth_optical_frame")
        self.declare_parameter("horizontal_fov", 1.0)
        self.declare_parameter("block_size", 0.09)
        self.declare_parameter("average_window_size", 30)
        self.declare_parameter("process_period", 0.2)
        self.declare_parameter("report_period", 1.0)

        self.world_frame = (
            self.get_parameter("world_frame").get_parameter_value().string_value
        )
        self.min_contour_area = (
            self.get_parameter("min_contour_area").get_parameter_value().double_value
        )
        self.optical_frame = (
            self.get_parameter("optical_frame").get_parameter_value().string_value
        )
        self.horizontal_fov = (
            self.get_parameter("horizontal_fov").get_parameter_value().double_value
        )
        self.block_size = (
            self.get_parameter("block_size").get_parameter_value().double_value
        )
        self.average_window_size = max(
            1,
            self.get_parameter("average_window_size")
            .get_parameter_value()
            .integer_value,
        )
        self.process_period = (
            self.get_parameter("process_period").get_parameter_value().double_value
        )
        self.report_period = (
            self.get_parameter("report_period").get_parameter_value().double_value
        )

        self.latest_color_msg: Optional[Image] = None
        self.latest_depth_msg: Optional[Image] = None
        self.latest_camera_info: Optional[CameraInfo] = None
        self.last_processed_frame_key: Optional[Tuple[int, int, int, int]] = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        color_topic = (
            self.get_parameter("color_topic").get_parameter_value().string_value
        )
        depth_topic = (
            self.get_parameter("depth_topic").get_parameter_value().string_value
        )
        camera_info_topic = (
            self.get_parameter("camera_info_topic").get_parameter_value().string_value
        )

        self.create_subscription(Image, color_topic, self.on_color_image, 10)
        self.create_subscription(Image, depth_topic, self.on_depth_image, 10)
        self.create_subscription(CameraInfo, camera_info_topic, self.on_camera_info, 10)
        self.process_timer = self.create_timer(
            self.process_period, self.process_latest_frame
        )
        self.report_timer = self.create_timer(
            self.report_period, self.report_averaged_results
        )

        self.color_names = ("red", "blue", "purple")
        self.ground_truth: Dict[str, Tuple[float, float, float]] = {
            "red": (0.42, 0.48, 0.045),
            "blue": (0.38, -0.32, 0.045),
            "purple": (0.42, 0.26, 0.045),
        }

        self.color_ranges: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {
            "red": [
                (np.array([0, 100, 80]), np.array([10, 255, 255])),
                (np.array([170, 100, 80]), np.array([180, 255, 255])),
            ],
            "blue": [
                (np.array([95, 100, 80]), np.array([130, 255, 255])),
            ],
            "purple": [
                (np.array([130, 80, 80]), np.array([165, 255, 255])),
            ],
        }
        self.detection_history: Dict[str, Deque[Dict[str, object]]] = {
            color_name: deque(maxlen=self.average_window_size)
            for color_name in self.color_names
        }

        self.get_logger().info("Block pose error tester started.")

    def on_color_image(self, msg: Image) -> None:
        self.latest_color_msg = msg

    def on_depth_image(self, msg: Image) -> None:
        self.latest_depth_msg = msg

    def on_camera_info(self, msg: CameraInfo) -> None:
        self.latest_camera_info = msg

    def process_latest_frame(self) -> None:
        missing = []
        if self.latest_color_msg is None:
            missing.append("color_image")
        if self.latest_depth_msg is None:
            missing.append("depth_image")
        if self.latest_camera_info is None:
            missing.append("camera_info")

        if missing:
            return

        frame_key = (
            self.latest_color_msg.header.stamp.sec,
            self.latest_color_msg.header.stamp.nanosec,
            self.latest_depth_msg.header.stamp.sec,
            self.latest_depth_msg.header.stamp.nanosec,
        )
        if frame_key == self.last_processed_frame_key:
            return

        try:
            color_image = self.ros_image_to_cv_color(self.latest_color_msg)
            depth_image = self.ros_image_to_cv_depth(self.latest_depth_msg)
        except ValueError as exc:
            self.get_logger().error(str(exc))
            return

        for color_name in self.color_names:
            detection = self.detect_one_block(color_name, color_image, depth_image)
            if detection is not None:
                self.detection_history[color_name].append(detection)

        self.last_processed_frame_key = frame_key

    def report_averaged_results(self) -> None:
        missing = []
        if self.latest_color_msg is None:
            missing.append("color_image")
        if self.latest_depth_msg is None:
            missing.append("depth_image")
        if self.latest_camera_info is None:
            missing.append("camera_info")

        if missing:
            self.get_logger().info(f"Waiting for: {', '.join(missing)}")
            return

        report_lines = []
        for color_name in self.color_names:
            history = self.detection_history[color_name]
            if not history:
                report_lines.append(f"{color_name}: not detected")
                continue

            center_points = np.array(
                [detection["center"] for detection in history], dtype=np.float64
            )
            avg_center = center_points.mean(axis=0)
            avg_pixel = np.array(
                [detection["pixel"] for detection in history], dtype=np.float64
            ).mean(axis=0)
            truth = self.ground_truth[color_name]
            dx = avg_center[0] - truth[0]
            dy = avg_center[1] - truth[1]
            dz = avg_center[2] - truth[2]
            xy_error = math.hypot(dx, dy)
            report_lines.append(
                (
                    f"{color_name}: "
                    f"n={len(history)}/{self.average_window_size} "
                    f"u={int(round(avg_pixel[0]))} v={int(round(avg_pixel[1]))} "
                    f"center=({avg_center[0]:.3f}, {avg_center[1]:.3f}, {avg_center[2]:.3f}) "
                    f"gt=({truth[0]:.3f}, {truth[1]:.3f}, {truth[2]:.3f}) "
                    f"err=({dx:+.3f}, {dy:+.3f}, {dz:+.3f}) "
                    f"xy={xy_error:.3f}m"
                )
            )

        self.get_logger().info(" | ".join(report_lines))

    def detect_one_block(
        self, color_name: str, color_image: np.ndarray, depth_image: np.ndarray
    ) -> Optional[Dict[str, object]]:
        hsv = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for lower, upper in self.color_ranges[color_name]:
            mask |= cv2.inRange(hsv, lower, upper)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < self.min_contour_area:
            return None

        moments = cv2.moments(contour)
        if abs(moments["m00"]) < 1e-6:
            return None

        u = int(moments["m10"] / moments["m00"])
        v = int(moments["m01"] / moments["m00"])

        depth_m = self.sample_depth_meters(depth_image, u, v)
        if depth_m is None:
            return None

        camera_xyz = self.project_pixel_to_camera(u, v, depth_m, self.latest_camera_info)
        world_xyz = self.transform_point_to_world(
            camera_xyz, self.latest_depth_msg.header.frame_id
        )
        if world_xyz is None:
            return None

        return {
            "pixel": (u, v),
            "depth_m": depth_m,
            "camera": camera_xyz,
            "world": world_xyz,
            "center": (
                world_xyz[0],
                world_xyz[1],
                world_xyz[2] - self.block_size / 2.0,
            ),
        }

    def sample_depth_meters(
        self, depth_image: np.ndarray, u: int, v: int, window_radius: int = 3
    ) -> Optional[float]:
        height, width = depth_image.shape[:2]
        u0 = max(0, u - window_radius)
        u1 = min(width, u + window_radius + 1)
        v0 = max(0, v - window_radius)
        v1 = min(height, v + window_radius + 1)

        window = depth_image[v0:v1, u0:u1]
        valid = window[np.isfinite(window) & (window > 0.0)]
        if valid.size == 0:
            return None

        return float(np.median(valid))

    def project_pixel_to_camera(
        self, u: int, v: int, depth_m: float, camera_info: CameraInfo
    ) -> Tuple[float, float, float]:
        fx, fy, cx, cy = self.get_intrinsics(camera_info)

        x = (u - cx) * depth_m / fx
        y = (v - cy) * depth_m / fy
        z = depth_m
        return (x, y, z)

    def get_intrinsics(self, camera_info: CameraInfo) -> Tuple[float, float, float, float]:
        width = float(camera_info.width)
        height = float(camera_info.height)

        fx = float(camera_info.k[0])
        fy = float(camera_info.k[4])
        cx = float(camera_info.k[2])
        cy = float(camera_info.k[5])

        expected_cx = width / 2.0
        expected_cy = height / 2.0
        intrinsics_look_wrong = (
            fx <= 0.0
            or fy <= 0.0
            or abs(cx - expected_cx) > width * 0.15
            or abs(cy - expected_cy) > height * 0.15
        )

        if not intrinsics_look_wrong:
            return fx, fy, cx, cy

        derived_fx = width / (2.0 * math.tan(self.horizontal_fov / 2.0))
        derived_fy = derived_fx
        return derived_fx, derived_fy, expected_cx, expected_cy

    def transform_point_to_world(
        self, camera_xyz: Tuple[float, float, float], source_frame: str
    ) -> Optional[Tuple[float, float, float]]:
        transform = None
        errors = []
        for candidate in self.build_frame_candidates(source_frame):
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.world_frame,
                    candidate,
                    rclpy.time.Time(),
                )
                break
            except TransformException as exc:
                errors.append(f"{candidate}: {exc}")

        if transform is None:
            self.get_logger().warning(
                "TF lookup failed for candidates: " + " | ".join(errors)
            )
            return None

        translation = transform.transform.translation
        rotation = transform.transform.rotation

        rot_matrix = self.quaternion_to_rotation_matrix(
            rotation.x, rotation.y, rotation.z, rotation.w
        )
        camera_vec = np.array(camera_xyz, dtype=np.float64)
        world_vec = rot_matrix @ camera_vec + np.array(
            [translation.x, translation.y, translation.z], dtype=np.float64
        )

        return (float(world_vec[0]), float(world_vec[1]), float(world_vec[2]))

    def build_frame_candidates(self, source_frame: str) -> List[str]:
        candidates = []

        def add_candidate(frame: str) -> None:
            if frame and frame not in candidates:
                candidates.append(frame)

        add_candidate(self.optical_frame)
        add_candidate(source_frame)

        if "/" in source_frame:
            prefix = source_frame.split("/", 1)[0]
            add_candidate(f"{prefix}/{self.optical_frame}")

        if "camera_base_link/depth_camera" in source_frame:
            replaced = source_frame.replace(
                "camera_base_link/depth_camera",
                "camera_depth_optical_frame",
            )
            add_candidate(replaced)
            if "/" in replaced:
                add_candidate(replaced.split("/", 1)[1])

        if "/" in source_frame:
            add_candidate(source_frame.split("/", 1)[1])

        return candidates

    def quaternion_to_rotation_matrix(
        self, x: float, y: float, z: float, w: float
    ) -> np.ndarray:
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z

        return np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=np.float64,
        )

    def ros_image_to_cv_color(self, msg: Image) -> np.ndarray:
        if msg.encoding not in ("rgb8", "bgr8", "rgba8", "bgra8"):
            raise ValueError(f"Unsupported color encoding: {msg.encoding}")

        channels = 4 if "a8" in msg.encoding else 3
        image = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            (msg.height, msg.width, channels)
        )

        if msg.encoding == "rgb8":
            return image.copy()
        if msg.encoding == "bgr8":
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if msg.encoding == "rgba8":
            return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

    def ros_image_to_cv_depth(self, msg: Image) -> np.ndarray:
        if msg.encoding == "32FC1":
            return np.frombuffer(msg.data, dtype=np.float32).reshape(
                (msg.height, msg.width)
            )
        if msg.encoding == "16UC1":
            depth_mm = np.frombuffer(msg.data, dtype=np.uint16).reshape(
                (msg.height, msg.width)
            )
            return depth_mm.astype(np.float32) / 1000.0

        raise ValueError(f"Unsupported depth encoding: {msg.encoding}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BlockPoseErrorTester()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
