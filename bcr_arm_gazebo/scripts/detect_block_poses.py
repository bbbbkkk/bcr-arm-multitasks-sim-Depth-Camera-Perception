#!/usr/bin/env python3

from collections import deque
import math
from typing import Deque, Dict, List, Optional, Tuple

import cv2
from geometry_msgs.msg import PoseStamped
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener


class BlockPoseDetector(Node):
    def __init__(self) -> None:
        super().__init__("block_pose_detector")

        self.declare_parameter("world_frame", "world")
        self.declare_parameter("color_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("min_contour_area", 800.0)
        self.declare_parameter("place_min_contour_area", 3000.0)
        self.declare_parameter("optical_frame", "camera_depth_optical_frame")
        self.declare_parameter("horizontal_fov", 1.0)
        self.declare_parameter("block_size", 0.09)
        self.declare_parameter("average_window_size", 30)
        self.declare_parameter("min_publish_samples", 10)
        self.declare_parameter("max_missing_frames", 2)
        self.declare_parameter("process_period", 0.2)
        self.declare_parameter("publish_period", 0.5)
        self.declare_parameter("topic_prefix", "/detected_blocks")
        self.declare_parameter("place_topic", "/detected_workspace/place_pose")

        self.world_frame = (
            self.get_parameter("world_frame").get_parameter_value().string_value
        )
        self.min_contour_area = (
            self.get_parameter("min_contour_area").get_parameter_value().double_value
        )
        self.place_min_contour_area = (
            self.get_parameter("place_min_contour_area")
            .get_parameter_value()
            .double_value
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
        self.min_publish_samples = max(
            1,
            self.get_parameter("min_publish_samples")
            .get_parameter_value()
            .integer_value,
        )
        self.max_missing_frames = max(
            1,
            self.get_parameter("max_missing_frames")
            .get_parameter_value()
            .integer_value,
        )
        self.process_period = (
            self.get_parameter("process_period").get_parameter_value().double_value
        )
        self.publish_period = (
            self.get_parameter("publish_period").get_parameter_value().double_value
        )
        self.topic_prefix = (
            self.get_parameter("topic_prefix").get_parameter_value().string_value
        ).rstrip("/")
        self.place_topic = (
            self.get_parameter("place_topic").get_parameter_value().string_value
        )

        self.latest_color_msg: Optional[Image] = None
        self.latest_depth_msg: Optional[Image] = None
        self.latest_camera_info: Optional[CameraInfo] = None
        self.last_processed_frame_key: Optional[Tuple[int, int, int, int]] = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.color_names = ("red", "blue", "purple")
        self.detection_history: Dict[str, Deque[Dict[str, object]]] = {
            color_name: deque(maxlen=self.average_window_size)
            for color_name in self.color_names
        }
        self.missing_counts: Dict[str, int] = {
            color_name: 0 for color_name in self.color_names
        }
        self.place_history: Deque[Dict[str, object]] = deque(
            maxlen=self.average_window_size
        )
        self.place_missing_count = 0

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
            "place": [
                (np.array([40, 80, 80]), np.array([90, 255, 255])),
            ],
        }

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

        self.pose_publishers: Dict[str, object] = {}
        for color_name in self.color_names:
            topic_name = f"{self.topic_prefix}/{color_name}_pose"
            self.pose_publishers[color_name] = self.create_publisher(
                PoseStamped, topic_name, 10
            )
        self.place_pose_publisher = self.create_publisher(
            PoseStamped, self.place_topic, 10
        )

        self.process_timer = self.create_timer(
            self.process_period, self.process_latest_frame
        )
        self.publish_timer = self.create_timer(
            self.publish_period, self.publish_averaged_poses
        )

        self.get_logger().info(
            "Block pose detector started. Publishing world poses to "
            + ", ".join(
                f"{color_name}={self.topic_prefix}/{color_name}_pose"
                for color_name in self.color_names
            )
            + f", place={self.place_topic}"
        )

    def on_color_image(self, msg: Image) -> None:
        self.latest_color_msg = msg

    def on_depth_image(self, msg: Image) -> None:
        self.latest_depth_msg = msg

    def on_camera_info(self, msg: CameraInfo) -> None:
        self.latest_camera_info = msg

    def process_latest_frame(self) -> None:
        if (
            self.latest_color_msg is None
            or self.latest_depth_msg is None
            or self.latest_camera_info is None
        ):
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
                self.missing_counts[color_name] = 0
            else:
                self.missing_counts[color_name] += 1
                if self.missing_counts[color_name] >= self.max_missing_frames:
                    self.detection_history[color_name].clear()

        place_detection = self.detect_place_area(color_image, depth_image)
        if place_detection is not None:
            self.place_history.append(place_detection)
            self.place_missing_count = 0
        else:
            self.place_missing_count += 1
            if self.place_missing_count >= self.max_missing_frames:
                self.place_history.clear()

        self.last_processed_frame_key = frame_key

    def publish_averaged_poses(self) -> None:
        missing_inputs = []
        if self.latest_color_msg is None:
            missing_inputs.append("color_image")
        if self.latest_depth_msg is None:
            missing_inputs.append("depth_image")
        if self.latest_camera_info is None:
            missing_inputs.append("camera_info")
        if missing_inputs:
            self.get_logger().info(f"Waiting for: {', '.join(missing_inputs)}")
            return

        lines = ["Perception update:"]
        for color_name in self.color_names:
            history = self.detection_history[color_name]
            if len(history) < self.min_publish_samples:
                lines.append(
                    f"  {color_name:<6} n={len(history):>2}/{self.min_publish_samples:<2}  waiting"
                )
                continue

            center_points = np.array(
                [detection["center"] for detection in history], dtype=np.float64
            )
            avg_center = center_points.mean(axis=0)
            avg_pixel = np.array(
                [detection["pixel"] for detection in history], dtype=np.float64
            ).mean(axis=0)
            avg_yaw = self.average_square_yaws(
                [float(detection["yaw"]) for detection in history]
            )

            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = self.world_frame
            pose_msg.pose.position.x = float(avg_center[0])
            pose_msg.pose.position.y = float(avg_center[1])
            pose_msg.pose.position.z = float(avg_center[2])
            qx, qy, qz, qw = self.quaternion_from_yaw(avg_yaw)
            pose_msg.pose.orientation.x = qx
            pose_msg.pose.orientation.y = qy
            pose_msg.pose.orientation.z = qz
            pose_msg.pose.orientation.w = qw
            self.pose_publishers[color_name].publish(pose_msg)

            lines.append(
                (
                    f"  {color_name:<6} "
                    f"n={len(history):>2}/{self.average_window_size:<2}  "
                    f"uv=({int(round(avg_pixel[0])):>3}, {int(round(avg_pixel[1])):>3})  "
                    f"xyz=({avg_center[0]:>6.3f}, {avg_center[1]:>6.3f}, {avg_center[2]:>6.3f})  "
                    f"yaw={math.degrees(avg_yaw):>6.1f} deg"
                )
            )

        if len(self.place_history) < self.min_publish_samples:
            lines.append(
                f"  {'place':<6} n={len(self.place_history):>2}/{self.min_publish_samples:<2}  waiting"
            )
        else:
            place_surface_points = np.array(
                [detection["surface"] for detection in self.place_history],
                dtype=np.float64,
            )
            avg_place_surface = place_surface_points.mean(axis=0)
            avg_place_pixel = np.array(
                [detection["pixel"] for detection in self.place_history],
                dtype=np.float64,
            ).mean(axis=0)
            avg_place_yaw = self.average_square_yaws(
                [float(detection["yaw"]) for detection in self.place_history]
            )

            place_pose_msg = PoseStamped()
            place_pose_msg.header.stamp = self.get_clock().now().to_msg()
            place_pose_msg.header.frame_id = self.world_frame
            place_pose_msg.pose.position.x = float(avg_place_surface[0])
            place_pose_msg.pose.position.y = float(avg_place_surface[1])
            place_pose_msg.pose.position.z = float(avg_place_surface[2])
            qx, qy, qz, qw = self.quaternion_from_yaw(avg_place_yaw)
            place_pose_msg.pose.orientation.x = qx
            place_pose_msg.pose.orientation.y = qy
            place_pose_msg.pose.orientation.z = qz
            place_pose_msg.pose.orientation.w = qw
            self.place_pose_publisher.publish(place_pose_msg)

            lines.append(
                (
                    f"  {'place':<6} "
                    f"n={len(self.place_history):>2}/{self.average_window_size:<2}  "
                    f"uv=({int(round(avg_place_pixel[0])):>3}, {int(round(avg_place_pixel[1])):>3})  "
                    f"xyz=({avg_place_surface[0]:>6.3f}, {avg_place_surface[1]:>6.3f}, {avg_place_surface[2]:>6.3f})  "
                    f"yaw={math.degrees(avg_place_yaw):>6.1f} deg"
                )
            )

        self.get_logger().info("\n".join(lines))

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
        yaw = self.estimate_block_yaw(contour, depth_m, self.latest_depth_msg.header.frame_id)
        if yaw is None:
            yaw = 0.0

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
            "yaw": yaw,
        }

    def detect_place_area(
        self, color_image: np.ndarray, depth_image: np.ndarray
    ) -> Optional[Dict[str, object]]:
        hsv = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for lower, upper in self.color_ranges["place"]:
            mask |= cv2.inRange(hsv, lower, upper)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < self.place_min_contour_area:
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

        yaw = self.estimate_block_yaw(contour, depth_m, self.latest_depth_msg.header.frame_id)
        if yaw is None:
            yaw = 0.0

        return {
            "pixel": (u, v),
            "depth_m": depth_m,
            "camera": camera_xyz,
            "world": world_xyz,
            "surface": world_xyz,
            "yaw": yaw,
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

    def estimate_block_yaw(
        self, contour: np.ndarray, depth_m: float, source_frame: str
    ) -> Optional[float]:
        rect = cv2.minAreaRect(contour)
        box_points = cv2.boxPoints(rect)

        world_points = []
        for point in box_points:
            u = int(round(point[0]))
            v = int(round(point[1]))
            camera_xyz = self.project_pixel_to_camera(
                u, v, depth_m, self.latest_camera_info
            )
            world_xyz = self.transform_point_to_world(camera_xyz, source_frame)
            if world_xyz is None:
                return None
            world_points.append(world_xyz)

        best_yaw = None
        best_length = 0.0
        for index in range(4):
            p0 = np.array(world_points[index][:2], dtype=np.float64)
            p1 = np.array(world_points[(index + 1) % 4][:2], dtype=np.float64)
            edge = p1 - p0
            length = float(np.linalg.norm(edge))
            if length < 1e-6:
                continue

            yaw = math.atan2(edge[1], edge[0])
            if length > best_length:
                best_length = length
                best_yaw = yaw

        if best_yaw is None:
            return None

        return self.canonicalize_square_yaw(best_yaw)

    def canonicalize_square_yaw(self, yaw: float) -> float:
        yaw = self.normalize_angle(yaw)
        while yaw >= math.pi / 4.0:
            yaw -= math.pi / 2.0
        while yaw < -math.pi / 4.0:
            yaw += math.pi / 2.0
        return yaw

    def average_square_yaws(self, yaws: List[float]) -> float:
        if not yaws:
            return 0.0

        wrapped = np.array([4.0 * yaw for yaw in yaws], dtype=np.float64)
        avg_wrapped = math.atan2(np.sin(wrapped).mean(), np.cos(wrapped).mean())
        return self.canonicalize_square_yaw(avg_wrapped / 4.0)

    def quaternion_from_yaw(self, yaw: float) -> Tuple[float, float, float, float]:
        half_yaw = yaw / 2.0
        return (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw))

    def normalize_angle(self, angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def transform_point_to_world(
        self, camera_xyz: Tuple[float, float, float], source_frame: str
    ) -> Optional[Tuple[float, float, float]]:
        transform = None
        for candidate in self.build_frame_candidates(source_frame):
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.world_frame,
                    candidate,
                    rclpy.time.Time(),
                )
                break
            except TransformException:
                continue

        if transform is None:
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
    node = BlockPoseDetector()

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
