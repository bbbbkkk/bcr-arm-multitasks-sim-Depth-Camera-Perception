#!/usr/bin/env python3

import math
import tkinter as tk
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
from geometry_msgs.msg import PoseStamped
import numpy as np
import rclpy

from detect_block_poses import BlockPoseDetector


@dataclass
class DisplayPose:
    x: float
    y: float
    z: float
    yaw: float
    updated_at: float
    pixel: Tuple[int, int]
    sample_count: int


class BlockPoseMonitorGui(BlockPoseDetector):
    def __init__(self) -> None:
        super().__init__()

        self.declare_parameter("stale_timeout", 1.0)
        self.declare_parameter("workspace_x_min", 0.20)
        self.declare_parameter("workspace_x_max", 0.72)
        self.declare_parameter("workspace_y_min", -0.50)
        self.declare_parameter("workspace_y_max", 0.55)
        self.declare_parameter("place_size", 0.18)
        self.declare_parameter("view_size", 420)

        self.stale_timeout = (
            self.get_parameter("stale_timeout").get_parameter_value().double_value
        )
        self.workspace_x_min = (
            self.get_parameter("workspace_x_min").get_parameter_value().double_value
        )
        self.workspace_x_max = (
            self.get_parameter("workspace_x_max").get_parameter_value().double_value
        )
        self.workspace_y_min = (
            self.get_parameter("workspace_y_min").get_parameter_value().double_value
        )
        self.workspace_y_max = (
            self.get_parameter("workspace_y_max").get_parameter_value().double_value
        )
        self.place_size = (
            self.get_parameter("place_size").get_parameter_value().double_value
        )
        self.view_size = (
            self.get_parameter("view_size").get_parameter_value().integer_value
        )

        self.workspace_center_x = 0.5 * (self.workspace_x_min + self.workspace_x_max)
        self.workspace_center_y = 0.5 * (self.workspace_y_min + self.workspace_y_max)
        (
            self.display_x_min,
            self.display_x_max,
            self.display_y_min,
            self.display_y_max,
        ) = self.compute_rotated_workspace_bounds()

        self.display_pose_data: Dict[str, Optional[DisplayPose]] = {
            "red": None,
            "blue": None,
            "purple": None,
            "place": None,
        }

        self.status_var = None
        self.data_vars: Dict[str, tk.StringVar] = {}
        self.root = tk.Tk()
        self.root.title("BCR Block Pose Monitor")
        self.root.geometry("1100x860")
        self.root.configure(bg="#f4f1eb")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.camera_label: Optional[tk.Label] = None
        self.camera_photo = None
        self.canvas: Optional[tk.Canvas] = None

        self.build_ui()
        self.root.after(30, self.schedule_spin)
        self.root.after(120, self.schedule_refresh)

        self.get_logger().info(
            "Block pose monitor GUI started. This GUI now runs detection and publishes pose topics."
        )

    def build_ui(self) -> None:
        container = tk.Frame(self.root, bg="#f4f1eb", padx=16, pady=16)
        container.pack(fill=tk.BOTH, expand=True)

        title = tk.Label(
            container,
            text="BCR Block Pose Monitor",
            font=("Helvetica", 18, "bold"),
            bg="#f4f1eb",
            fg="#1f1f1f",
        )
        title.pack(anchor="w")

        self.status_var = tk.StringVar(
            value="GUI detector starting. Waiting for color / depth / camera_info..."
        )
        subtitle = tk.Label(
            container,
            textvariable=self.status_var,
            font=("Helvetica", 11),
            bg="#f4f1eb",
            fg="#555555",
        )
        subtitle.pack(anchor="w", pady=(4, 12))

        top = tk.Frame(container, bg="#f4f1eb")
        top.pack(fill=tk.BOTH, expand=False)

        left_panel = tk.Frame(top, bg="#ffffff", bd=1, relief=tk.SOLID, padx=10, pady=10)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right_panel = tk.Frame(top, bg="#ffffff", bd=1, relief=tk.SOLID, padx=10, pady=10)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(16, 0))

        left_title = tk.Label(
            left_panel,
            text="Top View Monitor",
            font=("Helvetica", 13, "bold"),
            bg="#ffffff",
            fg="#222222",
        )
        left_title.pack(anchor="w")

        left_hint = tk.Label(
            left_panel,
            text=(
                "Top-down monitor only. Rotated 90° CCW for easier viewing.\n"
                "Shows detected world poses, not the arm animation."
            ),
            font=("Helvetica", 10),
            bg="#ffffff",
            fg="#666666",
            justify=tk.LEFT,
        )
        left_hint.pack(anchor="w", pady=(4, 8))

        self.canvas = tk.Canvas(
            left_panel,
            width=self.view_size,
            height=self.view_size,
            bg="#fcfaf6",
            highlightthickness=1,
            highlightbackground="#c9c2b8",
        )
        self.canvas.pack(anchor="center")

        right_title = tk.Label(
            right_panel,
            text="RGB-D Camera View",
            font=("Helvetica", 13, "bold"),
            bg="#ffffff",
            fg="#222222",
        )
        right_title.pack(anchor="w")

        right_hint = tk.Label(
            right_panel,
            text=(
                "Real-time color image from the depth camera.\n"
                "Detected red / blue / purple / place centers are overlaid."
            ),
            font=("Helvetica", 10),
            bg="#ffffff",
            fg="#666666",
            justify=tk.LEFT,
        )
        right_hint.pack(anchor="w", pady=(4, 8))

        self.camera_label = tk.Label(
            right_panel,
            width=self.view_size,
            height=self.view_size,
            bg="#111111",
            bd=0,
        )
        self.camera_label.pack(anchor="center")

        bottom = tk.Frame(container, bg="#ffffff", bd=1, relief=tk.SOLID)
        bottom.pack(fill=tk.BOTH, expand=True, pady=(16, 0))

        bottom_title = tk.Label(
            bottom,
            text="Detected Pose Data",
            font=("Helvetica", 13, "bold"),
            bg="#ffffff",
            fg="#222222",
            padx=16,
            pady=12,
        )
        bottom_title.pack(anchor="w")

        cards = tk.Frame(bottom, bg="#ffffff")
        cards.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 14))

        self.build_data_card(cards, "red", "#d7263d")
        self.build_data_card(cards, "blue", "#1d5fd0")
        self.build_data_card(cards, "purple", "#9b30c9")
        self.build_data_card(cards, "place", "#12a61e")

    def build_data_card(self, parent: tk.Widget, name: str, color: str) -> None:
        card = tk.Frame(parent, bg="#ffffff", bd=1, relief=tk.SOLID, padx=12, pady=10)
        card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6)

        title = tk.Label(
            card,
            text=name.capitalize(),
            font=("Helvetica", 12, "bold"),
            bg="#ffffff",
            fg=color,
        )
        title.pack(anchor="w")

        var = tk.StringVar(value="status : waiting")
        self.data_vars[name] = var
        body = tk.Label(
            card,
            textvariable=var,
            font=("Courier", 10),
            bg="#ffffff",
            fg="#222222",
            justify=tk.LEFT,
            anchor="w",
        )
        body.pack(anchor="w", pady=(6, 0))

    def publish_averaged_poses(self) -> None:
        missing_inputs = []
        if self.latest_color_msg is None:
            missing_inputs.append("color_image")
        if self.latest_depth_msg is None:
            missing_inputs.append("depth_image")
        if self.latest_camera_info is None:
            missing_inputs.append("camera_info")

        if missing_inputs:
            if self.status_var is not None:
                self.status_var.set(
                    "Waiting for: " + ", ".join(missing_inputs)
                )
            return

        fresh_count = 0

        for color_name in self.color_names:
            history = self.detection_history[color_name]
            if len(history) < self.min_publish_samples:
                continue

            avg_center, avg_pixel, avg_yaw = self.compute_average_from_history(history)
            pose_msg = self.build_pose_msg(avg_center, avg_yaw)
            self.pose_publishers[color_name].publish(pose_msg)
            self.display_pose_data[color_name] = DisplayPose(
                x=float(avg_center[0]),
                y=float(avg_center[1]),
                z=float(avg_center[2]),
                yaw=float(avg_yaw),
                updated_at=self.get_clock().now().nanoseconds / 1e9,
                pixel=(int(round(avg_pixel[0])), int(round(avg_pixel[1]))),
                sample_count=len(history),
            )
            fresh_count += 1

        if len(self.place_history) >= self.min_publish_samples:
            avg_center, avg_pixel, avg_yaw = self.compute_average_from_history(
                self.place_history, key="surface"
            )
            place_pose_msg = self.build_pose_msg(avg_center, avg_yaw)
            self.place_pose_publisher.publish(place_pose_msg)
            self.display_pose_data["place"] = DisplayPose(
                x=float(avg_center[0]),
                y=float(avg_center[1]),
                z=float(avg_center[2]),
                yaw=float(avg_yaw),
                updated_at=self.get_clock().now().nanoseconds / 1e9,
                pixel=(int(round(avg_pixel[0])), int(round(avg_pixel[1]))),
                sample_count=len(self.place_history),
            )
            fresh_count += 1

        if self.status_var is not None:
            self.status_var.set(
                f"GUI detector publishing poses. Fresh topics: {fresh_count}/4   "
                f"workspace x=[{self.workspace_x_min:.2f}, {self.workspace_x_max:.2f}]   "
                f"y=[{self.workspace_y_min:.2f}, {self.workspace_y_max:.2f}]"
            )

    def compute_average_from_history(
        self, history, key: str = "center"
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        centers = np.array([item[key] for item in history], dtype=np.float64)
        avg_center = centers.mean(axis=0)
        pixels = np.array([item["pixel"] for item in history], dtype=np.float64)
        avg_pixel = pixels.mean(axis=0)
        avg_yaw = self.average_square_yaws(
            [float(item["yaw"]) for item in history]
        )
        return avg_center, avg_pixel, avg_yaw

    def build_pose_msg(self, avg_center: np.ndarray, avg_yaw: float) -> PoseStamped:
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
        return pose_msg

    def schedule_spin(self) -> None:
        if not rclpy.ok():
            return
        rclpy.spin_once(self, timeout_sec=0.0)
        self.root.after(30, self.schedule_spin)

    def schedule_refresh(self) -> None:
        self.refresh_canvas()
        self.refresh_camera_view()
        self.refresh_data_cards()
        self.root.after(120, self.schedule_refresh)

    def refresh_canvas(self) -> None:
        if self.canvas is None:
            return

        self.canvas.delete("all")
        self.draw_grid()
        self.draw_axes()

        self.draw_pose_box("place", "#77e673", "#189c1f", self.place_size, self.place_size)
        self.draw_pose_box("red", "#ef3340", "#8f1022", self.block_size, self.block_size)
        self.draw_pose_box("blue", "#1d6ef2", "#0f3985", self.block_size, self.block_size)
        self.draw_pose_box("purple", "#b84ae6", "#6f1d8a", self.block_size, self.block_size)

    def refresh_camera_view(self) -> None:
        if self.camera_label is None:
            return

        if self.latest_color_msg is None:
            self.camera_label.configure(text="Waiting for camera image...", fg="#dddddd", image="")
            return

        try:
            color_image = self.ros_image_to_cv_color(self.latest_color_msg)
        except ValueError as exc:
            self.camera_label.configure(text=str(exc), fg="#ffaaaa", image="")
            return

        view = color_image.copy()
        overlays = {
            "red": ("red", (40, 40, 220)),
            "blue": ("blue", (220, 80, 30)),
            "purple": ("purple", (180, 60, 220)),
            "place": ("place", (20, 180, 40)),
        }

        for name, (label, bgr) in overlays.items():
            pose = self.display_pose_data[name]
            if pose is None or not self.is_pose_fresh(pose):
                continue

            u, v = pose.pixel
            cv2.circle(view, (u, v), 8, bgr, 2)
            cv2.putText(
                view,
                label,
                (u + 10, max(18, v - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                bgr,
                2,
                cv2.LINE_AA,
            )

        resized = cv2.resize(view, (self.view_size, self.view_size), interpolation=cv2.INTER_LINEAR)
        self.camera_photo = self.numpy_rgb_to_photoimage(resized)
        self.camera_label.configure(image=self.camera_photo, text="")

    def refresh_data_cards(self) -> None:
        now = self.get_clock().now().nanoseconds / 1e9
        for name, var in self.data_vars.items():
            pose = self.display_pose_data[name]
            if pose is None:
                var.set("status : waiting")
                continue

            age = now - pose.updated_at
            fresh = age <= self.stale_timeout
            state = "fresh" if fresh else "stale"
            var.set(
                "\n".join(
                    [
                        f"status : {state:>5}   age={age:4.1f}s",
                        f"samples : {pose.sample_count:>3}",
                        f"x      : {pose.x: .4f}",
                        f"y      : {pose.y: .4f}",
                        f"z      : {pose.z: .4f}",
                        f"yaw    : {math.degrees(pose.yaw): .1f} deg",
                        f"uv     : ({pose.pixel[0]:>3}, {pose.pixel[1]:>3})",
                    ]
                )
            )

    def draw_grid(self) -> None:
        step = 0.10
        x = math.floor(self.workspace_x_min / step) * step
        while x <= self.workspace_x_max + 1e-6:
            x0, y0 = self.world_to_canvas(x, self.workspace_y_min)
            x1, y1 = self.world_to_canvas(x, self.workspace_y_max)
            self.canvas.create_line(x0, y0, x1, y1, fill="#ece6db")
            x += step

        y = math.floor(self.workspace_y_min / step) * step
        while y <= self.workspace_y_max + 1e-6:
            x0, y0 = self.world_to_canvas(self.workspace_x_min, y)
            x1, y1 = self.world_to_canvas(self.workspace_x_max, y)
            self.canvas.create_line(x0, y0, x1, y1, fill="#ece6db")
            y += step

        x0, y0 = self.world_to_canvas(self.workspace_x_min, self.workspace_y_min)
        x1, y1 = self.world_to_canvas(self.workspace_x_max, self.workspace_y_max)
        self.canvas.create_rectangle(x0, y1, x1, y0, outline="#c7bfb4", width=2)

    def draw_axes(self) -> None:
        base_world_x = self.workspace_x_min + 0.04
        base_world_y = self.workspace_y_min + 0.05
        x0, y0 = self.world_to_canvas(base_world_x, base_world_y)
        x1, y1 = self.world_to_canvas(base_world_x + 0.08, base_world_y)
        x2, y2 = self.world_to_canvas(base_world_x, base_world_y + 0.08)
        self.canvas.create_line(x0, y0, x1, y1, fill="#555555", width=2, arrow=tk.LAST)
        self.canvas.create_line(x0, y0, x2, y2, fill="#555555", width=2, arrow=tk.LAST)
        self.canvas.create_text(x1 + 14, y1, text="+x", fill="#555555", font=("Helvetica", 10, "bold"))
        self.canvas.create_text(x2, y2 - 14, text="+y", fill="#555555", font=("Helvetica", 10, "bold"))

    def draw_pose_box(self, name: str, fill: str, outline: str, width_m: float, height_m: float) -> None:
        pose = self.display_pose_data[name]
        if pose is None:
            return

        fresh = self.is_pose_fresh(pose)
        self.draw_rotated_box(
            pose.x,
            pose.y,
            width_m,
            height_m,
            pose.yaw,
            fill=fill if fresh else "#d7d7d7",
            outline=outline if fresh else "#777777",
            label=name if fresh else f"{name}(stale)",
            label_color=outline if fresh else "#666666",
        )

    def draw_rotated_box(
        self,
        x: float,
        y: float,
        width_m: float,
        height_m: float,
        yaw: float,
        fill: str,
        outline: str,
        label: str,
        label_color: str,
    ) -> None:
        half_w = width_m / 2.0
        half_h = height_m / 2.0
        corners = [
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
            (-half_w, half_h),
        ]

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        points = []
        for cx, cy in corners:
            wx = x + cx * cos_yaw - cy * sin_yaw
            wy = y + cx * sin_yaw + cy * cos_yaw
            px, py = self.world_to_canvas(wx, wy)
            points.extend([px, py])

        self.canvas.create_polygon(points, fill=fill, outline=outline, width=2)

        cx, cy = self.world_to_canvas(x, y)
        ax = x + width_m * 0.6 * math.cos(yaw)
        ay = y + width_m * 0.6 * math.sin(yaw)
        apx, apy = self.world_to_canvas(ax, ay)
        self.canvas.create_line(cx, cy, apx, apy, fill=outline, width=2, arrow=tk.LAST)
        self.canvas.create_text(cx, cy - 18, text=label, fill=label_color, font=("Helvetica", 10, "bold"))

    def world_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        rotated_x, rotated_y = self.rotate_world_ccw_90(x, y)
        x_ratio = (rotated_x - self.display_x_min) / (self.display_x_max - self.display_x_min)
        y_ratio = (rotated_y - self.display_y_min) / (self.display_y_max - self.display_y_min)
        px = 20 + x_ratio * (self.view_size - 40)
        py = self.view_size - (20 + y_ratio * (self.view_size - 40))
        return px, py

    def rotate_world_ccw_90(self, x: float, y: float) -> tuple[float, float]:
        dx = x - self.workspace_center_x
        dy = y - self.workspace_center_y
        return -dy, dx

    def compute_rotated_workspace_bounds(self) -> tuple[float, float, float, float]:
        corners = [
            (self.workspace_x_min, self.workspace_y_min),
            (self.workspace_x_min, self.workspace_y_max),
            (self.workspace_x_max, self.workspace_y_min),
            (self.workspace_x_max, self.workspace_y_max),
        ]
        rotated = [self.rotate_world_ccw_90(x, y) for x, y in corners]
        xs = [value[0] for value in rotated]
        ys = [value[1] for value in rotated]
        return min(xs), max(xs), min(ys), max(ys)

    def numpy_rgb_to_photoimage(self, image_rgb: np.ndarray) -> tk.PhotoImage:
        if image_rgb.dtype != np.uint8:
            image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
        header = f"P6 {image_rgb.shape[1]} {image_rgb.shape[0]} 255\n".encode("ascii")
        data = header + image_rgb.tobytes()
        return tk.PhotoImage(data=data, format="PPM")

    def is_pose_fresh(self, pose: DisplayPose) -> bool:
        now = self.get_clock().now().nanoseconds / 1e9
        return (now - pose.updated_at) <= self.stale_timeout

    def on_close(self) -> None:
        self.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main(args=None) -> None:
    rclpy.init(args=args)
    gui = BlockPoseMonitorGui()
    try:
        gui.run()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            gui.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
