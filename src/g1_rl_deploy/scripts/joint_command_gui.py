#!/usr/bin/env python3
import sys
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist

try:
    from PyQt5.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout,
        QSlider, QLabel, QGroupBox, QPushButton,
    )
    from PyQt5.QtCore import Qt
except ImportError:
    print("PyQt5 not found. Install with: pip3 install PyQt5")
    sys.exit(1)

# Joint definitions: (name, index_in_29dof, min_rad, max_rad, default_rad)
JOINTS = {
    "Left Arm": [
        ("left_shoulder_pitch_joint", 15, -2.0, 2.0, 0.35),
        ("left_shoulder_roll_joint",  16, -1.0, 1.8, 0.18),
        ("left_elbow_joint",          18, -0.8, 1.7, 0.87),
    ],
    "Right Arm": [
        ("right_shoulder_pitch_joint", 22, -2.0, 2.0, 0.35),
        ("right_shoulder_roll_joint",  23, -1.8, 1.0, -0.18),
        ("right_elbow_joint",          25, -0.8, 1.7, 0.87),
    ],
    "Waist": [
        ("waist_roll_joint",  13, -0.4, 0.4, 0.0),
        ("waist_pitch_joint", 14, -0.4, 0.4, 0.0),
    ],
}

SLIDER_RESOLUTION = 1000  # steps per slider

# Velocity definitions: (label, min, max, default)
VELOCITIES = {
    "Linear Velocity": [
        ("lin_vel_x", -0.5, 1.0, 0.0),
        ("lin_vel_y", -0.5, 0.5, 0.0),
    ],
    "Angular Velocity": [
        ("ang_vel_z", -1.0, 1.0, 0.0),
    ],
}


class JointCommandGui(QWidget):
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.pub = node.create_publisher(JointState, "/joint_commands", 1)
        self.vel_pub = node.create_publisher(Twist, "/cmd_vel", 1)
        self.sliders = {}  # name -> (slider, value_label, lo, hi, default)
        self.vel_sliders = {}  # name -> (slider, value_label, lo, hi, default)
        self._build_ui()

        # Publish at 10 Hz
        self.timer = node.create_timer(0.1, self._publish)

    def _build_ui(self):
        self.setWindowTitle("G1 Upper Body Joint Commands")
        main_layout = QVBoxLayout()

        for group_name, joints in JOINTS.items():
            group = QGroupBox(group_name)
            group_layout = QVBoxLayout()

            for joint_name, _, lo, hi, default in joints:
                row = QHBoxLayout()

                label = QLabel(joint_name.replace("_joint", "").replace("_", " ").title())
                label.setFixedWidth(160)
                row.addWidget(label)

                slider = QSlider(Qt.Horizontal)
                slider.setMinimum(0)
                slider.setMaximum(SLIDER_RESOLUTION)
                slider.setValue(self._rad_to_tick(default, lo, hi))
                row.addWidget(slider)

                val_label = QLabel(f"{default:+.2f}")
                val_label.setFixedWidth(50)
                row.addWidget(val_label)

                slider.valueChanged.connect(
                    lambda v, lbl=val_label, l=lo, h=hi: lbl.setText(
                        f"{self._tick_to_rad(v, l, h):+.2f}"
                    )
                )

                self.sliders[joint_name] = (slider, val_label, lo, hi, default)
                group_layout.addLayout(row)

            group.setLayout(group_layout)
            main_layout.addWidget(group)

        # Velocity sliders
        for group_name, vels in VELOCITIES.items():
            group = QGroupBox(group_name)
            group_layout = QVBoxLayout()

            for vel_name, lo, hi, default in vels:
                row = QHBoxLayout()

                label = QLabel(vel_name.replace("_", " ").title())
                label.setFixedWidth(160)
                row.addWidget(label)

                slider = QSlider(Qt.Horizontal)
                slider.setMinimum(0)
                slider.setMaximum(SLIDER_RESOLUTION)
                slider.setValue(self._rad_to_tick(default, lo, hi))
                row.addWidget(slider)

                val_label = QLabel(f"{default:+.2f}")
                val_label.setFixedWidth(50)
                row.addWidget(val_label)

                slider.valueChanged.connect(
                    lambda v, lbl=val_label, l=lo, h=hi: lbl.setText(
                        f"{self._tick_to_rad(v, l, h):+.2f}"
                    )
                )

                self.vel_sliders[vel_name] = (slider, val_label, lo, hi, default)
                group_layout.addLayout(row)

            group.setLayout(group_layout)
            main_layout.addWidget(group)

        # Reset button
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self._reset)
        main_layout.addWidget(reset_btn)

        self.setLayout(main_layout)
        self.resize(500, 500)

    def _rad_to_tick(self, val, lo, hi):
        return int((val - lo) / (hi - lo) * SLIDER_RESOLUTION)

    def _tick_to_rad(self, tick, lo, hi):
        return lo + (tick / SLIDER_RESOLUTION) * (hi - lo)

    def _reset(self):
        for _, (slider, _, lo, hi, default) in self.sliders.items():
            slider.setValue(self._rad_to_tick(default, lo, hi))
        for _, (slider, _, lo, hi, default) in self.vel_sliders.items():
            slider.setValue(self._rad_to_tick(default, lo, hi))

    def _publish(self):
        msg = JointState()
        for joint_name, (slider, _, lo, hi, _) in self.sliders.items():
            msg.name.append(joint_name)
            msg.position.append(self._tick_to_rad(slider.value(), lo, hi))
        self.pub.publish(msg)

        def _vel_val(name):
            s, _, lo, hi, _ = self.vel_sliders[name]
            return self._tick_to_rad(s.value(), lo, hi)

        twist = Twist()
        twist.linear.x = _vel_val("lin_vel_x")
        twist.linear.y = _vel_val("lin_vel_y")
        twist.angular.z = _vel_val("ang_vel_z")
        self.vel_pub.publish(twist)


def main():
    rclpy.init()
    node = Node("joint_command_gui")

    app = QApplication(sys.argv)
    gui = JointCommandGui(node)
    gui.show()

    # Spin ROS in a background thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    app.exec_()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
