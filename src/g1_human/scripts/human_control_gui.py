#!/usr/bin/env python3
import math
import sys
import threading

import rclpy
from geometry_msgs.msg import Pose2D
from rclpy.node import Node
from sensor_msgs.msg import JointState

try:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import (
        QApplication,
        QCheckBox,
        QDoubleSpinBox,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSlider,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    print("PyQt5 not found. Install with: pip3 install PyQt5")
    sys.exit(1)


SLIDER_RESOLUTION = 1000
DEFAULT_X = 0.5
DEFAULT_Y = 0.0
DEFAULT_YAW_DEG = 0.0

POSE_CONTROLS = [
    ('x', 'X', 'm', -5.0, 5.0, DEFAULT_X, 0.01, 2),
    ('y', 'Y', 'm', -5.0, 5.0, DEFAULT_Y, 0.01, 2),
    ('yaw_deg', 'Yaw', 'deg', -180.0, 180.0, DEFAULT_YAW_DEG, 1.0, 1),
]

JOINT_GROUPS = {
    'Waist': [
        ('waist_yaw_joint', -2.094, 2.094, 0.0),
        ('waist_roll_joint', -0.416, 0.416, 0.0),
        ('waist_pitch_joint', -0.416, 0.416, 0.0),
    ],
    'Left Arm': [
        ('left_shoulder_pitch_joint', -2.471, 2.136, 0.35),
        ('left_shoulder_roll_joint', -1.271, 1.801, 0.18),
        ('left_shoulder_yaw_joint', -2.094, 2.094, 0.0),
        ('left_elbow_joint', -0.838, 1.676, 0.87),
    ],
    'Right Arm': [
        ('right_shoulder_pitch_joint', -2.471, 2.136, 0.35),
        ('right_shoulder_roll_joint', -1.801, 1.271, -0.18),
        ('right_shoulder_yaw_joint', -2.094, 2.094, 0.0),
        ('right_elbow_joint', -0.838, 1.676, 0.87),
    ],
}


class HumanControlGui(QWidget):
    def __init__(self, node, default_x, default_y, default_yaw_deg,
                 publish_joints_default):
        super().__init__()
        self.node = node
        self.pose_pub = node.create_publisher(Pose2D, '/human/pose_command', 10)
        self.joint_pub = node.create_publisher(
            JointState, '/human/joint_commands', 10,
        )
        self.pose_defaults = {
            'x': float(default_x),
            'y': float(default_y),
            'yaw_deg': float(default_yaw_deg),
        }
        self.publish_joints_default = self._as_bool(publish_joints_default)
        self.pose_controls = {}
        self.joint_controls = {}
        self._syncing = False

        self._build_ui()
        self.timer = node.create_timer(0.1, self._publish)
        self._publish()

    def _build_ui(self):
        self.setWindowTitle('Human Control')
        main_layout = QVBoxLayout()

        pose_group = QGroupBox('World Pose')
        pose_layout = QVBoxLayout()
        for name, label, suffix, lo, hi, _, step, decimals in POSE_CONTROLS:
            default = self.pose_defaults[name]
            self._add_control(
                pose_layout, self.pose_controls, name, label, suffix,
                lo, hi, default, step, decimals,
            )
        pose_buttons = QHBoxLayout()
        reset_pose_btn = QPushButton('Reset Pose')
        reset_pose_btn.clicked.connect(self._reset_pose)
        pose_buttons.addWidget(reset_pose_btn)
        pose_layout.addLayout(pose_buttons)
        pose_group.setLayout(pose_layout)
        main_layout.addWidget(pose_group)

        joints_group = QGroupBox('Joint Angles')
        joints_layout = QVBoxLayout()
        self.publish_joints_checkbox = QCheckBox('Publish Joint Commands')
        self.publish_joints_checkbox.setChecked(self.publish_joints_default)
        self.publish_joints_checkbox.toggled.connect(
            lambda _: self._publish_joints()
        )
        joints_layout.addWidget(self.publish_joints_checkbox)

        for group_name, joints in JOINT_GROUPS.items():
            group = QGroupBox(group_name)
            group_layout = QVBoxLayout()
            for joint_name, lo, hi, default in joints:
                label = joint_name.replace('_joint', '').replace('_', ' ').title()
                self._add_control(
                    group_layout, self.joint_controls, joint_name, label,
                    'rad', lo, hi, default, 0.01, 2,
                )
            group.setLayout(group_layout)
            joints_layout.addWidget(group)

        joint_buttons = QHBoxLayout()
        reset_joints_btn = QPushButton('Reset Joints')
        reset_joints_btn.clicked.connect(self._reset_joints)
        reset_all_btn = QPushButton('Reset All')
        reset_all_btn.clicked.connect(self._reset_all)
        joint_buttons.addWidget(reset_joints_btn)
        joint_buttons.addWidget(reset_all_btn)
        joints_layout.addLayout(joint_buttons)
        joints_group.setLayout(joints_layout)
        main_layout.addWidget(joints_group)

        self.setLayout(main_layout)
        self.resize(620, 760)

    def _add_control(self, layout, store, name, label_text, suffix, lo, hi,
                     default, step, decimals):
        row = QHBoxLayout()

        label = QLabel(label_text)
        label.setFixedWidth(170)
        row.addWidget(label)

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(SLIDER_RESOLUTION)
        slider.setValue(self._value_to_tick(default, lo, hi))
        row.addWidget(slider)

        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setSuffix(f' {suffix}')
        spin.setValue(default)
        spin.setFixedWidth(105)
        row.addWidget(spin)

        slider.valueChanged.connect(
            lambda tick, s=store, n=name: self._slider_changed(s, n, tick)
        )
        spin.valueChanged.connect(
            lambda value, s=store, n=name: self._spin_changed(s, n, value)
        )

        store[name] = {
            'slider': slider,
            'spin': spin,
            'lo': lo,
            'hi': hi,
            'default': default,
        }
        layout.addLayout(row)

    def _slider_changed(self, store, name, tick):
        if self._syncing:
            return
        control = store[name]
        value = self._tick_to_value(tick, control['lo'], control['hi'])
        self._syncing = True
        control['spin'].setValue(value)
        self._syncing = False
        self._publish()

    def _spin_changed(self, store, name, value):
        if self._syncing:
            return
        control = store[name]
        tick = self._value_to_tick(value, control['lo'], control['hi'])
        self._syncing = True
        control['slider'].setValue(tick)
        self._syncing = False
        self._publish()

    def _reset_pose(self):
        for name, default in self.pose_defaults.items():
            self.pose_controls[name]['spin'].setValue(default)
        self._publish_pose()

    def _reset_joints(self):
        for control in self.joint_controls.values():
            control['spin'].setValue(control['default'])
        self._publish_joints()

    def _reset_all(self):
        self._reset_pose()
        self._reset_joints()

    def _publish(self):
        self._publish_pose()
        self._publish_joints()

    def _publish_pose(self):
        msg = Pose2D()
        msg.x = float(self.pose_controls['x']['spin'].value())
        msg.y = float(self.pose_controls['y']['spin'].value())
        msg.theta = math.radians(
            float(self.pose_controls['yaw_deg']['spin'].value())
        )
        self.pose_pub.publish(msg)

    def _publish_joints(self):
        if not self.publish_joints_checkbox.isChecked():
            return
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        for joint_name, control in self.joint_controls.items():
            msg.name.append(joint_name)
            msg.position.append(float(control['spin'].value()))
        self.joint_pub.publish(msg)

    @staticmethod
    def _value_to_tick(value, lo, hi):
        ratio = (float(value) - lo) / (hi - lo)
        ratio = max(0.0, min(1.0, ratio))
        return int(round(ratio * SLIDER_RESOLUTION))

    @staticmethod
    def _tick_to_value(tick, lo, hi):
        return lo + (float(tick) / SLIDER_RESOLUTION) * (hi - lo)

    @staticmethod
    def _as_bool(value):
        if isinstance(value, str):
            return value.strip().lower() in ('1', 'true', 'yes', 'on')
        return bool(value)


def main():
    rclpy.init()
    node = Node('human_control_gui')
    node.declare_parameter('default_x', DEFAULT_X)
    node.declare_parameter('default_y', DEFAULT_Y)
    node.declare_parameter('default_yaw_deg', DEFAULT_YAW_DEG)
    node.declare_parameter('publish_joints_default', True)

    app = QApplication(sys.argv)
    gui = HumanControlGui(
        node,
        node.get_parameter('default_x').value,
        node.get_parameter('default_y').value,
        node.get_parameter('default_yaw_deg').value,
        node.get_parameter('publish_joints_default').value,
    )
    gui.show()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    app.exec_()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
