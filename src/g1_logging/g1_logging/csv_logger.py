"""ROS node that records low-rate CBF reconstruction inputs to a wide CSV."""

from dataclasses import dataclass
import json
import math
import threading
import time

import rclpy
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from tf2_ros import TransformException

from g1_cbf.human_capsules import transform_capsule_array
from g1_cbf.tf_pose import TfPoseLookup, normalize_frame
from g1_cbf_msg.msg import CapsuleArray, WorkspaceState

from g1_logging.async_csv import AsyncCsvWriter
from g1_logging.csv_schema import (
    CSV_FIELDS,
    JOINT_DEFAULTS,
    JOINT_NAMES,
    N_HUMAN_CAPSULES,
    extract_joint_state,
    human_slot_fields,
    joint_observed_field,
    joint_value_field,
    stamp_to_sec,
)
from g1_logging.paths import resolve_csv_output


SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

LATCHED_STATE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


@dataclass(frozen=True)
class TimedJointSample:
    sample: object
    stamp_sec: float


@dataclass(frozen=True)
class TimedOdomSample:
    stamp_sec: float
    frame_id: str
    child_frame_id: str
    x: float
    y: float


@dataclass(frozen=True)
class TimedWorkspaceSample:
    stamp_sec: float
    frame_id: str
    child_frame_id: str
    enabled: bool
    capture_pending: bool
    generation: int
    translation: tuple
    quaternion: tuple


@dataclass(frozen=True)
class WorkspaceActivation:
    generation: int
    stamp_sec: float
    frame_id: str
    child_frame_id: str
    x: float
    y: float
    quaternion: tuple


@dataclass(frozen=True)
class TimedWorkspacePathSample:
    sequence: int
    stamp_sec: float
    frame_id: str
    points_xy: tuple


@dataclass(frozen=True)
class PendingHumanSample:
    sequence: int
    receipt_sec: float
    message: object


@dataclass(frozen=True)
class HumanCapsule:
    name: str
    a: tuple
    b: tuple
    radius: float


@dataclass(frozen=True)
class HumanWorldSample:
    sequence: int
    stamp_sec: float
    source_frame: str
    frame_id: str
    received_count: int
    capsules: tuple


@dataclass(frozen=True)
class LoggerSnapshot:
    sample_index: int
    ros_time_sec: float
    elapsed_sec: float
    joints: object
    human: object
    odom: object
    workspace: object
    workspace_activation: object
    workspace_path: object
    cbf_enabled: bool
    external_enabled: bool


def _finite(values):
    return all(math.isfinite(float(value)) for value in values)


def _age_sec(now_sec, stamp_sec):
    age = float(now_sec) - float(stamp_sec)
    return age if math.isfinite(age) else float('nan')


def _message_time_sec(stamp, receipt_sec):
    measurement_sec = stamp_to_sec(stamp)
    return float(receipt_sec) if measurement_sec == 0.0 else measurement_sec


class CbfCsvLoggerNode(Node):
    """Sample the CBF inputs without adding work to the CBF process."""

    def __init__(self):
        super().__init__('cbf_csv_logger_node')

        self.declare_parameter('sample_rate_hz', 10.0)
        self.declare_parameter('output_filename', '')
        self.declare_parameter('overwrite_existing', False)
        self.declare_parameter('queue_size', 4)
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('pelvis_frame', 'pelvis')
        self.declare_parameter('human_timeout_sec', 0.5)
        self.declare_parameter('tf_lookup_timeout_sec', 0.0)
        self.declare_parameter('tf_stale_timeout_sec', 0.5)
        self.declare_parameter('world_circle_radius_m', 1.5)
        self.declare_parameter('collision_geometry', 'capsules')
        self.declare_parameter('internal_margin_phi', 0.05)
        self.declare_parameter('external_margin_phi', 0.5)
        self.declare_parameter('external_torso_margin_phi', 0.1)

        self._sample_rate_hz = float(
            self.get_parameter('sample_rate_hz').value
        )
        self._queue_size = int(self.get_parameter('queue_size').value)
        self._world_frame = normalize_frame(
            self.get_parameter('world_frame').value,
            'world',
        )
        self._pelvis_frame = normalize_frame(
            self.get_parameter('pelvis_frame').value,
            'pelvis',
        )
        self._human_timeout_sec = float(
            self.get_parameter('human_timeout_sec').value
        )
        self._tf_lookup_timeout_sec = float(
            self.get_parameter('tf_lookup_timeout_sec').value
        )
        self._tf_stale_timeout_sec = float(
            self.get_parameter('tf_stale_timeout_sec').value
        )
        self._workspace_radius_m = float(
            self.get_parameter('world_circle_radius_m').value
        )
        self._collision_geometry = str(
            self.get_parameter('collision_geometry').value
        ).strip().lower()
        self._internal_margin_phi_m = float(
            self.get_parameter('internal_margin_phi').value
        )
        self._external_margin_phi_m = float(
            self.get_parameter('external_margin_phi').value
        )
        self._external_torso_margin_phi_m = float(
            self.get_parameter('external_torso_margin_phi').value
        )
        self._validate_parameters()

        self._tf_pose_lookup = TfPoseLookup(
            self,
            self._world_frame,
            self._pelvis_frame,
            self._tf_lookup_timeout_sec,
        )

        self._state_lock = threading.Lock()
        self._joints = None
        self._pending_human = None
        self._human_sequence = 0
        self._odom = None
        self._workspace = None
        self._workspace_activation = None
        self._workspace_path = None
        self._workspace_path_sequence = 0
        self._cbf_enabled = False
        self._external_enabled = False
        self._sample_index = 0
        self._start_monotonic = time.monotonic()
        self._shutdown_started = False

        # This cache belongs exclusively to the CSV worker thread.
        self._human_world = None
        self._last_logged_workspace_path_sequence = None

        self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_states_cb,
            SENSOR_QOS,
        )
        self.create_subscription(
            CapsuleArray,
            '/human/colliders',
            self._human_cb,
            SENSOR_QOS,
        )
        self.create_subscription(
            Odometry,
            '/odom',
            self._odom_cb,
            SENSOR_QOS,
        )
        self.create_subscription(
            Path,
            '/workspace_path',
            self._workspace_path_cb,
            LATCHED_STATE_QOS,
        )
        self.create_subscription(
            Bool,
            '/cbf/enabled',
            self._cbf_enabled_cb,
            LATCHED_STATE_QOS,
        )
        self.create_subscription(
            Bool,
            '/cbf/external_enabled',
            self._external_enabled_cb,
            LATCHED_STATE_QOS,
        )
        self.create_subscription(
            WorkspaceState,
            '/cbf/workspace_state',
            self._workspace_state_cb,
            LATCHED_STATE_QOS,
        )

        output_path = resolve_csv_output(
            self.get_parameter('output_filename').value
        )
        overwrite = bool(
            self.get_parameter('overwrite_existing').value
        )
        self._writer = AsyncCsvWriter(
            path=output_path,
            fieldnames=CSV_FIELDS,
            row_builder=self._snapshot_to_row,
            max_queue_size=self._queue_size,
            error_callback=self._writer_error_cb,
            file_mode='w' if overwrite else 'x',
        )
        self._writer.start()
        self._timer = self.create_timer(
            1.0 / self._sample_rate_hz,
            self._sample_tick,
        )

        self.get_logger().info(
            f'CBF CSV logger recording at {self._sample_rate_hz:.3f} Hz '
            f'to {output_path}'
        )

    def _validate_parameters(self):
        finite_values = (
            self._sample_rate_hz,
            self._human_timeout_sec,
            self._tf_lookup_timeout_sec,
            self._tf_stale_timeout_sec,
            self._workspace_radius_m,
            self._internal_margin_phi_m,
            self._external_margin_phi_m,
            self._external_torso_margin_phi_m,
        )
        if not _finite(finite_values):
            raise ValueError('numeric logger parameters must be finite')
        if self._sample_rate_hz <= 0.0:
            raise ValueError('sample_rate_hz must be positive')
        if self._queue_size < 1:
            raise ValueError('queue_size must be at least 1')
        if not self._world_frame or not self._pelvis_frame:
            raise ValueError('world_frame and pelvis_frame must not be empty')
        if self._world_frame == self._pelvis_frame:
            raise ValueError('world_frame and pelvis_frame must differ')
        if self._human_timeout_sec < 0.0:
            raise ValueError('human_timeout_sec must be non-negative')
        if self._tf_lookup_timeout_sec < 0.0:
            raise ValueError('tf_lookup_timeout_sec must be non-negative')
        if self._tf_stale_timeout_sec < 0.0:
            raise ValueError('tf_stale_timeout_sec must be non-negative')
        if self._workspace_radius_m <= 0.0:
            raise ValueError('world_circle_radius_m must be positive')
        if min(
            self._internal_margin_phi_m,
            self._external_margin_phi_m,
            self._external_torso_margin_phi_m,
        ) < 0.0:
            raise ValueError('collision margins must be non-negative')
        if self._collision_geometry != 'capsules':
            raise ValueError(
                "g1_logging currently requires collision_geometry='capsules'"
            )

    def _now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _joint_states_cb(self, msg):
        receipt_sec = self._now_sec()
        sample = TimedJointSample(
            sample=extract_joint_state(msg.name, msg.position),
            stamp_sec=_message_time_sec(msg.header.stamp, receipt_sec),
        )
        with self._state_lock:
            self._joints = sample

    def _human_cb(self, msg):
        receipt_sec = self._now_sec()
        with self._state_lock:
            self._human_sequence += 1
            self._pending_human = PendingHumanSample(
                sequence=self._human_sequence,
                receipt_sec=receipt_sec,
                message=msg,
            )

    def _odom_cb(self, msg):
        receipt_sec = self._now_sec()
        position = msg.pose.pose.position
        sample = TimedOdomSample(
            stamp_sec=_message_time_sec(msg.header.stamp, receipt_sec),
            frame_id=str(msg.header.frame_id),
            child_frame_id=str(msg.child_frame_id),
            x=float(position.x),
            y=float(position.y),
        )
        with self._state_lock:
            self._odom = sample

    def _workspace_path_cb(self, msg):
        receipt_sec = self._now_sec()
        points_xy = tuple(
            (
                float(pose.pose.position.x),
                float(pose.pose.position.y),
            )
            for pose in msg.poses
        )
        if len(points_xy) < 2 or not _finite(
            value for point in points_xy for value in point
        ):
            self.get_logger().warn(
                'Ignoring a short or non-finite /workspace_path',
                throttle_duration_sec=2.0,
            )
            return

        frame_id = normalize_frame(msg.header.frame_id)
        with self._state_lock:
            previous = self._workspace_path
            if (
                previous is not None
                and previous.frame_id == frame_id
                and previous.points_xy == points_xy
            ):
                return
            self._workspace_path_sequence += 1
            self._workspace_path = TimedWorkspacePathSample(
                sequence=self._workspace_path_sequence,
                stamp_sec=_message_time_sec(msg.header.stamp, receipt_sec),
                frame_id=frame_id,
                points_xy=points_xy,
            )

    def _cbf_enabled_cb(self, msg):
        with self._state_lock:
            self._cbf_enabled = bool(msg.data)

    def _external_enabled_cb(self, msg):
        with self._state_lock:
            self._external_enabled = bool(msg.data)

    def _workspace_state_cb(self, msg):
        receipt_sec = self._now_sec()
        translation = msg.transform.translation
        rotation = msg.transform.rotation
        sample = TimedWorkspaceSample(
            stamp_sec=_message_time_sec(msg.header.stamp, receipt_sec),
            frame_id=str(msg.header.frame_id),
            child_frame_id=str(msg.child_frame_id),
            enabled=bool(msg.enabled),
            capture_pending=bool(msg.capture_pending),
            generation=int(msg.generation),
            translation=(
                float(translation.x),
                float(translation.y),
                float(translation.z),
            ),
            quaternion=(
                float(rotation.x),
                float(rotation.y),
                float(rotation.z),
                float(rotation.w),
            ),
        )

        with self._state_lock:
            previous = self._workspace
            activation_changed = (
                self._workspace_activation is None
                or sample.generation
                != self._workspace_activation.generation
                or (previous is not None and not previous.enabled)
            )
            if (
                sample.enabled
                and not sample.capture_pending
                and sample.generation > 0
                and activation_changed
                and _finite(sample.translation[:2])
            ):
                self._workspace_activation = WorkspaceActivation(
                    generation=sample.generation,
                    stamp_sec=sample.stamp_sec,
                    frame_id=sample.frame_id,
                    child_frame_id=sample.child_frame_id,
                    x=sample.translation[0],
                    y=sample.translation[1],
                    quaternion=sample.quaternion,
                )
            elif (
                previous is None
                and self._workspace_activation is None
                and not sample.enabled
                and sample.generation > 0
                and _finite(sample.translation[:2])
            ):
                # A transient-local disabled state can be the first state
                # observed when logging begins after a successful capture.
                # Keep its generation/center, but leave the activation time
                # unknown so the plotter starts at the first recorded odom.
                self._workspace_activation = WorkspaceActivation(
                    generation=sample.generation,
                    stamp_sec=float('nan'),
                    frame_id=sample.frame_id,
                    child_frame_id=sample.child_frame_id,
                    x=sample.translation[0],
                    y=sample.translation[1],
                    quaternion=sample.quaternion,
                )
            self._workspace = sample

    def _make_snapshot(self):
        ros_time_sec = self._now_sec()
        elapsed_sec = time.monotonic() - self._start_monotonic
        with self._state_lock:
            snapshot = LoggerSnapshot(
                sample_index=self._sample_index,
                ros_time_sec=ros_time_sec,
                elapsed_sec=elapsed_sec,
                joints=self._joints,
                human=self._pending_human,
                odom=self._odom,
                workspace=self._workspace,
                workspace_activation=self._workspace_activation,
                workspace_path=self._workspace_path,
                cbf_enabled=self._cbf_enabled,
                external_enabled=self._external_enabled,
            )
            self._sample_index += 1
        return snapshot

    def _sample_tick(self):
        dropped_before = self._writer.dropped_count
        accepted = self._writer.submit(self._make_snapshot())
        if not accepted:
            self.get_logger().error(
                'CSV writer is unavailable; this sample was not recorded',
                throttle_duration_sec=2.0,
            )
        elif self._writer.dropped_count != dropped_before:
            self.get_logger().warn(
                'CSV worker fell behind; dropped the oldest queued sample '
                f'(total dropped={self._writer.dropped_count})',
                throttle_duration_sec=2.0,
            )

    def _writer_error_cb(self, exc):
        self.get_logger().error(
            f'CSV worker error: {exc}',
            throttle_duration_sec=2.0,
        )

    def _refresh_human_world(self, pending):
        if pending is None:
            return
        if (
            self._human_world is not None
            and pending.sequence == self._human_world.sequence
        ):
            return

        msg = pending.message
        try:
            msg_world = transform_capsule_array(
                msg,
                self._world_frame,
                self._tf_pose_lookup.buffer,
            )
        except (TransformException, ValueError) as exc:
            self.get_logger().warn(
                'Could not transform the latest /human/colliders sample '
                f"from '{msg.header.frame_id}' to '{self._world_frame}': "
                f'{exc}; retaining the last valid sample',
                throttle_duration_sec=2.0,
            )
            return

        capsules = []
        for source in msg_world.capsules[:N_HUMAN_CAPSULES]:
            capsules.append(HumanCapsule(
                name=str(source.name),
                a=(
                    float(source.a.x),
                    float(source.a.y),
                    float(source.a.z),
                ),
                b=(
                    float(source.b.x),
                    float(source.b.y),
                    float(source.b.z),
                ),
                radius=float(source.radius),
            ))
        self._human_world = HumanWorldSample(
            sequence=pending.sequence,
            stamp_sec=_message_time_sec(
                msg_world.header.stamp,
                pending.receipt_sec,
            ),
            source_frame=str(msg.header.frame_id),
            frame_id=str(msg_world.header.frame_id),
            received_count=len(msg.capsules),
            capsules=tuple(capsules),
        )

    def _snapshot_to_row(self, snapshot):
        row = {field: '' for field in CSV_FIELDS}
        row.update({
            'sample_index': snapshot.sample_index,
            'ros_time_sec': snapshot.ros_time_sec,
            'elapsed_sec': snapshot.elapsed_sec,
            'cbf_enabled': int(snapshot.cbf_enabled),
            'external_enabled': int(snapshot.external_enabled),
            'workspace_radius_m': self._workspace_radius_m,
            'collision_geometry': self._collision_geometry,
            'internal_margin_phi_m': self._internal_margin_phi_m,
            'external_margin_phi_m': self._external_margin_phi_m,
            'external_torso_margin_phi_m':
                self._external_torso_margin_phi_m,
            'human_timeout_sec': self._human_timeout_sec,
            'tf_stale_timeout_sec': self._tf_stale_timeout_sec,
        })
        self._fill_joint_fields(row, snapshot)
        self._fill_human_fields(row, snapshot)
        self._fill_odom_fields(row, snapshot)
        self._fill_workspace_fields(row, snapshot)
        self._fill_workspace_path_fields(row, snapshot)
        self._fill_tf_fields(row, snapshot)
        return row

    def _fill_joint_fields(self, row, snapshot):
        timed = snapshot.joints
        row['joint_state_valid'] = int(timed is not None)
        for name in JOINT_NAMES:
            row[joint_value_field(name)] = JOINT_DEFAULTS[name]
            row[joint_observed_field(name)] = 0
        if timed is None:
            return

        age = _age_sec(snapshot.ros_time_sec, timed.stamp_sec)
        row['joint_state_stamp_sec'] = timed.stamp_sec
        row['joint_state_age_sec'] = age
        row['joint_state_observed_count'] = timed.sample.observed_count
        for name in JOINT_NAMES:
            row[joint_value_field(name)] = timed.sample.values[name]
            row[joint_observed_field(name)] = int(
                timed.sample.observed[name]
            )

    def _fill_human_fields(self, row, snapshot):
        self._refresh_human_world(snapshot.human)
        human = self._human_world
        row['human_valid'] = int(human is not None)
        if human is None:
            return

        age = _age_sec(snapshot.ros_time_sec, human.stamp_sec)
        fresh = (
            age >= 0.0
            and (
                self._human_timeout_sec <= 0.0
                or age <= self._human_timeout_sec
            )
        )
        row.update({
            'human_stamp_sec': human.stamp_sec,
            'human_age_sec': age,
            'human_source_frame': human.source_frame,
            'human_frame_id': human.frame_id,
            'human_received_count': human.received_count,
            'human_used_count': len(human.capsules),
        })
        for index, capsule in enumerate(human.capsules):
            fields = human_slot_fields(index)
            values = (
                1,
                int(fresh),
                capsule.name,
                *capsule.a,
                *capsule.b,
                capsule.radius,
            )
            row.update(zip(fields, values))

    def _fill_odom_fields(self, row, snapshot):
        odom = snapshot.odom
        valid = odom is not None and _finite((odom.x, odom.y))
        row['odom_valid'] = int(valid)
        if odom is None:
            return
        row.update({
            'odom_stamp_sec': odom.stamp_sec,
            'odom_age_sec': _age_sec(
                snapshot.ros_time_sec,
                odom.stamp_sec,
            ),
            'odom_frame_id': odom.frame_id,
            'odom_child_frame_id': odom.child_frame_id,
            'odom_x_m': odom.x,
            'odom_y_m': odom.y,
        })

    def _fill_workspace_fields(self, row, snapshot):
        workspace = snapshot.workspace
        valid = (
            workspace is not None
            and _finite(workspace.translation + workspace.quaternion)
        )
        row['workspace_valid'] = int(valid)
        if workspace is not None:
            row.update({
                'workspace_stamp_sec': workspace.stamp_sec,
                'workspace_age_sec': _age_sec(
                    snapshot.ros_time_sec,
                    workspace.stamp_sec,
                ),
                'workspace_frame_id': workspace.frame_id,
                'workspace_child_frame_id': workspace.child_frame_id,
                'workspace_enabled': int(workspace.enabled),
                'workspace_capture_pending':
                    int(workspace.capture_pending),
                'workspace_generation': workspace.generation,
                'workspace_center_x_m': workspace.translation[0],
                'workspace_center_y_m': workspace.translation[1],
                'workspace_center_z_m': workspace.translation[2],
                'workspace_qx': workspace.quaternion[0],
                'workspace_qy': workspace.quaternion[1],
                'workspace_qz': workspace.quaternion[2],
                'workspace_qw': workspace.quaternion[3],
            })

        activation = snapshot.workspace_activation
        row['workspace_activation_valid'] = int(activation is not None)
        if activation is not None:
            row.update({
                'workspace_activation_generation':
                    activation.generation,
                'workspace_activation_stamp_sec': activation.stamp_sec,
                'workspace_activation_frame_id': activation.frame_id,
                'workspace_activation_child_frame_id':
                    activation.child_frame_id,
                'workspace_activation_center_x_m': activation.x,
                'workspace_activation_center_y_m': activation.y,
                'workspace_activation_qx': activation.quaternion[0],
                'workspace_activation_qy': activation.quaternion[1],
                'workspace_activation_qz': activation.quaternion[2],
                'workspace_activation_qw': activation.quaternion[3],
            })

    def _fill_workspace_path_fields(self, row, snapshot):
        path = snapshot.workspace_path
        row['workspace_path_valid'] = int(path is not None)
        if path is None:
            return

        row.update({
            'workspace_path_stamp_sec': path.stamp_sec,
            'workspace_path_age_sec': _age_sec(
                snapshot.ros_time_sec,
                path.stamp_sec,
            ),
            'workspace_path_frame_id': path.frame_id,
            'workspace_path_sequence': path.sequence,
            'workspace_path_point_count': len(path.points_xy),
        })
        if path.sequence == self._last_logged_workspace_path_sequence:
            return

        row['workspace_path_xy_json'] = json.dumps(
            path.points_xy,
            allow_nan=False,
            separators=(',', ':'),
        )
        self._last_logged_workspace_path_sequence = path.sequence

    def _fill_tf_fields(self, row, snapshot):
        row['tf_world_frame'] = self._world_frame
        row['tf_pelvis_frame'] = self._pelvis_frame
        pose, reason = self._tf_pose_lookup.lookup()
        if pose is None:
            row['tf_valid'] = 0
            if reason:
                self.get_logger().warn(
                    f'TF lookup failed for {self._world_frame} -> '
                    f'{self._pelvis_frame}: {reason}',
                    throttle_duration_sec=2.0,
                )
            return

        values = tuple(pose.position) + tuple(pose.quat)
        stamp_sec = pose.stamp.nanoseconds * 1e-9
        age = (
            _age_sec(snapshot.ros_time_sec, stamp_sec)
            if pose.stamp.nanoseconds != 0
            else None
        )
        valid = _finite(values)
        if age is not None:
            valid = (
                valid
                and age >= 0.0
                and (
                    self._tf_stale_timeout_sec <= 0.0
                    or age <= self._tf_stale_timeout_sec
                )
            )
        row.update({
            'tf_valid': int(valid),
            'tf_stamp_sec': stamp_sec,
            'tf_age_sec': '' if age is None else age,
            'tf_world_pelvis_tx_m': pose.position[0],
            'tf_world_pelvis_ty_m': pose.position[1],
            'tf_world_pelvis_tz_m': pose.position[2],
            'tf_world_pelvis_qx': pose.quat[0],
            'tf_world_pelvis_qy': pose.quat[1],
            'tf_world_pelvis_qz': pose.quat[2],
            'tf_world_pelvis_qw': pose.quat[3],
        })

    def shutdown(self):
        """Stop sampling and synchronously drain the worker once."""
        if self._shutdown_started:
            return
        self._shutdown_started = True
        self._timer.cancel()
        self._writer.submit(self._make_snapshot())
        self._writer.close()
        if rclpy.ok():
            self.get_logger().info(
                f'CSV closed: rows={self._writer.written_count}, '
                f'dropped={self._writer.dropped_count}, '
                f'processing_errors={self._writer.processing_error_count}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = CbfCsvLoggerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
