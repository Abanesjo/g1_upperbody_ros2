import csv
import json
from types import SimpleNamespace
import threading

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import numpy as np

from g1_cbf.jax_kinematics import (
    CONTROLLED_JOINTS as CBF_CONTROLLED_JOINTS,
    CONTROLLED_JOINT_DEFAULTS as CBF_CONTROLLED_DEFAULTS,
    LEG_JOINTS as CBF_LEG_JOINTS,
)
from g1_logging import paths
from g1_logging.async_csv import AsyncCsvWriter
from g1_logging.csv_schema import (
    CSV_FIELDS,
    CONTROLLED_JOINT_DEFAULTS,
    CONTROLLED_JOINT_NAMES,
    JOINT_NAMES,
    LEG_JOINT_NAMES,
    N_HUMAN_CAPSULES,
    extract_joint_state,
    human_slot_fields,
    joint_observed_field,
    joint_value_field,
    stamp_to_sec,
)
from g1_logging.csv_logger import (
    CbfCsvLoggerNode,
    TimedWorkspacePathSample,
)


def test_schema_is_unique_and_covers_every_fixed_slot():
    assert isinstance(CSV_FIELDS, tuple)
    assert len(CSV_FIELDS) == len(set(CSV_FIELDS))
    assert N_HUMAN_CAPSULES == 11

    for name in JOINT_NAMES:
        assert joint_value_field(name) in CSV_FIELDS
        assert joint_observed_field(name) in CSV_FIELDS

    all_human_fields = {
        field
        for index in range(N_HUMAN_CAPSULES)
        for field in human_slot_fields(index)
    }
    assert len(all_human_fields) == N_HUMAN_CAPSULES * 10
    assert all_human_fields.issubset(CSV_FIELDS)

    required_reconstruction_fields = {
        'odom_x_m',
        'odom_y_m',
        'workspace_activation_generation',
        'workspace_activation_stamp_sec',
        'workspace_activation_frame_id',
        'workspace_activation_child_frame_id',
        'workspace_activation_center_x_m',
        'workspace_activation_center_y_m',
        'workspace_activation_qx',
        'workspace_activation_qy',
        'workspace_activation_qz',
        'workspace_activation_qw',
        'workspace_path_valid',
        'workspace_path_stamp_sec',
        'workspace_path_age_sec',
        'workspace_path_frame_id',
        'workspace_path_sequence',
        'workspace_path_point_count',
        'workspace_path_xy_json',
        'workspace_radius_m',
        'tf_world_pelvis_tx_m',
        'tf_world_pelvis_ty_m',
        'tf_world_pelvis_tz_m',
        'tf_world_pelvis_qx',
        'tf_world_pelvis_qy',
        'tf_world_pelvis_qz',
        'tf_world_pelvis_qw',
    }
    assert required_reconstruction_fields.issubset(CSV_FIELDS)


def test_joint_schema_cannot_drift_from_cbf_runtime():
    assert CONTROLLED_JOINT_NAMES == tuple(CBF_CONTROLLED_JOINTS)
    assert LEG_JOINT_NAMES == tuple(CBF_LEG_JOINTS)
    np.testing.assert_array_equal(
        np.asarray(CONTROLLED_JOINT_DEFAULTS),
        np.asarray(CBF_CONTROLLED_DEFAULTS),
    )


def test_extract_joint_state_uses_cbf_defaults_and_observation_flags():
    sample = extract_joint_state(
        [
            'unrelated_joint',
            'left_elbow_joint',
            'right_hip_yaw_joint',
        ],
        [99.0, 1.25, -0.75],
    )

    assert sample.observed_count == 2
    assert sample.values['left_elbow_joint'] == 1.25
    assert sample.values['right_hip_yaw_joint'] == -0.75
    assert sample.observed['left_elbow_joint']
    assert sample.observed['right_hip_yaw_joint']

    assert sample.values['left_shoulder_pitch_joint'] == 0.35
    assert sample.values['left_hip_pitch_joint'] == 0.0
    assert not sample.observed['left_shoulder_pitch_joint']
    assert not sample.observed['left_hip_pitch_joint']


def test_stamp_to_sec_preserves_nanosecond_fraction():
    stamp = SimpleNamespace(sec=12, nanosec=345_678_901)

    assert stamp_to_sec(stamp) == 12.345678901


def test_output_directories_are_created_for_a_fresh_source_root(
    monkeypatch,
    tmp_path,
):
    source_root = tmp_path / 'g1_logging'
    source_root.mkdir()
    (source_root / 'package.xml').write_text(
        '<package><name>g1_logging</name></package>',
        encoding='utf-8',
    )
    (source_root / 'g1_logging').mkdir()
    monkeypatch.setenv(paths.SOURCE_ROOT_ENV, str(source_root))

    assert paths.package_source_root() == source_root.resolve()
    assert paths.data_dir() == source_root / 'data'
    assert paths.plot_dir() == source_root / 'plot'
    assert (source_root / 'data').is_dir()
    assert (source_root / 'plot').is_dir()
    assert paths.resolve_csv_output('recording') == (
        source_root / 'data' / 'recording.csv'
    )


def _workspace_path(*points):
    message = Path()
    message.header.frame_id = '/workspace'
    for x, y in points:
        pose = PoseStamped()
        pose.pose.position.x = x
        pose.pose.position.y = y
        message.poses.append(pose)
    return message


def test_workspace_path_callback_ignores_identical_geometry():
    node = SimpleNamespace(
        _state_lock=threading.Lock(),
        _workspace_path=None,
        _workspace_path_sequence=0,
        _now_sec=lambda: 12.0,
        get_logger=lambda: SimpleNamespace(warn=lambda *args, **kwargs: None),
    )
    first = _workspace_path((0.0, 0.0), (1.0, 0.0))

    CbfCsvLoggerNode._workspace_path_cb(node, first)
    assert node._workspace_path_sequence == 1
    assert node._workspace_path.frame_id == 'workspace'
    assert node._workspace_path.points_xy == ((0.0, 0.0), (1.0, 0.0))

    CbfCsvLoggerNode._workspace_path_cb(node, first)
    assert node._workspace_path_sequence == 1

    changed = _workspace_path((0.0, 0.0), (0.0, 1.0))
    CbfCsvLoggerNode._workspace_path_cb(node, changed)
    assert node._workspace_path_sequence == 2
    assert node._workspace_path.points_xy == ((0.0, 0.0), (0.0, 1.0))


def test_workspace_path_json_is_written_once_per_changed_sequence():
    node = SimpleNamespace(_last_logged_workspace_path_sequence=None)
    first_path = TimedWorkspacePathSample(
        sequence=1,
        stamp_sec=10.0,
        frame_id='workspace',
        points_xy=((0.0, 0.0), (1.0, 0.0)),
    )

    first_row = {field: '' for field in CSV_FIELDS}
    CbfCsvLoggerNode._fill_workspace_path_fields(
        node,
        first_row,
        SimpleNamespace(ros_time_sec=10.1, workspace_path=first_path),
    )
    assert first_row['workspace_path_valid'] == 1
    assert first_row['workspace_path_point_count'] == 2
    assert json.loads(first_row['workspace_path_xy_json']) == [
        [0.0, 0.0],
        [1.0, 0.0],
    ]

    repeated_row = {field: '' for field in CSV_FIELDS}
    CbfCsvLoggerNode._fill_workspace_path_fields(
        node,
        repeated_row,
        SimpleNamespace(ros_time_sec=10.2, workspace_path=first_path),
    )
    assert repeated_row['workspace_path_valid'] == 1
    assert repeated_row['workspace_path_xy_json'] == ''

    changed_path = TimedWorkspacePathSample(
        sequence=2,
        stamp_sec=11.0,
        frame_id='workspace',
        points_xy=((0.0, 0.0), (0.0, 1.0)),
    )
    changed_row = {field: '' for field in CSV_FIELDS}
    CbfCsvLoggerNode._fill_workspace_path_fields(
        node,
        changed_row,
        SimpleNamespace(ros_time_sec=11.1, workspace_path=changed_path),
    )
    assert json.loads(changed_row['workspace_path_xy_json']) == [
        [0.0, 0.0],
        [0.0, 1.0],
    ]


def test_async_writer_close_drains_every_accepted_row(tmp_path):
    output = tmp_path / 'nested' / 'samples.csv'
    writer = AsyncCsvWriter(
        output,
        ('sample_index', 'value'),
        max_queue_size=32,
    )

    writer.start()
    for index in range(10):
        assert writer.submit({
            'sample_index': index,
            'value': index * 2,
        })
    writer.close()

    with output.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))

    assert [int(row['sample_index']) for row in rows] == list(range(10))
    assert [int(row['value']) for row in rows] == [
        index * 2 for index in range(10)
    ]
    assert writer.written_count == 10
    assert writer.dropped_count == 0
    assert writer.error is None
    assert not writer.submit({'sample_index': 10, 'value': 20})


def test_async_writer_drops_oldest_without_blocking_producer(tmp_path):
    entered_builder = threading.Event()
    release_builder = threading.Event()

    def blocking_builder(item):
        if item['sample_index'] == 0:
            entered_builder.set()
            assert release_builder.wait(timeout=3.0)
        return item

    output = tmp_path / 'bounded.csv'
    writer = AsyncCsvWriter(
        output,
        ('sample_index',),
        row_builder=blocking_builder,
        max_queue_size=2,
    )
    writer.start()
    assert writer.submit({'sample_index': 0})
    assert entered_builder.wait(timeout=3.0)

    assert writer.submit({'sample_index': 1})
    assert writer.submit({'sample_index': 2})
    assert writer.submit({'sample_index': 3})
    assert writer.dropped_count == 1

    release_builder.set()
    writer.close()

    with output.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))

    assert [int(row['sample_index']) for row in rows] == [0, 2, 3]
    assert writer.written_count == 3


def test_async_writer_reports_bad_samples_and_continues(tmp_path):
    reported = []

    def build_row(item):
        if item == 'bad':
            raise ValueError('invalid sample')
        return {'sample_index': item}

    output = tmp_path / 'recover.csv'
    writer = AsyncCsvWriter(
        output,
        ('sample_index',),
        row_builder=build_row,
        max_queue_size=8,
        error_callback=reported.append,
    )
    writer.start()
    assert writer.submit(1)
    assert writer.submit('bad')
    assert writer.submit(2)
    writer.close()

    with output.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))

    assert [int(row['sample_index']) for row in rows] == [1, 2]
    assert writer.processing_error_count == 1
    assert len(reported) == 1
    assert isinstance(reported[0], ValueError)
    assert writer.error is None
