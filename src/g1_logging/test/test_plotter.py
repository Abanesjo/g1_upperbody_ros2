import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle
import numpy as np
import pytest

from g1_logging.csv_schema import (
    CSV_FIELDS,
    JOINT_DEFAULTS,
    JOINT_NAMES,
    joint_value_field,
)
from g1_logging.plotter import (
    CsvTable,
    create_collision_figure,
    create_odom_workspace_figure,
    generate_plots,
    reconstruct_collision_series,
    select_latest_workspace_trajectory,
)
from g1_logging.plotter_node import discover_csv_files


def _table(**columns):
    lengths = {len(values) for values in columns.values()}
    assert len(lengths) == 1
    sample_count = lengths.pop()
    rows = tuple(
        {
            field: '' if values[index] is None else str(values[index])
            for field, values in columns.items()
        }
        for index in range(sample_count)
    )
    return CsvTable(
        path=Path('/tmp/synthetic.csv'),
        fieldnames=tuple(columns),
        rows=rows,
    )


def test_discover_csv_files_finds_all_csvs_in_filename_order(tmp_path):
    (tmp_path / 'z_run.csv').write_text('header\n', encoding='utf-8')
    (tmp_path / 'a_run.CSV').write_text('header\n', encoding='utf-8')
    (tmp_path / 'notes.txt').write_text('ignore\n', encoding='utf-8')
    (tmp_path / 'nested.csv').mkdir()

    assert discover_csv_files(tmp_path) == (
        tmp_path / 'a_run.CSV',
        tmp_path / 'z_run.csv',
    )


def test_collision_figure_adds_correct_thick_black_row_minimum():
    distances = np.array([
        [0.5, 0.3, np.nan],
        [0.4, 0.2, np.nan],
        [np.nan, 0.1, np.nan],
        [np.nan, np.nan, np.nan],
    ])
    figure = create_collision_figure(
        np.arange(4.0),
        distances,
        ('pair a', 'pair b', 'never observed'),
        'Collision distances',
    )
    try:
        axis = figure.axes[0]
        assert len(axis.lines) == 4
        minimum = next(
            line
            for line in axis.lines
            if line.get_label() == 'Minimum clearance'
        )
        assert minimum.get_label() == 'Minimum clearance'
        assert minimum.get_color() == 'black'
        assert minimum.get_linewidth() > max(
            line.get_linewidth()
            for line in axis.lines
            if line.get_label() not in {
                'Minimum clearance',
                '_nolegend_',
            }
        )
        np.testing.assert_allclose(
            minimum.get_ydata(),
            [0.3, 0.2, 0.1, np.nan],
            equal_nan=True,
        )
        zero_line = next(
            line
            for line in axis.lines
            if line.get_label() == '_nolegend_'
        )
        np.testing.assert_allclose(zero_line.get_ydata(), [0.0, 0.0])
        assert zero_line.get_color() == 'red'
        assert zero_line.get_linestyle() == '--'
        assert [
            text.get_text() for text in axis.get_legend().get_texts()
        ] == ['Robot link pair', 'Minimum clearance']
        assert axis.title.get_fontsize() == 48.0
        assert axis.xaxis.label.get_fontsize() == 40.0
        assert axis.yaxis.label.get_fontsize() == 40.0
        assert {
            text.get_fontsize()
            for text in axis.get_legend().get_texts()
        } == {24.0}
        np.testing.assert_allclose(
            axis.xaxis.label.get_position(),
            (0.5, -0.21),
        )
    finally:
        plt.close(figure)


def test_external_collision_figure_uses_human_robot_legend_label():
    figure = create_collision_figure(
        np.arange(2.0),
        np.asarray([[0.5], [0.4]]),
        ('human-robot pair',),
        'External collision capsule clearances',
        pair_legend_label='Human-Robot link pair',
    )
    try:
        figure.canvas.draw()
        legend = figure.axes[0].get_legend()
        assert [
            text.get_text()
            for text in legend.get_texts()
        ] == ['Human-Robot link pair', 'Minimum clearance']
        renderer = figure.canvas.get_renderer()
        for text, line in zip(legend.get_texts(), legend.get_lines()):
            text_bounds = text.get_window_extent(renderer)
            text_center_y = 0.5 * (text_bounds.y0 + text_bounds.y1)
            line_points = line.get_transform().transform(line.get_xydata())
            assert abs(line_points[0, 1] - text_center_y) < 3.0
    finally:
        plt.close(figure)


def test_collision_figure_is_blank_when_no_human_distance_exists():
    figure = create_collision_figure(
        np.arange(3.0),
        np.full((3, 77), np.nan),
        tuple(f'pair {index}' for index in range(77)),
        'External collision capsule clearances',
    )
    try:
        axis = figure.axes[0]
        assert axis.get_title() == 'External collision capsule clearances'
        assert len(axis.lines) == 1
        np.testing.assert_allclose(axis.lines[0].get_ydata(), [0.0, 0.0])
        assert axis.lines[0].get_color() == 'red'
        assert axis.lines[0].get_linestyle() == '--'
        assert axis.get_legend() is None
    finally:
        plt.close(figure)


def test_latest_workspace_activation_slices_and_rebases_odom():
    table = _table(
        ros_time_sec=[100.0, 101.0, 102.0, 103.0, 104.0],
        elapsed_sec=[0.0, 1.0, 2.0, 3.0, 4.0],
        odom_valid=[1, 1, 1, 1, 1],
        odom_x_m=[5.0, 6.0, 7.0, 8.0, 9.0],
        odom_y_m=[1.0, 1.5, 2.0, 4.0, 7.0],
        workspace_activation_valid=[1, 1, 1, 1, 1],
        workspace_activation_generation=[1, 1, 2, 2, 2],
        workspace_activation_stamp_sec=[
            100.0, 100.0, 102.5, 102.5, 102.5,
        ],
        workspace_activation_center_x_m=[10, 10, 20, 20, 20],
        workspace_activation_center_y_m=[15, 15, 30, 30, 30],
        workspace_radius_m=[1.5, 1.5, 1.75, 1.75, 1.75],
    )

    trajectory = select_latest_workspace_trajectory(table)

    assert trajectory is not None
    assert trajectory.generation == 2
    assert trajectory.source_start_index == 3
    np.testing.assert_allclose(trajectory.time_sec, [0.0, 1.0])
    np.testing.assert_allclose(trajectory.x_m, [0.0, 1.0])
    np.testing.assert_allclose(trajectory.y_m, [0.0, 3.0])
    assert trajectory.center_x_m == 12.0
    assert trajectory.center_y_m == 26.0
    assert trajectory.radius_m == 1.75
    assert trajectory.path_x_m.size == 0
    assert trajectory.path_y_m.size == 0


def test_workspace_cutoff_rejects_repeated_pre_activation_odom_stamp():
    table = _table(
        ros_time_sec=[100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        elapsed_sec=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        odom_valid=[1, 1, 1, 1, 1, 1],
        # The timer logged two new rows while /odom still held an old sample.
        odom_stamp_sec=[100.0, 101.0, 101.0, 101.0, 103.0, 104.0],
        odom_x_m=[0.0, 1.0, 50.0, 60.0, 4.0, 5.0],
        odom_y_m=[0.0, 1.0, 50.0, 60.0, 8.0, 9.0],
        workspace_activation_valid=[1, 1, 1, 1, 1, 1],
        workspace_activation_generation=[1, 1, 2, 2, 2, 2],
        workspace_activation_stamp_sec=[
            100.0, 100.0, 102.5, 102.5, 102.5, 102.5,
        ],
        workspace_activation_center_x_m=[0, 0, 4.5, 4.5, 4.5, 4.5],
        workspace_activation_center_y_m=[0, 0, 8.5, 8.5, 8.5, 8.5],
        workspace_radius_m=[1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
    )

    trajectory = select_latest_workspace_trajectory(table)

    assert trajectory is not None
    assert trajectory.source_start_index == 4
    np.testing.assert_allclose(trajectory.x_m, [0.0, 1.0])
    np.testing.assert_allclose(trajectory.y_m, [0.0, 1.0])
    assert trajectory.center_x_m == 0.5
    assert trajectory.center_y_m == 0.5


def test_odom_figure_draws_configured_circle_and_rebased_trajectory():
    table = _table(
        ros_time_sec=[10.0, 11.0],
        odom_valid=[1, 1],
        odom_x_m=[2.0, 2.5],
        odom_y_m=[3.0, 4.0],
        workspace_activation_valid=[1, 1],
        workspace_activation_generation=[7, 7],
        workspace_activation_stamp_sec=[10.0, 10.0],
        workspace_activation_center_x_m=[2.25, 2.25],
        workspace_activation_center_y_m=[3.5, 3.5],
        workspace_radius_m=[1.25, 1.25],
    )
    trajectory = select_latest_workspace_trajectory(table)

    figure = create_odom_workspace_figure(trajectory)
    try:
        axis = figure.axes[0]
        circles = [
            patch for patch in axis.patches if isinstance(patch, Circle)
        ]
        assert len(circles) == 1
        assert circles[0].center == (0.25, 0.5)
        assert circles[0].radius == 1.25
        assert circles[0].get_label() == 'Workspace bounds'
        assert circles[0].get_edgecolor() == to_rgba('tab:blue')
        assert circles[0].get_linestyle() == '-'
        assert axis.get_aspect() == 1.0
        assert axis.get_title() == 'Planar CBF Test'
        assert axis.get_xlabel() == 'x(m)'
        assert axis.get_ylabel() == 'Y (m)'

        odom_line = next(
            line
            for line in axis.lines
            if line.get_label() == 'Robot position'
        )
        np.testing.assert_allclose(odom_line.get_xdata(), [0.0, 0.5])
        np.testing.assert_allclose(odom_line.get_ydata(), [0.0, 1.0])
        assert odom_line.get_color() == 'black'
        assert odom_line.get_linestyle() == '-'

        legend = axis.get_legend()
        assert [text.get_text() for text in legend.get_texts()] == [
            'Workspace bounds',
            'Robot position',
        ]
        workspace_handle = legend.get_lines()[0]
        assert workspace_handle.get_color() == 'tab:blue'
        assert workspace_handle.get_linestyle() == '-'
    finally:
        plt.close(figure)


def test_workspace_path_is_rotated_rebased_and_drawn():
    quarter_turn = 0.5 * math.pi
    path_json = json.dumps([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 2.0],
    ])
    table = _table(
        ros_time_sec=[10.0, 11.0, 12.0],
        odom_valid=[1, 1, 1],
        odom_stamp_sec=[10.0, 11.0, 12.0],
        odom_x_m=[10.0, 10.5, 11.0],
        odom_y_m=[20.0, 20.5, 21.0],
        workspace_activation_valid=[1, 1, 1],
        workspace_activation_generation=[4, 4, 4],
        workspace_activation_stamp_sec=[10.0, 10.0, 10.0],
        workspace_activation_frame_id=['world', 'world', 'world'],
        workspace_activation_child_frame_id=[
            'workspace',
            'workspace',
            'workspace',
        ],
        workspace_activation_center_x_m=[11.0, 11.0, 11.0],
        workspace_activation_center_y_m=[22.0, 22.0, 22.0],
        workspace_activation_qx=[0.0, 0.0, 0.0],
        workspace_activation_qy=[0.0, 0.0, 0.0],
        workspace_activation_qz=[
            math.sin(0.5 * quarter_turn),
            math.sin(0.5 * quarter_turn),
            math.sin(0.5 * quarter_turn),
        ],
        workspace_activation_qw=[
            math.cos(0.5 * quarter_turn),
            math.cos(0.5 * quarter_turn),
            math.cos(0.5 * quarter_turn),
        ],
        workspace_generation=[4, 4, 4],
        workspace_radius_m=[1.5, 1.5, 1.5],
        workspace_path_valid=[1, 1, 1],
        workspace_path_frame_id=['workspace', 'workspace', 'workspace'],
        workspace_path_point_count=[3, 3, 3],
        workspace_path_xy_json=['', path_json, ''],
    )

    trajectory = select_latest_workspace_trajectory(table)

    assert trajectory is not None
    np.testing.assert_allclose(trajectory.path_x_m, [2.0, 3.0, 3.0])
    np.testing.assert_allclose(trajectory.path_y_m, [-1.0, -1.0, 1.0])
    np.testing.assert_allclose(trajectory.x_m, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(trajectory.y_m, [0.0, -0.5, -1.0])
    assert trajectory.center_x_m == pytest.approx(2.0)
    assert trajectory.center_y_m == pytest.approx(-1.0)

    figure = create_odom_workspace_figure(trajectory)
    try:
        path_line = next(
            line
            for line in figure.axes[0].lines
            if line.get_label() == 'Sample nominal path'
        )
        np.testing.assert_allclose(path_line.get_xdata(), [2.0, 3.0, 3.0])
        np.testing.assert_allclose(path_line.get_ydata(), [-1.0, -1.0, 1.0])
        assert path_line.get_color() == 'tab:green'
        assert path_line.get_linestyle() == '--'
    finally:
        plt.close(figure)


def test_workspace_path_with_wrong_frame_is_omitted():
    table = _table(
        ros_time_sec=[10.0],
        odom_valid=[1],
        odom_x_m=[0.0],
        odom_y_m=[0.0],
        workspace_activation_valid=[1],
        workspace_activation_generation=[1],
        workspace_activation_stamp_sec=[10.0],
        workspace_activation_frame_id=['world'],
        workspace_activation_child_frame_id=['workspace'],
        workspace_activation_center_x_m=[0.0],
        workspace_activation_center_y_m=[0.0],
        workspace_activation_qx=[0.0],
        workspace_activation_qy=[0.0],
        workspace_activation_qz=[0.0],
        workspace_activation_qw=[1.0],
        workspace_radius_m=[1.5],
        workspace_path_valid=[1],
        workspace_path_frame_id=['unrelated'],
        workspace_path_point_count=[2],
        workspace_path_xy_json=[json.dumps([[0.0, 0.0], [1.0, 0.0]])],
    )

    trajectory = select_latest_workspace_trajectory(table)

    assert trajectory is not None
    assert trajectory.path_x_m.size == 0
    assert trajectory.path_y_m.size == 0


def _minimal_csv_row():
    row = {field: '' for field in CSV_FIELDS}
    row.update({
        'sample_index': 0,
        'ros_time_sec': 10.0,
        'elapsed_sec': 0.0,
        'joint_state_valid': 1,
        'human_valid': 0,
        'tf_valid': 0,
        'odom_valid': 0,
        'workspace_activation_valid': 0,
        'workspace_radius_m': 1.5,
    })
    for joint_name in JOINT_NAMES:
        row[joint_value_field(joint_name)] = JOINT_DEFAULTS[joint_name]
    return row


def test_generate_plots_creates_csv_named_folder_and_three_pngs(tmp_path):
    csv_path = tmp_path / 'analysis_run.csv'
    with csv_path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerow(_minimal_csv_row())

    outputs = generate_plots(csv_path, output_root=tmp_path / 'plots')

    assert outputs.output_dir == tmp_path / 'plots' / 'analysis_run'
    generated = (
        outputs.internal_png,
        outputs.external_png,
        outputs.odom_png,
    )
    assert all(path.parent == outputs.output_dir for path in generated)
    assert all(
        path.is_file() and path.stat().st_size > 0
        for path in generated
    )
    assert not tuple(outputs.output_dir.glob('*.pdf'))

    collisions = reconstruct_collision_series(
        CsvTable(
            csv_path,
            CSV_FIELDS,
            (dict(_minimal_csv_row()),),
        )
    )
    assert collisions.internal_m.shape == (1, 21)
    assert collisions.external_m.shape == (1, 77)
    assert np.isfinite(collisions.internal_m).all()
    assert np.isnan(collisions.external_m).all()
