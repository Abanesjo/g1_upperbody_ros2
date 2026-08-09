"""Load a G1 state CSV, reconstruct capsule distances, and create plots."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.text import Text
import numpy as np

from g1_logging.csv_schema import (
    CONTROLLED_JOINT_NAMES,
    JOINT_DEFAULTS,
    LEG_JOINT_NAMES,
    N_HUMAN_CAPSULES,
    human_slot_fields,
    joint_value_field,
)
from g1_logging.geometry import (
    EXTERNAL_PAIR_LABELS,
    EXTERNAL_ROBOT_BODY_NAMES,
    INTERNAL_PAIR_LABELS,
    compute_external_clearances,
    compute_internal_clearances,
)
from g1_logging.paths import plot_dir


PLOT_FONT_SCALE = 2.0
COLLISION_HEADING_FONT_SCALE = 2.0
PNG_DPI = 50


@dataclass(frozen=True)
class CsvTable:
    """A small typed facade over rows loaded by :class:`csv.DictReader`."""

    path: Path
    fieldnames: Tuple[str, ...]
    rows: Tuple[Mapping[str, str], ...]

    def __len__(self) -> int:
        return len(self.rows)

    def has(self, field: str) -> bool:
        return field in self.fieldnames

    def float_column(
        self,
        field: str,
        default: float = np.nan,
    ) -> np.ndarray:
        return np.asarray(
            [_parse_float(row.get(field), default) for row in self.rows],
            dtype=np.float64,
        )

    def bool_column(
        self,
        field: str,
        default: bool = False,
    ) -> np.ndarray:
        return np.asarray(
            [_parse_bool(row.get(field), default) for row in self.rows],
            dtype=bool,
        )

    def string_column(self, field: str) -> np.ndarray:
        return np.asarray(
            [str(row.get(field) or '').strip() for row in self.rows],
            dtype=object,
        )


@dataclass(frozen=True)
class CollisionSeries:
    """Reconstructed signed surface clearances for every logged sample."""

    time_sec: np.ndarray
    internal_m: np.ndarray
    external_m: np.ndarray
    internal_labels: Tuple[str, ...]
    external_labels: Tuple[str, ...]


@dataclass(frozen=True)
class WorkspaceTrajectory:
    """Rebased odometry and workspace geometry for the latest activation."""

    generation: int
    source_start_index: int
    time_sec: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    center_x_m: float
    center_y_m: float
    radius_m: float
    path_x_m: np.ndarray
    path_y_m: np.ndarray


@dataclass(frozen=True)
class PlotOutputs:
    """Paths produced for one CSV analysis run."""

    output_dir: Path
    internal_png: Path
    external_png: Path
    odom_png: Path


def load_csv(path) -> CsvTable:
    """Load a logger CSV without requiring pandas."""

    csv_path = Path(path).expanduser().resolve()
    with csv_path.open('r', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f'CSV has no header: {csv_path}')
        fieldnames = tuple(str(field).strip() for field in reader.fieldnames)
        if not all(fieldnames):
            raise ValueError(f'CSV contains an empty header field: {csv_path}')
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError(
                f'CSV contains duplicate header fields: {csv_path}'
            )
        rows = tuple(
            {
                str(key).strip(): ('' if value is None else str(value).strip())
                for key, value in row.items()
                if key is not None
            }
            for row in reader
        )
    return CsvTable(csv_path, fieldnames, rows)


def reconstruct_collision_series(table: CsvTable) -> CollisionSeries:
    """Reconstruct all configured internal and external capsule distances."""

    count = len(table)
    time_sec = _elapsed_time(table)
    internal = np.full(
        (count, len(INTERNAL_PAIR_LABELS)),
        np.nan,
        dtype=np.float64,
    )
    external = np.full(
        (count, len(EXTERNAL_PAIR_LABELS)),
        np.nan,
        dtype=np.float64,
    )
    if count == 0:
        return CollisionSeries(
            time_sec,
            internal,
            external,
            INTERNAL_PAIR_LABELS,
            EXTERNAL_PAIR_LABELS,
        )

    controlled = _joint_matrix(table, CONTROLLED_JOINT_NAMES)
    legs = _joint_matrix(table, LEG_JOINT_NAMES)
    joint_values_finite = (
        np.all(np.isfinite(controlled), axis=1)
        & np.all(np.isfinite(legs), axis=1)
    )
    if table.has('joint_state_valid'):
        joint_valid = table.bool_column('joint_state_valid')
    else:
        joint_valid = joint_values_finite.copy()
    joint_valid &= joint_values_finite

    tf_translation = np.column_stack([
        table.float_column('tf_world_pelvis_tx_m'),
        table.float_column('tf_world_pelvis_ty_m'),
        table.float_column('tf_world_pelvis_tz_m'),
    ])
    tf_quaternion = np.column_stack([
        table.float_column('tf_world_pelvis_qx'),
        table.float_column('tf_world_pelvis_qy'),
        table.float_column('tf_world_pelvis_qz'),
        table.float_column('tf_world_pelvis_qw'),
    ])
    tf_values_finite = (
        np.all(np.isfinite(tf_translation), axis=1)
        & np.all(np.isfinite(tf_quaternion), axis=1)
        & (np.linalg.norm(tf_quaternion, axis=1) > 1.0e-9)
    )
    if table.has('tf_valid'):
        tf_valid = table.bool_column('tf_valid')
    else:
        tf_valid = tf_values_finite.copy()
    tf_valid &= tf_values_finite

    human_a, human_b, human_radii, human_present = _human_arrays(table)
    if table.has('human_valid'):
        human_global_valid = table.bool_column('human_valid')
        human_present &= human_global_valid[:, None]

    for row_index in np.flatnonzero(joint_valid):
        try:
            internal[row_index] = compute_internal_clearances(
                controlled[row_index],
                legs[row_index],
            )
        except ValueError:
            continue

        if not tf_valid[row_index] or not np.any(human_present[row_index]):
            continue
        try:
            external[row_index] = compute_external_clearances(
                controlled[row_index],
                legs[row_index],
                tf_translation[row_index],
                tf_quaternion[row_index],
                human_a[row_index],
                human_b[row_index],
                human_radii[row_index],
                human_present[row_index],
            ).reshape(-1)
        except ValueError:
            continue

    return CollisionSeries(
        time_sec=time_sec,
        internal_m=internal,
        external_m=external,
        internal_labels=INTERNAL_PAIR_LABELS,
        external_labels=_external_labels(table),
    )


def select_latest_workspace_trajectory(
    table: CsvTable,
) -> Optional[WorkspaceTrajectory]:
    """Select and rebase odometry after the most recent workspace capture."""

    count = len(table)
    if count == 0:
        return None

    activation_valid = table.bool_column('workspace_activation_valid')
    activation_generation = table.float_column(
        'workspace_activation_generation'
    )
    candidates = activation_valid & np.isfinite(activation_generation)

    # Backward-compatible fallback for a CSV that has workspace state but no
    # retained activation fields.
    using_retained_activation = np.any(candidates)
    if not using_retained_activation:
        workspace_valid = table.bool_column(
            'workspace_valid',
            default=True,
        )
        workspace_enabled = table.bool_column('workspace_enabled')
        activation_generation = table.float_column('workspace_generation')
        candidates = (
            workspace_valid
            & workspace_enabled
            & np.isfinite(activation_generation)
        )
    if not np.any(candidates):
        return None

    latest_generation_value = float(np.max(activation_generation[candidates]))
    chosen_generation = (
        candidates
        & np.isclose(
            activation_generation,
            latest_generation_value,
            rtol=0.0,
            atol=1.0e-9,
        )
    )
    chosen_indices = np.flatnonzero(chosen_generation)
    if chosen_indices.size == 0:
        return None

    start_index = int(chosen_indices[0])
    activation_stamp = np.nan
    cutoff_time = None
    if using_retained_activation:
        activation_stamp = _last_finite(
            table.float_column('workspace_activation_stamp_sec')[
                chosen_generation
            ]
        )
        if np.isfinite(activation_stamp):
            odom_stamp = table.float_column('odom_stamp_sec')
            if table.has('odom_stamp_sec') and np.any(
                np.isfinite(odom_stamp)
            ):
                cutoff_time = odom_stamp
            else:
                cutoff_time = table.float_column('ros_time_sec')
            at_or_after = np.flatnonzero(
                np.isfinite(cutoff_time)
                & (cutoff_time >= activation_stamp - 1.0e-9)
            )
            if at_or_after.size:
                start_index = int(at_or_after[0])
            elif np.any(np.isfinite(cutoff_time)):
                return None

    odom_x = table.float_column('odom_x_m')
    odom_y = table.float_column('odom_y_m')
    odom_valid = np.isfinite(odom_x) & np.isfinite(odom_y)
    if table.has('odom_valid'):
        odom_valid &= table.bool_column('odom_valid')
    odom_valid[:start_index] = False
    if cutoff_time is not None and np.isfinite(activation_stamp):
        odom_valid &= (
            np.isfinite(cutoff_time)
            & (cutoff_time >= activation_stamp - 1.0e-9)
        )
    selected_odom = np.flatnonzero(odom_valid)
    if selected_odom.size == 0:
        return None

    first_odom_index = int(selected_odom[0])
    origin_x = float(odom_x[first_odom_index])
    origin_y = float(odom_y[first_odom_index])

    if using_retained_activation:
        center_x = _last_finite(
            table.float_column('workspace_activation_center_x_m')[
                chosen_generation
            ]
        )
        center_y = _last_finite(
            table.float_column('workspace_activation_center_y_m')[
                chosen_generation
            ]
        )
    else:
        center_x = np.nan
        center_y = np.nan
    if not np.isfinite(center_x):
        center_x = _last_finite(
            table.float_column('workspace_center_x_m')[chosen_generation]
        )
    if not np.isfinite(center_y):
        center_y = _last_finite(
            table.float_column('workspace_center_y_m')[chosen_generation]
        )

    radius = _last_finite(
        table.float_column('workspace_radius_m')[chosen_generation],
        positive=True,
    )
    if not (
        np.isfinite(center_x)
        and np.isfinite(center_y)
        and np.isfinite(radius)
    ):
        return None

    elapsed = _elapsed_time(table)
    selected_time = elapsed[selected_odom]
    if selected_time.size and np.isfinite(selected_time[0]):
        selected_time = selected_time - selected_time[0]

    path_x, path_y = _workspace_path_for_activation(
        table,
        chosen_generation,
        latest_generation_value,
        center_x,
        center_y,
        origin_x,
        origin_y,
    )
    workspace_generation = table.float_column('workspace_generation')
    workspace_generation_mask = (
        np.isfinite(workspace_generation)
        & np.isclose(
            workspace_generation,
            latest_generation_value,
            rtol=0.0,
            atol=1.0e-9,
        )
    )
    if not np.any(workspace_generation_mask):
        workspace_generation_mask = chosen_generation
    activation_yaw = _yaw_from_quaternion(_activation_quaternion(
        table,
        chosen_generation,
        workspace_generation_mask,
    ))

    odom_plot_x, odom_plot_y = _world_xy_to_robot_axes(
        odom_x[selected_odom] - origin_x,
        odom_y[selected_odom] - origin_y,
        activation_yaw,
    )
    center_plot_x, center_plot_y = _world_xy_to_robot_axes(
        np.asarray([center_x - origin_x], dtype=np.float64),
        np.asarray([center_y - origin_y], dtype=np.float64),
        activation_yaw,
    )
    path_plot_x, path_plot_y = _world_xy_to_robot_axes(
        path_x,
        path_y,
        activation_yaw,
    )

    return WorkspaceTrajectory(
        generation=int(round(latest_generation_value)),
        source_start_index=start_index,
        time_sec=selected_time,
        x_m=odom_plot_x,
        y_m=odom_plot_y,
        center_x_m=float(center_plot_x[0]),
        center_y_m=float(center_plot_y[0]),
        radius_m=float(radius),
        path_x_m=path_plot_x,
        path_y_m=path_plot_y,
    )


def create_collision_figure(
    time_sec: np.ndarray,
    distances_m: np.ndarray,
    labels: Sequence[str],
    title: str,
    pair_legend_label: str = 'Robot link pair',
) -> Figure:
    """Create a distance figure with a thick black finite-row minimum."""

    time_array = np.asarray(time_sec, dtype=np.float64).reshape(-1)
    distance_array = np.asarray(distances_m, dtype=np.float64)
    if distance_array.ndim != 2:
        raise ValueError('distances_m must have shape (samples, pairs)')
    if distance_array.shape[0] != time_array.size:
        raise ValueError('time_sec and distances_m sample counts differ')
    if distance_array.shape[1] != len(labels):
        raise ValueError('labels count does not match distances_m columns')

    figure, axis = plt.subplots(figsize=(14.0, 8.0), constrained_layout=True)
    finite_columns = np.flatnonzero(
        np.any(np.isfinite(distance_array), axis=0)
    )
    colors = plt.get_cmap('tab20')
    for plotted_index, column_index in enumerate(finite_columns):
        axis.plot(
            time_array,
            distance_array[:, column_index],
            color=colors(plotted_index % 20),
            linewidth=0.9,
            alpha=0.72,
            label=labels[column_index],
        )

    if finite_columns.size:
        finite_values = np.isfinite(distance_array)
        row_minimum = np.min(
            np.where(finite_values, distance_array, np.inf),
            axis=1,
        )
        row_minimum[~np.any(finite_values, axis=1)] = np.nan
        axis.plot(
            time_array,
            row_minimum,
            color='black',
            linewidth=3.0,
            alpha=1.0,
            label='Minimum clearance',
            zorder=20,
        )

    axis.axhline(
        0.0,
        color='red',
        linewidth=1.5,
        linestyle='--',
        label='_nolegend_',
        zorder=10,
    )
    axis.set_title(title)
    axis.set_xlabel('Elapsed time (s)')
    axis.set_ylabel('Capsule Distance (m)')
    axis.grid(True, linewidth=0.45, alpha=0.35)
    _scale_figure_fonts(figure)
    for text_artist in (
        axis.title,
        axis.xaxis.label,
        axis.yaxis.label,
    ):
        text_artist.set_fontsize(
            text_artist.get_fontsize() * COLLISION_HEADING_FONT_SCALE
        )

    if finite_columns.size:
        legend_handles = (
            Line2D(
                [0.0, 1.0],
                [0.0, 0.0],
                color=colors(0),
                linewidth=2.0,
                label=pair_legend_label,
            ),
            Line2D(
                [0.0, 1.0],
                [0.0, 0.0],
                color='black',
                linewidth=3.0,
                label='Minimum clearance',
            ),
        )
        axis.legend(
            handles=legend_handles,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.09),
            borderaxespad=0.0,
            fontsize=(
                6.0
                * PLOT_FONT_SCALE
                * COLLISION_HEADING_FONT_SCALE
            ),
            ncol=2,
        )
        axis.xaxis.set_label_coords(0.5, -0.21)
    return figure


def create_odom_workspace_figure(
    trajectory: Optional[WorkspaceTrajectory],
) -> Figure:
    """Create an equal-aspect XY trajectory and workspace-circle figure."""

    figure, axis = plt.subplots(figsize=(9.0, 9.0), constrained_layout=True)
    axis.set_title('Planar CBF Test')
    axis.set_xlabel('x(m)')
    axis.set_ylabel('Y (m)')
    axis.grid(True, linewidth=0.45, alpha=0.35)
    axis.set_aspect('equal', adjustable='datalim')

    if trajectory is None:
        _scale_figure_fonts(figure)
        return figure

    workspace_circle = Circle(
        (trajectory.center_x_m, trajectory.center_y_m),
        trajectory.radius_m,
        fill=False,
        edgecolor='tab:blue',
        linewidth=2.5,
        linestyle='-',
        label='Workspace bounds',
    )
    axis.add_patch(workspace_circle)
    legend_handles = [
        Line2D(
            [0.0, 1.0],
            [0.0, 0.0],
            color='tab:blue',
            linewidth=1.8,
            linestyle='-',
            label='Workspace bounds',
        ),
    ]
    if trajectory.path_x_m.size:
        path_line, = axis.plot(
            trajectory.path_x_m,
            trajectory.path_y_m,
            color='tab:green',
            linewidth=2.0,
            linestyle='--',
            label='Sample nominal path',
            zorder=2,
        )
        legend_handles.append(path_line)
    odom_line, = axis.plot(
        trajectory.x_m,
        trajectory.y_m,
        color='black',
        linewidth=1.8,
        linestyle='-',
        label='Robot position',
        zorder=3,
    )
    legend_handles.append(odom_line)

    radius = trajectory.radius_m
    center_x = trajectory.center_x_m
    center_y = trajectory.center_y_m
    axis.update_datalim(np.asarray([
        [center_x - radius, center_y - radius],
        [center_x + radius, center_y + radius],
    ]))
    axis.autoscale_view()
    axis.legend(
        handles=legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.09),
        borderaxespad=0.0,
        ncol=2,
    )
    axis.xaxis.set_label_coords(0.5, -0.24)
    _scale_figure_fonts(figure)
    return figure


def _scale_figure_fonts(
    figure: Figure,
    scale: float = PLOT_FONT_SCALE,
) -> None:
    """Scale every text artist in a completed plot by the same factor."""

    for text_artist in figure.findobj(match=Text):
        text_artist.set_fontsize(text_artist.get_fontsize() * scale)


def generate_plots(csv_path, output_root=None) -> PlotOutputs:
    """Generate PNG plots for one CSV and return their paths."""

    table = load_csv(csv_path)
    collisions = reconstruct_collision_series(table)
    trajectory = select_latest_workspace_trajectory(table)

    root = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else plot_dir().resolve()
    )
    output_directory = root / table.path.stem
    output_directory.mkdir(parents=True, exist_ok=True)

    internal_base = output_directory / 'internal_collision_distances'
    external_base = output_directory / 'external_collision_distances'
    odom_base = output_directory / 'odom_workspace'

    internal_figure = create_collision_figure(
        collisions.time_sec,
        collisions.internal_m,
        collisions.internal_labels,
        'Self-collision capsule clearances',
    )
    _save_figure(internal_figure, internal_base)

    external_figure = create_collision_figure(
        collisions.time_sec,
        collisions.external_m,
        collisions.external_labels,
        'External collision capsule clearances',
        pair_legend_label='Human-Robot link pair',
    )
    _save_figure(external_figure, external_base)

    odom_figure = create_odom_workspace_figure(trajectory)
    _save_figure(odom_figure, odom_base)

    return PlotOutputs(
        output_dir=output_directory,
        internal_png=internal_base.with_suffix('.png'),
        external_png=external_base.with_suffix('.png'),
        odom_png=odom_base.with_suffix('.png'),
    )


def _elapsed_time(table: CsvTable) -> np.ndarray:
    if table.has('elapsed_sec'):
        elapsed = table.float_column('elapsed_sec')
        if np.any(np.isfinite(elapsed)):
            return elapsed

    if table.has('ros_time_sec'):
        ros_time = table.float_column('ros_time_sec')
        finite_indices = np.flatnonzero(np.isfinite(ros_time))
        if finite_indices.size:
            return ros_time - ros_time[finite_indices[0]]

    if table.has('sample_index'):
        return table.float_column('sample_index')
    return np.arange(len(table), dtype=np.float64)


def _joint_matrix(
    table: CsvTable,
    joint_names: Sequence[str],
) -> np.ndarray:
    return np.column_stack([
        table.float_column(
            joint_value_field(name),
            default=float(JOINT_DEFAULTS[name]),
        )
        for name in joint_names
    ])


def _human_arrays(table: CsvTable):
    sample_count = len(table)
    human_a = np.full(
        (sample_count, N_HUMAN_CAPSULES, 3),
        np.nan,
        dtype=np.float64,
    )
    human_b = np.full_like(human_a, np.nan)
    human_radii = np.full(
        (sample_count, N_HUMAN_CAPSULES),
        np.nan,
        dtype=np.float64,
    )
    present = np.zeros(
        (sample_count, N_HUMAN_CAPSULES),
        dtype=bool,
    )

    for human_index in range(N_HUMAN_CAPSULES):
        (
            present_field,
            fresh_field,
            _,
            a_x_field,
            a_y_field,
            a_z_field,
            b_x_field,
            b_y_field,
            b_z_field,
            radius_field,
        ) = human_slot_fields(human_index)
        human_a[:, human_index] = np.column_stack([
            table.float_column(a_x_field),
            table.float_column(a_y_field),
            table.float_column(a_z_field),
        ])
        human_b[:, human_index] = np.column_stack([
            table.float_column(b_x_field),
            table.float_column(b_y_field),
            table.float_column(b_z_field),
        ])
        human_radii[:, human_index] = table.float_column(radius_field)

        inferred_present = (
            np.all(np.isfinite(human_a[:, human_index]), axis=1)
            & np.all(np.isfinite(human_b[:, human_index]), axis=1)
            & np.isfinite(human_radii[:, human_index])
            & (human_radii[:, human_index] > 0.0)
        )
        slot_present = (
            table.bool_column(present_field)
            if table.has(present_field)
            else inferred_present
        )
        slot_fresh = (
            table.bool_column(fresh_field)
            if table.has(fresh_field)
            else slot_present
        )
        present[:, human_index] = (
            slot_present
            & slot_fresh
            & inferred_present
        )
    return human_a, human_b, human_radii, present


def _external_labels(table: CsvTable) -> Tuple[str, ...]:
    labels = []
    for human_index in range(N_HUMAN_CAPSULES):
        name_field = human_slot_fields(human_index)[2]
        names = [
            value
            for value in table.string_column(name_field)
            if value
        ]
        most_common_name = Counter(names).most_common(1)
        human_label = f'human[{human_index:02d}]'
        if most_common_name:
            human_label += f' {most_common_name[0][0]}'
        labels.extend(
            f'{body_name} \u2194 {human_label}'
            for body_name in EXTERNAL_ROBOT_BODY_NAMES
        )
    return tuple(labels)


def _last_finite(values: np.ndarray, positive: bool = False) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    valid = np.isfinite(array)
    if positive:
        valid &= array > 0.0
    indices = np.flatnonzero(valid)
    if indices.size == 0:
        return np.nan
    return float(array[indices[-1]])


def _workspace_path_for_activation(
    table: CsvTable,
    chosen_generation: np.ndarray,
    generation_value: float,
    center_x: float,
    center_y: float,
    origin_x: float,
    origin_y: float,
) -> Tuple[np.ndarray, np.ndarray]:
    empty = np.empty(0, dtype=np.float64)
    path_frame, points = _latest_workspace_path(table)
    if path_frame is None:
        return empty, empty

    parent_frame = _last_nonempty(
        table.string_column('workspace_activation_frame_id')[
            chosen_generation
        ]
    )
    child_frame = _last_nonempty(
        table.string_column('workspace_activation_child_frame_id')[
            chosen_generation
        ]
    )

    workspace_generation = table.float_column('workspace_generation')
    workspace_generation_mask = (
        np.isfinite(workspace_generation)
        & np.isclose(
            workspace_generation,
            generation_value,
            rtol=0.0,
            atol=1.0e-9,
        )
    )
    if not np.any(workspace_generation_mask):
        workspace_generation_mask = chosen_generation
    if not parent_frame:
        parent_frame = _last_nonempty(
            table.string_column('workspace_frame_id')[
                workspace_generation_mask
            ]
        )
    if not child_frame:
        child_frame = _last_nonempty(
            table.string_column('workspace_child_frame_id')[
                workspace_generation_mask
            ]
        )

    path_frame = _normalize_frame_id(path_frame)
    parent_frame = _normalize_frame_id(parent_frame)
    child_frame = _normalize_frame_id(child_frame)
    if path_frame == child_frame and child_frame:
        quaternion = _activation_quaternion(
            table,
            chosen_generation,
            workspace_generation_mask,
        )
        yaw = _yaw_from_quaternion(quaternion)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        world_x = (
            float(center_x)
            + cos_yaw * points[:, 0]
            - sin_yaw * points[:, 1]
        )
        world_y = (
            float(center_y)
            + sin_yaw * points[:, 0]
            + cos_yaw * points[:, 1]
        )
    elif path_frame == parent_frame and parent_frame:
        world_x = points[:, 0]
        world_y = points[:, 1]
    else:
        return empty, empty

    return (
        np.asarray(world_x - float(origin_x), dtype=np.float64),
        np.asarray(world_y - float(origin_y), dtype=np.float64),
    )


def _latest_workspace_path(
    table: CsvTable,
) -> Tuple[Optional[str], np.ndarray]:
    empty = np.empty((0, 2), dtype=np.float64)
    if not table.has('workspace_path_xy_json'):
        return None, empty

    raw_paths = table.string_column('workspace_path_xy_json')
    frame_ids = table.string_column('workspace_path_frame_id')
    point_counts = table.float_column('workspace_path_point_count')
    path_valid = (
        table.bool_column('workspace_path_valid')
        if table.has('workspace_path_valid')
        else np.ones(len(table), dtype=bool)
    )
    for index in reversed(np.flatnonzero(path_valid & (raw_paths != ''))):
        try:
            points = np.asarray(
                json.loads(raw_paths[index]),
                dtype=np.float64,
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if (
            points.ndim != 2
            or points.shape[0] < 2
            or points.shape[1] != 2
            or not np.all(np.isfinite(points))
        ):
            continue
        expected_count = point_counts[index]
        if (
            np.isfinite(expected_count)
            and int(round(expected_count)) != points.shape[0]
        ):
            continue
        frame_id = _normalize_frame_id(frame_ids[index])
        if frame_id:
            return frame_id, points
    return None, empty


def _activation_quaternion(
    table: CsvTable,
    activation_mask: np.ndarray,
    workspace_mask: np.ndarray,
) -> Tuple[float, float, float, float]:
    quaternion = _last_valid_quaternion(
        table,
        (
            'workspace_activation_qx',
            'workspace_activation_qy',
            'workspace_activation_qz',
            'workspace_activation_qw',
        ),
        activation_mask,
    )
    if quaternion is None:
        quaternion = _last_valid_quaternion(
            table,
            ('workspace_qx', 'workspace_qy', 'workspace_qz', 'workspace_qw'),
            workspace_mask,
        )
    return quaternion or (0.0, 0.0, 0.0, 1.0)


def _last_valid_quaternion(
    table: CsvTable,
    fields: Tuple[str, str, str, str],
    mask: np.ndarray,
) -> Optional[Tuple[float, float, float, float]]:
    values = np.column_stack([
        table.float_column(field)
        for field in fields
    ])
    valid = (
        np.asarray(mask, dtype=bool)
        & np.all(np.isfinite(values), axis=1)
        & (np.linalg.norm(values, axis=1) > 1.0e-12)
    )
    indices = np.flatnonzero(valid)
    if indices.size == 0:
        return None
    quaternion = values[indices[-1]]
    quaternion /= np.linalg.norm(quaternion)
    return tuple(float(value) for value in quaternion)


def _yaw_from_quaternion(
    quaternion: Tuple[float, float, float, float],
) -> float:
    x, y, z, w = quaternion
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _world_xy_to_robot_axes(
    world_x: np.ndarray,
    world_y: np.ndarray,
    activation_yaw: float,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(world_x, dtype=np.float64)
    y = np.asarray(world_y, dtype=np.float64)
    cos_yaw = math.cos(activation_yaw)
    sin_yaw = math.sin(activation_yaw)
    return (
        cos_yaw * x + sin_yaw * y,
        -sin_yaw * x + cos_yaw * y,
    )


def _last_nonempty(values: np.ndarray) -> str:
    for value in reversed(np.asarray(values, dtype=object).reshape(-1)):
        text = str(value or '').strip()
        if text:
            return text
    return ''


def _normalize_frame_id(value) -> str:
    return str(value or '').strip().lstrip('/')


def _parse_float(value, default: float) -> float:
    if value is None:
        return float(default)
    text = str(value).strip()
    if not text:
        return float(default)
    try:
        result = float(text)
    except (TypeError, ValueError):
        return float(default)
    return result


def _parse_bool(value, default: bool) -> bool:
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if not text:
        return bool(default)
    if text in ('1', 'true', 'yes', 'on'):
        return True
    if text in ('0', 'false', 'no', 'off'):
        return False
    try:
        return bool(float(text))
    except ValueError:
        return bool(default)


def _save_figure(figure: Figure, output_base: Path) -> None:
    try:
        figure.savefig(
            output_base.with_suffix('.png'),
            dpi=PNG_DPI,
            bbox_inches='tight',
        )
    finally:
        plt.close(figure)


__all__ = [
    'CollisionSeries',
    'CsvTable',
    'PlotOutputs',
    'WorkspaceTrajectory',
    'create_collision_figure',
    'create_odom_workspace_figure',
    'generate_plots',
    'load_csv',
    'reconstruct_collision_series',
    'select_latest_workspace_trajectory',
]
