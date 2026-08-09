"""One-shot ROS node for plotting every CSV in the logging data directory."""

from pathlib import Path

import rclpy
from rclpy.node import Node

from g1_logging.plotter import generate_plots


CSV_DIRECTORY = Path(
    '/workspace/ros2_ws/src/g1_logging/data'
)


def discover_csv_files(directory=CSV_DIRECTORY):
    """Return every CSV file directly under ``directory``, sorted by name."""
    directory = Path(directory).expanduser()
    if not directory.is_dir():
        return ()
    return tuple(sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == '.csv'
    ))


class CbfCsvPlotterNode(Node):
    """Generate analysis plots for every recorded CSV and then exit."""

    def __init__(self, csv_directory=CSV_DIRECTORY):
        super().__init__('cbf_csv_plotter')
        self.csv_directory = Path(csv_directory).expanduser()

    def generate(self):
        csv_paths = discover_csv_files(self.csv_directory)
        if not csv_paths:
            self.get_logger().warn(
                f'No CSV files found in {self.csv_directory}'
            )
            return (), ()

        outputs = []
        failures = []
        for csv_path in csv_paths:
            try:
                result = generate_plots(csv_path)
            except Exception as error:
                failures.append((csv_path, error))
                self.get_logger().error(
                    f'Could not generate plots from {csv_path}: {error}'
                )
                continue
            outputs.append(result)
            self.get_logger().info(
                f'Generated plots for {csv_path.name} in '
                f'{result.output_dir}'
            )

        self.get_logger().info(
            f'Plotting complete: {len(outputs)} succeeded, '
            f'{len(failures)} failed'
        )
        return tuple(outputs), tuple(failures)


def main(args=None):
    rclpy.init(args=args)
    node = CbfCsvPlotterNode()
    exit_code = 0
    try:
        _, failures = node.generate()
        if failures:
            exit_code = 1
    except Exception as error:
        node.get_logger().error(
            f'Could not scan CSV directory {node.csv_directory}: {error}'
        )
        exit_code = 1
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return exit_code
