"""Source-tree paths for mutable logging artifacts.

The source symlink and fixed Docker workspace are preferred, with a
working-directory search retained for other development layouts.
"""

from datetime import datetime
import os
from pathlib import Path


PACKAGE_NAME = 'g1_logging'
SOURCE_ROOT_ENV = 'G1_LOGGING_ROOT'
DOCKER_SOURCE_ROOT = Path('/workspace/ros2_ws/src/g1_logging')


def _is_source_root(path):
    path = Path(path)
    return (
        (path / 'package.xml').is_file()
        and (path / 'g1_logging').is_dir()
    )


def package_source_root():
    """Return the writable source package containing ``data`` and ``plot``."""
    override = os.environ.get(SOURCE_ROOT_ENV, '').strip()
    if override:
        candidate = Path(override).expanduser().resolve()
        if not _is_source_root(candidate):
            raise RuntimeError(
                f'{SOURCE_ROOT_ENV} does not point to a g1_logging source '
                f'package: {candidate}'
            )
        return candidate

    module_root = Path(__file__).resolve().parents[1]
    if _is_source_root(module_root):
        return module_root

    if _is_source_root(DOCKER_SOURCE_ROOT):
        return DOCKER_SOURCE_ROOT

    cwd = Path.cwd().resolve()
    for parent in (cwd, *cwd.parents):
        for candidate in (
            parent / 'src' / PACKAGE_NAME,
            parent / PACKAGE_NAME,
        ):
            if _is_source_root(candidate):
                return candidate

    raise RuntimeError(
        'Could not locate the writable g1_logging source package. Run from '
        'the ROS workspace built with --symlink-install, or set '
        f'{SOURCE_ROOT_ENV}.'
    )


def data_dir():
    """Return the package's source ``data`` directory."""
    path = package_source_root() / 'data'
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_dir():
    """Return the package's source ``plot`` directory."""
    path = package_source_root() / 'plot'
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_csv_output(filename=''):
    """Resolve a configured output name under ``data``.

    Absolute paths are honored. Relative paths are deliberately rooted in the
    package data directory so launch location does not change where logs land.
    """
    value = str(filename or '').strip()
    if not value:
        value = datetime.now().strftime('cbf_log_%Y%m%d_%H%M%S.csv')

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = data_dir() / path
    if path.suffix.lower() != '.csv':
        path = path.with_suffix('.csv')
    return path.resolve()
