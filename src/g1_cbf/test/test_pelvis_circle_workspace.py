import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

from builtin_interfaces.msg import Time


MODULE_PATH = (
    Path(__file__).parents[1] / 'scripts' / 'pelvis_circle_publisher_node.py'
)
SPEC = importlib.util.spec_from_file_location(
    'pelvis_circle_publisher_node', MODULE_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
PelvisCirclePublisherNode = MODULE.PelvisCirclePublisherNode


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


def test_circle_is_local_to_workspace_and_keeps_configured_height():
    node = PelvisCirclePublisherNode.__new__(PelvisCirclePublisherNode)
    node.radius = 2.0
    node.height = 0.5
    node.segments = 4
    node.line_width = 0.02
    node.workspace_frame = 'workspace'
    node.pub = _Publisher()
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(to_msg=lambda: Time())
    )

    node._publish_circle()

    assert len(node.pub.messages) == 1
    marker = node.pub.messages[0]
    assert marker.header.frame_id == 'workspace'
    assert len(marker.points) == 5
    assert all(point.z == 0.5 for point in marker.points)
    assert math.isclose(marker.points[0].x, 2.0)
    assert math.isclose(marker.points[0].y, 0.0, abs_tol=1e-12)
