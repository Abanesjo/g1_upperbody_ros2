import importlib.util
from pathlib import Path
from types import SimpleNamespace

from geometry_msgs.msg import Twist
from std_msgs.msg import Bool


MODULE_PATH = (
    Path(__file__).parents[1] / 'scripts' / 'cmd_vel_cbf_node.py'
)
SPEC = importlib.util.spec_from_file_location('cmd_vel_cbf_node', MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
CmdVelCBFNode = MODULE.CmdVelCBFNode


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


def test_disabled_cbf_uses_clipped_passthrough_without_state_or_solver():
    node = CmdVelCBFNode.__new__(CmdVelCBFNode)
    node._cbf_enabled = True
    node._latest_cmd = Twist()
    node._latest_cmd.linear.x = 3.0
    node._latest_cmd.linear.y = -2.0
    node._latest_cmd.linear.z = 0.4
    node._latest_cmd.angular.z = 0.7
    node._lin_vel_x_limits = [-1.0, 2.0]
    node._lin_vel_y_limits = [-1.0, 1.0]
    node._cmd_pub = _Publisher()
    node.get_parameter = lambda _name: SimpleNamespace(value=True)

    def _unexpected_call():
        raise AssertionError('disabled CBF must not inspect state or solve')

    node._state_ready = _unexpected_call
    node._filter_planar_velocity = _unexpected_call

    node._cbf_enabled_cb(Bool(data=False))
    node._tick()

    assert not node._cbf_enabled
    assert len(node._cmd_pub.messages) == 1
    safe = node._cmd_pub.messages[0]
    assert safe.linear.x == 2.0
    assert safe.linear.y == -1.0
    assert safe.linear.z == 0.4
    assert safe.angular.z == 0.7


def test_reenabled_cbf_returns_to_existing_missing_state_guard():
    node = CmdVelCBFNode.__new__(CmdVelCBFNode)
    node._cbf_enabled = False
    node._latest_cmd = Twist()
    node._latest_cmd.linear.x = 0.8
    node._latest_cmd.angular.z = 0.3
    node._lin_vel_x_limits = [-1.0, 2.0]
    node._lin_vel_y_limits = [-1.0, 1.0]
    node._cmd_pub = _Publisher()
    node.get_parameter = lambda _name: SimpleNamespace(value=True)
    node.get_logger = lambda: SimpleNamespace(
        warn=lambda *_args, **_kwargs: None,
    )

    state_checks = []

    def _missing_state():
        state_checks.append(True)
        return False, 'test state unavailable'

    def _unexpected_filter(_command):
        raise AssertionError('missing state must prevent the CBF solve')

    node._state_ready = _missing_state
    node._filter_planar_velocity = _unexpected_filter

    node._cbf_enabled_cb(Bool(data=True))
    node._tick()

    assert node._cbf_enabled
    assert state_checks == [True]
    assert len(node._cmd_pub.messages) == 1
    safe = node._cmd_pub.messages[0]
    assert safe.linear.x == 0.0
    assert safe.linear.y == 0.0
    assert safe.angular.z == 0.3
