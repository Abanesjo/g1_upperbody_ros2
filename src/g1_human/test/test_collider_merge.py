import pytest
from g1_cbf_msg.msg import Capsule, CapsuleArray
from rclpy.time import Time

from g1_human.collider_merge import ColliderMerger


def _array(names, frame_id='world', stamp_sec=10.0):
    msg = CapsuleArray()
    msg.header.frame_id = frame_id
    msg.header.stamp = Time(seconds=stamp_sec).to_msg()
    for name in names:
        capsule = Capsule()
        capsule.name = name
        capsule.radius = 0.1
        msg.capsules.append(capsule)
    return msg


def _names(msg):
    return [capsule.name for capsule in msg.capsules]


def test_fresh_sources_are_concatenated_in_source_order():
    merger = ColliderMerger(3, stale_timeout_sec=0.2)
    merger.update(2, _array(['c0']), Time(seconds=10.0))
    merger.update(0, _array(['a0', 'a1']), Time(seconds=10.0))
    merger.update(1, _array(['b0']), Time(seconds=10.05))

    merged, skipped = merger.merge(Time(seconds=10.1))

    assert _names(merged) == ['a0', 'a1', 'b0', 'c0']
    assert merged.header.frame_id == 'world'
    assert skipped == []


def test_stale_source_is_dropped_while_the_others_stay():
    merger = ColliderMerger(3, stale_timeout_sec=0.2)
    merger.update(0, _array(['a0']), Time(seconds=10.0))
    merger.update(1, _array(['b0']), Time(seconds=9.7))
    merger.update(2, _array(['c0']), Time(seconds=10.0))

    merged, skipped = merger.merge(Time(seconds=10.1))

    assert _names(merged) == ['a0', 'c0']
    assert skipped == [(1, 'stale by 0.400s')]


def test_source_in_another_frame_is_left_out():
    merger = ColliderMerger(2, stale_timeout_sec=0.2)
    merger.update(0, _array(['a0'], frame_id='world'), Time(seconds=10.0))
    merger.update(1, _array(['b0'], frame_id='fusion_world'), Time(seconds=10.0))

    merged, skipped = merger.merge(Time(seconds=10.0))

    assert _names(merged) == ['a0']
    assert merged.header.frame_id == 'world'
    assert skipped == [(1, "frame 'fusion_world' is not 'world'")]


def test_merged_stamp_is_the_oldest_nonzero_stamp():
    merger = ColliderMerger(3, stale_timeout_sec=0.2)
    merger.update(0, _array(['a0'], stamp_sec=9.95), Time(seconds=10.0))
    merger.update(1, _array(['b0'], stamp_sec=9.9), Time(seconds=10.0))
    merger.update(2, _array(['c0'], stamp_sec=0.0), Time(seconds=10.0))

    merged, _ = merger.merge(Time(seconds=10.0))

    assert merged.header.stamp == Time(seconds=9.9).to_msg()


def test_nothing_is_merged_while_no_source_is_fresh():
    merger = ColliderMerger(2, stale_timeout_sec=0.2)

    assert merger.merge(Time(seconds=10.0)) == (
        None, [(0, 'nothing received'), (1, 'nothing received')],
    )

    merger.update(1, _array(['b0']), Time(seconds=10.0))
    merged, _ = merger.merge(Time(seconds=11.0))
    assert merged is None


def test_zero_timeout_keeps_every_received_source():
    merger = ColliderMerger(1, stale_timeout_sec=0.0)
    merger.update(0, _array(['a0']), Time(seconds=1.0))

    merged, skipped = merger.merge(Time(seconds=100.0))

    assert _names(merged) == ['a0']
    assert skipped == []


@pytest.mark.parametrize(
    ('num_sources', 'stale_timeout_sec'), [(0, 0.2), (1, -0.1)],
)
def test_invalid_arguments_are_rejected(num_sources, stale_timeout_sec):
    with pytest.raises(ValueError):
        ColliderMerger(num_sources, stale_timeout_sec)
