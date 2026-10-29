"""Concatenate the latest CapsuleArray of several sources into one."""

from rclpy.time import Time

from g1_cbf.human_capsules import human_data_is_fresh
from g1_cbf_msg.msg import CapsuleArray


class ColliderMerger:
    """Keep the latest array of each source and concatenate the fresh ones.

    A source is fresh while its latest array arrived within
    ``stale_timeout_sec`` of ``now`` (0 disables the timeout). Receipt times
    come from the merging node's clock, so clock offsets between the sources
    cannot drop one. The merged array takes the frame of the first fresh
    source, and a fresh source in another frame is left out. Its stamp is the
    oldest nonzero stamp of the merged sources, so a consumer's freshness gate
    judges the oldest measurement in it.
    """

    def __init__(self, num_sources, stale_timeout_sec):
        if num_sources < 1:
            raise ValueError('num_sources must be at least 1')
        if stale_timeout_sec < 0.0:
            raise ValueError('stale_timeout_sec must be non-negative')
        self._latest = [None] * num_sources
        self._stale_timeout_sec = float(stale_timeout_sec)

    def update(self, index, msg, receipt_time):
        """Store ``msg``, received at ``receipt_time``, for source ``index``."""
        self._latest[index] = (msg, receipt_time)

    def merge(self, now):
        """Return ``(merged, skipped)``.

        ``merged`` is a CapsuleArray, or None when no source is fresh.
        ``skipped`` lists ``(index, reason)`` for every source left out.
        """
        merged = None
        oldest_stamp = None
        skipped = []
        for index, latest in enumerate(self._latest):
            if latest is None:
                skipped.append((index, 'nothing received'))
                continue
            msg, receipt_time = latest
            if not human_data_is_fresh(
                receipt_time, now, self._stale_timeout_sec,
            ):
                age = (now - receipt_time).nanoseconds * 1e-9
                skipped.append((index, f'stale by {age:.3f}s'))
                continue

            frame_id = str(msg.header.frame_id)
            if merged is None:
                merged = CapsuleArray()
                merged.header.frame_id = frame_id
            elif frame_id != merged.header.frame_id:
                skipped.append((
                    index,
                    f"frame '{frame_id}' is not "
                    f"'{merged.header.frame_id}'",
                ))
                continue

            merged.capsules.extend(msg.capsules)
            stamp = Time.from_msg(msg.header.stamp)
            if stamp.nanoseconds and (
                oldest_stamp is None or stamp < oldest_stamp
            ):
                oldest_stamp = stamp

        if merged is not None and oldest_stamp is not None:
            merged.header.stamp = oldest_stamp.to_msg()
        return merged, skipped
