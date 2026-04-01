"""CRC32 computation for Unitree G1 LowCmd messages.

Ported from g1_bridge/crc.py (motor_crc_hg.cpp). The CRC must be computed
over the raw C struct byte layout.
"""

import struct

_HEADER_FMT = '<BBxx'
_MOTOR_FMT = '<BxxxfffffI'
_RESERVE_FMT = '<4I'
_TOTAL_SIZE = 4 + 35 * 28 + 16 + 4  # 1004 bytes
_CRC_WORDS = (_TOTAL_SIZE - 4) // 4  # 250 words (exclude crc field)
_POLYNOMIAL = 0x04C11DB7


def _crc32_core(words):
    crc = 0xFFFFFFFF
    for word in words:
        xbit = 1 << 31
        data = word
        for _ in range(32):
            if crc & 0x80000000:
                crc = ((crc << 1) & 0xFFFFFFFF) ^ _POLYNOMIAL
            else:
                crc = (crc << 1) & 0xFFFFFFFF
            if data & xbit:
                crc ^= _POLYNOMIAL
            xbit >>= 1
    return crc


def compute_crc(msg):
    """Compute and set the CRC field on a unitree_hg/msg/LowCmd message."""
    buf = bytearray(_TOTAL_SIZE)

    struct.pack_into(_HEADER_FMT, buf, 0, msg.mode_pr, msg.mode_machine)

    offset = 4
    for i in range(35):
        mc = msg.motor_cmd[i]
        struct.pack_into(
            _MOTOR_FMT, buf, offset,
            mc.mode, mc.q, mc.dq, mc.tau, mc.kp, mc.kd, mc.reserve,
        )
        offset += 28

    struct.pack_into(
        _RESERVE_FMT, buf, offset,
        msg.reserve[0], msg.reserve[1],
        msg.reserve[2], msg.reserve[3],
    )

    words = struct.unpack_from(f'<{_CRC_WORDS}I', buf, 0)
    msg.crc = _crc32_core(words)
