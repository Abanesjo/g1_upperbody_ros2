"""
CRC32 computation for Unitree G1 LowCmd messages.

Ports the C++ implementation from motor_crc_hg.cpp. The CRC must be computed
over the raw C struct byte layout, not the ROS message fields directly.
"""

import struct

# C struct layout:
#   LowCmd (1004 bytes):
#     uint8  modePr          (1 byte)
#     uint8  modeMachine     (1 byte)
#     <2 bytes padding>
#     MotorCmd[35]           (35 * 28 = 980 bytes)
#     uint32 reserve[4]      (16 bytes)
#     uint32 crc             (4 bytes)  <-- excluded from CRC
#
#   MotorCmd (28 bytes):
#     uint8  mode            (1 byte)
#     <3 bytes padding>
#     float  q               (4 bytes)
#     float  dq              (4 bytes)
#     float  tau             (4 bytes)
#     float  kp              (4 bytes)
#     float  kd              (4 bytes)
#     uint32 reserve         (4 bytes)

_HEADER_FMT = '<BBxx'          # modePr, modeMachine, 2 pad bytes = 4 bytes
_MOTOR_FMT = '<BxxxfffffI'     # mode, 3 pad, q, dq, tau, kp, kd, reserve = 28 bytes
_RESERVE_FMT = '<4I'           # reserve[4] = 16 bytes
_TOTAL_SIZE = 4 + 35 * 28 + 16 + 4  # 1004 bytes
_CRC_WORDS = (_TOTAL_SIZE - 4) // 4  # 250 words (exclude crc field)

_POLYNOMIAL = 0x04C11DB7


def _crc32_core(words):
    """Compute CRC32 over a sequence of uint32 words.

    Direct port of crc32_core() from motor_crc_hg.cpp.
    """
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
    """Compute and set the CRC field on a unitree_hg/msg/LowCmd message.

    Packs the message into the C struct binary layout, computes CRC32,
    and sets msg.crc.
    """
    buf = bytearray(_TOTAL_SIZE)

    # Pack header
    struct.pack_into(_HEADER_FMT, buf, 0, msg.mode_pr, msg.mode_machine)

    # Pack 35 motor commands
    offset = 4
    for i in range(35):
        mc = msg.motor_cmd[i]
        struct.pack_into(
            _MOTOR_FMT, buf, offset,
            mc.mode, mc.q, mc.dq, mc.tau, mc.kp, mc.kd, mc.reserve
        )
        offset += 28

    # Pack reserve
    struct.pack_into(_RESERVE_FMT, buf, offset,
                     msg.reserve[0], msg.reserve[1],
                     msg.reserve[2], msg.reserve[3])

    # Interpret first 1000 bytes as 250 uint32 words (little-endian)
    words = struct.unpack_from(f'<{_CRC_WORDS}I', buf, 0)

    msg.crc = _crc32_core(words)
