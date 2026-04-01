#ifndef G1_RL_DEPLOY_MOTOR_CRC_HG_H_
#define G1_RL_DEPLOY_MOTOR_CRC_HG_H_

#include <stdint.h>
#include <array>
#include "unitree_hg/msg/low_cmd.hpp"

typedef struct {
  uint8_t mode;
  float q;
  float dq;
  float tau;
  float Kp;
  float Kd;
  uint32_t reserve = 0;
} MotorCmd_;

typedef struct {
  uint8_t modePr;
  uint8_t modeMachine;
  std::array<MotorCmd_, 35> motorCmd;
  std::array<uint32_t, 4> reserve;
  uint32_t crc;
} LowCmd_;

uint32_t crc32_core(uint32_t *ptr, uint32_t len);
void get_crc(unitree_hg::msg::LowCmd &msg);

#endif
