# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Constants for G1."""

from etils import epath

from mujoco_playground._src import mjx_env

ROOT_PATH = mjx_env.ROOT_PATH / "locomotion" / "g1"
FEET_ONLY_FLAT_TERRAIN_XML = (
    ROOT_PATH / "xmls" / "scene_mjx_feetonly_flat_terrain.xml"
)
FEET_ONLY_ROUGH_TERRAIN_XML = (
    ROOT_PATH / "xmls" / "scene_mjx_feetonly_rough_terrain.xml"
)


def task_to_xml(task_name: str) -> epath.Path:
  return {
      "flat_terrain": FEET_ONLY_FLAT_TERRAIN_XML,
      "rough_terrain": FEET_ONLY_ROUGH_TERRAIN_XML,
  }[task_name]


FEET_SITES = [
    "left_foot",
    "right_foot",
]

HAND_SITES = [
    "left_palm",
    "right_palm",
]

LEFT_FEET_GEOMS = ["left_foot"]
RIGHT_FEET_GEOMS = ["right_foot"]
FEET_GEOMS = LEFT_FEET_GEOMS + RIGHT_FEET_GEOMS

ROOT_BODY = "torso_link"

GRAVITY_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"

# Joint index i (0-28) used throughout joystick.py, as defined in
# xmls/g1_mjx_feetonly.xml (the same for the flat and rough terrain scenes).
# Index i is the same joint in all of these:
#   qpos[7 + i]   (qpos[0:7] is the floating base: xyz + quaternion)
#   qvel[6 + i]   (qvel[0:6] is the floating base: linear + angular velocity)
#   ctrl[i], action[i], motor_targets[i], actuator_force[i], _default_pose[i]
#   jnt_range[1 + i]   (jnt_range[0] is the floating base)
#
# Index groups built in joystick.py (Joystick._post_init):
#   _hip_indices   = [1, 2, 7, 8]       hip roll and yaw, left and right
#   _knee_indices  = [3, 9]
#   _waist_indices = [12, 13, 14]
#   _arm_indices   = [16, 17, 19, 20, 21, 23, 24, 26, 27, 28]
#                    shoulder roll/yaw and wrist roll/pitch/yaw, left and right
#   _weights: 0.01 at hip pitch and knee (0, 3, 6, 9), 1.0 elsewhere.

RESTRICTED_JOINT_RANGE = (
    # Left leg.
    (-1.57, 1.57),  #  0: left_hip_pitch_joint
    (-0.5, 0.5),  #  1: left_hip_roll_joint
    (-0.7, 0.7),  #  2: left_hip_yaw_joint
    (0, 1.57),  #  3: left_knee_joint
    (-0.4, 0.4),  #  4: left_ankle_pitch_joint
    (-0.2, 0.2),  #  5: left_ankle_roll_joint
    # Right leg.
    (-1.57, 1.57),  #  6: right_hip_pitch_joint
    (-0.5, 0.5),  #  7: right_hip_roll_joint
    (-0.7, 0.7),  #  8: right_hip_yaw_joint
    (0, 1.57),  #  9: right_knee_joint
    (-0.4, 0.4),  # 10: right_ankle_pitch_joint
    (-0.2, 0.2),  # 11: right_ankle_roll_joint
    # Waist.
    (-2.618, 2.618),  # 12: waist_yaw_joint
    (-0.52, 0.52),  # 13: waist_roll_joint
    (-0.52, 0.52),  # 14: waist_pitch_joint
    # Left arm.
    (-3.0892, 2.6704),  # 15: left_shoulder_pitch_joint
    (-1.5882, 2.2515),  # 16: left_shoulder_roll_joint
    (-2.618, 2.618),  # 17: left_shoulder_yaw_joint
    (-1.0472, 2.0944),  # 18: left_elbow_joint
    (-1.97222, 1.97222),  # 19: left_wrist_roll_joint
    (-1.61443, 1.61443),  # 20: left_wrist_pitch_joint
    (-1.61443, 1.61443),  # 21: left_wrist_yaw_joint
    # Right arm.
    (-3.0892, 2.6704),  # 22: right_shoulder_pitch_joint
    (-2.2515, 1.5882),  # 23: right_shoulder_roll_joint
    (-2.618, 2.618),  # 24: right_shoulder_yaw_joint
    (-1.0472, 2.0944),  # 25: right_elbow_joint
    (-1.97222, 1.97222),  # 26: right_wrist_roll_joint
    (-1.61443, 1.61443),  # 27: right_wrist_pitch_joint
    (-1.61443, 1.61443),  # 28: right_wrist_yaw_joint
)
