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

"""Constants for H2."""

from etils import epath

from mujoco_playground._src import mjx_env

ROOT_PATH = mjx_env.ROOT_PATH / "locomotion" / "h2"
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

LEFT_FEET_GEOMS = ["left_ankle_pitch_link"]
RIGHT_FEET_GEOMS = ["right_ankle_pitch_link"]
FEET_GEOMS = LEFT_FEET_GEOMS + RIGHT_FEET_GEOMS

ROOT_BODY = "pelvis"

GRAVITY_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "imu_acc"
GYRO_SENSOR = "imu_gyro"

# Joints in order:
#    0: left_hip_pitch_joint
#    1: left_hip_roll_joint
#    2: left_hip_yaw_joint
#    3: left_knee_joint
#    4: left_ankle_roll_joint
#    5: left_ankle_pitch_joint
#    6: right_hip_pitch_joint
#    7: right_hip_roll_joint
#    8: right_hip_yaw_joint
#    9: right_knee_joint
#   10: right_ankle_roll_joint
#   11: right_ankle_pitch_joint
#   12: waist_yaw_joint
#   13: waist_roll_joint
#   14: waist_pitch_joint
#   15: head_pitch_joint
#   16: head_yaw_joint
#   17: left_shoulder_pitch_joint
#   18: left_shoulder_roll_joint
#   19: left_shoulder_yaw_joint
#   20: left_elbow_joint
#   21: left_wrist_roll_joint
#   22: left_wrist_pitch_joint
#   23: left_wrist_yaw_joint
#   24: right_shoulder_pitch_joint
#   25: right_shoulder_roll_joint
#   26: right_shoulder_yaw_joint
#   27: right_elbow_joint
#   28: right_wrist_roll_joint
#   29: right_wrist_pitch_joint
#   30: right_wrist_yaw_joint
RESTRICTED_JOINT_RANGE = (
    # Left hip.
    (-2.4526, 2.77542),  # left_hip_pitch_joint
    (-0.467441, 2.16886),  # left_hip_roll_joint
    (-2.827, 2.827),  # left_hip_yaw_joint
    # Left.
    (-0.08725, 2.53025),  # left_knee_joint
    # Left ankle.
    (-0.349066, 0.296706),  # left_ankle_roll_joint
    (-1.13446, 0.610865),  # left_ankle_pitch_joint
    # Right hip.
    (-2.4526, 2.77542),  # right_hip_pitch_joint
    (-2.16886, 0.467441),  # right_hip_roll_joint
    (-2.827, 2.827),  # right_hip_yaw_joint
    # Right.
    (-0.08725, 2.53025),  # right_knee_joint
    # Right ankle.
    (-0.296706, 0.349066),  # right_ankle_roll_joint
    (-1.13446, 0.610865),  # right_ankle_pitch_joint
    # Waist.
    (-1.7453, 1.7453),  # waist_yaw_joint
    (-0.5236, 0.5236),  # waist_roll_joint
    (-0.43633, 0.5236),  # waist_pitch_joint
    # Head.
    (-0.5236, 0.83775),  # head_pitch_joint
    (-1.7453, 1.7453),  # head_yaw_joint
    # Left shoulder.
    (-2.618, 1.833),  # left_shoulder_pitch_joint
    (-0.517, 2.494),  # left_shoulder_roll_joint
    (-2.618, 2.618),  # left_shoulder_yaw_joint
    # Left.
    (-0.986, 3.071),  # left_elbow_joint
    # Left wrist.
    (-2.618, 2.618),  # left_wrist_roll_joint
    (-0.576, 0.576),  # left_wrist_pitch_joint
    (-1.22, 1.22),  # left_wrist_yaw_joint
    # Right shoulder.
    (-2.618, 1.833),  # right_shoulder_pitch_joint
    (-2.494, 0.517),  # right_shoulder_roll_joint
    (-2.618, 2.618),  # right_shoulder_yaw_joint
    # Right.
    (-0.986, 3.071),  # right_elbow_joint
    # Right wrist.
    (-2.618, 2.618),  # right_wrist_roll_joint
    (-0.576, 0.576),  # right_wrist_pitch_joint
    (-1.22, 1.22),  # right_wrist_yaw_joint
)
