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

# Joint indices. Model index i (0-30) is qpos[7 + i], qvel[6 + i], ctrl[i]
# (order in RESTRICTED_JOINT_RANGE). The joystick policy drives the 29 joints
# without the head: policy index p = model index for p <= 14, p + 2 for the
# arms. p is the same joint as G1's index p, except the ankles: 4/10 are roll
# and 5/11 pitch (G1: pitch, roll).

# Joint groups used by joystick.py, as in g1/joystick.py.
WAIST_JOINT_NAMES = ["waist_yaw", "waist_roll", "waist_pitch"]
ARM_JOINT_NAMES = [
    "shoulder_roll",
    "shoulder_yaw",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
]
HIP_JOINT_NAMES = ["hip_roll", "hip_yaw"]
KNEE_JOINT_NAMES = ["knee"]
# Held at the keyframe pose; not controlled by the joystick policy.
HEAD_JOINT_NAMES = ["head_pitch", "head_yaw"]

# UNSURE: these are the full joint ranges (no restriction yet). Only used when
# restricted_joint_range=True.
RESTRICTED_JOINT_RANGE = (
    # Left leg.
    (-2.4526, 2.77542),  #  0: left_hip_pitch_joint
    (-0.467441, 2.16886),  #  1: left_hip_roll_joint
    (-2.827, 2.827),  #  2: left_hip_yaw_joint
    (-0.08725, 2.53025),  #  3: left_knee_joint
    (-0.349066, 0.296706),  #  4: left_ankle_roll_joint
    (-1.13446, 0.610865),  #  5: left_ankle_pitch_joint
    # Right leg.
    (-2.4526, 2.77542),  #  6: right_hip_pitch_joint
    (-2.16886, 0.467441),  #  7: right_hip_roll_joint
    (-2.827, 2.827),  #  8: right_hip_yaw_joint
    (-0.08725, 2.53025),  #  9: right_knee_joint
    (-0.296706, 0.349066),  # 10: right_ankle_roll_joint
    (-1.13446, 0.610865),  # 11: right_ankle_pitch_joint
    # Waist.
    (-1.7453, 1.7453),  # 12: waist_yaw_joint
    (-0.5236, 0.5236),  # 13: waist_roll_joint
    (-0.43633, 0.5236),  # 14: waist_pitch_joint
    # Head.
    (-0.5236, 0.83775),  # 15: head_pitch_joint
    (-1.7453, 1.7453),  # 16: head_yaw_joint
    # Left arm.
    (-2.618, 1.833),  # 17: left_shoulder_pitch_joint
    (-0.517, 2.494),  # 18: left_shoulder_roll_joint
    (-2.618, 2.618),  # 19: left_shoulder_yaw_joint
    (-0.986, 3.071),  # 20: left_elbow_joint
    (-2.618, 2.618),  # 21: left_wrist_roll_joint
    (-0.576, 0.576),  # 22: left_wrist_pitch_joint
    (-1.22, 1.22),  # 23: left_wrist_yaw_joint
    # Right arm.
    (-2.618, 1.833),  # 24: right_shoulder_pitch_joint
    (-2.494, 0.517),  # 25: right_shoulder_roll_joint
    (-2.618, 2.618),  # 26: right_shoulder_yaw_joint
    (-0.986, 3.071),  # 27: right_elbow_joint
    (-2.618, 2.618),  # 28: right_wrist_roll_joint
    (-0.576, 0.576),  # 29: right_wrist_pitch_joint
    (-1.22, 1.22),  # 30: right_wrist_yaw_joint
)
