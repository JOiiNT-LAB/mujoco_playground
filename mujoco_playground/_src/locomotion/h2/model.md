# Unitree H2 model

H2 port of the G1 joystick task (`H2JoystickFlatTerrain`, `H2JoystickRoughTerrain`). Rewards and training
notes: [joystick.md](joystick.md).

## Files

| File | Content |
|---|---|
| `xmls/h2_mjx_feetonly.xml` | Robot: bodies, joints, PD gains, colliders, contact pairs, actuators, sensors |
| `xmls/scene_mjx_feetonly_flat_terrain.xml` | Flat scene and keyframes |
| `xmls/scene_mjx_feetonly_rough_terrain.xml` | Heightfield scene (as G1), keyframes 4.5 cm higher |
| `xmls/sensor.xml` | Contact sensors (same as G1) |

Meshes come from `unitree_h2/assets` in the menagerie fork
[JOiiNT-LAB/mujoco_menagerie](https://github.com/JOiiNT-LAB/mujoco_menagerie), pinned by
`MENAGERIE_COMMIT_SHA` in `mjx_env.py`.

## Sources

| Tag | Source | Used for |
|---|---|---|
| `[MJCF]` | [unitree_mujoco `h2_mujoco.xml`](https://github.com/unitreerobotics/unitree_mujoco/blob/main/unitree_robots/h2/h2_mujoco.xml) | Kinematics, inertias, joint ranges, torque limits, armature, friction loss |
| `[URDF]` | [unitree_ros `H2.urdf`](https://github.com/unitreerobotics/unitree_ros/blob/master/robots/h2_description/H2.urdf) | Pelvis inertial (missing in `[MJCF]`) |
| `[MJLAB]` | [unitree_rl_mjlab `h2_constants.py`](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/main/src/assets/robots/unitree_h2/h2_constants.py) | PD gains, `knees_bent` joint values |

## Joints

31 model joints (`#` = qpos/ctrl order). The policy drives 29 (`Policy #`); the head is held at the keyframe
pose. Policy index `p` is G1's joint `p`, except the ankles (H2: 4/10 roll, 5/11 pitch; G1: the reverse).
The restricted range (`RESTRICTED_JOINT_RANGE`, used only with `restricted_joint_range=True`) is currently the
full range.

| # | Policy # | Joint | Range [rad] | Restricted range [rad] | Torque limit [Nm] | kp / kd |
|---|---|---|---|---|---|---|
| 0 | 0 | left_hip_pitch | [-2.4526, 2.77542] | [-2.4526, 2.77542] | 360 | 200 / 4 |
| 1 | 1 | left_hip_roll | [-0.467441, 2.16886] | [-0.467441, 2.16886] | 360 | 200 / 4 |
| 2 | 2 | left_hip_yaw | [-2.827, 2.827] | [-2.827, 2.827] | 360 | 200 / 4 |
| 3 | 3 | left_knee | [-0.08725, 2.53025] | [-0.08725, 2.53025] | 360 | 200 / 4 |
| 4 | 4 | left_ankle_roll | [-0.349066, 0.296706] | [-0.349066, 0.296706] | 19 | 40 / 2 |
| 5 | 5 | left_ankle_pitch | [-1.13446, 0.610865] | [-1.13446, 0.610865] | 66.88 | 40 / 2 |
| 6 | 6 | right_hip_pitch | [-2.4526, 2.77542] | [-2.4526, 2.77542] | 360 | 200 / 4 |
| 7 | 7 | right_hip_roll | [-2.16886, 0.467441] | [-2.16886, 0.467441] | 360 | 200 / 4 |
| 8 | 8 | right_hip_yaw | [-2.827, 2.827] | [-2.827, 2.827] | 360 | 200 / 4 |
| 9 | 9 | right_knee | [-0.08725, 2.53025] | [-0.08725, 2.53025] | 360 | 200 / 4 |
| 10 | 10 | right_ankle_roll | [-0.296706, 0.349066] | [-0.296706, 0.349066] | 19 | 40 / 2 |
| 11 | 11 | right_ankle_pitch | [-1.13446, 0.610865] | [-1.13446, 0.610865] | 66.88 | 40 / 2 |
| 12 | 12 | waist_yaw | [-1.7453, 1.7453] | [-1.7453, 1.7453] | 120 | 150 / 3 |
| 13 | 13 | waist_roll | [-0.5236, 0.5236] | [-0.5236, 0.5236] | 180 | 150 / 3 |
| 14 | 14 | waist_pitch | [-0.43633, 0.5236] | [-0.43633, 0.5236] | 180 | 150 / 3 |
| 15 | – | head_pitch | [-0.5236, 0.83775] | [-0.5236, 0.83775] | 50 | 20 / 1 |
| 16 | – | head_yaw | [-1.7453, 1.7453] | [-1.7453, 1.7453] | 50 | 20 / 1 |
| 17 | 15 | left_shoulder_pitch | [-2.618, 1.833] | [-2.618, 1.833] | 130 | 40 / 2 |
| 18 | 16 | left_shoulder_roll | [-0.517, 2.494] | [-0.517, 2.494] | 60 | 40 / 2 |
| 19 | 17 | left_shoulder_yaw | [-2.618, 2.618] | [-2.618, 2.618] | 60 | 40 / 2 |
| 20 | 18 | left_elbow | [-0.986, 3.071] | [-0.986, 3.071] | 60 | 40 / 2 |
| 21 | 19 | left_wrist_roll | [-2.618, 2.618] | [-2.618, 2.618] | 60 | 40 / 2 |
| 22 | 20 | left_wrist_pitch | [-0.576, 0.576] | [-0.576, 0.576] | 10 | 20 / 1 |
| 23 | 21 | left_wrist_yaw | [-1.22, 1.22] | [-1.22, 1.22] | 10 | 20 / 1 |
| 24 | 22 | right_shoulder_pitch | [-2.618, 1.833] | [-2.618, 1.833] | 130 | 40 / 2 |
| 25 | 23 | right_shoulder_roll | [-2.494, 0.517] | [-2.494, 0.517] | 60 | 40 / 2 |
| 26 | 24 | right_shoulder_yaw | [-2.618, 2.618] | [-2.618, 2.618] | 60 | 40 / 2 |
| 27 | 25 | right_elbow | [-0.986, 3.071] | [-0.986, 3.071] | 60 | 40 / 2 |
| 28 | 26 | right_wrist_roll | [-2.618, 2.618] | [-2.618, 2.618] | 60 | 40 / 2 |
| 29 | 27 | right_wrist_pitch | [-0.576, 0.576] | [-0.576, 0.576] | 10 | 20 / 1 |
| 30 | 28 | right_wrist_yaw | [-1.22, 1.22] | [-1.22, 1.22] | 10 | 20 / 1 |

Gains: `[MJLAB]` (head: no official value). kd is applied as joint damping, as in G1, plus 0.05 passive damping
`[MJCF]`. Ranges and torque limits: `[MJCF]`.

Other differences from G1: the hip pitch axis is tilted 30°, the forearm points forward at elbow = 0, and the
foot site is at the sole (G1: at the ankle, 0.037 m above the sole).

## Model

- Mass 75.59 kg (G1: 33.3 kg): waist + torso 18.98, legs 16.12 / 16.08, pelvis 8.55, arms 6.37 / 6.37, head 3.12.
- Armature 0.01, friction loss 0.2 on all joints.
- Contacts as in G1: foot–floor (friction 0.6), hand–thigh, foot–foot.
- Keyframes: `knees_bent` (task default, `[MJLAB]` joint values) and `home` (knee 0.3, legs solved for flat feet).

## Open points

- **IMU sites** are at the link origins as in `[MJCF]`. `[URDF]` places them at pelvis `(-0.055, 0, -0.0589)`
  and torso `(-0.06382, -0.06538, 0.29588)` (no rotation).
- **Head:** no official gains (`[MJLAB]` welds the head). The head wobbles when the body is shaken.
- **Thigh, shin, hand colliders and palm sites** are sized from the meshes (only used by contact sensors).
- **`knees_bent`:** the soles are about 2° from flat with 7° toe-out on these kinematics.
- **Unitree files disagree** (`[MJCF]` kept): arm torque limits 130 / 60 Nm vs 54 Nm `[MJLAB]`, wrist pitch/yaw
  10 vs 25 Nm, wrist pitch range ±0.576 vs ±0.436. Check against the real robot before deployment.
