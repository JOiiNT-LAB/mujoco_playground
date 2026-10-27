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

31 model joints, in qpos/ctrl order:

- legs, left then right: hip pitch, hip roll, hip yaw, knee, ankle roll, ankle pitch
- waist: yaw, roll, pitch
- head: pitch, yaw
- arms, left then right: shoulder pitch, roll, yaw, elbow, wrist roll, pitch, yaw

The policy drives 29 joints; the head is held at the keyframe pose. Policy index `p` is G1's joint `p`, except
the ankles (H2: 4/10 roll, 5/11 pitch; G1: the reverse).

Other differences from G1: the hip pitch axis is tilted 30°, the forearm points forward at elbow = 0, and the
foot site is 0.058 m above the sole (G1: 0.037 m).

## Model

- Mass 75.59 kg (G1: 33.3 kg): waist + torso 18.98, legs 16.12 / 16.08, pelvis 8.55, arms 6.37 / 6.37, head 3.12.
- PD gains `[MJLAB]`; kd is applied as joint damping (as in G1) plus 0.05 passive damping `[MJCF]`.

  | Joints | kp | kd |
  |---|---|---|
  | Hip pitch/roll/yaw, knee | 200 | 4 |
  | Ankle roll, ankle pitch | 40 | 2 |
  | Waist | 150 | 3 |
  | Shoulders, elbow, wrist roll | 40 | 2 |
  | Wrist pitch/yaw | 20 | 1 |
  | Head | 20 | 1 (no official value) |

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
