# H2 joystick task

Same task, rewards and parameters as G1 (`g1/joystick.py`), except: the policy drives 29 joints (head held at
the keyframe pose), `max_contact_force` is 1100 N (G1: 500 N, scaled for the ~2.2x heavier robot) and
`njmax` covers 31 joints. Model details: [model.md](model.md). All other values: `default_config()`.

Per-step reward = Σ (scale × term) × `ctrl_dt`. Joint terms use the 29 policy joints.

## Rewards

| Reward | Scale | Computes |
|---|---|---|
| tracking_lin_vel | 1.0 | exp(−‖cmd_xy − v_xy‖² / σ), pelvis local velocity |
| tracking_ang_vel | 0.75 | exp(−(cmd_yaw − ω_z)² / σ), pelvis gyro |
| lin_vel_z | 0.0 | v_z² of torso + v_z² of pelvis |
| ang_vel_xy | −0.25 | ‖ω_xy‖², torso |
| orientation | −2.5 | ‖torso z-axis − [0.073, 0, 1]‖² |
| base_height | 0.0 | (root z − base_height_target)² |
| torques | −0.0001 | Σ\|τ\| |
| action_rate | −0.02 | Σ(a − a_prev)² |
| energy | −0.003 | Σ\|q̇\|·\|τ\| |
| dof_acc | −2e-6 | Σq̈² |
| feet_clearance | 0.0 | Σ\|z_foot − max_foot_height\|·√‖v_foot_xy‖ |
| feet_air_time | 2.0 | Σ clip(t_air − 0.2, max 0.3), on first contact |
| feet_slip | −0.25 | Σ‖v_foot_xy‖ while in contact |
| feet_height | 0.0 | Σ(swing_peak / max_foot_height − 1)², on first contact |
| feet_phase | 1.0 | exp(−Σ(z_foot − z_ref(phase))² / 0.01), only when moving or commanded |
| alive | 0.0 | 1 |
| stand_still | −1.5 | Σ\|q − q_default\| when ‖cmd‖ < 0.01 |
| termination | −100.0 | 1 on termination |
| collision | −0.1 | Hand–thigh contact, either side |
| contact_force | −0.01 | Σ clip(\|F_z\| − max_contact_force, 0), foot force sensors |
| joint_deviation_knee | −0.1 | Σ\|q_knee − q_default\| |
| joint_deviation_hip | −0.25 | Σ\|q_hip − q_default\| (roll, yaw); roll weight 0 when cmd_y > 0.1 |
| dof_pos_limits | −1.0 | Violation of soft joint limits |
| pose | −0.2 | Σ w·(q − q_default)²; w = 0.01 for hip pitch and knee, 1.0 otherwise |

Termination: torso upside down, foot–foot or foot–shin contact, or NaN in qpos/qvel.

## To address for H2

G1 values that likely hinder H2 training, from comparing both robots at their default pose:

1. **`feet_phase` can't be earned fully.** The target foot height starts at 0, but the foot site is 0.058 m
   above the floor in stance (G1: 0.037 m). A perfect gait scores at most 0.43 (G1: 0.80). Subtract the stance
   site height in `_reward_feet_phase` (and in `feet_clearance` / `feet_height` if enabled), or move the foot
   site to the sole.
2. **`torques` and `energy` penalties are ~5x larger.** Holding the default pose takes Σ|τ| = 337 Nm
   (G1: 64 Nm). Scale them down ~3–5x; compare `reward/torques` and `reward/energy` in TensorBoard.
3. **Orientation target `[0.073, 0, 1]`** is G1's forward-leaning torso. H2's default torso is upright:
   use `[0, 0, 1]`.
4. **`action_scale = 0.5` for all joints.** [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/main/src/assets/robots/unitree_h2/h2_constants.py) uses 0.25 × torque limit / kp per joint: legs 0.45,
   ankle pitch 0.41, waist 0.2, arms 0.34, wrists 0.31, ankle roll 0.12. With 0.5, exploration saturates the
   19 Nm ankle roll.
5. **`joint_deviation_hip` is 2–4x weaker than G1's effective value.** G1's index arrays have shape (n, 1),
   which broadcasts its hip weights into a 4x4 matrix; H2's are fixed. Raise toward −0.5 to −1.0 if the hips
   drift.
6. **IMU sites** should use the `H2.urdf` positions (see [model.md](model.md)); they affect the pelvis
   velocity and gyro observations.
7. **Torso mass randomization (±1 kg)** is ±5% of H2's 18.5 kg torso (G1: ±13%). Use about ±2.5 kg.
8. Optional: the gait frequency (U(1.25, 1.5) Hz, hardcoded in `reset`) may be fast for H2's longer legs;
   `soft_joint_pos_limit_factor` is 0.95 (unitree_rl_mjlab: 0.9).

`command_config` a/b are read but unused by `sample_command`, as in G1.
