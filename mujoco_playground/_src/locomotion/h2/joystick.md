# H2 joystick task

Same task and rewards as G1 (`g1/joystick.py`), except:

- the policy drives 29 joints (head held at the keyframe pose);
- per-joint action scale `action_scale` (0.25) × effort limit / kp [rad per unit action], as in
  unitree_rl_mjlab (G1: 0.5 rad for all joints);
- the foot sites are at the sole, so foot-height terms measure sole height (G1: ankle, 0.037 m above the sole);
- orientation target `[0, 0, 1]` (G1: `[0.073, 0, 1]`, its forward-leaning default torso);
- `torques`, `action_rate`, `energy`, `dof_acc` are 0, as in upstream Playground (with the G1 values the
  policy learned to fall: ~−0.8 net reward per step against a one-off −100 termination);
- `max_contact_force` 1100 N (G1: 500 N) and `njmax` for 31 joints.

Model details: [model.md](model.md). All other values: `default_config()`.

Per-step reward = Σ (scale × term) × `ctrl_dt`. Joint terms use the 29 policy joints.

## Rewards

| Reward | Scale | Computes |
|---|---|---|
| tracking_lin_vel | 1.0 | exp(−‖cmd_xy − v_xy‖² / σ), pelvis local velocity |
| tracking_ang_vel | 0.75 | exp(−(cmd_yaw − ω_z)² / σ), pelvis gyro |
| lin_vel_z | 0.0 | v_z² of torso + v_z² of pelvis |
| ang_vel_xy | −0.25 | ‖ω_xy‖², torso |
| orientation | −2.5 | ‖torso z-axis − [0, 0, 1]‖² |
| base_height | 0.0 | (root z − base_height_target)² |
| torques | 0.0 | Σ\|τ\| |
| action_rate | 0.0 | Σ(a − a_prev)² |
| energy | 0.0 | Σ\|q̇\|·\|τ\| |
| dof_acc | 0.0 | Σq̈² |
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

1. **`max_foot_height` 0.15** is now real sole clearance (unitree_rl_mjlab: 0.10).
2. **Effort penalties** (`torques`, `energy`, ...): re-enable by fine-tuning from a walking checkpoint, about
   5x smaller than G1's (H2 needs Σ|τ| = 337 vs 64 Nm to hold its pose).
3. **`joint_deviation_hip`** is 2–4x weaker than G1's effective value (G1's (n, 1) index arrays broadcast its
   hip weights into a 4x4 matrix; H2's are fixed). Raise toward −0.5 to −1.0 if the hips drift.
4. **IMU sites** should use the `H2.urdf` positions (see [model.md](model.md)).
5. **Torso mass randomization (±1 kg)** is ±5% of H2's 18.5 kg torso (G1: ±13%). Use about ±2.5 kg.
6. Optional: gait frequency U(1.25, 1.5) Hz (hardcoded in `reset`) may be fast for H2's longer legs;
   `soft_joint_pos_limit_factor` 0.95 (unitree_rl_mjlab: 0.9).

`command_config` a/b are read but unused by `sample_command`, as in G1.
