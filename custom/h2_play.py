"""Interactive and headless evaluation of H2 joystick policies.

Runs the H2 joystick task in native MuJoCo on the CPU and reproduces the
observations, actions, pushes and terminations of `h2/joystick.py`.

Policies (--checkpoint):
  * Brax PPO: a checkpoint step folder, or a run's checkpoints/ folder (latest step).
  * TorchScript (.pt / .jit): input float32 [1, 103] raw observation, output [1, 29]
    action. Observation normalization must be part of the exported model.
All actions are clipped to [-1, 1].

Environment config: the env is built from the run's checkpoints/config.json when
present, so ranges and scales match training. Flags take precedence over it.

Episodes are reproducible: for a given --seed, episode k has the same start state,
commands, pushes, observation noise and gait frequency for every policy and flag
combination. --start_episode k replays headless episode k in the GUI.

GUI keys:
  Up / Down          forward speed +/- 0.1 m/s
  Left / Right       turn rate +/- 0.1 rad/s
  PageUp / PageDown  lateral speed +/- 0.1 m/s
  Home               zero command
  8 / 9              gait frequency -/+ 0.05 Hz
  6 / 7              manual push speed -/+ 0.25 m/s
  Insert             push in a random direction
  Delete             toggle automatic pushes
  Enter              reset
  End                pause / resume
Mouse: double-click the pelvis, then Ctrl + right-drag to push, Ctrl + left-drag to twist.
"""

import argparse
import dataclasses
import datetime
import json
import os
import tempfile
import time


def parse_args():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--checkpoint", required=True,
                 help="Brax checkpoint step folder, a run's checkpoints/ folder, or a TorchScript file.")
  p.add_argument("--env_name", default="H2JoystickFlatTerrain")
  p.add_argument("--env_config", default=None,
                 help="Env config JSON. Default: config.json next to the checkpoint.")
  p.add_argument("--default_env_config", action="store_true",
                 help="Use the env defaults instead of the run's config.json.")
  p.add_argument("--cmd", type=float, nargs=3, default=[0.5, 0.0, 0.0],
                 metavar=("VX", "VY", "YAW"), help="Fixed command (without --random_commands).")
  p.add_argument("--random_commands", action="store_true",
                 help="Sample commands as in training, every 10 s.")
  p.add_argument("--pushes", action="store_true", help="Enable automatic pushes.")
  p.add_argument("--push_mag_range", type=float, nargs=2, default=None,
                 help="Push speed range [m/s]. Default: env config.")
  p.add_argument("--push_interval_range", type=float, nargs=2, default=None,
                 help="Time between pushes [s]. Default: env config.")
  p.add_argument("--manual_push", type=float, default=1.0, help="Insert-key push speed [m/s].")
  p.add_argument("--obs_noise", action="store_true", help="Add the training observation noise.")
  p.add_argument("--random_init", action="store_true", help="Randomize the start state as in training.")
  p.add_argument("--gait_freq", type=float, default=None,
                 help="Fixed gait frequency [Hz]. Default: sampled from the env config range.")
  p.add_argument("--allow_ood", action="store_true",
                 help="Allow commands and gait frequencies outside the training ranges.")
  p.add_argument("--episode_sec", type=float, default=None,
                 help="Episode length [s]. Default: unlimited in the GUI, 20 s headless.")
  p.add_argument("--headless", type=int, default=0, metavar="N",
                 help="Run N episodes without a window and print statistics.")
  p.add_argument("--out", default=None, help="Write headless results to this JSON file.")
  p.add_argument("--start_episode", type=int, default=1, help="Number of the first episode.")
  p.add_argument("--speed", type=float, default=1.0, help="GUI playback speed.")
  p.add_argument("--reset_delay", type=float, default=1.5,
                 help="GUI: seconds before resetting after a fall.")
  p.add_argument("--no_auto_reset", action="store_true", help="GUI: wait for Enter after a fall.")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--jax_gpu", action="store_true", help="Run a Brax policy on the GPU.")
  return p.parse_args()


# JAX reads JAX_PLATFORMS at import time.
ARGS = parse_args()
if not ARGS.jax_gpu:
  os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# pylint: disable=g-import-not-at-top,wrong-import-position,protected-access
import jax
import jax.numpy as jp
import mujoco
import mujoco.viewer
import numpy as np
from brax.training import checkpoint as brax_checkpoint
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks
from etils import epath
from mujoco_playground import registry

# Gait frequency limits for keys 8 / 9 with --allow_ood.
OOD_GAIT_FREQ_RANGE = (0.5, 3.0)
# Gait frequency range of runs saved before it became a config entry.
LEGACY_GAIT_FREQ_RANGE = [1.25, 1.5]

# GLFW key codes.
KEY_ENTER, KEY_INSERT, KEY_DELETE = 257, 260, 261
KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP = 262, 263, 264, 265
KEY_PAGE_UP, KEY_PAGE_DOWN, KEY_HOME, KEY_END = 266, 267, 268, 269
KEY_6, KEY_7, KEY_8, KEY_9 = 54, 55, 56, 57


# Policies

def brax_policy(ckpt: epath.Path):
  """Deterministic Brax PPO policy.

  Null entries are removed from the network config first, since
  brax_checkpoint.load_config fails on them.
  """
  cfg = json.loads((ckpt / "ppo_network_config.json").read_text())
  cfg["network_factory_kwargs"] = {
      k: v for k, v in cfg["network_factory_kwargs"].items() if v is not None}
  with tempfile.TemporaryDirectory() as tmp:
    cfg_path = epath.Path(tmp) / "ppo_network_config.json"
    cfg_path.write_text(json.dumps(cfg))
    config = brax_checkpoint.load_config(cfg_path)
  network = brax_checkpoint.get_network(config, ppo_networks.make_ppo_networks)
  params = ppo_checkpoint.load(ckpt)
  fn = jax.jit(ppo_networks.make_inference_fn(network)(params, deterministic=True))
  # The policy only reads "state", but the normalizer expects every key.
  extra = {k: jp.zeros(v["shape"]) for k, v in cfg["observation_size"].items() if k != "state"}
  key = jax.random.PRNGKey(0)

  def act(obs: np.ndarray) -> np.ndarray:
    action, _ = fn({"state": jp.asarray(obs), **extra}, key)
    return np.asarray(action, dtype=np.float64)

  return act


def torchscript_policy(path: epath.Path):
  import torch
  model = torch.jit.load(str(path), map_location="cpu").eval()

  def act(obs: np.ndarray) -> np.ndarray:
    with torch.no_grad():
      return model(torch.from_numpy(obs)[None]).numpy()[0].astype(np.float64)

  return act


def load_policy(path: str):
  """Returns (act, description), with act mapping an observation to an action in [-1, 1]."""
  path = epath.Path(path).resolve()
  if path.is_file() and path.suffix in (".pt", ".jit"):
    act, desc = torchscript_policy(path), f"TorchScript {path}"
  else:
    if not (path / "ppo_network_config.json").exists():
      steps = sorted(p for p in path.iterdir() if p.is_dir() and p.name.isdigit())
      if not steps:
        raise FileNotFoundError(f"No checkpoint found in {path}")
      path = steps[-1]
    act, desc = brax_policy(path), f"Brax PPO {path}"
  return (lambda obs: np.clip(act(obs), -1.0, 1.0)), desc


# Environment config

def flatten(d: dict, prefix: str = ""):
  for k, v in d.items():
    if isinstance(v, dict):
      yield from flatten(v, f"{prefix}{k}.")
    else:
      yield f"{prefix}{k}", v


def env_overrides(args) -> dict:
  """Config overrides that rebuild the env with the run's saved settings."""
  overrides = {"impl": "jax"}
  if args.default_env_config:
    print("env config: env defaults")
    return overrides
  path = args.env_config
  if path is None:
    ck = epath.Path(args.checkpoint).resolve()
    path = next((str(c) for c in (ck / "config.json", ck.parent / "config.json")
                 if c.exists()), None)
  if path is None:
    print("env config: no config.json found, using the env defaults")
    return overrides

  saved = dict(flatten(json.loads(epath.Path(path).read_text())))
  default = dict(flatten(registry.get_default_config(args.env_name).to_dict()))
  diff = {k: v for k, v in saved.items() if k in default and k != "impl" and v != default[k]}
  missing = sorted(k for k in default if k not in saved)
  if "gait_frequency" in missing:
    diff["gait_frequency"] = LEGACY_GAIT_FREQ_RANGE
    missing.remove("gait_frequency")

  print(f"env config: {path}")
  for k, v in diff.items():
    print(f"  {k} = {v} (default {default[k]})")
  if missing:
    print(f"  not in the saved config, defaults used: {', '.join(missing)}")
  overrides.update(diff)
  return overrides


# Simulation

@dataclasses.dataclass
class Push:
  time: float
  speed: float
  label: str             # direction relative to the robot heading
  direction: np.ndarray  # unit vector in the world x/y plane


class H2Sim:
  """Single H2 robot in native MuJoCo, stepped like Joystick.step."""

  def __init__(self, env, policy, args):
    self.env, self.policy, self.args = env, policy, args
    self.m = env.mj_model
    self.d = mujoco.MjData(self.m)
    self.dt = env.dt
    self.n_substeps = env.n_substeps
    cfg = env._config

    self.pid = np.asarray(env._policy_joint_ids)
    self.action_scale = np.asarray(env._action_scale)
    self.default_ctrl = np.asarray(env._default_ctrl)
    self.default_pose = np.asarray(env._default_pose)
    self.init_q = np.asarray(env._init_q)
    self.pelvis_imu = self.m.site("imu_in_pelvis").id
    self.pelvis_id = self.m.body("pelvis").id
    self.torso_id = self.m.body("torso_link").id
    self.mass = float(self.m.body_mass.sum())

    self.noise_level = cfg.noise_config.level
    self.noise = cfg.noise_config.scales
    self.gait_range = tuple(cfg.gait_frequency)
    self.cmd_low = np.array([cfg.lin_vel_x[0], cfg.lin_vel_y[0], cfg.ang_vel_yaw[0]])
    self.cmd_high = np.array([cfg.lin_vel_x[1], cfg.lin_vel_y[1], cfg.ang_vel_yaw[1]])
    self.push_mag_range = args.push_mag_range or list(cfg.push_config.magnitude_range)
    self.push_interval_range = args.push_interval_range or list(cfg.push_config.interval_range)
    # Extra fall check, not used in training: pelvis below half its start height.
    self.fall_height = 0.5 * self.init_q[2]

    def sensor(name):
      i = self.m.sensor(name).id
      return slice(self.m.sensor_adr[i], self.m.sensor_adr[i] + self.m.sensor_dim[i])
    self.s_gyro = sensor("gyro_pelvis")
    self.s_linvel = sensor("local_linvel_pelvis")
    self.s_up_torso = sensor("upvector_torso")
    self.s_feet = sensor("right_foot_left_foot_found")
    self.s_lfoot_rshin = sensor("left_foot_right_shin_found")
    self.s_rfoot_lshin = sensor("right_foot_left_shin_found")
    self.s_foot_up = [sensor("left_foot_upvector"), sensor("right_foot_upvector")]
    self.foot_geoms = [self.m.geom("left_foot").id, self.m.geom("right_foot").id]

    self.command = np.array(args.cmd, dtype=np.float64)
    self.auto_push = args.pushes
    self.manual_push = args.manual_push
    self.episode = args.start_episode - 1

  def heading(self) -> float:
    w, x, y, z = self.d.qpos[3:7]
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

  def velocity(self) -> np.ndarray:
    """Pelvis forward, lateral and yaw velocity."""
    return np.array([*self.d.sensordata[self.s_linvel][:2], self.d.sensordata[self.s_gyro][2]])

  def sample_command(self) -> np.ndarray:
    cmd = self.rng_cmd.uniform(self.cmd_low, self.cmd_high)
    return np.zeros(3) if self.rng_cmd.uniform() < 0.1 else cmd

  def set_gait_freq(self, freq: float):
    lo, hi = OOD_GAIT_FREQ_RANGE if self.args.allow_ood else self.gait_range
    self.gait_freq = float(np.clip(freq, lo, hi))
    self.phase_dt = 2 * np.pi * self.dt * self.gait_freq

  def reset(self):
    self.episode += 1
    # One random stream per episode and purpose, so that changing one setting
    # does not shift the random numbers of the others.
    self.rng_init, self.rng_cmd, self.rng_push, self.rng_noise, rng_gait = (
        np.random.default_rng([self.args.seed, self.episode, i]) for i in range(5))

    m, d, rng = self.m, self.d, self.rng_init
    mujoco.mj_resetData(m, d)
    qpos = self.init_q.copy()
    qvel = np.zeros(m.nv)
    if self.args.random_init:
      qpos[0:2] += rng.uniform(-0.5, 0.5, 2)
      yaw_quat, new_quat = np.zeros(4), np.zeros(4)
      mujoco.mju_axisAngle2Quat(yaw_quat, np.array([0.0, 0.0, 1.0]), rng.uniform(-3.14, 3.14))
      mujoco.mju_mulQuat(new_quat, qpos[3:7].copy(), yaw_quat)
      qpos[3:7] = new_quat
      qpos[7 + self.pid] *= rng.uniform(0.5, 1.5, len(self.pid))
      qvel[0:6] = rng.uniform(-0.5, 0.5, 6)
    d.qpos[:] = qpos
    d.qvel[:] = qvel
    d.ctrl[:] = qpos[7:]
    mujoco.mj_forward(m, d)

    self.set_gait_freq(self.args.gait_freq or rng_gait.uniform(*self.gait_range))
    self.phase = np.array([0.0, np.pi])
    self.last_act = np.zeros(len(self.pid))
    if self.args.random_commands:
      self.command = self.sample_command()

    self.steps = 0
    self.done = False
    self.cause = ""
    self.pushes = []
    self.next_push_t = self.rng_push.uniform(*self.push_interval_range)

    self.err_sq = np.zeros(3)     # squared tracking error (vx, vy, yaw), summed
    self.energy = 0.0             # actuator work |force * velocity| dt [J]
    self.distance = 0.0           # pelvis path length in the x/y plane [m]
    self.target_rate_sq = 0.0     # squared joint-target change per step [rad^2], summed
    self.foot_samples = []        # (heel z, toe z, tilt deg) per foot and step
    self.history = []             # (time, cmd vx, vx, cmd yaw, yaw rate), last 5 s
    self.obs = self.build_obs()

  def build_obs(self) -> np.ndarray:
    """The 103-value `state` observation of Joystick._get_obs."""
    d = self.d
    gyro = d.sensordata[self.s_gyro].copy()
    gravity = d.site_xmat[self.pelvis_imu].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
    joint_angles = d.qpos[7:][self.pid].copy()
    joint_vel = d.qvel[6:][self.pid].copy()
    linvel = d.sensordata[self.s_linvel].copy()
    if self.args.obs_noise:
      def noise(x, scale):
        return self.noise_level * scale * (2 * self.rng_noise.uniform(size=x.shape) - 1)
      gyro += noise(gyro, self.noise.gyro)
      gravity += noise(gravity, self.noise.gravity)
      joint_angles += noise(joint_angles, self.noise.joint_pos)
      joint_vel += noise(joint_vel, self.noise.joint_vel)
      linvel += noise(linvel, self.noise.linvel)
    return np.concatenate([
        linvel, gyro, gravity, self.command,
        joint_angles - self.default_pose, joint_vel, self.last_act,
        np.cos(self.phase), np.sin(self.phase),
    ]).astype(np.float32)

  def push(self, speed: float, theta: float):
    """Adds `speed` m/s to the base velocity in direction `theta`, as the env does."""
    direction = np.array([np.cos(theta), np.sin(theta)])
    self.d.qvel[0:2] += speed * direction
    rel = np.degrees((theta - self.heading() + np.pi) % (2 * np.pi) - np.pi)
    label = ("forward" if abs(rel) < 45 else "backward" if abs(rel) > 135
             else "to the left" if rel > 0 else "to the right")
    self.pushes.append(Push(self.d.time, speed, label, direction))

  def manual_push_now(self):
    self.push(self.manual_push, np.random.uniform(0, 2 * np.pi))

  def step(self):
    action = self.policy(self.obs)

    if self.auto_push and self.d.time >= self.next_push_t:
      speed = self.rng_push.uniform(*self.push_mag_range)
      self.push(speed, self.rng_push.uniform(0, 2 * np.pi))
      self.next_push_t = self.d.time + self.rng_push.uniform(*self.push_interval_range)

    ctrl = self.default_ctrl.copy()
    ctrl[self.pid] += action * self.action_scale
    self.d.ctrl[:] = ctrl
    xy_before = self.d.qpos[0:2].copy()
    for _ in range(self.n_substeps):
      mujoco.mj_step(self.m, self.d)
      self.energy += np.abs(self.d.actuator_force * self.d.actuator_velocity).sum() * self.m.opt.timestep

    # Like the env, the new observation still uses the previous phase and action.
    self.obs = self.build_obs()
    self.check_termination()
    self.target_rate_sq += np.sum(((action - self.last_act) * self.action_scale) ** 2)
    self.phase = np.fmod(self.phase + self.phase_dt + np.pi, 2 * np.pi) - np.pi
    self.last_act = action
    self.steps += 1

    self.distance += float(np.linalg.norm(self.d.qpos[0:2] - xy_before))
    vel = self.velocity()
    self.err_sq += (self.command - vel) ** 2
    self.history.append((self.d.time, self.command[0], vel[0], self.command[2], vel[2]))
    del self.history[:-250]
    self.record_feet()
    if self.args.random_commands and self.steps % 500 == 0:
      self.command = self.sample_command()

  def record_feet(self):
    """Heel and toe height (bottom edges of the foot box) and foot tilt."""
    for g, s in zip(self.foot_geoms, self.s_foot_up):
      rot, c, size = self.d.geom_xmat[g].reshape(3, 3), self.d.geom_xpos[g], self.m.geom_size[g]
      heel = (c + rot @ np.array([-size[0], 0.0, -size[2]]))[2]
      toe = (c + rot @ np.array([size[0], 0.0, -size[2]]))[2]
      tilt = np.degrees(np.arccos(np.clip(self.d.sensordata[s][2], -1.0, 1.0)))
      self.foot_samples.append((heel, toe, tilt))

  def check_termination(self):
    sd = self.d.sensordata
    causes = []
    if sd[self.s_up_torso][2] < 0.0:
      causes.append("torso tilted past horizontal")
    if sd[self.s_feet][0] > 0:
      causes.append("feet touched each other")
    if sd[self.s_lfoot_rshin][0] > 0 or sd[self.s_rfoot_lshin][0] > 0:
      causes.append("foot hit the other shin")
    if np.isnan(self.d.qpos).any() or np.isnan(self.d.qvel).any():
      causes.append("NaN in the simulation")
    if not causes and self.d.qpos[2] < self.fall_height:
      causes.append(f"pelvis below {self.fall_height:.2f} m")
    if causes:
      self.done, self.cause = True, ", ".join(causes)

  # Reporting

  def last_push_info(self) -> str:
    if not self.pushes:
      return "no push this episode"
    p = self.pushes[-1]
    return f"{p.speed:.2f} m/s {p.label}, {self.d.time - p.time:.1f} s ago"

  def episode_result(self) -> dict:
    n = max(self.steps, 1)
    rms = np.sqrt(self.err_sq / n)
    t = max(self.d.time, 1e-9)
    feet = np.array(self.foot_samples) if self.foot_samples else np.full((1, 3), np.nan)
    return dict(
        episode=self.episode, time_s=round(self.d.time, 3), fell=self.done, cause=self.cause,
        pushes=len(self.pushes),
        fell_within_2s_of_push=bool(self.done and self.pushes
                                    and self.d.time - self.pushes[-1].time < 2.0),
        last_push=self.last_push_info(), gait_freq_hz=round(self.gait_freq, 3),
        rms_err_vx=rms[0], rms_err_vy=rms[1], rms_err_yaw=rms[2],
        mean_power_w=self.energy / t,
        cost_of_transport=(self.energy / (self.mass * 9.81 * self.distance)
                           if self.distance > 0.5 else float("nan")),
        rms_target_rate_rad=float(np.sqrt(self.target_rate_sq / (n * len(self.pid)))),
        energy_j=self.energy, distance_m=self.distance, weight_n=self.mass * 9.81,
        # 95th percentiles over the episode and both feet, close to the swing peak.
        heel_height_p95_m=float(np.percentile(feet[:, 0], 95)),
        toe_height_p95_m=float(np.percentile(feet[:, 1], 95)),
        foot_tilt_p95_deg=float(np.percentile(feet[:, 2], 95)),
    )


def summary_line(r: dict) -> str:
  end = f"FELL: {r['cause']} | last push: {r['last_push']}" if r["fell"] else "survived"
  return (f"episode {r['episode']:3d}  {r['time_s']:6.1f} s  pushes {r['pushes']:2d}  "
          f"rms err vx {r['rms_err_vx']:.2f} vy {r['rms_err_vy']:.2f} yaw {r['rms_err_yaw']:.2f}  "
          f"power {r['mean_power_w']:5.0f} W  CoT {r['cost_of_transport']:.2f}  {end}")


def summarize(results: list) -> dict:
  falls = [r for r in results if r["fell"]]
  mean = lambda k: float(np.nanmean([r[k] for r in results]))  # noqa: E731
  return dict(
      episodes=len(results), falls=len(falls), fall_rate=len(falls) / len(results),
      falls_within_2s_of_push=sum(r["fell_within_2s_of_push"] for r in falls),
      fall_causes={c: sum(r["cause"] == c for r in falls) for c in sorted({r["cause"] for r in falls})},
      mean_survival_s=mean("time_s"),
      rms_err_vx=mean("rms_err_vx"), rms_err_vy=mean("rms_err_vy"), rms_err_yaw=mean("rms_err_yaw"),
      mean_power_w=mean("mean_power_w"),
      # Total energy over total distance, so that near-standing episodes do not dominate.
      cost_of_transport=(sum(r["energy_j"] for r in results)
                         / (results[0]["weight_n"] * max(sum(r["distance_m"] for r in results), 1e-9))),
      rms_target_rate_rad=mean("rms_target_rate_rad"),
      heel_height_p95_m=mean("heel_height_p95_m"), toe_height_p95_m=mean("toe_height_p95_m"),
      foot_tilt_p95_deg=mean("foot_tilt_p95_deg"),
  )


def run_headless(sim: H2Sim, n_episodes: int, policy_desc: str):
  limit = sim.args.episode_sec or 20.0
  results = []
  for _ in range(n_episodes):
    sim.reset()
    while not sim.done and sim.d.time < limit:
      sim.step()
    results.append(sim.episode_result())
    print(summary_line(results[-1]), flush=True)

  s = summarize(results)
  print("\n" + "=" * 78)
  print(f"policy: {policy_desc}")
  print(f"episodes {sim.args.start_episode}-{sim.episode}, {limit:.0f} s each, seed {sim.args.seed}, "
        f"auto pushes {sim.auto_push} ({sim.push_mag_range} m/s every {sim.push_interval_range} s)")
  print(f"falls: {s['falls']} ({100 * s['fall_rate']:.0f} %), within 2 s after a push: "
        f"{s['falls_within_2s_of_push']}, mean survival {s['mean_survival_s']:.1f} s")
  for cause, n in s["fall_causes"].items():
    print(f"  cause '{cause}': {n}")
  print(f"rms tracking error: vx {s['rms_err_vx']:.2f} m/s, vy {s['rms_err_vy']:.2f} m/s, "
        f"yaw {s['rms_err_yaw']:.2f} rad/s")
  print(f"mean power {s['mean_power_w']:.0f} W, cost of transport {s['cost_of_transport']:.2f}, "
        f"rms joint-target change per step {s['rms_target_rate_rad']:.4f} rad")
  print(f"feet (95th percentile): heel height {s['heel_height_p95_m']:.3f} m, toe height "
        f"{s['toe_height_p95_m']:.3f} m, foot tilt {s['foot_tilt_p95_deg']:.1f} deg")

  if sim.args.out:
    out = epath.Path(sim.args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        created=datetime.datetime.now().isoformat(timespec="seconds"),
        policy=policy_desc, settings=vars(sim.args), summary=s, episodes=results,
    ), indent=2, default=float))
    print(f"results written to {out}")


# GUI

class Keys:
  """Collects key presses from the viewer thread for the main loop."""

  def __init__(self, sim: H2Sim):
    self.sim = sim
    self.reset = self.push = self.paused = False

  def __call__(self, key: int):
    sim = self.sim
    delta = {KEY_UP: (0, 0.1), KEY_DOWN: (0, -0.1), KEY_PAGE_UP: (1, 0.1),
             KEY_PAGE_DOWN: (1, -0.1), KEY_LEFT: (2, 0.1), KEY_RIGHT: (2, -0.1)}
    if key in delta:
      i, dv = delta[key]
      cmd = sim.command.copy()
      cmd[i] = np.round(cmd[i] + dv, 2)
      if not sim.args.allow_ood:
        cmd[i] = np.clip(cmd[i], sim.cmd_low[i], sim.cmd_high[i])
      sim.command = cmd
    elif key == KEY_HOME:
      sim.command = np.zeros(3)
    elif key in (KEY_8, KEY_9):
      sim.set_gait_freq(sim.gait_freq + (0.05 if key == KEY_9 else -0.05))
    elif key in (KEY_6, KEY_7):
      sim.manual_push = max(0.0, sim.manual_push + (0.25 if key == KEY_7 else -0.25))
    elif key == KEY_ENTER:
      self.reset = True
    elif key == KEY_INSERT:
      self.push = True
    elif key == KEY_DELETE:
      sim.auto_push = not sim.auto_push
    elif key == KEY_END:
      self.paused = not self.paused


HELP = ("Up/Down\nLeft/Right\nPgUp/PgDn\nHome\n8 / 9\n6 / 7\nInsert\nDelete\nEnter\nEnd\n"
        "Ctrl+R-drag\nCtrl+L-drag",
        "forward speed\nturn\nlateral speed\nstop\ngait frequency -/+\nmanual push speed -/+\n"
        "push now\nauto pushes on/off\nreset\npause\npush (double-click pelvis first)\ntwist")


def make_figure() -> mujoco.MjvFigure:
  fig = mujoco.MjvFigure()
  mujoco.mjv_defaultFigure(fig)
  fig.title = "tracking (last 5 s)"
  fig.flg_extend = 1
  fig.gridsize = np.array([5, 5])
  for i, (name, rgb) in enumerate([("vx cmd", (0.5, 0.5, 1.0)), ("vx", (0.2, 0.2, 1.0)),
                                   ("yaw cmd", (1.0, 0.6, 0.4)), ("yaw rate", (1.0, 0.3, 0.1))]):
    fig.linename[i] = name
    fig.linergb[i] = rgb
  return fig


def update_figure(fig: mujoco.MjvFigure, history: list):
  if not history:
    return
  h = np.array(history)
  t = h[:, 0] - h[-1, 0]
  for i in range(4):
    fig.linepnt[i] = len(h)
    fig.linedata[i, 0:2 * len(h):2] = t
    fig.linedata[i, 1:2 * len(h):2] = h[:, i + 1]


def draw_push_arrow(scn: mujoco.MjvScene, sim: H2Sim):
  """Red arrow along the last push, shown for 1 s."""
  scn.ngeom = 0
  if not sim.pushes or sim.d.time - sim.pushes[-1].time > 1.0:
    return
  p = sim.pushes[-1]
  direction = np.array([*p.direction, 0.0])
  tip = sim.d.xpos[sim.pelvis_id] - 0.15 * direction
  tail = tip - (0.2 + 0.4 * p.speed) * direction
  geom = scn.geoms[0]
  mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                      np.zeros(9), np.array([1.0, 0.1, 0.1, 1.0], dtype=np.float32))
  mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.04, tail, tip)
  scn.ngeom = 1


def run_gui(sim: H2Sim, policy_desc: str):
  keys = Keys(sim)
  fig = make_figure()
  limit = sim.args.episode_sec
  sim.reset()
  fell_wall = None
  last_time = sim.d.time
  rtf = 1.0
  print(f"policy: {policy_desc}")

  with mujoco.viewer.launch_passive(sim.m, sim.d, key_callback=keys,
                                    show_left_ui=False, show_right_ui=False) as viewer:
    with viewer.lock():
      viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
      viewer.cam.trackbodyid = sim.torso_id
      viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 4.0, -15.0, 135.0

    while viewer.is_running():
      t0 = time.perf_counter()
      with viewer.lock():
        # The viewer's own reset (button or Backspace) shows up as time going back.
        gui_reset = sim.d.time < last_time - 1e-9
        auto_reset = (fell_wall is not None and not sim.args.no_auto_reset
                      and time.perf_counter() - fell_wall > sim.args.reset_delay)
        timed_out = limit is not None and sim.d.time >= limit
        if keys.reset or gui_reset or auto_reset or timed_out:
          if sim.steps:
            print(summary_line(sim.episode_result()), flush=True)
          sim.reset()
          keys.reset, fell_wall = False, None
        if keys.push:
          sim.manual_push_now()
          keys.push = False
        if not keys.paused and not sim.done:
          sim.step()
          if sim.done:
            fell_wall = time.perf_counter()
        last_time = sim.d.time
        draw_push_arrow(viewer.user_scn, sim)
        vel = sim.velocity()

      ood = not (sim.gait_range[0] - 1e-9 <= sim.gait_freq <= sim.gait_range[1] + 1e-9)
      status = (f"episode {sim.episode}   t = {sim.d.time:5.1f} s"
                + ("   PAUSED" if keys.paused else "")
                + (f"\nFELL: {sim.cause}" if sim.done else "")
                + f"\ncommand  vx {sim.command[0]:+.2f}  vy {sim.command[1]:+.2f}  "
                  f"yaw {sim.command[2]:+.2f}"
                + f"\nactual   vx {vel[0]:+.2f}  vy {vel[1]:+.2f}  yaw {vel[2]:+.2f}"
                + f"\npelvis height {sim.d.qpos[2]:.2f} m   gait {sim.gait_freq:.2f} Hz"
                + (" (outside training range)" if ood else "")
                + f"\nauto pushes {'ON' if sim.auto_push else 'off'} "
                  f"({sim.push_mag_range[0]:.1f}-{sim.push_mag_range[1]:.1f} m/s)"
                  f"   manual push {sim.manual_push:.2f} m/s"
                + f"\nlast push: {sim.last_push_info()}"
                + f"\nspeed x{sim.args.speed:.2f}   real-time factor {rtf:.2f}")
      viewer.set_texts([
          (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, *HELP),
          (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPRIGHT, status, ""),
      ])
      update_figure(fig, sim.history)
      vp = viewer.viewport
      if vp is not None and vp.width > 0:
        w, h = vp.width // 3, vp.height // 3
        viewer.set_figures((mujoco.MjrRect(vp.width - w, 0, w, h), fig))
      viewer.sync()

      target = sim.dt / sim.args.speed
      elapsed = time.perf_counter() - t0
      if elapsed < target:
        time.sleep(target - elapsed)
      rtf = 0.9 * rtf + 0.1 * sim.dt / max(time.perf_counter() - t0, 1e-6)


def main():
  policy, desc = load_policy(ARGS.checkpoint)
  env = registry.load(ARGS.env_name, config_overrides=env_overrides(ARGS))
  sim = H2Sim(env, policy, ARGS)
  out = policy(np.zeros(103, dtype=np.float32))
  assert out.shape == (len(sim.pid),), f"policy outputs {out.shape}, expected {len(sim.pid)}"
  if ARGS.headless:
    run_headless(sim, ARGS.headless, desc)
  else:
    run_gui(sim, desc)


if __name__ == "__main__":
  main()
