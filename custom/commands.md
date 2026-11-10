### 1. RUN THE VIEWER WITH A SPECIFIC CHECKPOINT
```sh
python viewer.py \
  --env_name G1JoystickFlatTerrain \
  --checkpoint /home/rl_h2_project_container/ros2_ws/src/mujoco_playground/logs/g1_baseline/G1JoystickFlatTerrain-20260727-125712/checkpoints/000170393600
```

### 2. START A BRAND NEW TRAINING RUN
(Saves to a new log folder so you don't overwrite the old one)
```sh
train-jax-ppo \
  --env_name G1JoystickFlatTerrain \
  --logdir ./logs/g1_baseline \
  --use_tb=True
```

### 3. RESUME TRAINING FROM A SPECIFIC CHECKPOINT 
(Saves to a new log folder so you don't overwrite the old one)
```sh
train-jax-ppo \
  --env_name G1JoystickFlatTerrain \
  --logdir ./logs/g1_baseline_resumed \
  --load_checkpoint_path /home/rl_h2_project_container/ros2_ws/src/mujoco_playground/logs/g1_baseline/G1JoystickFlatTerrain-20260727-125712/checkpoints \
  --use_tb=True
```

Fine tune the new policy with new rewards, resuming from checkpoint while changing the reward function or weights:
```sh
train-jax-ppo \
  --env_name G1JoystickFlatTerrain \
  --logdir ./logs/g1_optimized_1 \
  --load_checkpoint_path ./logs/g1_baseline_resumed/G1JoystickFlatTerrain-20260730-143621/checkpoints \
  --use_tb=True
```

### 4. TENSORBOARD
(Saves to a new log folder so you don't overwrite the old one)
```sh
tensorboard --logdir ./logs/g1_baseline_resumed --bind_all
```





### 5. TEST AN H2 POLICY: h2_play.py
Native MuJoCo physics + policy on the CPU (does not use the GPU). Keys and flags: `python h2_play.py --help`.
`--checkpoint` takes one step folder, or a run's `checkpoints/` folder (then the latest step is used).
```sh
source ~/rl_venvs/playground_venv/bin/activate
cd ~/ros2_ws/src/mujoco_playground/custom
CK=logs/ppo01/H2JoystickFlatTerrain-20261109-134934/checkpoints

# GUI, pushes off (default), clean start, no observation noise
python h2_play.py --checkpoint $CK
# GUI, training pushes (0.1-2.0 m/s every 5-10 s)
python h2_play.py --checkpoint $CK --pushes
# GUI, weaker pushes (Unitree mjlab H2: up to ~0.7 m/s every 5-6 s)
python h2_play.py --checkpoint $CK --pushes --push_mag_range 0.1 0.7 --push_interval_range 5 6
# GUI, closest to the training evaluation
python h2_play.py --checkpoint $CK --pushes --obs_noise --random_init --random_commands --episode_sec 20

# Headless statistics (20 episodes of 20 s, fall rate and causes)
python h2_play.py --checkpoint $CK --headless 20 --random_commands
python h2_play.py --checkpoint $CK --headless 20 --random_commands --pushes

# Save results (settings + per-episode + summary) for later comparison
python h2_play.py --checkpoint $CK --headless 20 --random_commands --pushes --out results/ppo01_pushes.json
# Replay headless episode 7 in the GUI (same start, commands and pushes; use the same flags)
python h2_play.py --checkpoint $CK --random_commands --pushes --start_episode 7
# Gait frequency outside the training range (keys 8 / 9)
python h2_play.py --checkpoint $CK --allow_ood
```
Episode k is identical (start state, commands, push times/speeds/directions, noise) for every
policy and flag combination with the same `--seed`, so results files can be compared episode by episode.
`--checkpoint` also accepts a TorchScript `.pt` file (103 raw observations in, 29 actions out)
for the PyTorch algorithms.

### 6. H2 RUN WITH CHANGED SETTINGS (ppo02: gait, feet, effort, pushes)
The env defaults reproduce ppo01 exactly; the changes are only in `configs/h2_ppo02_gait.json`,
passed as overrides. The run's `checkpoints/config.json` stores the full effective env config,
and `h2_play.py` loads it automatically, so each policy is evaluated on its own settings.
```sh
source ~/rl_venvs/playground_venv/bin/activate
cd ~/ros2_ws/src/mujoco_playground/custom
train-jax-ppo \
  --env_name H2JoystickFlatTerrain \
  --logdir ./logs/ppo02_gait \
  --playground_config_overrides "$(cat configs/h2_ppo02_gait.json)" \
  --use_tb=True
```
Check the printed "Environment Config Overrides" at the start of the run.
