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






