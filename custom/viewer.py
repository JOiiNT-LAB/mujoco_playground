import argparse
import argparse
import json
import os
import time
from collections import namedtuple

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.80"


import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from mujoco_playground import registry
from orbax import checkpoint as ocp

# --- GLOBAL JOYSTICK STATE ---
# We start with a forward velocity of 0.5. 
# Walking forward is much more stable than standing still!
current_command = np.array([0.2, 0.0, 0.0])

def key_callback(keycode):
    """Listens to keyboard inputs and updates the target velocity."""
    global current_command
    
    # Safely parse the keycode
    try:
        char = chr(keycode).lower()
    except:
        char = ''
        
    vx, vy, yaw = current_command
    step = 0.2  # How much the velocity changes per key press
    
    if char == 'w': vx += step
    elif char == 's': vx -= step
    elif char == 'a': yaw += step
    elif char == 'd': yaw -= step
    elif char == ' ': vx, vy, yaw = 0.0, 0.0, 0.0  # Spacebar to stop
    
    current_command = np.array([vx, vy, yaw])
    print(f"Target Command -> Forward: {vx:.2f}, Lateral: {vy:.2f}, Turn: {yaw:.2f}")

def dict_to_obj(name, d):
    if not isinstance(d, dict) or not d:
        return d
    return namedtuple(name, d.keys())(**d)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="G1JoystickFlatTerrain")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    print(f"Loading environment: {args.env_name}")
    env = registry.load(args.env_name)

    # 1. Load Architecture Config
    config_path = os.path.join(args.checkpoint, "ppo_network_config.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)
        net_kwargs = cfg.get("network_factory_kwargs", {})
        policy_sizes = tuple(net_kwargs.get("policy_hidden_layer_sizes", (512, 256, 128)))
        value_sizes = tuple(net_kwargs.get("value_hidden_layer_sizes", (512, 256, 128)))
        use_norm = cfg.get("normalize_observations", True)

    # 2. Build Network Architecture
    ppo_network = ppo_networks.make_ppo_networks(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=running_statistics.normalize if use_norm else lambda x, y: x,
        policy_hidden_layer_sizes=policy_sizes,
        value_hidden_layer_sizes=value_sizes,
    )

    # 3. Load Checkpoint Raw Data
    ckpt = ocp.PyTreeCheckpointer().restore(args.checkpoint)
    
    # 4. Clean Unpacking
    norm_dict = ckpt.get("normalizer_params", {}) if isinstance(ckpt, dict) else ckpt[0]
    net_params = ckpt.get("params", ckpt) if isinstance(ckpt, dict) else ckpt[1]
    policy_params = net_params[0] if isinstance(net_params, (list, tuple)) else net_params.get('policy', net_params)

    # 5. Build Dynamic Normalizer Object
    norm_params_obj = dict_to_obj("NestedMeanStd", norm_dict)

    # Compile the inference function
    inference_fn = jax.jit(ppo_networks.make_inference_fn(ppo_network)((norm_params_obj, policy_params)))
    jit_step = jax.jit(env.step)
    
    # Randomize the seed based on system time so it acts differently every run!
    rng = jax.random.PRNGKey(int(time.time()))
    state = jax.jit(env.reset)(rng)

    mj_model = env.mj_model
    mj_data = mujoco.MjData(mj_model)

    print("\n" + "="*50)
    print("LAUNCHING MUJOCO VIEWER")
    print(">>> IMPORTANT: Click anywhere inside the 3D window first! <<<")
    print("Use W/A/S/D to steer, Spacebar to stop. Press ESC to exit.")
    print("="*50 + "\n")
    
    # Inject the key_callback into the viewer
    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            step_start = time.time()
            rng, act_rng = jax.random.split(rng)
            
            # --- FORCE-INJECT JOYSTICK COMMAND ---
            cmd_jax = jnp.array(current_command)
            
            # Update the 'info' dict (used for rewards)
            if hasattr(state, 'info') and hasattr(state.info, 'keys'):
                new_info = dict(state.info)
                new_info['command'] = cmd_jax
                state = state.replace(info=new_info)
                
            # CRITICAL: Update the 'obs' dict so the brain actually sees our inputs!
            if hasattr(state, 'obs') and hasattr(state.obs, 'keys') and 'command' in state.obs:
                new_obs = dict(state.obs)
                new_obs['command'] = cmd_jax
                state = state.replace(obs=new_obs)
            # ---------------------------------------
            
            # Get Action & Step Physics
            ctrl, _ = inference_fn(state.obs, act_rng)
            state = jit_step(state, ctrl)

            # Extract physics state safely
            physics_state = getattr(state, 'data', getattr(state, 'pipeline_state', None))
            
            if physics_state is not None:
                mj_data.qpos[:] = np.array(physics_state.qpos)
                mj_data.qvel[:] = np.array(physics_state.qvel)
                mujoco.mj_forward(mj_model, mj_data)
            
            viewer.sync()
            
            # Enforce real-time rendering
            elapsed = time.time() - step_start
            if elapsed < env.dt:
                time.sleep(env.dt - elapsed)

if __name__ == "__main__":
    main()