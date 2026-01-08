import os
import sys
import numpy as np
import shutil
from collections import deque
from stable_baselines3 import DQN
from sumo_env import SumoEnv
from sim_config import CONFIG_4WAY_160M 

LOG_DIR = os.path.join("logs", "tests")
if os.path.exists(LOG_DIR):
    shutil.rmtree(LOG_DIR)
os.makedirs(LOG_DIR)

MODELS_DIR = os.path.join("models", "dqn")

MODEL_NAME = "model_parallel"
TEST_EPISODES = 10
N_STACK = 4

model_path = os.path.join(MODELS_DIR, f"{MODEL_NAME}.zip")

if not os.path.exists(model_path):
    print(f"ERROR: model not found in {model_path}")
    sys.exit()


print(f"Loading model from {model_path}")

env = SumoEnv(sim_config=CONFIG_4WAY_160M, 
            sim_step=0.5, 
            action_step=10, 
            episode_duration=3600, 
            log_folder=LOG_DIR,
            rank=0,
            episode_offset=0,
            enable_measure=True)

model = DQN.load(model_path)

print(f"Running {TEST_EPISODES} test episodes.")


try:
    for ep in range(1, TEST_EPISODES + 1):
        obs, _ = env.reset()
        stacked_frames = deque([obs for _ in range(N_STACK)], maxlen=N_STACK)

        done = False
        truncated = False
        episode_reward = 0
        step_counter = 0
        
        print(f"Episode {ep}/{TEST_EPISODES} started...")
        
        while not (done or truncated):
            obs_stack = np.concatenate(stacked_frames, axis=-1)
            action, _state = model.predict(obs_stack, deterministic=True)
            
            new_obs, reward, done, truncated, info = env.step(action)
            stacked_frames.append(new_obs)
            episode_reward += reward
            step_counter += 1
            
        
        print(f"Episode {ep} terminated.")
        print(f" -> Length: {step_counter} steps")
        print(f" -> Total reward: {episode_reward:.2f}")
        print("------------------------------------------------")
        measures = env.get_measures()
        episode_measures_file = os.path.join(LOG_DIR, f"test_episode_{ep}_measures.csv")
        env.dump_vehicle_population(os.path.join(LOG_DIR, f"test_episode_{ep}_vehicle_pop.yaml"))

        if len(measures) > 0:
            with open(episode_measures_file, 'w') as fd:
                keys = list(measures[0].keys())
                for k in keys:
                    print(f"{k}", file=fd, end=";")
                print(file=fd)

                for m in measures:
                    for k in keys:
                        print(f"{m[k]}", file=fd, end=";")
                    print(file=fd)

except KeyboardInterrupt:
    print("\nUser interruption.")

finally:
    env.close()