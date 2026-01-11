import os
import sys
import numpy as np
import shutil
import argparse
from collections import deque
from stable_baselines3 import DQN
from sumo_env import SumoEnv
from sim_config import CONFIG_4WAY_160M 

def write_measures(measures, summary_filename, ep_measures_file_name, ep):
    averages = {}
    n_measures = len(measures)
    target_keys = ['totalTravelTime', 'totalWaitingTime', 'totalCO2Emissions']
    summary_averages_file = os.path.join(LOG_DIR, f"{summary_filename}")
    
    with open(summary_averages_file, 'a') as f:
        print(f"--- Episode: {ep} ---", file=f)
        for key in target_keys:
            if key in measures[0]:
                total = sum(m[key] for m in measures)
                averages[key] = total / n_measures
                
                print(f"Average for {key}: {averages[key]}", file=f)
        print("", file=f)
    
    episode_measures_file = os.path.join(LOG_DIR, f"{ep_measures_file_name}")
    with open(episode_measures_file, 'w') as f:
        keys = list(measures[0].keys())
        for k in keys:
            print(f"{k}", file=f, end=";")
        print(file=f)

        for m in measures:
            for k in keys:
                print(f"{m[k]}", file=f, end=";")
            print(file=f)

parser = argparse.ArgumentParser(description="Run tests on specific DQN model")
parser.add_argument("--id", type=int, required=True, help="Training ID")
parser.add_argument("--skip-stl", action="store_true", required=False, help="Skip STL tests")
args = parser.parse_args()

MODEL_RUN = f"train_id_{args.id}"
LOG_DIR = os.path.join("logs", "tests", MODEL_RUN)
if os.path.exists(LOG_DIR):
    shutil.rmtree(LOG_DIR)
os.makedirs(LOG_DIR)

MODELS_DIR = os.path.join("models", "dqn", MODEL_RUN)

MODEL_NAME = f"DQN_{args.id}"
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
            
        measures = env.get_measures()
        env.dump_vehicle_population(os.path.join(LOG_DIR, f"test_episode_{ep}_vehicle_pop.yaml"))
        print(f"Episode {ep} terminated.")
        print("------------------------------------------------")
        write_measures(measures, "dqn_summary.txt", f"dqn_measures_ep{ep}.txt", ep)

        if not args.skip_stl:
            # STL without improvments
            env.run_smart_traffic_light([])
            measures = env.get_measures()
            write_measures(measures, "stl_summary.txt", f"stl_measures_ep{ep}.txt", ep)

            # STL with improvment 1 (K = 5)
            env.run_smart_traffic_light([1])
            measures = env.get_measures()
            write_measures(measures, "stl1_summary.txt", f"stl1_measures_ep{ep}.txt", ep)

            # STL with improvment 2 (skip safe guard)
            env.run_smart_traffic_light([2])
            measures = env.get_measures()
            write_measures(measures, "stl2_summary.txt", f"stl2_measures_ep{ep}.txt", ep)

            # STL with improvments 1 and 2
            env.run_smart_traffic_light([1,2])
            measures = env.get_measures()
            write_measures(measures, "stl12_summary.txt", f"stl12_measures_ep{ep}.txt", ep)

except KeyboardInterrupt:
    print("\nUser interruption.")

finally:
    env.close()