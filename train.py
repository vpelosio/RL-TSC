import os
import shutil
import time
import datetime
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from sumo_env import SumoEnv
from sim_config import CONFIG_4WAY_160M

NUM_CPU = 16
TIMESTEPS = 10_000_000 # very high limit, never reached for 500 episodes
SUMO_WORKSPACE = "sumo_workspace"
BASE_MODELS_DIR = "models/ppo"
BASE_LOG_DIR = os.path.join("logs", "training")

def get_next_train_id(base_dir):
    if not os.path.exists(base_dir):
        return 1
    
    # all folders that starts with "train_id"
    existing_runs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("train_id_")]
    
    if not existing_runs:
        return 1

    # get train ids
    train_ids = []
    for d in existing_runs:
        try:
            train_ids.append(int(d.split("_")[2]))
        except (IndexError, ValueError):
            pass
            
    if not train_ids:
        return 1
        
    return max(train_ids) + 1

def setup_run_directories():
    train_id = get_next_train_id(BASE_LOG_DIR)
    run_name = f"train_id_{train_id}"

    current_models_dir = os.path.join(BASE_MODELS_DIR, run_name)
    current_log_dir = os.path.join(BASE_LOG_DIR, run_name)

    os.makedirs(current_models_dir, exist_ok=True)
    os.makedirs(current_log_dir, exist_ok=True)
    os.makedirs(SUMO_WORKSPACE, exist_ok=True) 
    
    print(f"--- Run Config ---")
    print(f"Train ID: {train_id}")
    print(f"Models Dir: {current_models_dir}")
    print(f"Logs Dir: {current_log_dir}")
    print(f"--------------------------")

    return current_models_dir, current_log_dir, train_id

class StopAtMaxEpisodesVec(BaseCallback):
    def __init__(self, max_episodes: int, verbose=1):
        super().__init__(verbose)
        self.max_episodes = max_episodes
        self.episode_count = 0

    def _on_step(self) -> bool:
        dones = self.locals['dones']
        
        finished_now = np.sum(dones)
        
        if finished_now > 0:
            self.episode_count += finished_now
            if self.verbose > 0:
                print("---------------------------------------------------------------")
                print(f"Completed episodes: {self.episode_count} / {self.max_episodes}")
                print("---------------------------------------------------------------")


        if self.episode_count >= self.max_episodes:
            if self.verbose > 0:
                print("---------------------------------------------------------------")
                print("---------- Episode limit reached. Stop Training. --------------")
                print("---------------------------------------------------------------")

            return False
            
        return True

def make_env(rank, log_dir, seed=0):
    def _init():
        episode_offset = rank * 2000 
        
        env = SumoEnv(
            sim_config=CONFIG_4WAY_160M, 
            sim_step=0.5, 
            action_step=10, 
            episode_duration=3600, 
            log_folder=log_dir,
            rank=rank,          # Proc ID
            episode_offset=episode_offset # Offset
        )
        
        env.reset(seed=seed + rank)
        return env
    
    return _init

if __name__ == "__main__":
    models_dir, log_dir, train_id = setup_run_directories()

    print(f"Parallel training on {NUM_CPU} processes")
    
    env = SubprocVecEnv([make_env(i, log_dir) for i in range(NUM_CPU)])
    
    env = VecMonitor(env, filename=os.path.join(log_dir, "monitor.csv"))

    model = PPO(
        "MlpPolicy", 
        env, 
        tensorboard_log=log_dir,
        device="auto"
    )

    print(f"Start training...")
    start_time = time.perf_counter()
    callback_max_episodes = StopAtMaxEpisodesVec(max_episodes=2000, verbose=1)
    model.learn(total_timesteps=TIMESTEPS, callback=callback_max_episodes)

    end_time = time.perf_counter()
    elapsed = int(end_time - start_time)
    formatted_time = str(datetime.timedelta(seconds=elapsed))

    print(f"Training completed in: {formatted_time}")
    model.save(f"{models_dir}/PPO_{train_id}")
    
    env.close()