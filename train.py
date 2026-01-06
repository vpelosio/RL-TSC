import os
import shutil
import time
import datetime
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from sumo_env import SumoEnv
from sim_config import CONFIG_4WAY_160M

NUM_CPU = 32
EPISODES = 500
TIMESTEPS = EPISODES * 360  # Approximation
MODELS_DIR = "models/dqn"
LOG_DIR = "logs"
SUMO_WORKSPACE = "sumo_workspace"

def setup_directories():
    if os.path.exists(MODELS_DIR):
        shutil.rmtree(MODELS_DIR)
    os.makedirs(MODELS_DIR)

    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)
    os.makedirs(LOG_DIR)

    if os.path.exists(SUMO_WORKSPACE):
        shutil.rmtree(SUMO_WORKSPACE)
    os.makedirs(SUMO_WORKSPACE)
    
def make_env(rank, seed=0):
    def _init():
        episode_offset = rank * 2000 
        
        env = SumoEnv(
            sim_config=CONFIG_4WAY_160M, 
            sim_step=0.5, 
            action_step=10, 
            episode_duration=3600, 
            log_folder=LOG_DIR,
            rank=rank,          # Proc ID
            episode_offset=episode_offset, # Offset
            gui=False
        )
        
        env.reset(seed=seed + rank)
        return env
    return _init

if __name__ == "__main__":
    setup_directories()

    print(f"Parallel training on {NUM_CPU} processes")
    
    env = SubprocVecEnv([make_env(i) for i in range(NUM_CPU)])
    
    env = VecMonitor(env, filename=os.path.join(LOG_DIR, "monitor.csv"))

    model = DQN(
        "MlpPolicy", 
        env, 
        verbose=1,
        tensorboard_log=LOG_DIR,
        device="auto",
        learning_rate=1e-4, # TBD
        learning_starts=1000, # TBD
        batch_size=64, # TBD
        gamma=0.99, # TBD
        train_freq=4, # TBD
        target_update_interval=1000, # TBD 
        exploration_fraction=0.3, # TBD
        exploration_final_eps=0.05, # TBD
    )

    print(f"Start training...")
    start_time = time.perf_counter()
    
    model.learn(total_timesteps=TIMESTEPS, progress_bar=True)

    end_time = time.perf_counter()
    elapsed = int(end_time - start_time)
    formatted_time = str(datetime.timedelta(seconds=elapsed))

    print(f"Training completed in: {formatted_time}")
    
    model.save(f"{MODELS_DIR}/final_model_parallel")
    
    env.close()