from stable_baselines3 import DQN
from sumo_env import SumoEnv
from sim_config import CONFIG_4WAY_160M
import os
import time
import datetime

models_dir = "models/dqn"
log_dir = "logs"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

def get_next_log_path(log_dir, base_name):
    max_run = 0
    for filename in os.listdir(log_dir):
        if filename.startswith(base_name) and filename.endswith(".csv"):
            try:
                number_part = filename.replace(base_name, "").replace(".csv", "")
                run_number = int(number_part)
                
                if run_number > max_run:
                    max_run = run_number
            except ValueError:
                continue

    new_filename = f"{base_name}{max_run + 1}.csv"
    return os.path.join(log_dir, new_filename)

env_log_path = get_next_log_path(log_dir, "train_data")


env = SumoEnv(CONFIG_4WAY_160M, 0.5, 10, 3600, env_log_path)

model = DQN(
    "MlpPolicy", 
    env, 
    verbose=1,
    tensorboard_log=log_dir,
    device="auto",
    learning_rate=1e-4, # TBD
    learning_starts=1000, # warm up steps ~ 3 episodes
    batch_size=64, # TBD
    gamma=0.99, # TBD
    train_freq=4, # TBD
    target_update_interval=1000, # TBD
    exploration_fraction=0.3, # TBD
    exploration_final_eps=0.05 # TBD
)

EPISODES = 500
TIMESTEPS = EPISODES * 360

print(f"Inizio training: {EPISODES} Episodi ({TIMESTEPS} steps totali)...")
start_time = time.perf_counter()
model.learn(total_timesteps=TIMESTEPS, progress_bar=True)

end_time = time.perf_counter()
elapsed = int(end_time - start_time)
formatted_time = str(datetime.timedelta(seconds=elapsed))

print(f"Training completato in: {formatted_time}")
model.save(f"{models_dir}/final_model_500ep_10s")
env.close()