import sys
import os

src_root_dir = os.getcwd()

sys.path.append(src_root_dir)
sys.path.append(src_root_dir+"/src")
sys.path.append(src_root_dir+"/smina")
print(sys.path)

import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env  import DummyVecEnv

from gym import spaces
import hydra

from config.config import cs

from ChemSpaceEnv import ChemSpaceEnv

@hydra.main(config_path  = src_root_dir+"/src/config", config_name = "config")
def runRL(config):
    """
    Initializes and runs a reinforcement learning model using the Proximal Policy Optimization (PPO) algorithm
    on a custom environment 'ChemSpaceEnv'.
    
    The function creates a vectorized environment to facilitate the training of the model and then trains the model
    using the PPO algorithm. After training, the model is saved and then reloaded for further use.
    
    Parameters:
    - config: Configuration settings loaded through Hydra, based on a configuration file.
    
    The script sets an environment variable 'SCRATCH' and appends a run directory to the system arguments,
    allowing Hydra to manage the configuration and logging directories dynamically.
    """
    env = DummyVecEnv([lambda: ChemSpaceEnv(config)])

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1000000)
    model.save("main_ppo_1")

    model = PPO.load("main_ppo_1")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)

if __name__ == '__main__':
    os.environ['SCRATCH']="/Users/ruhichoudhary/runs_data2"
    run_path=os.environ['SCRATCH']
    rundir_param='hydra.run.dir='+run_path
    print(rundir_param)
    sys.argv.append(rundir_param)
    runRL()

