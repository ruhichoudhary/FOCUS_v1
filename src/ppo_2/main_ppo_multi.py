
import sys
import os
sys.path.append("/Users/ruhichoudhary/code/gitrepos/DD_Reinforcement4")
sys.path.append("/Users/ruhichoudhary/code/gitrepos/DD_Reinforcement4/src")
sys.path.append("/Users/ruhichoudhary/code/gitrepos/DD_Reinforcement4/src/smina")

import numpy as np

import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env  import DummyVecEnv, SubprocVecEnv
#from stable_baselines3.common import set_global_seeds
from stable_baselines3.common.evaluation import evaluate_policy

from gym import spaces
import hydra


from config.config import cs

from ChemSpaceEnv import ChemSpaceEnv

def make_env(config, rank, seed=0):
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = ChemSpaceEnv(config)
        env.seed(seed + rank)
        return env
    
    #set_global_seeds(seed)
    return _init


@hydra.main(config_path  = "../config", config_name = "config")
def runRL(config):
    prev_dir=os.getcwd()
    os.chdir("/Users/ruhichoudhary/runs_data/run2")

    num_cpu = 2  # Number of processes to use
    # Create the vectorized environment

    envs= [make_env(config, i) for i in range(num_cpu)]

    envs = SubprocVecEnv(envs, start_method='fork')

    envs.reset()

    model = PPO('MlpPolicy', envs, verbose=1)

    # Random Agent, before training
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')
    n_timesteps = 1000

    # Multiprocessed RL Training
    start_time = time.time()
    model.learn(n_timesteps)
    total_time_multi = time.time() - start_time

    print("Took {:.2f}s for multiprocessed version - {:.2f} FPS".format(total_time_multi, n_timesteps / total_time_multi))

    model.save("DD_Reinforcement")
    os.chdir(prev_dir)


if __name__ == '__main__':
    runRL()

