

import gym 
from gym import spaces
from gym.spaces import Dict, Discrete, MultiBinary, MultiDiscrete
import numpy as np
import sys
import os
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import hydra
from config.config import cs

import mcts
from mcts.Utils.utils import read_smilesset, parse_smiles, convert_smiles
from mcts.Utils.reward import getReward
from mcts.mcts import runEpisode
import time 
import random 
import pandas as pd

"""
Custom Environment that follows gym interface, designed specifically for 
exploring chemical spaces using reinforcement learning.

Attributes:
    max_iter (int): Maximum number of iterations for the environment.
    total_iter (int): Counter for the total number of iterations.
    observation_space (gym.spaces): Defines the space of possible observations.
    observation (np.array): Current observation of the environment.
    action_space (gym.spaces): Defines the space of possible actions.
    input_smiles (list): List of SMILES strings read from the input file.
    next_smile (str): Next SMILES string to be processed.
    reward (function): Reward function used for evaluating actions.
    n_invalid (int): Counter for the number of invalid actions.
    n_valid (int): Counter for the number of valid actions.
    smina_index (int): Index used for tracking SMINA evaluations.
    episodeIndex (int): Counter for the number of episodes.
    config (dict): Configuration dictionary containing settings for the environment.
    prev_reward (list): List tracking the rewards from the last five steps.
    episodestep (int): Counter for the number of steps in the current episode.
    restartSmile (str or None): SMILES string used to restart the environment (if any).

Methods:
    __init__(self, config): Initializes the environment with a given configuration.
"""

class ChemSpaceEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, config):
        """
        Initialize the ChemSpaceEnv environment with the provided configuration.

        Parameters:
            config (dict): A configuration dictionary with settings for the environment.
                            It should include keys like 'mcts' to specify MCTS-related
                            settings and 'reward_name' to define the reward function.

        The initialization process sets up the observation space, action space,
        reads the input SMILES strings, and initializes various counters and state
        variables used throughout the environment's operation.
        """
        super(ChemSpaceEnv, self).__init__()
        self.max_iter = 1000
        self.total_iter = 0 
        """
        self.observation_space = Dict({"inner_loop_reward":spaces.Box(low = 0, high = 3, dtype = np.float32, shape=(1,0)), 
                                        "smina_NOD2": spaces.Box(low = -9, high = 1, dtype = np.float32, shape=(1,0)), 
                                        "smina_PEP": spaces.Box(low = -9, high = 1, dtype = np.float32, shape=(1,0)),
                                        "smina_MCT": spaces.Box(low = -9, high = 1, dtype = np.float32, shape=(1,0)),
                                            "hydrophobicContacts_NOD2":spaces.Box(low = 0, high = 45, dtype = np.uint8, shape=(1,0)),
                                             "hydrogenBonds_NOD2": spaces.Box(low = 0, high = 6, dtype = np.uint8, shape=(1,0)), 
                                             "piPiStackingInteractions_NOD2":spaces.Box(low = 0, high = 3, dtype = np.uint8, shape=(1,0)),
                                             "tStackingInteractions_NOD2":spaces.Box(low = 0, high = 3, dtype = np.uint8, shape=(1,0)),
                                             "cationPiInteractions_NOD2":spaces.Box(low = 0, high = 3, dtype = np.uint8, shape=(1,0)),
                                             "saltBridges_NOD2":spaces.Box(low = 0, high = 20, dtype = np.uint8, shape=(1,0)),
                                             "electrostaticEnergies_NOD2":spaces.Box(low = 0, high = 30, dtype = np.uint8, shape=(1,0)),
                                             "ASN_PEP":spaces.Box(low = 0, high = 10, dtype = np.uint8, shape=(1,0)),
                                             "GLU_PEP":spaces.Box(low = 0, high = 10, dtype = np.uint8, shape=(1,0)),
                                             "LYS_MCT":spaces.Box(low = 0, high = 10, dtype = np.uint8, shape=(1,0)),
                                             "hydrophobicContacts_MCT":spaces.Box(low = 0, high = 50, dtype = np.uint8, shape=(1,0))  })
        """
        self.observation_space = spaces.Box(low=-9, high=50, shape=(15,), dtype=np.float32)
        #self.observation =np.array([False,False,False,False,False,False,False,False,False,False,False,False,False,False, False, False,False,False,False]) 
        self.observation = self.reset()
        self.action_space = spaces.Box(low=1, high=2, shape=(5,), dtype=np.float32)
        self.input_smiles = read_smilesset(hydra.utils.get_original_cwd()+config['mcts']['in_smiles_file'])
        self.next_smile = self.input_smiles[0]
        self.reward = getReward(name = config['mcts']['reward_name'], init_smiles = self.input_smiles)
        self.n_invalid = 0 
        self.n_valid = 0
        self.smina_index=0
        self.episodeIndex = 0 
        self.config = config
        self.prev_reward = [0]*5
        self.episodestep = 0

        self.episodeIndex = 19
        self.smina_index = 245
        #restartdf = pd.read_csv( "./data/"+ "No-{:04d}.csv".format(19))
        ######filtered_smiles = generated_smiles.loc[generated_smiles['smina_MDP'] != 0]
        #filtered_smiles = restartdf.loc[restartdf['OutReward'].idxmax()]
        #new_smiles = filtered_smiles['SMILES']
        #self.restartSmile = new_smiles
        self.restartSmile=None

    def reset(self):
        self.episodestep = 0
        self.prev_reward = [0]*5 
        return np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],dtype=np.float32)  


    def step(self, action):

        """
        Executes one step in the environment based on the given action, updates the environment's state,
        and calculates the reward.

        This method takes an action, applies it to the environment, and evolves the state of the environment
        to the next step. It also computes the reward based on the new state and checks for the convergence
        condition of the current episode.

        Parameters:
            action (np.array): An array representing the action to be taken. The action is first adjusted to
                            match the required range for the MCTS algorithm.

        Returns:
            tuple: A tuple containing:
                - np.array: The new observation (state) of the environment after the action is applied.
                - float: The reward obtained after applying the action.
                - bool: A boolean flag indicating whether the episode has converged.
                - dict: Additional information as a dictionary (currently empty).

        The method updates the internal state of the environment, increments the episode index, processes the
        restart mechanism if needed, and logs the rewards. It also checks for convergence by examining the sequence
        of the last five rewards. If all recent rewards are above a certain threshold, the episode is considered
        as having converged. The method then updates the episodestep counter and returns the new observation,
        the current reward, the convergence status, and an info dictionary.
        """
        
        #elf.total_iter += np.sum(action)
        action = (action*0.5) + 1.5 #setting the action to required ranges for mcts
        done = 1
        info = {}
        costs = 0 

        self.episodeIndex += 1 
        #action = self.action_space.sample()
        if self.restartSmile is not None:
            self.next_smile = self.restartSmile
            self.restartSmile = None
        [self.next_smile, self.n_valid, self.n_invalid, outerreward, state, self.smina_index] = runEpisode(self.config, self.next_smile, self.n_valid, self.n_invalid, self.reward, self.episodeIndex, action, self.smina_index)
        self.observation = state

        fp = open("rewards.txt", "at")
        fp.write(str(self.episodestep)+ ','+str(self.reward.max_r)+ ','+str(outerreward)+ ',' + str(self.observation)+ str(action)+"\n")
        print(str(self.reward.max_r)+ ','+str(outerreward)+',' + str(self.observation)+"\n")
        fp.close()
        converged = False
        self.prev_reward[self.episodestep%5] = outerreward[0]
        if (self.episodestep > 10) & ((self.prev_reward[0] >= 1.5) & (self.prev_reward[1] >= 1.5)&(self.prev_reward[2] >= 1.5) & (self.prev_reward[3] >= 1.5)& (self.prev_reward[4] >= 1.5)):
            converged = True
        self.episodestep = self.episodestep + 1
        currentreward = outerreward[0]
        if currentreward < 0:
            currentreward = 0
        return self.observation, currentreward, converged , info 