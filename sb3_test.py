import gymnasium as gym
import numpy as np
import pygame
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

from env_game import AirHockeyEnv

# Create and wrap the environment
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

# Create and wrap the environment
env = make_vec_env(lambda: AirHockeyEnv(), n_envs=1)

# Instantiate the agent with the double_q parameter in policy_kwargs
model = DQN('MlpPolicy', env, verbose=1)

#To reload the agent
model = DQN.load("air_hockey_ddqn")

#Test the trained agent

for i_episode in range(100):
    print(f'Episode: {i_episode}')

    # Initialize the environment and the state
    obs = env.reset()
    done = False
    score = 0
    
    # Run the episode
    while not done:
        # Select an action and take a step in the environment
        action, _states = model.predict(obs, deterministic=True)
        print(action)
        obs, rewards, dones, info = env.step(action)
        env.render()