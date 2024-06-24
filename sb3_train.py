import gymnasium as gym
import numpy as np
import pygame
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

from game_env_without_pygame import AirHockeyEnv

# Create and wrap the environment
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

# Create and wrap the environment
env = make_vec_env(lambda: AirHockeyEnv(), n_envs=1)

# Define the policy kwargs to enable Double DQN
policy_kwargs = {
    'double_q': True
}

# Instantiate the agent with the double_q parameter in policy_kwargs
model = DQN('MlpPolicy', env, verbose=1)

# Train the agent
model.learn(total_timesteps=100000)

# Save the agent
model.save("air_hockey_ddqn")

# To reload the agent
# model = DQN.load("air_hockey_ddqn")

# Test the trained agent
# obs = env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, rewards, dones, info = env.step(action)
#     #env.render()