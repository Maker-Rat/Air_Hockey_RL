import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import shutil

from game_env_without_pygame import AirHockeyEnv

from ddqn import DDQNAgent

# Create the environment
#env = gym.make('CartPole-v0')
env = AirHockeyEnv()

# Get the state and action sizes
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Set the random seed
seed = 0

# Create the DDQN agent
agent = DDQNAgent(state_size, action_size, seed)

# agent.qnetwork_local.load_state_dict(torch.load("DDQN_Model_1.pt"))
# agent.qnetwork_target.load_state_dict(torch.load("DDQN_Model_1.pt"))

# Set the number of episodes and the maximum number of steps per episode
num_episodes = 10000
max_steps = 750

# Set the exploration rate
eps = eps_start = 1.0
eps_end = 0.01
eps_decay = 0.999

# Set the rewards and scores lists
rewards = []
scores = []


def save_ckp(state):
    f_path = 'checkpoint.pt'
    torch.save(state, f_path)


# Run the training loop
for i_episode in range(num_episodes):
    print(f'Episode: {i_episode}')

    # Initialize the environment and the state
    state = env.reset()
    done = False
    score = 0
    # eps = eps_end + (eps_start - eps_end) * np.exp(-i_episode / eps_decay)
    # Update the exploration rate
    eps = max(eps_end, eps_decay * eps)

    # Run the episode
    while not done:
        # Select an action and take a step in the environment
        action = agent.act(state, eps)

        next_state, reward, done, _ = env.step(action)

        # Store the experience in the replay buffer and learn from it
        agent.step(state, action, reward, next_state, done)
        # env.render(should_render=False)
        # Update the state and the score
        state = next_state
        score += reward

    print(f"\tScore: {score}, Epsilon: {eps}")
    # Save the rewards and scores
    rewards.append(score)
    scores.append(np.mean(rewards[-100:]))


    if (i_episode + 1) % 100 == 0:
        torch.save(agent.qnetwork_local.state_dict(), "DDQN_Model.pt")

        checkpoint = {
            'epoch': i_episode + 1,
            'state_dict': agent.qnetwork_local.state_dict(),
            'optimizer': agent.optimizer.state_dict()
        }

        save_ckp(checkpoint)
        print("Saved checkpoint...")

# Close the environment
env.close()

plt.ylabel("Score")
plt.xlabel("Episode")
plt.plot(range(len(rewards)), rewards)
plt.plot(range(len(rewards)), scores)
plt.legend(['Reward', "Score"])
plt.show()