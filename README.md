# **Air Hockey RL**

Repo for Reinforcement Learning based implementation in Air Hockey environment.


## Inspiration
The idea for this project came up as a fun interactive demonstration for our college's technical fest Quark '24. We were able to somewhat implement a Neural Network agent trained using imitation learning techniques. The expert for this approach was the computer of a mobile based air hockey game. Data was extracted from a half-hour long video playthrough of the game through computer vision techniques. The agent was trained using a DNN in keras. This was then deployed on hardware built completely in-house which used a stepper motor actuated gantry to actuate the paddle and a single camera to gather relevant information.


## Current Progress
The initial goal for the project was to implement RL on the environment and compare it with the neural network opponent. Due to limited time during Quark we were unable to do this. I then took up this project again this summer to try and make some progress on it. I was able to create a much better and reliable environment than before to train my agent. I then trained the agent using a double DQN strategyb over 10,000 episodes. The setup used a simple reward function defined as follows:
  - 1000 for scoring a goal
  - -1000 for having a goal scored against it
  - -1 for every frame the puck is in the agent's half

This resulted in the agent learning decently well how to attack the puck [as soon as it enters it's half](/RL_vs_NN.mp4). It's defensive capabilities however are next to none.


## Further Work
I plan on designing a better reward function and test out other algorithms like VPG or PPO. I would then look into possibilities of somehow benchmarking the agent's performance maybe through some sort of ranked online game if possible.
One more interesting idea would be to experiment with intrinsic rewards and curiosity based exploration as the Air Hockey environment is one with scarce rewards and a natural low pass tendency. This would be a great way of testing out the effectiveness of the ICM module in a simple dynamic environment.


