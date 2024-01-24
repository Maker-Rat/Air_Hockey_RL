from Air_Hockey_Env import *

if __name__ == "__main__":
    env = AirHockeyEnv()

    # Example of using the environment
    observation = env.reset()
    for _ in range(1000):  # Run for 1000 steps
        action = env.action_space.sample()  # Replace this with your own logic
        observation, reward, done, _ = env.step(action)
        print(observation)

        if done:
            env.reset()

    env.close()