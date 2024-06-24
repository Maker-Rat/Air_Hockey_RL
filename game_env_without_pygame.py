import gym
from gym import spaces
import numpy as np
import tensorflow
from keras.models import load_model
import time
from math import sqrt

class AirHockeyEnv(gym.Env):
    def __init__(self):
        super(AirHockeyEnv, self).__init__()

        self.width = 800
        self.height = 400

        # Define dimensions
        self.play_area = {
            'left': 50,
            'right': 750,
            'top': 50,
            'bottom': 350,
            'centerx': 400,
            'centery': 200
        }

        # Define game objects' sizes
        self.puck_radius = self.height / 11.25 / 2
        self.paddle_radius = self.height / 8.18 / 2

        self.goal_width = 150
        self.goal_height = 5

        self.count = 0

        self.friction = 0.995
        self.max_velocity = 20

        self.x_velocities = [-15, -7.5, 0, 7.5, 15]
        self.y_velocities = [-15, -7.5, 0, 7.5, 15]

        self.prev_pos = 0
        self.cur_pos = 0

        self.done = False

        self.velocities = []
        for i in self.x_velocities:
            for j in self.y_velocities:
                self.velocities.append([i, j])

        self.info = [[[0.5, 0.5, 0, 0],[0.083, 0.5, 0, 0]], [[0.5, 0.5, 0, 0],[0.083, 0.5, 0, 0]]]
        self.state = []

        # Define action space
        self.action_space = spaces.Discrete(25)

        # Define observation space
        self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)

        # Load model
        # self.model = load_model('my_model_new.h5')
        self.model = tensorflow.keras.models.load_model('my_model_new.h5', compile=False)
        self.model.compile()

        self.reset()

    def step(self, action):
        # Convert action to velocity
        self.player_vel[0], self.player_vel[1] = self.velocities[action]

        # Update player position
        self.player_pos[0] += self.player_vel[0]
        self.player_pos[1] += self.player_vel[1]

        # Ensure the paddle stays within bounds
        self.player_pos[0] = max(self.play_area['left'] + self.paddle_radius, min(self.play_area['centerx'] - self.paddle_radius, self.player_pos[0]))
        self.player_pos[1] = max(self.play_area['top'] + self.paddle_radius, min(self.play_area['bottom'] - self.paddle_radius, self.player_pos[1]))

        # Update game state
        self.ai_control()
        self.update_puck()
        reward = self.calculate_reward()
        self.check_goal()
        state = self.get_observation()
        done = self.game_over()

        self.count += 1

        return state, reward, done, {}

    def reset(self):
        self.puck_pos = np.array([self.width / 2, self.height / 2], dtype=np.float64)
        self.puck_vel = np.array([0, 0], dtype=np.float64)
        self.player_pos = np.array([self.play_area['left'] + self.paddle_radius + 10, self.height / 2], dtype=np.float64)
        self.player_vel = np.array([0, 0], dtype=np.float64)
        self.ai_pos = np.array([self.play_area['right'] - self.paddle_radius - 10, self.height / 2], dtype=np.float64)
        self.ai_vel = np.array([0, 0], dtype=np.float64)
        self.done = False
        self.count = 0
        return self.get_observation()

    def render(self, should_render=False):
        if not should_render:
            return

    def close(self):
        pass

    def handle_collision(self, paddle_pos, paddle_vel, paddle_radius):
        dist = np.linalg.norm(self.puck_pos - paddle_pos)
        if dist <= self.puck_radius + paddle_radius:
            # Calculate the direction of the bounce
            normal = (self.puck_pos - paddle_pos) / dist
            relative_velocity = self.puck_vel - paddle_vel
            speed = np.dot(relative_velocity, normal)

            # Only update if moving towards each other
            if speed < 0:
                self.puck_vel -= 2 * speed * normal

            # Move puck out of collision
            overlap = self.puck_radius + paddle_radius - dist
            self.puck_pos += normal * overlap

    def update_puck(self):
        # Update puck position
        vel = list(self.puck_vel)
        self.puck_pos += self.puck_vel

        # Handle collisions with walls, allowing puck to pass through the goal areas
        if (self.puck_pos[0] <= self.play_area['left'] + self.puck_radius and not (self.play_area['centery'] - self.goal_width / 2 <= self.puck_pos[1] <= self.play_area['centery'] + self.goal_width / 2)):
            self.puck_pos[0] = self.play_area['left'] + self.puck_radius
            self.puck_vel[0] = -self.puck_vel[0]
        if (self.puck_pos[0] >= self.play_area['right'] - self.puck_radius and not (self.play_area['centery'] - self.goal_width / 2 <= self.puck_pos[1] <= self.play_area['centery'] + self.goal_width / 2)):
            self.puck_pos[0] = self.play_area['right'] - self.puck_radius
            self.puck_vel[0] = -self.puck_vel[0]
        if self.puck_pos[1] <= self.play_area['top'] + self.puck_radius:
            self.puck_pos[1] = self.play_area['top'] + self.puck_radius
            self.puck_vel[1] = -self.puck_vel[1]
        if self.puck_pos[1] >= self.play_area['bottom'] - self.puck_radius:
            self.puck_pos[1] = self.play_area['bottom'] - self.puck_radius
            self.puck_vel[1] = -self.puck_vel[1]

        # Handle collisions with paddles
        self.handle_collision(self.player_pos, self.player_vel, self.paddle_radius)
        self.handle_collision(self.ai_pos, self.ai_vel, self.paddle_radius)

        # Update velocities based on friction
        self.puck_vel *= self.friction

        # Limit maximum velocity
        speed = np.linalg.norm(self.puck_vel)
        if speed > self.max_velocity:
            self.puck_vel = (self.puck_vel / speed) * self.max_velocity

        self.prev_pos = self.cur_pos
        self.cur_pos = (self.puck_pos[0] > self.play_area['centerx'])

        self.info.append([[round((self.width-self.puck_pos[0])/self.width, 3), round((self.height-self.puck_pos[1])/self.height, 3),
                    -1 * round((vel[0])/self.width, 3), -1 * round((vel[1])/self.height, 3)]])

    def check_goal(self):
        if (self.puck_pos[0] < self.play_area['left'] - 2 * self.puck_radius and
            self.play_area['centery'] - self.goal_width / 2 <= self.puck_pos[1] <= self.play_area['centery'] + self.goal_width / 2):
            self.done = True

        elif (self.puck_pos[0] > self.play_area['right'] - 2 * self.puck_radius and
            self.play_area['centery'] - self.goal_width / 2 <= self.puck_pos[1] <= self.play_area['centery'] + self.goal_width / 2):
            self.done = True

    def get_observation(self):
        return np.array([self.puck_pos[0] / self.width,
                         self.puck_pos[1] / self.height,
                         self.puck_vel[0] / self.width,
                         self.puck_vel[1] / self.height,
                         self.player_pos[0] / self.width,
                         self.player_pos[1] / self.height,
                         self.player_vel[0] / self.width,
                         self.player_vel[1] / self.height], dtype=np.float32)

    def calculate_reward(self):
        reward = 0

        if (self.puck_pos[0] < self.play_area['left'] - 2 * self.puck_radius and
            self.play_area['centery'] - self.goal_width / 2 <= self.puck_pos[1] <= self.play_area['centery'] + self.goal_width / 2):
            reward -= 2000

        elif (self.puck_pos[0] > self.play_area['right'] - 2 * self.puck_radius and
            self.play_area['centery'] - self.goal_width / 2 <= self.puck_pos[1] <= self.play_area['centery'] + self.goal_width / 2):
            reward += 2000

        elif (self.puck_pos[0] <= self.play_area['centerx']):
            reward -= 0.1

        if (self.cur_pos) and (not self.prev_pos):
          reward += self.puck_vel[0] * 1.5

        reward += abs(sqrt(self.player_vel[0]**2 + self.player_vel[1]**2)) * 0.01

        return reward

    def game_over(self):
        if self.count >= 750:
            self.done = True
        return self.done

    def ai_control(self):
        self.info[-1].append([round((self.width-self.ai_pos[0])/self.width, 3), round((self.height-self.ai_pos[1])/self.height, 3),
                     -1 * round((self.ai_vel[0])/self.width, 3), -1 * round((self.ai_vel[1])/self.height, 3)])

        if len(self.info) == 3:
            state = np.array(self.info[0][1] + self.info[0][0] +
                            self.info[1][1] + self.info[1][0] +
                            self.info[2][1] + self.info[2][0] )

            self.info.pop(0)
            #predicted_movement = self.model.predict(state.reshape(1, -1), verbose=0)[0]
            predicted_movement = self.model(state.reshape(1, -1), training=False)[0]
        else:
            predicted_movement = [0, 0]

        predicted_movement = [0, 0]

        self.ai_vel[0], self.ai_vel[1] = -1 * predicted_movement[0] * self.width, -1 * predicted_movement[1] * self.height

        # Update AI paddle position
        self.ai_pos += self.ai_vel

        # Ensure the paddle stays within bounds
        self.ai_pos[0] = max(self.play_area['centerx'] - self.paddle_radius, min(self.play_area['right'] - self.paddle_radius, self.ai_pos[0]))
        self.ai_pos[1] = max(self.play_area['top'] + self.paddle_radius, min(self.play_area['bottom'] - self.paddle_radius, self.ai_pos[1]))


# #Test the environment
# ct = time.time()
# env = AirHockeyEnv()
# obs = env.reset()
# done = False
# count = 200
# while not done and count>=0:
#     action = env.action_space.sample()
#     obs, reward, done, _ = env.step(action)
#     print(reward)
#     if count == 100:
#         ct = time.time()
#     #env.render(should_render=True)
#     count -= 1
# print(time.time() - ct)
# env.close()