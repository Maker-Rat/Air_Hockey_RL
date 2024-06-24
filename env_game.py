import gym
from gym import spaces
import numpy as np
import pygame
import tensorflow
from keras.models import load_model

class AirHockeyEnv(gym.Env):
    def __init__(self):
        super(AirHockeyEnv, self).__init__()
        # Initialize Pygame
        pygame.init()
        self.width = 800
        self.height = 400
        self.screen = pygame.display.set_mode([self.width, self.height])
        
        # Define dimensions
        self.play_area = pygame.Rect(50, 50, self.width - 100, self.height - 100)
        
        # Define game objects' sizes
        self.puck_radius = self.height / 11.25 / 2
        self.paddle_radius = self.height / 8.18 / 2

        self.goal_width = 150
        self.goal_height = 5

        self.friction = 0.995
        self.max_velocity = 20

        self.x_velocities = [-15, -7.5, 0, 7.5, 15]
        self.y_velocities = [-15, -7.5, 0, 7.5, 15]

        self.done = False

        self.velocities = []
        for i in self.x_velocities:
            for j in self.y_velocities:
                self.velocities.append([i, j])

        self.info = [[[0.5, 0.5, 0, 0],[0.083, 0.5, 0, 0]], [[0.5, 0.5, 0, 0],[0.083, 0.5, 0, 0]]]
        self.state = []

        self.count = 0
        
        # Define action space
        self.action_space = spaces.Discrete(25)
        
        # Define observation space
        self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)
        
        # Load model
        #self.model = load_model('my_model_new.h5')
        self.model = tensorflow.keras.models.load_model('my_model_new.h5', compile=False)
        self.model.compile()
        
        # Other initialization
        self.clock = pygame.time.Clock()
        self.fps = 60
        self.reset()

    def step(self, action):
        # Convert action to velocity
        self.player_vel[0], self.player_vel[1] = self.velocities[action]

        # Update player position
        self.player_pos[0] += self.player_vel[0]
        self.player_pos[1] += self.player_vel[1]

        # Ensure the paddle stays within bounds
        self.player_pos[0] = max(self.play_area.left + self.paddle_radius, min(self.play_area.centerx - self.paddle_radius, self.player_pos[0]))
        self.player_pos[1] = max(self.play_area.top + self.paddle_radius, min(self.play_area.bottom - self.paddle_radius, self.player_pos[1]))

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
        self.player_pos = np.array([self.play_area.left + self.paddle_radius + 10, self.height / 2], dtype=np.float64)
        self.player_vel = np.array([0, 0], dtype=np.float64)
        self.ai_pos = np.array([self.play_area.right - self.paddle_radius - 10, self.height / 2], dtype=np.float64)
        self.ai_vel = np.array([0, 0], dtype=np.float64)
        self.done = False
        self.count = 0
        return self.get_observation()

    def render(self, should_render=True):
        if not should_render:
            return 
        
        # Fill the background with white
        self.screen.fill((255, 255, 255))

        # Draw the play area
        pygame.draw.rect(self.screen, (0, 0, 0), self.play_area, 2)

        # Draw vertical center
        pygame.draw.line(self.screen, (0, 0, 0), (self.play_area.centerx, self.play_area.top), (self.play_area.centerx, self.play_area.bottom), 2)

        # Draw the puck
        pygame.draw.circle(self.screen, (0, 0, 0), (int(self.puck_pos[0]), int(self.puck_pos[1])), int(self.puck_radius))

        # Draw the player's paddle
        pygame.draw.circle(self.screen, (0, 0, 255), (int(self.player_pos[0]), int(self.player_pos[1])), int(self.paddle_radius))

        # Draw the AI's paddle
        pygame.draw.circle(self.screen, (255, 0, 0), (int(self.ai_pos[0]), int(self.ai_pos[1])), int(self.paddle_radius))

        # Draw the goals
        pygame.draw.rect(self.screen, (0, 255, 0), (self.play_area.left - self.goal_height, self.play_area.centery - self.goal_width / 2, self.goal_height, self.goal_width))
        pygame.draw.rect(self.screen, (0, 255, 0), (self.play_area.right, self.play_area.centery - self.goal_width / 2, self.goal_height, self.goal_width))

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        self.clock.tick(self.fps)

    def close(self):
        pygame.quit()

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
        if (self.puck_pos[0] <= self.play_area.left + self.puck_radius and not (self.play_area.centery - self.goal_width / 2 <= self.puck_pos[1] <= self.play_area.centery + self.goal_width / 2)):
            self.puck_pos[0] = self.play_area.left + self.puck_radius
            self.puck_vel[0] = -self.puck_vel[0]
        if (self.puck_pos[0] >= self.play_area.right - self.puck_radius and not (self.play_area.centery - self.goal_width / 2 <= self.puck_pos[1] <= self.play_area.centery + self.goal_width / 2)):
            self.puck_pos[0] = self.play_area.right - self.puck_radius
            self.puck_vel[0] = -self.puck_vel[0]
        if self.puck_pos[1] <= self.play_area.top + self.puck_radius:
            self.puck_pos[1] = self.play_area.top + self.puck_radius
            self.puck_vel[1] = -self.puck_vel[1]
        if self.puck_pos[1] >= self.play_area.bottom - self.puck_radius:
            self.puck_pos[1] = self.play_area.bottom - self.puck_radius
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

        self.info.append([[round((self.width-self.puck_pos[0])/self.width, 3), round((self.height-self.puck_pos[1])/self.height, 3),
                    -1 * round((vel[0])/self.width, 3), -1 * round((vel[1])/self.height, 3)]])

    def check_goal(self):
        if (self.puck_pos[0] < self.play_area.left - 2 * self.puck_radius and
            self.play_area.centery - self.goal_width / 2 <= self.puck_pos[1] <= self.play_area.centery + self.goal_width / 2):
            print("Goal by Player!")  # Goal by Player means puck went into AI's goal
            self.done = True
            #self.reset()

        elif (self.puck_pos[0] > self.play_area.right - 2 * self.puck_radius and
            self.play_area.centery - self.goal_width / 2 <= self.puck_pos[1] <= self.play_area.centery + self.goal_width / 2):
            print("Goal by AI!")  # Goal by AI means puck went into Player's goal
            self.done = True
            #self.reset()

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
        if (self.puck_pos[0] < self.play_area.left - 2 * self.puck_radius and
            self.play_area.centery - self.goal_width / 2 <= self.puck_pos[1] <= self.play_area.centery + self.goal_width / 2):
            return -1000

        elif (self.puck_pos[0] > self.play_area.right - 2 * self.puck_radius and
            self.play_area.centery - self.goal_width / 2 <= self.puck_pos[1] <= self.play_area.centery + self.goal_width / 2):
            return 1000
        
        elif (self.puck_pos[0] <= self.play_area.centerx):
            return -1
        
        else:
            return 0

    def game_over(self): 
        if self.count >= 750:
            self.done = True
        return self.done

    def ai_control(self):
        self.info[-1].append([round((self.width-self.ai_pos[0])/self.width, 3), round((self.height-self.ai_pos[1])/self.height, 3),
                     -1 * round((self.ai_vel[0])/self.width, 3), -1 * round((self.ai_vel[1])/self.height, 3)])

        #print(self.info)
        
        if len(self.info) == 3:
            state = np.array(self.info[0][1] + self.info[0][0] +
                            self.info[1][1] + self.info[1][0] +
                            self.info[2][1] + self.info[2][0] )
            
            self.info.pop(0)
            predicted_movement = self.model.predict(state.reshape(1, -1), verbose=0)[0]
            # print(predicted_movement[0], predicted_movement[1])
            #predicted_movement = [0, 0]
            #predicted_movement = self.model(state.reshape(1, -1), training=False)[0]
        else:
            predicted_movement = [0, 0]

        self.ai_vel[0], self.ai_vel[1] = -1 * predicted_movement[0] * self.width, -1 * predicted_movement[1] * self.height

        # Update AI paddle position
        self.ai_pos += self.ai_vel

        # Ensure the paddle stays within bounds
        self.ai_pos[0] = max(self.play_area.centerx - self.paddle_radius, min(self.play_area.right- self.paddle_radius, self.ai_pos[0]))
        self.ai_pos[1] = max(self.play_area.top + self.paddle_radius, min(self.play_area.bottom - self.paddle_radius, self.ai_pos[1]))

# Test the environment
# env = AirHockeyEnv()
# obs = env.reset()
# done = False
# while not done:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             done = True
#     action = env.action_space.sample()
#     obs, reward, done, _ = env.step(action)
#     print(reward)
#     env.render(should_render=True)
# env.close()
