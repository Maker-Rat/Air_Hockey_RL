import pygame
import numpy as np
import tensorflow as tf
from keras.models import load_model
import tensorflow

import gymnasium as gym
import numpy as np
from ddqn import DDQNAgent
import matplotlib.pyplot as plt
import torch
import shutil
from env_game import AirHockeyEnv


#--------------------------------------------------------------------------------------------------

MODE = 1

if MODE:
    env = AirHockeyEnv()

    # Get the state and action sizes
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Set the random seed
    seed = 0

    # Create the DDQN agent
    agent = DDQNAgent(state_size, action_size, seed)

    agent.qnetwork_local.load_state_dict(torch.load("Test1/DDQN_Model_1.pt"))
    agent.qnetwork_target.load_state_dict(torch.load("Test1/DDQN_Model_1.pt"))


#--------------------------------------------------------------------------------------------------


model = tensorflow.keras.models.load_model('my_model_new.h5', compile=False)
model.compile()

# Initialize Pygame
pygame.init()

# Define dimensions
width = 800
height = 400
play_area = pygame.Rect(50, 50, width, height)

# Define colors
white = (255, 255, 255)
black = (0, 0, 0)

# Define game objects' sizes
puck_radius = height / 11.25 / 2
paddle_radius = height / 8.18 / 2

# Goal area dimensions
goal_width = 150
goal_height = 5

# Set up display
window = pygame.display.set_mode((width + 100, height + 100))
pygame.display.set_caption('Air Hockey Game')

# Define initial positions (ensure floats for calculations)
initial_puck_pos = np.array([width / 2, height / 2], dtype=np.float64)
puck_pos = np.array([width / 2, height / 2], dtype=np.float64)
puck_vel = np.array([0, 0], dtype=np.float64)

player_pos = np.array([paddle_radius + 10, height / 2], dtype=np.float64)
player_vel = np.array([0, 0], dtype=np.float64)

ai_pos = np.array([width - paddle_radius - 10, height / 2], dtype=np.float64)
ai_vel = np.array([0, 0], dtype=np.float64)

clock = pygame.time.Clock()
fps = 60

# Friction coefficient and max velocity
friction = 0.995
max_velocity = 20

info = []
state = []

def draw_play_area():
    window.fill(white)
    pygame.draw.rect(window, black, play_area, 2)
    pygame.draw.line(window, black, (play_area.centerx, play_area.top), (play_area.centerx, play_area.bottom), 2)
    pygame.draw.rect(window, black, (play_area.left, play_area.centery - goal_width / 2, goal_height, goal_width))
    pygame.draw.rect(window, black, (play_area.right - goal_height, play_area.centery - goal_width / 2, goal_height, goal_width))

def draw_puck(position):
    pygame.draw.circle(window, black, (int(position[0]) + 50, int(position[1]) + 50), int(puck_radius))

def draw_paddle(position):
    pygame.draw.circle(window, black, (int(position[0]) + 50, int(position[1]) + 50), int(paddle_radius))

def draw():
    draw_play_area()
    draw_puck(puck_pos)
    draw_paddle(player_pos)
    draw_paddle(ai_pos)
    pygame.display.flip()

def handle_input():
    global player_vel, player_pos
    keys = pygame.key.get_pressed()
    player_vel[0] = 0
    player_vel[1] = 0
    if keys[pygame.K_LEFT]:
        player_vel[0] = -15
    if keys[pygame.K_RIGHT]:
        player_vel[0] = 15
    if keys[pygame.K_UP]:
        player_vel[1] = -15
    if keys[pygame.K_DOWN]:
        player_vel[1] = 15

    x_velocities = [-15, -7.5, 0, 7.5, 15]
    y_velocities = [-15, -7.5, 0, 7.5, 15]


    if MODE:
        velocities = []
        for i in x_velocities:
            for j in y_velocities:
                velocities.append([i, j])

        state = np.array([puck_pos[0] / width, 
                            puck_pos[1] / height,  
                            puck_vel[0] / width, 
                            puck_vel[1] / height,
                            player_pos[0] / width, 
                            player_pos[1] / height,
                            player_vel[0] / width, 
                            player_vel[1] / height], dtype=np.float32)
        
        player_vel = velocities[agent.act(state, eps=0.05)]

    # Update player position
    player_pos[0] += player_vel[0]
    player_pos[1] += player_vel[1]

    # Ensure the paddle stays within bounds
    player_pos[0] = max(paddle_radius, min(play_area.centerx - 3 * paddle_radius, player_pos[0]))
    player_pos[1] = max(paddle_radius, min(height - paddle_radius, player_pos[1]))

def handle_collision(paddle_pos, paddle_vel, paddle_radius):
    global puck_pos, puck_vel

    dist = np.linalg.norm(puck_pos - paddle_pos)
    if dist <= puck_radius + paddle_radius:
        # Calculate the direction of the bounce
        normal = (puck_pos - paddle_pos) / dist
        relative_velocity = puck_vel - paddle_vel
        speed = np.dot(relative_velocity, normal)
        
        # Only update if moving towards each other
        if speed < 0:
            puck_vel -= 2 * speed * normal

        # Move puck out of collision
        overlap = puck_radius + paddle_radius - dist
        puck_pos += normal * overlap

def update_puck():
    global puck_pos, puck_vel, info

    # Update puck position
    vel = list(puck_vel)
    puck_pos += puck_vel

    # Handle collisions with walls, allowing puck to pass through the goal areas
    if (puck_pos[0] <= puck_radius and not (play_area.centery - goal_width / 2 <= puck_pos[1] <= play_area.centery + goal_width / 2)):
        puck_pos[0] = puck_radius
        puck_vel[0] = -puck_vel[0]
    if (puck_pos[0] >= width - puck_radius and not (play_area.centery - goal_width / 2 <= puck_pos[1] <= play_area.centery + goal_width / 2)):
        puck_pos[0] = width - puck_radius
        puck_vel[0] = -puck_vel[0]
    if puck_pos[1] <= puck_radius:
        puck_pos[1] = puck_radius
        puck_vel[1] = -puck_vel[1]
    if puck_pos[1] >= height - puck_radius:
        puck_pos[1] = height - puck_radius
        puck_vel[1] = -puck_vel[1]

    # Handle collisions with paddles
    handle_collision(player_pos, player_vel, paddle_radius)
    handle_collision(ai_pos, ai_vel, paddle_radius)

    # Update velocities based on friction
    puck_vel *= friction

    # Limit maximum velocity
    speed = np.linalg.norm(puck_vel)
    if speed > max_velocity:
        puck_vel = (puck_vel / speed) * max_velocity

    info.append([[round((width-puck_pos[0])/width, 3), round((height-puck_pos[1])/height, 3),
                 -1 * round((vel[0])/width, 3), -1 * round((vel[1])/height, 3)]])


def check_goal():
    global puck_pos, puck_vel, player_pos, ai_pos, player_vel, ai_vel
    if (puck_pos[0] < play_area.left - 2 * puck_radius and
        play_area.centery - goal_width / 2 <= puck_pos[1] <= play_area.centery + goal_width / 2):
        print("Goal by Player!")  # Goal by Player means puck went into AI's goal
        reset_game()

    elif (puck_pos[0] > play_area.right - 2 * puck_radius and
          play_area.centery - goal_width / 2 <= puck_pos[1] <= play_area.centery + goal_width / 2):
        print("Goal by AI!")  # Goal by AI means puck went into Player's goal
        reset_game()


def reset_game():
    global puck_pos, puck_vel, player_pos, ai_pos, player_vel, ai_vel
    puck_pos = initial_puck_pos.copy()
    puck_vel = np.array([0, 0], dtype=np.float64)
    player_pos = np.array([paddle_radius + 10, height / 2], dtype=np.float64)
    player_vel = np.array([0, 0], dtype=np.float64)
    ai_pos = np.array([width - paddle_radius - 10, height / 2], dtype=np.float64)
    ai_vel = np.array([0, 0], dtype=np.float64)

def update_game():
    handle_input()
    update_puck()
    check_goal()
    draw()



# Placeholder for model loading and prediction logic
def ai_control():
    global puck_pos, puck_vel, ai_pos, ai_vel, info, state

    info[-1].append([round((width-ai_pos[0])/width, 3), round((height-ai_pos[1])/height, 3),
                     -1 * round((ai_vel[0])/width, 3), -1 * round((ai_vel[1])/height, 3)])

    #print(info)
    
    if len(info) == 3:
        state = np.array(info[0][1] + info[0][0] +
                         info[1][1] + info[1][0] +
                         info[2][1] + info[2][0] )
        
        info.pop(0)
        predicted_movement = model.predict(state.reshape(1, -1), verbose=0)[0]
        # print(predicted_movement[0], predicted_movement[1])
        #predicted_movement = [0, 0]
    else:
        predicted_movement = [0, 0]

    ai_vel[0], ai_vel[1] = -0.9 * predicted_movement[0] * width, -1.1 * predicted_movement[1] * height

    # Update AI paddle position
    ai_pos += ai_vel

    # Ensure the paddle stays within bounds
    ai_pos[0] = max(play_area.centerx - paddle_radius, min(width - paddle_radius, ai_pos[0]))
    ai_pos[1] = max(paddle_radius, min(height - paddle_radius, ai_pos[1]))


def main():
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        update_game()
        ai_control()
        clock.tick(fps)

    pygame.quit()

if __name__ == "__main__":
    main()
