import gym
from gym import spaces
import tkinter as tk
import numpy as np

import math
import sys, random

if sys.version_info.major > 2:
    import tkinter as tk
else:
    import Tkinter as tk

import keras
from keras.models import load_model
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler

import numpy as np

model = load_model('new_nn_model.h5')

RED, BLACK, WHITE, DARK_RED, BLUE = "red", "black", "white", "dark red", "blue"
ZERO = 2 
LOWER, UPPER = "lower", "upper"
HOME, AWAY = "Player 1", "Player 2"
START_SCORE = {HOME: 0, AWAY: 0}
FPS = 125 # fps
FONT = "ms 50"
MAX_SPEED, PADDLE_SPEED, MIN_SPEED = 6, 4, 0.1

SCREEN_WIDTH = 380
SCREEN_HEIGHT = 800


def str_dict(dic):
    return "%s: %d, %s: %d" % (HOME, dic[HOME], AWAY, dic[AWAY])
    
def rand():
    return random.choice(((1, 1), (1, -1), (-1, 1), (-1, -1))) 



class AirHockeyEnv(gym.Env):
    def __init__(self):
        super(AirHockeyEnv, self).__init__()

        self.screen_width = SCREEN_WIDTH
        self.screen_height = SCREEN_HEIGHT
        self.score = START_SCORE

        self.velocities = [-4, -2, 0, 2, 4]
        self.actions = []
        for vel_x in self.velocities:
            for vel_y in self.velocities:
                self.actions.append([vel_x, vel_y])

        self.action_space = spaces.Discrete(25)
        self.observation_space = spaces.Box(low=-15, high=15, shape=(12,), dtype=np.float32)

        self.root = None  # Keep a reference to the Tkinter root

        self.reset()

    def reset(self):
        if self.root is not None:
            self.root.destroy()  # Destroy the existing Tkinter window if it exists

        self.root = tk.Tk()
        self.home = Home(self.root, (self.screen_width, self.screen_height))
        self.state = self.get_state()

        #self.root.after(10, self.step)
        #self.root.mainloop()


        return self.state

    def step(self, action):
        self.home.p2.vx, self.home.p2.vy = self.actions[action]

        # print("Action:", action)

        self.home.update()
        state = self.get_state()

        # Modify the reward and done criteria based on your game logic
        reward = 0.0
        if self.home.score[HOME] > self.score[HOME]:
            reward = -1000
            done = True
        elif self.home.score[AWAY] > self.score[AWAY]:
            reward = 1000
            done = True
        else:
            if (self.home.puck.y/self.screen_height) > 0.5:
                reward = -1
            done = False

        self.score = self.home.score

        #self.root.update_idletasks()  # Manually update the Tkinter window in a non-blocking way

        return state, reward, done, {}

    def render(self, mode='human'):
        # Rendering is done through the Tkinter GUI, so no specific render function needed
        pass

    def close(self):
        if self.root is not None:
            self.root.destroy()  # Destroy the Tkinter window on close

    def get_state(self):
        state = [
            self.home.p1.x / self.screen_width, self.home.p1.y / self.screen_height, self.home.p1.vx, self.home.p1.vy,
            self.home.p2.x / self.screen_width, self.home.p2.y / self.screen_height, self.home.p2.vx, self.home.p2.vy,
            self.home.puck.x / self.screen_width, self.home.puck.y / self.screen_height, self.home.puck.vx, self.home.puck.vy
        ]
        return np.array(state)


       
class Circle(object):
    def __init__(self, canvas, radius, position, color):
        self.can, self.radius = canvas, radius
        self.x, self.y = position
        
        self.Object = self.can.create_oval(self.x-self.radius, self.y-self.radius, 
                                    self.x+self.radius, self.y+self.radius, fill=color)
    def update(self, position):
        self.x, self.y = position
        self.can.coords(self.Object, self.x-self.radius, self.y-self.radius,
                                     self.x+self.radius, self.y+self.radius)
    def __eq__(self, other):
        overlapping = self.can.find_overlapping(self.x-self.radius, self.y-self.radius,
                                                self.x+self.radius, self.y+self.radius)
        return other.get_object() in overlapping
        
    def get_width(self):
        return self.radius
    def get_position(self):
        return self.x, self.y
    def get_object(self):
        return self.Object
        
class PuckManager(Circle):
    def __init__(self, canvas, radius, position):
        Circle.__init__(self, canvas, radius, position, BLACK)
        
class Paddle(Circle): 
    def __init__(self, canvas, radius, position):
        Circle.__init__(self, canvas, radius, position, RED)
        self.handle = self.can.create_oval(self.x-self.radius/2, self.y-self.radius/2, self.x+self.radius/2, self.y+self.radius/2, fill=DARK_RED)
        
    def update(self, position):
        Circle.update(self, position)
        self.can.coords(self.handle, self.x-self.radius/2, self.y-self.radius/2, self.x+self.radius/2, self.y+self.radius/2)
                                   
class Background(object):
    def __init__(self, canvas, screen, goal_w):
        self.can, self.goal_w = canvas, goal_w     
        self.width, self.height = screen
        
        self.draw_bg()
    
    def draw_bg(self):
        self.can.config(bg=WHITE, width=self.width, height=self.height)
        d = self.goal_w/3.5
        self.can.create_oval(self.width/2-d, self.height/2-d, self.width/2+d, self.height/2+d, fill=WHITE, outline=BLUE)
        self.can.create_line(ZERO, self.height/2, self.width, self.height/2, fill=BLUE)
        self.can.create_line(ZERO, ZERO, ZERO, self.height, fill=BLUE)
        self.can.create_line(self.width, ZERO, self.width, self.height, fill=BLUE)
    
        self.can.create_line(ZERO, ZERO, self.width/2-self.goal_w/2, ZERO, fill=BLUE)
        self.can.create_line(self.width/2+self.goal_w/2, ZERO, self.width, ZERO, fill=BLUE)
        
        self.can.create_line(ZERO, self.height, self.width/2-self.goal_w/2, self.height, fill=BLUE)
        self.can.create_line(self.width/2+self.goal_w/2, self.height, self.width, self.height, fill=BLUE)
                                                                     
    def is_position_valid(self, position, width, constraint=None):
        x, y = position

        if constraint == None and self.is_in_goal(position, width):
            return True
        elif (x - width < ZERO or x + width > self.width or 
            y - width < ZERO or y + width > self.height):
            return False
        elif constraint == LOWER:
            return y - width > self.height/2
        elif constraint == UPPER:
            return y + width < self.height/2
        else:
            return True    

    def is_in_goal(self, position, radius):
        x, y = position
        if (y - radius <= ZERO and x - radius > self.width/2 - self.goal_w/2 and x + radius < self.width/2 + self.goal_w/2):
            return HOME
        elif (y + radius >= self.height and x - radius > self.width/2 - self.goal_w/2 and x + radius < self.width/2 + self.goal_w/2):
            return AWAY
        else:
            return False
            
    def get_screen(self):
        return self.width, self.height   
    
    def get_goal_w(self):
        return self.goal_w
        
class Puck(object):
    def __init__(self, canvas, background):
        self.background = background
        self.screen = self.background.get_screen()
        self.x, self.y = self.screen[0]/2, self.screen[1]/2
        self.can, self.radius = canvas, 16 
        self.vx, self.vy = 1, -1
        self.friction = 0.99
        self.cushion = self.radius*0.05
        self.cooldown = 0
        
        self.puck = PuckManager(canvas, self.radius, (self.y, self.x))
        
    def update(self):
        
        if abs(self.vx) > MIN_SPEED: self.vx *= self.friction
        if abs(self.vy) > MIN_SPEED: self.vy *= self.friction

        self.vx = min(self.vx, MAX_SPEED)
        self.vx = max(self.vx, -MAX_SPEED)
        self.vy = min(self.vy, MAX_SPEED)
        self.vy = max(self.vy, -MAX_SPEED)
        
        x, y = self.x + (self.vx/FPS*self.screen[0]), self.y + (self.vy/FPS*self.screen[1])

        if not self.background.is_position_valid((x, y), self.radius):
            if x - self.radius < ZERO or x + self.radius > self.screen[0]:
                self.vx *= -1
            if y - self.radius < ZERO or y + self.radius > self.screen[1]:
                self.vy *= -1

            x, y = self.x + (self.vx/FPS*self.screen[0]), self.y + (self.vy/FPS*self.screen[1])
            
        self.x, self.y = x, y
        if self.cooldown > 0:
            self.cooldown -= 1
        self.puck.update((self.x, self.y))

    def hit(self, paddle, moving):
        x, y = paddle.x, paddle.y

        if moving:        
            if (x > self.x - self.cushion and x < self.x + self.cushion or 
                                                    abs(self.vx) > MAX_SPEED):
                xpower = 1
            else:
                xpower = 5 if self.vx < 1 else 2
            if (y > self.y - self.cushion and y < self.y + self.cushion or 
                                                    abs(self.vy) > MAX_SPEED):
                ypower = 1
            else:
                ypower = 5 if self.vy < 1 else 2
        else:
            xpower, ypower = 1, 1
            
        if self.x + self.cushion < x:
            xpower *= -1
        if self.y + self.cushion < y:
            ypower *= -1
        
        self.vx = abs(self.vx)*xpower
        self.vy = abs(self.vy)*ypower

    
    def __eq__(self, other):
        return other == self.puck
    def in_goal(self):
        return self.background.is_in_goal((self.x, self.y), self.radius)

class Player(object):
    def __init__(self, master, canvas, background, puck, constraint):
        self.puck, self.background = puck, background
        self.constraint, self.v = constraint, PADDLE_SPEED
        self.screen = self.background.get_screen()
        self.x = self.screen[0]/2
        self.y = 60 if self.constraint == UPPER else self.screen[1] - 50

        self.vx = 0
        self.vy = 0

        self.radius = 22

        self.paddle = Paddle(canvas, self.radius, (self.x, self.y))
        self.up, self.down, self.left, self.right = False, False, False, False
        
        
    def update(self, state):
        x, y = self.x, self.y
        if self.constraint == LOWER:
            vx, vy = self.vx, self.vy
            x, y = self.x + (self.vx/FPS*self.screen[0]), self.y + (self.vy/FPS*self.screen[1])

            if vx > 0:
                self.MoveRight()
            elif vx < 0:
                self.MoveLeft()
            else:
                self.LeftRelease()
                self.RightRelease

            if vy > 0:
                self.MoveDown()
            elif vy < 0:
                self.MoveUp()
            else:
                self.UpRelease()
                self.DownRelease

        else:
            state_x = np.array([state])
            [vx, vy] = model.predict([state_x], verbose = 0)[0]
            vx *= 3.5
            vy *= 3.5
            self.vx, self.vy = vx, vy
            x, y = self.x + (self.vx/FPS*self.screen[0]), self.y + (self.vy/FPS*self.screen[1])

            if vx > 0:
                self.MoveRight()
            elif vx < 0:
                self.MoveLeft()
            else:
                self.LeftRelease()
                self.RightRelease

            if vy > 0:
                self.MoveDown()
            elif vy < 0:
                self.MoveUp()
            else:
                self.UpRelease()
                self.DownRelease

        if not self.background.is_position_valid((x, y), self.paddle.get_width(), self.constraint):
            x = max(x, ZERO + self.radius)
            x = min(x, self.background.get_screen()[0] - self.radius)
            if self.constraint == LOWER:
                y = max(y, self.background.get_screen()[1]/2 + self.radius)
                y = min(y, self.background.get_screen()[1] - self.radius)
            else:
                y = max(y, ZERO + self.radius)
                y = min(y, self.background.get_screen()[1]/2 - self.radius)

        self.x, self.y = x, y
        self.paddle.update((self.x, self.y))

        if self.puck == self.paddle:
            moving = any((self.up, self.down, self.left, self.right))
            self.puck.hit(self, moving)
    
    def MoveUp(self, callback=False):
        self.up = True
    def MoveDown(self, callback=False):
        self.down = True
    def MoveLeft(self, callback=False):
        self.left = True
    def MoveRight(self, callback=False):
        self.right = True
    def UpRelease(self, callback=False):
        self.up = False
    def DownRelease(self, callback=False):
        self.down = False
    def LeftRelease(self, callback=False):
        self.left = False
    def RightRelease(self, callback=False):
        self.right = False
        
class Home(object):
    def __init__(self, master, screen, score=START_SCORE.copy()):
        self.frame = tk.Frame(master)
        self.frame.pack()
        self.can = tk.Canvas(self.frame)
        self.can.pack()
        background = Background(self.can, screen, screen[0]*0.45)
        self.puck = Puck(self.can, background)
        self.p1 = Player(master, self.can, background, self.puck, UPPER)
        self.p2 = Player(master, self.can, background, self.puck, LOWER)
        
        master.bind("<Return>", self.reset)
        master.bind("<r>", self.reset)
        
        master.title(str_dict(score))
        
        self.master, self.screen, self.score = master, screen, score
        
        self.update()
        
    def reset(self, callback=False):
        self.frame.destroy()
        self.__init__(self.master, self.screen, self.score)
        
    def update(self):
        self.puck.update()
        state = [self.p1.x / self.screen[0], self.p1.y / self.screen[1], self.p1.vx, self.p1.vy,
                 self.p2.x / self.screen[0], self.p2.y / self.screen[1], self.p2.vx, self.p2.vy,
                 self.puck.x / self.screen[0], self.puck.y / self.screen[1], self.puck.vx, self.puck.vy]
        
        self.p1.update(state)
        self.p2.update(state)
        if not self.puck.in_goal():
            self.frame.after(int(1/FPS*1000), self.update) 
        else:
            winner = HOME if self.puck.in_goal() == AWAY else AWAY
            self.update_score(winner)
            
    def update_score(self, winner):
        self.score[winner] += 1
        self.reset()

            