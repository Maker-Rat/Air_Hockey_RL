import math
import sys, random

if sys.version_info.major > 2:
    import tkinter as tk
else:
    import Tkinter as tk

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
MAX_SPEED, PADDLE_SPEED, MIN_SPEED = 5, 4, 0.1

SCREEN_WIDTH = 380
SCREEN_HEIGHT = 800


def str_dict(dic):
    return "%s: %d, %s: %d" % (HOME, dic[HOME], AWAY, dic[AWAY])
    
def rand():
    return random.choice(((1, 1), (1, -1), (-1, 1), (-1, -1))) 
    

        
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
        print(self.vx, self.vy)
        print("yupppa")

        if moving:        
            if (x > self.x - self.cushion and x < self.x + self.cushion or 
                                                    abs(self.vx) > MAX_SPEED):
                if abs(self.vx) > MAX_SPEED:
                    print("KAPP")
                xpower = 1
            else:
                xpower = 5 if self.vx < 1 else 2
            if (y > self.y - self.cushion and y < self.y + self.cushion or 
                                                    abs(self.vy) > MAX_SPEED):
                ypower = 1
                print("yesss")
            else:
                ypower = 5 if self.vy < 1 else 2
        else:
            xpower, ypower = 1, 1
            print("yes")
            
        if self.x + self.cushion < x:
            xpower *= -1
        if self.y + self.cushion < y:
            ypower *= -1
        
        self.vx = abs(self.vx)*xpower
        self.vy = abs(self.vy)*ypower

        print(self.vx, self.vy)

    def hit1(self, paddle):
        puck_pos = [self.x, self.y]
        mallet_pos = [paddle.x, paddle.y]

        dx = puck_pos[0] - mallet_pos[0]
        dy = puck_pos[1] - mallet_pos[1]
        distance = math.sqrt(dx ** 2 + dy ** 2)

        self.x += dx
        self.y += dy

        if distance < self.radius + paddle.radius and self.cooldown == 0:
            print("yupppa")
            angle = math.atan2(dy, dx)
            relative_speed = (self.vx - paddle.vx, self.vy - paddle.vy)
            speed = math.sqrt(relative_speed[0] ** 2 + relative_speed[1] ** 2)
            self.vx = speed * math.cos(angle)
            self.vy = speed * math.sin(angle)
            self.cooldown = 5

    
    def __eq__(self, other):
        return other == self.puck
    def in_goal(self):
        return self.background.is_in_goal((self.x, self.y), self.radius)

class Player(object):
    def __init__(self, master, canvas, background, puck, constraint):
        self.puck, self.background = puck, background
        self.constraint, self.v = constraint, PADDLE_SPEED
        self.screen = self.background.get_screen()
        self.x = screen[0]/2
        self.y = 60 if self.constraint == UPPER else screen[1] - 50

        self.vx = 0
        self.vy = 0

        self.radius = 22

        self.paddle = Paddle(canvas, self.radius, (self.x, self.y))
        self.up, self.down, self.left, self.right = False, False, False, False
        
        if self.constraint == LOWER:
            master.bind('<Up>', self.MoveUp)
            master.bind('<Down>', self.MoveDown)
            master.bind('<KeyRelease-Up>', self.UpRelease)
            master.bind('<KeyRelease-Down>', self.DownRelease)
            master.bind('<Right>', self.MoveRight)
            master.bind('<Left>', self.MoveLeft)
            master.bind('<KeyRelease-Right>', self.RightRelease)
            master.bind('<KeyRelease-Left>', self.LeftRelease)
        else:
            pass
        
    def update(self, state):
        x, y = self.x, self.y
        if self.constraint == LOWER:
            if self.up:
                y = self.y - (self.v/FPS*self.screen[1])
                self.vy = -self.v
            if self.down: 
                y = self.y + (self.v/FPS*self.screen[1])
                self.vy = self.v
            if self.left: 
                x = self.x - (self.v/FPS*self.screen[0])
                self.vx = -self.v
            if self.right: 
                x = self.x + (self.v/FPS*self.screen[0])
                self.vx = self.v

        else:
            state_x = np.array([state])
            [vx, vy] = model.predict([state_x])[0]
            vx *= 3.5
            vy *= 3.5
            print(vx, vy)
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
        
        print(state)
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

                             
def play(screen):
    root = tk.Tk()
    Home(root, screen)
    root.mainloop()
            
if __name__ == "__main__":
    
    screen = SCREEN_WIDTH, SCREEN_HEIGHT
    
    play(screen)
