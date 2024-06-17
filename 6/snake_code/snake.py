import math

from cube import Cube
from constants import *
from utility import *
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from model import QNet, Training
# from model import Conv_QNet

LEFT = 0
DOWN = 3
RIGHT = 1
UP = 2

IMPOSSIBLE = 0
POSSIBLE = 1

class Snake:
    body = []
    turns = {}

    def __init__(self, color, pos, file_name=None, name=''):
        self.color = color
        self.head = Cube(pos, color=color)
        self.body.append(self.head)
        self.dirnx = 0
        self.dirny = 1
        self.name = name
        try:
            self.q_table = np.load(file_name)
        except:
            self.q_table = QNet(12, 256, 4)
            # self.q_table = Conv_QNet(12, 256, 4)

        # self.q_table = np.zeros((256, 4))
        self.lr = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.1
        self.training = Training(self.q_table, self.lr, self.discount_factor)
 

    def get_optimal_policy(self, state):
        converted_state = torch.tensor(state, dtype=torch.float)
        print(converted_state)
        prediction = self.q_table.forward(converted_state)
        print(prediction)
        move = torch.argmax(prediction).item()
        print(self.name + ' Movement -------> ' + str(move))
        return move

    def make_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        else:
            return self.get_optimal_policy(state)

    def update_q_table(self, state, action, next_state, reward):
        self.training.train_in_iteration(state, action, next_state, reward)


    def will_the_snake_lose(self, x=None,y=None, other_snake=None):

        if x >= 19 or x < 1 or y >= 19 or y < 1:
            return True

        if self.head.pos in list(map(lambda z: z.pos, self.body[1:])):
            return True
        
        if self.head.pos in list(map(lambda z: z.pos, other_snake.body)):
            if self.head.pos != other_snake.head.pos:
                return True
            else:
                if len(self.body) > len(other_snake.body):
                    return False
                elif len(self.body) == len(other_snake.body):
                    return True
                else:
                    return True

        return False

    def state_checker(self, snack, other_snake):
        # print(self.head.pos[0],self.head.pos[1])

        dir_l = (self.dirnx == -1 and self.dirny == 0)
        dir_r = (self.dirnx == 1 and self.dirny == 0)
        dir_u = (self.dirnx == 0 and self.dirny == -1)
        dir_d = (self.dirnx == 0 and self.dirny == 1)

        state = [

            (dir_r and self.will_the_snake_lose(self.head.pos[0]+1,self.head.pos[1], other_snake)) or
            (dir_l and self.will_the_snake_lose(self.head.pos[0]-1,self.head.pos[1], other_snake)) or
            (dir_u and self.will_the_snake_lose(self.head.pos[0],self.head.pos[1]-1, other_snake)) or
            (dir_d and self.will_the_snake_lose(self.head.pos[0],self.head.pos[1]+1, other_snake)),

            (dir_u and self.will_the_snake_lose(self.head.pos[0]+1,self.head.pos[1], other_snake)) or
            (dir_d and self.will_the_snake_lose(self.head.pos[0]-1,self.head.pos[1], other_snake)) or
            (dir_l and self.will_the_snake_lose(self.head.pos[0],self.head.pos[1]-1, other_snake)) or
            (dir_r and self.will_the_snake_lose(self.head.pos[0],self.head.pos[1]+1, other_snake)),

            (dir_d and self.will_the_snake_lose(self.head.pos[0]+1,self.head.pos[1], other_snake)) or
            (dir_u and self.will_the_snake_lose(self.head.pos[0]-1,self.head.pos[1], other_snake)) or
            (dir_r and self.will_the_snake_lose(self.head.pos[0],self.head.pos[1]-1, other_snake)) or
            (dir_l and self.will_the_snake_lose(self.head.pos[0],self.head.pos[1]+1, other_snake)),

            False, # Going back will cause it to hit itself

            dir_l,
            dir_r,
            dir_u,
            dir_d,

            snack.pos[0] < self.head.pos[0],
            snack.pos[0] > self.head.pos[0],  
            snack.pos[1] < self.head.pos[1],  
            snack.pos[1] > self.head.pos[1]
        ]
        return np.array(state, dtype=int)


    def move(self, snack, other_snake):
        state = self.state_checker(snack, other_snake)
        action = self.make_action(state)

        if action == 0: # Left
            self.dirnx = -1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 1: # Right
            self.dirnx = 1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 2: # Up
            self.dirny = -1
            self.dirnx = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 3: # Down
            self.dirny = 1
            self.dirnx = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

        for i, c in enumerate(self.body):
            p = c.pos[:]
            if p in self.turns:
                turn = self.turns[p]
                c.move(turn[0], turn[1])
                if i == len(self.body) - 1:
                    self.turns.pop(p)
            else:
                c.move(c.dirnx, c.dirny)

        new_state = (self.state_checker(snack,other_snake))
        return state, new_state, action

    def check_out_of_board(self):
        headPos = self.head.pos
        if headPos[0] >= 19 or headPos[0] < 1 or headPos[1] >= 19 or headPos[1] < 1:
            self.reset((random.randint(1, 18), random.randint(1, 18)))
            return True
        return False

    def calc_reward(self, snack, other_snake):
        reward = 0
        win_self, win_other = False, False

        if self.check_out_of_board():
            reward = -100
            win_other = True
            reset(self)

        if self.head.pos == snack.pos:
            self.addCube()
            snack = Cube(randomSnack(ROWS, self), color=(0, 255, 0))
            reward = 200

        if self.head.pos in list(map(lambda z: z.pos, self.body[1:])):
            reward = -70
            win_other = True
            reset(self)

        if self.head.pos in list(map(lambda z: z.pos, other_snake.body)):
            if self.head.pos != other_snake.head.pos:
                reward = -100
                win_other = True
                reset(self)

            else:
                if len(self.body) > len(other_snake.body):
                    reward = 150
                    win_self = True
                    reset(other_snake)

                elif len(self.body) == len(other_snake.body):
                    reward = 30
                    reset(other_snake)
                    reset(self)

                else:
                    reward = -120
                    win_other = True
                    reset(self)

        return snack, reward, win_self, win_other

    def reset(self, pos):
        self.head = Cube(pos, color=self.color)
        self.body = []
        self.body.append(self.head)
        self.turns = {}
        self.dirnx = 0
        self.dirny = 1

    def addCube(self):
        tail = self.body[-1]
        dx, dy = tail.dirnx, tail.dirny

        if dx == 1 and dy == 0:
            self.body.append(Cube((tail.pos[0] - 1, tail.pos[1]), color=self.color))
        elif dx == -1 and dy == 0:
            self.body.append(Cube((tail.pos[0] + 1, tail.pos[1]), color=self.color))
        elif dx == 0 and dy == 1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] - 1), color=self.color))
        elif dx == 0 and dy == -1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] + 1), color=self.color))

        self.body[-1].dirnx = dx
        self.body[-1].dirny = dy

    def draw(self, surface):
        for i, c in enumerate(self.body):
            if i == 0:
                c.draw(surface, True)
            else:
                c.draw(surface)

