from snake import *
from utility import *
from cube import *

import pygame
import numpy as np
from tkinter import messagebox
from snake import Snake
import matplotlib.pyplot as plt


def main():
    pygame.init()
    win = pygame.display.set_mode((WIDTH, HEIGHT))

    snake_1 = Snake((255, 0, 0), (15, 15), SNAKE_1_Q_TABLE,'snake1')
    snake_2 = Snake((255, 255, 0), (5, 5), SNAKE_2_Q_TABLE,'snake2')
    snake_1.addCube()
    snake_2.addCube()

    snack = Cube(randomSnack(ROWS, snake_1), color=(0, 255, 0))

    clock = pygame.time.Clock()
    rewards_1 = []
    rewards_2 = []



    for i in range(80):
        reward_calculator_1 = 0
        reward_calculator_2 = 0
        while True:
            pygame.time.delay(50)
            clock.tick(30)

            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    if messagebox.askokcancel("Quit", "Do you want to save the Q-tables?"):
                        save(snake_1, snake_2)
                    pygame.quit()
                    exit()

                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    torch.save(snake_1.q_table, SNAKE_1_Q_TABLE)
                    torch.save(snake_2.q_table, SNAKE_2_Q_TABLE)
                    pygame.time.delay(1000)

            state_1, new_state_1, action_1 = snake_1.move(snack, snake_2)
            state_2, new_state_2, action_2 = snake_2.move(snack, snake_1)

            # print(state_1)
            # print(new_state_1)
            # print()


            snack, reward_1, win_1, win_2 = snake_1.calc_reward(snack, snake_2)
            snack, reward_2, win_2, win_1 = snake_2.calc_reward(snack, snake_1)

            reward_calculator_1 += reward_1
            reward_calculator_2 += reward_2

            snake_1.update_q_table(state_1, action_1, new_state_1, reward_1)
            snake_2.update_q_table(state_2, action_2, new_state_2, reward_2)
            if win_1 or win_2:
                break

            redrawWindow(snake_1, snake_2, snack, win)
        if i > 0:
            rewards_1.append(reward_calculator_1/ i)
            rewards_2.append(reward_calculator_2/ i)

    np.save(SNAKE_1_Q_TABLE, snake_1.q_table)
    np.save(SNAKE_2_Q_TABLE, snake_2.q_table)
    return rewards_1,rewards_2

def plot_rewards(rewards_1, rewards_2):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_1, label="Snake 1")
    plt.plot(rewards_2, label="Snake 2")
    plt.xlabel("iteration")
    plt.ylabel("average reward per iteration")
    plt.title("Avg rewards")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    rewards_1, rewards_2 = main()
    plot_rewards(rewards_1, rewards_2)