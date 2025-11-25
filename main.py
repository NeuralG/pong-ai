import torch
import random
import numpy as np
from collections import deque
from game import PongGame
from agent import Agent
import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, meanScores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(meanScores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(meanScores)-1, meanScores[-1], str(meanScores[-1]))
    plt.show(block=False)
    plt.pause(.1)

def train():
    plotScores = []
    plotMeanScores = []
    totalScore = 0
    highScore = 0 

    game = PongGame()
    agent = Agent()

    agent.loadAgent()

    print("Starting training...")

    while True:

        game.reset()
        stateOld = game.getState()
        done = False

        while not done:
            action = agent.getAction(stateOld)
            reward, score, done = game.playStep(action)
            stateNew = game.getState()
            agent.remember(stateOld, action, reward, stateNew, done)
            agent.trainShortMemory(stateOld, action, reward, stateNew, done)
            stateOld = stateNew

        agent.n_games += 1
        agent.trainLongMemory()

        if agent.n_games % 50 == 0:
            agent.saveAgent()

        if score > highScore:
            highScore = score
        totalScore += score
        meanScore = totalScore / agent.n_games

        plotScores.append(highScore)
        plotMeanScores.append(meanScore)

        plot(plotScores, plotMeanScores)

if __name__ == "__main__":
    train()