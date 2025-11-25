import torch
import random
from collections import deque
from model import Linear_QNet, QTrainer
import os 
import pickle 
import math

MAX_MEMORY = 250_000
BATCH_SIZE = 1500
LR = 1e-4
DECAY_RATE = 0.005

MODEL_PATH = 'model/model.pth'
AGENT_STATE_PATH = 'model/agent_state.pkl'

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon_start = 1.0
        self.epsilon_min = 0.01
        self.epsilon = self.epsilon_start 
        self.gamma = 0.99
        self.memory = deque(maxlen=MAX_MEMORY)

        self.model = Linear_QNet(7, 128, 64, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def getAction(self, state):

        self.epsilon = self.epsilon_min + \
            (self.epsilon_start - self.epsilon_min) * \
            math.exp(-DECAY_RATE * self.n_games)
        
        finalMove = [0, 0, 0]

        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            finalMove[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            finalMove[move] = 1

        return finalMove

    def remember(self, state, action, reward, nextState, done):
        self.memory.append((state, action, reward, nextState, done))

    def trainShortMemory(self, state, action, reward, nextState, done):
        self.trainer.trainStep(state, action, reward, nextState, done)

    def trainLongMemory(self):
        if len(self.memory) >= BATCH_SIZE:
            arr = random.sample(self.memory, BATCH_SIZE)
        else:
            arr = self.memory

        states, actions, rewards, nextStates, dones = zip(*arr)
        self.trainer.trainStep(states, actions, rewards, nextStates, dones)

    def saveAgent(self, fileNameModel=MODEL_PATH, fileNameState=AGENT_STATE_PATH):
        os.makedirs(os.path.dirname(fileNameModel), exist_ok=True)
        torch.save(self.model.state_dict(), fileNameModel)
        agentState = {
            'nGames': self.n_games, 
            'epsilon': self.epsilon,
            'epsilonStart': self.epsilon_start,
            'epsilonMin': self.epsilon_min,
            'gamma': self.gamma,
            'memory': self.memory, 
        }
        
        with open(fileNameState, 'wb') as f:
            pickle.dump(agentState, f)
        
        print(f"Agent state and model weights successfully saved.")

    def loadAgent(self, fileNameModel=MODEL_PATH, fileNameState=AGENT_STATE_PATH):
        
        if os.path.exists(fileNameModel):
            self.model.load_state_dict(torch.load(fileNameModel))
            self.model.eval() 
            print(f"Model weights loaded.")
        else:
            print("No recorded model weights found. Starting from scratch.")
            return

        if os.path.exists(fileNameState):
            with open(fileNameState, 'rb') as f:
                agentState = pickle.load(f)
            
            self.n_games = agentState.get('nGames', 0)
            self.epsilon = agentState.get('epsilon', self.epsilon_start)
            self.gamma = agentState.get('gamma', self.gamma) 
            self.memory = agentState.get('memory', deque(maxlen=MAX_MEMORY))
            
            print(f"Agent state loaded (Games: {self.n_games}, Epsilon: {self.epsilon:.2f}).")
        else:
            print("No recorded agent state file found.")