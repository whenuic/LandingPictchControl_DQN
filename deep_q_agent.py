import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

from pathlib import Path
from collections import deque
from itertools import repeat


script_name = Path(__file__).stem
seed_val = int(time.time())
print(seed_val)
random.seed(int(time.time()))


PRINT_SCORE_FREQUENCY = 1  # print score every * episode

MAX_MEMORY = 100_000
BATCH_SIZE = 2000
LR = 0.005
GAMMA = 0.95
EPSILON = 200
EXPLORATION_RATE = 2

LOAD_MODEL_FROM_FILE = True
LOAD_MODEL_FROM_FILE_NAME = ""


class Q_Network(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=1024):
        super().__init__()
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x), inplace=True)
        x = F.relu(self.linear2(x), inplace=True)
        x = self.linear3(x)
        return x

    def save(self, record):
        file_name = (
            "model_" + str(record) + "_" + time.strftime("%Y%m%d_%H%M%S") + ".pth"
        )
        model_folder_path = (
            "/Users/wenjiang/Desktop/python/aircraft_simulation/" + script_name
        )
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float).to(torch.device("mps"))
        next_state = torch.tensor(next_state, dtype=torch.float).to(torch.device("mps"))
        action = torch.tensor(action, dtype=torch.long).to(torch.device("mps"))
        reward = torch.tensor(reward, dtype=torch.float).to(torch.device("mps"))
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0).to(torch.device("mps"))
            next_state = torch.unsqueeze(next_state, 0).to(torch.device("mps"))
            action = torch.unsqueeze(action, 0).to(torch.device("mps"))
            reward = torch.unsqueeze(reward, 0).to(torch.device("mps"))
            done = (done,)

        # 1: predicted Q values with current state
        # pred is a three element vector due to three possible actions
        pred = self.model(state)
        td_target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(
                    self.model(next_state[idx])
                )

            td_target[idx][torch.argmax(action).item()] = Q_new

        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        self.optimizer.zero_grad()
        loss = self.criterion(td_target, pred)
        loss.backward()

        self.optimizer.step()


class Agent:

    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.epsilon = 0  # control the randomness of exploration and exploitation
        self.gamma = GAMMA  # discounted rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        model_file_loaded_successfully = False
        if LOAD_MODEL_FROM_FILE:
            model_folder_path = (
                "/Users/wenjiang/Desktop/python/aircraft_simulation/" + script_name
            )
            file = os.path.join(model_folder_path, LOAD_MODEL_FROM_FILE_NAME)
            if os.path.isfile(file):
                self.model = Q_Network(
                    self.state_dim,  # input
                    self.action_dim,  # output
                    1024,  # hidden
                )
                self.model.load_state_dict(
                    torch.load(file, map_location=torch.device("mps"))
                )
                self.model.eval()
                model_file_loaded_successfully = True
                print("Model loaded!!")

        if model_file_loaded_successfully == False:
            self.model = Q_Network(
                self.state_dim,
                self.action_dim,
                1024,
            )

        self.random_epsilon = EPSILON
        self.model.to(torch.device("mps"))

        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(
            (state, action, reward, next_state, done)
        )  # popleft if MAX_MEMORY is reach

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def GetAction(self, state, num_of_episodes):
        # random moves: tradeoff between exploration and exploitation
        self.epsilon = (self.random_epsilon - num_of_episodes) / 2
        final_move = list(repeat(0, 21))
        random_value = random.randint(0, self.random_epsilon)
        if random_value < self.epsilon or random_value <= EXPLORATION_RATE:
            move = random.randint(0, 20)
            final_move[move] = 1
            # print(f"random action: {np.argmax(final_move)}")
        else:
            current_state = torch.tensor(state, dtype=torch.float).to(
                torch.device("mps")
            )
            model_output = self.model(current_state)
            move = torch.argmax(model_output).item()
            final_move[move] = 1
            # print(f"network action: {np.argmax(final_move)}")

        return final_move
