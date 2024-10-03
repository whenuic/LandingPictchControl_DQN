import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from collections import deque

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device type: {device}")

# hyperparameters
LR_ACTOR = 1e-5
LR_CRITIC = 1e-5
GAMMA = 0.95
MEMORY_SIZE = 100000
BATCH_SIZE = 128
TAU = 1e-3


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)

        # nn.init.zeros_(self.fc1.weight)
        # nn.init.zeros_(self.fc2.weight)
        # nn.init.zeros_(self.fc3.weight)
        # nn.init.zeros_(self.fc4.weight)
        # nn.init.normal_(self.fc1.bias, mean=0, std=0.5)
        # nn.init.normal_(self.fc2.bias, mean=0, std=0.5)
        # nn.init.normal_(self.fc3.bias, mean=0, std=0.5)
        # nn.init.normal_(self.fc4.bias, mean=0, std=0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x)) * 0.2 - 0.1
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, out_features=1)

        # nn.init.zeros_(self.fc1.weight)
        # nn.init.zeros_(self.fc2.weight)
        # nn.init.zeros_(self.fc3.weight)
        # nn.init.zeros_(self.fc4.weight)
        # nn.init.normal_(self.fc1.bias, mean=0, std=0.5)
        # nn.init.normal_(self.fc2.bias, mean=0, std=0.5)
        # nn.init.normal_(self.fc3.bias, mean=0, std=0.5)
        # nn.init.normal_(self.fc4.bias, mean=0, std=0.5)

    def forward(self, x, a):
        x = torch.cat((x, a), 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add_memo(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(state, 0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size)
        )
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)


class DDPGAgent:
    def __init__(self, state_dim, action_dim):
        torch.manual_seed(2024)

        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.replay_buffer = ReplayMemory(MEMORY_SIZE)

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.actor(state)
        return action.detach().cpu().numpy()[0]

    def update(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            BATCH_SIZE
        )
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        # update critic, td learning
        next_actions = self.actor_target(next_states)
        target_Q = self.critic_target(next_states.squeeze(1), next_actions.squeeze(1))
        target_Q = rewards + (GAMMA * target_Q * (1 - dones))
        current_Q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_Q, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update target networks
        for (
            target_param,
            param,
        ) in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        for (
            target_param,
            param,
        ) in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
