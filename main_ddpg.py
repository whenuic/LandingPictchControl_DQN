import gym
import numpy as np
import random
import os
import torch
import time

from agent_ddpg import DDPGAgent

env = gym.make(id="Pendulum-v1")
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]

agent = DDPGAgent(STATE_DIM, ACTION_DIM)

# hyperparameters
NUM_EPISODE = 100000
NUM_STEP = 200
EPISILON_START = 0.5
EPISILON_END = 0.001
EPSILON_DECAY = 100000

REWARD_BUFFER = np.empty(shape=NUM_EPISODE)

for episode_i in range(NUM_EPISODE):
    state, others = env.reset()
    episode_reward = 0

    for step_i in range(NUM_STEP):
        epsilon = np.interp(
            x=episode_i * NUM_STEP + step_i,
            xp=[0, EPSILON_DECAY],
            fp=[EPISILON_START, EPISILON_END],
        )
        random_sample = random.random()
        if random_sample <= epsilon:
            action = np.random.uniform(low=-2, high=2, size=ACTION_DIM)
        else:
            action = agent.get_action(state)

        next_state, reward, done, truncation, info = env.step(action)

        agent.replay_buffer.add_memo(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        agent.update()
        if done:
            break

    REWARD_BUFFER[episode_i] = episode_reward
    print(f"Episode: {episode_i+1}, Reward: {round(episode_reward, 2)}")

current_path = os.path.dirname(os.path.realpath(__file__))
model_folder_path = current_path + "/models/"
if not os.path.exists(model_folder_path):
    os.makedirs(model_folder_path)
timestamp = time.strftime("%Y%m%d%H%M%S")

# save models
torch.save(
    agent.actor.state_dict(),
    os.path.join(model_folder_path, f"ddpg_actor_{timestamp}.pth"),
)
torch.save(
    agent.critic.state_dict(),
    os.path.join(model_folder_path, f"ddpg_critic_{timestamp}.pth"),
)


env.close()
