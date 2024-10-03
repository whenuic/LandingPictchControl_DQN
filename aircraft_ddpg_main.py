import numpy as np
import pygame
import random

from aircraft import Aircraft
from agent_ddpg import DDPGAgent


def ProcessState(state):
    state[0] /= 152.4
    state[1] /= 2606.0
    state[2] /= 35
    state[3] = state[3] / 40 + 0.5
    state[4] = state[4] / 800 + 1
    state[5] /= 3.0
    return state


if __name__ == "__main__":

    aircraft = Aircraft(
        152.4,  # height in meters, 500ft = 152.4m
        2606.0,  # x dist to landing point in meters
    )

    state_dim = 6
    action_dim = 1

    agent = DDPGAgent(state_dim, action_dim)
    # hyperparameters
    NUM_EPISODE = 10000
    NUM_STEP = 10000
    EPISILON_START = 0.5
    EPISILON_END = 0.001
    EPSILON_DECAY = 50000000

    REWARD_BUFFER = np.empty(shape=NUM_EPISODE)
    reward_record = -10000000

    fixed_simulation_step = 0.01
    control_update_step = 0.01
    debug_display_step = 0.01
    simulation_length = 0.2  # in seconds

    clock = pygame.time.Clock()
    fps = 60
    dt = 1 / fps

    for episode_i in range(NUM_EPISODE):
        aircraft.Reset(152.4)
        raw_state = aircraft.GetState()
        raw_state = ProcessState(raw_state)
        state = np.array(raw_state)
        episode_reward = 0
        done_info = ""
        action_hold = None
        compute_new_action = False
        reward_since_last_compute = 0

        for step_i in range(NUM_STEP):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
            aircraft.Draw()

            if step_i % 10 == 0:
                compute_new_action = True
            if compute_new_action:
                epsilon = np.interp(
                    x=episode_i * NUM_STEP + step_i,
                    xp=[0, EPSILON_DECAY],
                    fp=[EPISILON_START, EPISILON_END],
                )
                random_sample = random.random()
                # print(f"random_sample: {random_sample}, epsilon: {epsilon}")
                if random_sample <= epsilon:
                    action = np.random.uniform(low=-0.1, high=0.1, size=action_dim)
                    if step_i % 200 == 0:
                        print(f"action random: {action}")
                else:
                    action = agent.get_action(state)
                    if step_i % 200 == 0:
                        print(f"action network: {action}")
                action_hold = action
            else:
                action = action_hold

            next_raw_state, reward, done, info = aircraft.Step(action, dt)
            reward_since_last_compute += reward
            if compute_new_action:
                next_raw_state = ProcessState(next_raw_state)
                next_state = np.array(next_raw_state)

                agent.replay_buffer.add_memo(
                    state, action, reward_since_last_compute, next_state, done
                )
                reward_since_last_compute = 0
                state = next_state

                agent.update()

            episode_reward += reward

            if done:
                done_info = info
                break

        REWARD_BUFFER[episode_i] = episode_reward
        if episode_reward > reward_record:
            reward_record = episode_reward
        print(
            f"Episode: {episode_i+1}, Reward: {round(episode_reward, 5)}, Record: {round(reward_record, 5)}, done_info: {done_info}"
        )
