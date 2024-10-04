import os
import pygame  #

from aircraft import Aircraft
from deep_q_agent import Agent
from utils_snake import *
from pathlib import Path

script_name = Path(__file__).stem

TRAIN_LONG_MEMORY = True
REMEMBER = True


def ProcessState(state, agent, dt):
    state_out = [0, 0, 0, 0, 0, 0]
    state_out[0] = state[0] / 152.4
    state_out[1] = state[1] / 2606.0
    state_out[2] = state[2] / 35.0
    state_out[3] = state[3] / 20
    state_out[4] = (state[4] + 300) / 300
    state_out[5] = (state[5] - 3.0) / 3.0

    length = len(agent.memory)
    # append vertical rate change, that's the acceleration of vertical speed
    if length >= 1:
        state_out.append((state[5] - agent.memory[length - 1][0][5]) / dt / 50)
    else:
        state_out.append(0)

    if length >= 30:
        state_out.append(agent.memory[length - 10][0][4])
        state_out.append(agent.memory[length - 10][0][5])
        state_out.append(agent.memory[length - 10][0][6])
        state_out.append(agent.memory[length - 20][0][4])
        state_out.append(agent.memory[length - 20][0][5])
        state_out.append(agent.memory[length - 20][0][6])
        state_out.append(agent.memory[length - 30][0][4])
        state_out.append(agent.memory[length - 30][0][5])
        state_out.append(agent.memory[length - 30][0][6])
    elif length >= 20:
        state_out.append(agent.memory[length - 10][0][4])
        state_out.append(agent.memory[length - 10][0][5])
        state_out.append(agent.memory[length - 10][0][6])
        state_out.append(agent.memory[length - 20][0][4])
        state_out.append(agent.memory[length - 20][0][5])
        state_out.append(agent.memory[length - 20][0][6])
        state_out.append(0)
        state_out.append(0)
        state_out.append(0)
    elif length >= 10:
        state_out.append(agent.memory[length - 10][0][4])
        state_out.append(agent.memory[length - 10][0][5])
        state_out.append(agent.memory[length - 10][0][6])
        state_out.append(0)
        state_out.append(0)
        state_out.append(0)
        state_out.append(0)
        state_out.append(0)
        state_out.append(0)
    else:
        state_out.append(0)
        state_out.append(0)
        state_out.append(0)
        state_out.append(0)
        state_out.append(0)
        state_out.append(0)
        state_out.append(0)
        state_out.append(0)
        state_out.append(0)

    return state_out


if __name__ == "__main__":
    plot_scores = []
    plot_average_scores = []
    plot_last_100_average_scores = []
    total_score = 0
    model_folder_path = (
        "/Users/wenjiang/Desktop/python/aircraft_simulation/" + script_name
    )
    plot_file_name = os.path.join(model_folder_path, "result.png")
    record = -10000000
    num_of_episodes = 1
    episode_reward = 0
    FPS = 60
    dt = 0.1

    state_dim = 16
    action_dim = 21
    aircraft = Aircraft(
        152.4,  # height in meters, 500ft = 152.4m
        2606.0,  # x dist to landing point in meters
    )

    agent = Agent(state_dim, action_dim)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                agent.model.save(record)
                quit()
        aircraft.Draw()
        # Step 1: get the old state
        state_old = aircraft.GetState()
        state_old_processed = ProcessState(state_old, agent, dt)

        # Step 2: get move
        final_move = agent.GetAction(state_old_processed, num_of_episodes)

        # Step 3: perform move
        index = final_move.index(1)
        action = [(index - 10.0) / 200.0]
        state_new, reward, done, done_info = aircraft.Step(action, dt)
        state_new_processed = ProcessState(state_new, agent, dt)

        episode_reward += reward

        pygame.display.set_caption(
            f"{round(state_new[5], 2)},  {round(action[0] + 0.63, 3)}, {round(state_new[4], 3)}, {episode_reward}"
        )

        # train short memory
        agent.train_short_memory(
            state_old_processed, final_move, reward, state_new_processed, done
        )

        # remember
        if TRAIN_LONG_MEMORY or REMEMBER:
            agent.remember(
                state_old_processed, final_move, reward, state_new_processed, done
            )

        if done:
            # train long memory, plot result
            aircraft.Reset(152.4)
            if TRAIN_LONG_MEMORY:
                agent.train_long_memory()

            if episode_reward > record:
                record = episode_reward
                agent.model.save(record)

            plot_scores.append(episode_reward)
            total_score += episode_reward
            average_score = total_score / num_of_episodes
            plot_average_scores.append(average_score)
            last_100_average_score = (
                sum(plot_scores[-100:]) / 100
                if len(plot_scores) >= 100
                else sum(plot_scores) / len(plot_scores)
            )
            plot_last_100_average_scores.append(last_100_average_score)

            print(
                f"Episode: {num_of_episodes}, Reward: {round(episode_reward, 5)}, Record: {round(record, 5)}, Last_100: {last_100_average_score}, done_info: {done_info}"
            )
            num_of_episodes += 1
            episode_reward = 0

            Plot(
                plot_scores,
                plot_average_scores,
                plot_last_100_average_scores,
                plot_file_name,
            )

            # PrintEpisodeResult(
            #     episode_reward,
            #     average_score,
            #     last_100_average_score,
            #     num_of_episodes,
            #     record,
            # )
