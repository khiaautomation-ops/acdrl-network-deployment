import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from RABSEnvironment import BaseStationDeploymentEnv
import time
import csv

max_episodes = 1000
num_steps = 24
epsilon_decay_rate = 0.99

start_time = time.time()

def q_learning_model(env, alpha, gamma):
    Q = np.zeros((env.num_locations, env.num_base_stations))

    epsilon = 1.0  # Start with high exploration
    all_rewards = []
    all_traffics = []
    all_energies = []
    all_fairness =[]

    # if (env.num_locations == 10):
    #     episode_traffic = np.load('demand_data_10_1000.npy')
    #     actual_episode_traffic = np.load('mixed_data_10_1000.npy')  # Actual traffic load at each location
    # elif (env.num_locations == 50):
    #     episode_traffic = np.load('demand_data_50_1000.npy')
    #     actual_episode_traffic = np.load('mixed_data_50_1000.npy')  # Actual traffic load at each location
    # elif (env.num_locations == 75):
    #     episode_traffic = np.load('demand_data_75_1000.npy')
    #     actual_episode_traffic = np.load('mixed_data_75_1000.npy')  # Actual traffic load at each location
    # else:
    #     episode_traffic = np.load('demand_data_120_1000.npy')
    #     actual_episode_traffic = np.load('mixed_data_120_1000.npy')  # Actual traffic load at each location

    episode_traffic = np.load('demand_data_120_1000_diurnal.npy')  # Actual traffic load at each location
    actual_episode_traffic = episode_traffic[:, :, :env.num_locations]

    for episode in range(max_episodes):
        state = env.reset()
        rewards = []
        actions = []
        traffics = []
        energies =[]
        fairness = []

        epsilon = max(0.01, epsilon * epsilon_decay_rate) # Exploration rate decay

        traffic = episode_traffic[episode]
        actual_traffic = actual_episode_traffic[episode]

        for env.current_step in range(num_steps):
            random_slot = random.randint(0, num_steps - 1)
            random_slot = env.current_step

            env.traffic_load = actual_traffic[random_slot]

            Q_action = 0

            # Initialize action array for each base station
            actions = np.zeros(env.num_base_stations, dtype=int)

            # Track available locations to ensure uniqueness
            available_locations = list(range(env.num_locations))

            for rabs in range(env.num_base_stations):
                if np.random.rand() < epsilon:
                    # Exploration: Randomly select a unique location for each base station
                    action = np.random.choice(available_locations)  # Choose from remaining locations
                else:
                    # Exploitation: Select the location with the highest Q-value from available options
                    # Filter Q-values to include only available locations
                    q_values_for_available_locations = Q[available_locations, rabs]
                    best_location_index = np.argmax(q_values_for_available_locations)
                    action = available_locations[best_location_index]

                # Assign the selected action to the current base station
                actions[rabs] = action

                # Remove the selected action from the available locations to ensure uniqueness
                available_locations.remove(action)

            # print('Action', actions)

            next_state, reward, done, fairness_index, total_traffic, total_energy, _ = env.step(actions)

            rewards.append(reward)
            # print('Reward per action:', reward)
            fairness.append(fairness_index)
            traffics.append(total_traffic)
            energies.append(total_energy)

            # Update Q-value
            next_Q_action = np.argmax(next_state['locations'])
            best_next_action = Q[next_Q_action]
            # print('Next q', best_next_action)

            # Update Q Table
            Q[Q_action] = Q[Q_action] + alpha * (
                    reward + gamma * Q[next_Q_action] - Q[Q_action])

            state = next_state

        # if episode % 10 == 0:
        #     sys.stdout.write("episode: {}, reward: {} \n".format(episode, np.sum(rewards)))
        #     sys.stdout.write("episode: {}, traffic: {} \n".format(episode, np.sum(traffics)))
        #     sys.stdout.write("episode: {}, energy: {} \n".format(episode, np.sum(energies)))
        #     sys.stdout.write("episode: {}, fairness: {} \n".format(episode, np.average(fairness)))

        all_rewards.append(np.sum(rewards))
        all_traffics.append(np.sum(traffics)/1e3)
        all_energies.append(np.sum(energies)/1e3)
        all_fairness.append(np.average(fairness))

    smoothed_rewards = np.convolve(all_rewards, np.ones(10) / 10, mode='valid')

    print('Average Efficiency:', np.average(all_rewards))
    # print('Variance:', np.var(all_rewards))

    # # Plot the two lines
    # plt.plot(all_rewards, label='Energy Efficiencies')
    # plt.plot(smoothed_rewards, label='Smoothed Energy Efficiencies')
    # plt.plot(all_energies, label='Energy Consumption')
    # plt.plot(all_traffics, label='Traffic')
    plt.plot(all_fairness, label='Fairness')

    # Set x and y axis values explicitly
    plt.xticks([0, 200, 400, 600, 800, 1000])  # Set specific x-axis values
    # plt.yticks([0, 10, 20, 30, 40, 50, 60, 70])  # Set specific y-axis values

    # Add labels and legend
    plt.xlabel('Episode')
    # plt.ylabel('Energy Efficiency')
    plt.ylabel('Fairness')
    plt.legend()

    # Show the plot
    plt.show()

    # # # Export evaluation results to .csv file
    # set the file name and location
    filename = 'evaluation_results_03_QLearningModel_diurnal.csv'

    # open the file for writing
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)

        # Write the column titles as the first row
        writer.writerow(all_rewards)
        writer.writerow(all_traffics)
        writer.writerow(all_energies)
        writer.writerow(all_fairness)

if __name__ == "__main__":
    env = BaseStationDeploymentEnv(120, 25)
    q_learning_model(env, alpha=0.1, gamma=0.99)

end_time = time.time()
execution_time = end_time - start_time
print(execution_time)