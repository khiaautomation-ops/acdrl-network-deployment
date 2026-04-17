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

start_time = time.time()

def gauss_markov_model(env, alpha, sigma):
    all_rewards = []
    all_traffics = []
    all_energies = []
    all_fairness = []

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
        observation = env.reset()
        rewards = []
        actions = []
        traffics = []
        energies = []
        fairness = []

        previous_locations = env.locations
        # print('Locations', previous_locations)

        traffic = episode_traffic[episode]
        actual_traffic = actual_episode_traffic[episode]

        for steps in range(num_steps):
            # Dynamic traffic assignment
            random_slot = random.randint(0, num_steps - 1)
            random_slot = steps

            env.traffic_load = actual_traffic[random_slot]

            # Generate actions using the Gauss-Markov process
            noise = np.random.normal(0, sigma, size=env.num_base_stations)
            action = alpha * previous_locations + (1 - alpha) * noise
            action = np.clip(action, 1, env.num_locations).astype(int)
            # print('Action', action)

            actions.append(action)

            new_observation, reward, done, fairness_index, total_traffic, total_energy, _ = env.step(action)

            rewards.append(reward)
            fairness.append(fairness_index)
            traffics.append(total_traffic)
            energies.append(total_energy)
            # print('Reward per action:', reward)

            # print(reward)
            previous_locations = action

        # if episode % 10 == 0:
        #     sys.stdout.write("episode: {}, reward: {} \n".format(episode, np.sum(rewards)))
        #     sys.stdout.write("episode: {}, traffic: {} \n".format(episode, np.sum(traffics)))
        #     sys.stdout.write("episode: {}, energy: {} \n".format(episode, np.sum(energies)))
        #     sys.stdout.write("episode: {}, fairness: {} \n".format(episode, np.average(fairness)))

        # all_rewards.append(np.sum(rewards))
        all_rewards.append(np.sum(rewards))
        all_traffics.append(np.sum(traffics) / 1e3)
        all_energies.append(np.sum(energies) / 1e3)
        all_fairness.append(np.average(fairness))

    smoothed_rewards = np.convolve(all_rewards, np.ones(10) / 10, mode='valid')

    print('Average Efficiency:', np.average(all_rewards))
    print('Maximum:', np.max(all_rewards))

    # # Plot the two lines
    plt.plot(all_rewards, label='Energy Efficiencies')
    plt.plot(smoothed_rewards, label='Smoothed Energy Efficiencies')
    # plt.plot(all_fairness, label='Fairness')

    # Set x and y axis values explicitly
    plt.xticks([0, 200, 400, 600, 800, 1000])  # Set specific x-axis values
    # plt.yticks([0, 10, 20, 30, 40, 50, 60, 70])  # Set specific y-axis values

    # Add labels and legend
    plt.xlabel('Episode')
    plt.ylabel('Energy Efficiency')
    plt.legend()

    # Show the plot
    plt.show()

    # # # Export evaluation results to .csv file
    # set the file name and location
    filename = 'evaluation_results_02_GaussMarkovModel_diurnal.csv'

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
    gauss_markov_model(env, 0.5, 1.0)

end_time = time.time()
execution_time = end_time - start_time
print(execution_time)