import sys
import torch
import gym
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from RABSEnvironment import BaseStationDeploymentEnv
import time
import csv

start_time = time.time()

# hyperparameters
hidden_size = 256
learning_rate = 1e-4
lr_decay_factor = 0.99
epsilon_decay_rate = 0.99

# Entropy coefficient for exploration; starts high, then decays over time
initial_entropy_coeff = 0.02
entropy_decay_rate = 0.99 # Decay rate for entropy coefficient

# Constants
GAMMA = 0.99
num_steps = 24
max_episodes = 1000

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate):
        super(ActorCritic, self).__init__()

        # LSTM layer to capture temporal dependencies
        self.lstm = nn.LSTM(input_size=num_inputs, hidden_size=hidden_size, batch_first=True)

        # Critic Network
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        # Actor Network
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        # value = self.critic(state)
        # policy_dist = self.actor(state)

        # lstm
        lstm_out, _ = self.lstm(state.unsqueeze(0))
        lstm_out = lstm_out.squeeze(0)
        value = self.critic(lstm_out)
        policy_dist = self.actor(lstm_out)

        return value, policy_dist

    def select_action(self, value, policy_dist, epsilon):
        # Convert policy_dist to numpy array
        policy_dist_np = policy_dist.detach().numpy()
        # print('policy',policy_dist_np)

        policy_dist_np = np.clip(policy_dist_np, -500, 500)  # Clip logits to avoid extreme values
        exp_values = np.exp(policy_dist_np - np.max(policy_dist_np))  # Softmax with stabilization

        # Check for NaNs in policy_dist
        if np.isnan(policy_dist_np).any():
            raise ValueError(f"NaN detected in policy_dist_np: {policy_dist_np}")

        # Ensure exp_values doesn't contain NaNs
        if np.isnan(exp_values).any():
            raise ValueError(f"NaN detected in exp_values: {exp_values}")

        sum_exp = np.sum(exp_values)
        if sum_exp == 0:
            sum_exp = 1e-10  # Avoid division by zero

        probabilities = exp_values / sum_exp

        # Clip probabilities to avoid NaNs or extreme values
        probabilities = np.clip(probabilities, 1e-10, 1.0)
        probabilities /= np.sum(probabilities)  # Ensure they sum to 1

        # Check for NaNs in final probabilities
        if np.isnan(probabilities).any():
            raise ValueError(f"NaN detected in probabilities after clipping: {probabilities}")

        # Initialize action array for each base station
        actions = np.zeros(env.num_base_stations, dtype=int)

        # Track available locations to ensure uniqueness
        available_locations = list(range(env.num_locations))

        # Assume policy_dist is now a 2D array of shape [env.num_base_stations, env.num_locations]
        for rabs in range(env.num_base_stations):
            if np.random.rand() < epsilon:
                # Exploration: Randomly select a unique location for each base station
                action = np.random.choice(available_locations)  # Choose from remaining available locations
            else:
                # Exploitation: Select the location based on the policy distribution for the current base station
                available_policy_dist = policy_dist[rabs]  # policy_dist[rabs] is now a vector for locations
                # Choose the location with the highest probability from the available locations
                best_location_index = torch.argmax(torch.tensor(available_policy_dist)).item()
                action = available_locations[best_location_index]

            # Assign the selected action to the current base station
            actions[rabs] = action

            # Remove the selected action from the available locations to ensure uniqueness
            available_locations.remove(action)

        # print('Actions from function', actions)

        return actions

def a2c(env):
    observation = env.reset()

    # Preprocess the observation to create input tensors
    traffic_load_tensor = torch.tensor(observation['traffic_load'], dtype=torch.float32)
    traffic_load_tensor = traffic_load_tensor.reshape(env.num_locations)
    # print('traffic shape', traffic_load_tensor.shape)
    energy_consumption_tensor = torch.tensor(observation['energy_consumption'], dtype=torch.float32)
    # print('energy', energy_consumption_tensor.shape)
    locations_tensor = torch.tensor(observation['locations'], dtype=torch.int32)
    traffic_served_tensor = torch.tensor(observation['traffic_served'], dtype=torch.float32)
    # signal_strength_tensor = torch.tensor(observation['signal_strength'], dtype=torch.float32)
    # # print('signal', signal_strength_tensor.shape)
    # interference_tensor = torch.tensor(observation['interference'], dtype=torch.float32)
    # noise_tensor = torch.tensor(observation['noise'], dtype=torch.float32)
    # historical_data_tensor = torch.tensor(observation['historical_data'], dtype=torch.float32)

    # Concatenate the input tensors
    input_tensor = torch.cat([
        traffic_load_tensor,
        energy_consumption_tensor,
        locations_tensor,
        traffic_served_tensor
        # , signal_strength_tensor,
        # interference_tensor,
        # noise_tensor,
        # historical_data_tensor
    ],dim=0)

    # Determine the input dimension based on the observation
    num_inputs = (
            observation['traffic_load'].shape[0] +
            observation['energy_consumption'].shape[0] +
            observation['locations'].shape[0] +
            observation['traffic_served'].shape[0]
            # + observation['signal_strength'].shape[0] +
            # observation['interference'].shape[0] +
            # observation['noise'].shape[0] +
            # observation['historical_data'].shape[0]
    )

    num_outputs = env.action_space.n

    print("Input dimension:", num_inputs)
    print("Output dimension (number of actions):", num_outputs)

    actor_critic = ActorCritic(num_inputs, num_outputs, hidden_size, learning_rate)
    ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)

    all_lengths = 0
    average_lengths = 0
    all_rewards = []
    all_traffics = []
    all_energies = []
    all_fairness =[]

    entropy_term = 0
    epsilon = 1.0  # Start with high exploration

    # Initialize lists to store the losses
    actor_losses = []
    critic_losses = []

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

        log_probs = []
        values = []
        rewards = []
        actions = []
        traffics = []
        energies = []
        fairness = []

        # epsilon = max(0.01, 1 - epsilon_decay_rate * (episode / max_episodes))  # Exploration rate decay
        epsilon = max(0.01, epsilon * epsilon_decay_rate)

        adjusted_lr = max(1e-4, learning_rate * (lr_decay_factor ** (episode // 10)))
        for param_group in ac_optimizer.param_groups:
            param_group['lr'] = adjusted_lr

        # actor_critic = ActorCritic(num_inputs, num_outputs, hidden_size, adjusted_lr)
        # # ac_optimizer = optim.RMSprop(actor_critic.parameters(), lr=adjusted_lr)
        # ac_optimizer = optim.Adam(actor_critic.parameters(), lr=adjusted_lr)
        traffic = episode_traffic[episode]
        actual_traffic = actual_episode_traffic[episode]

        for steps in range(num_steps):
            # print('current step', steps)

            value, policy_dist = actor_critic.forward(input_tensor)
            # print('Value',value)
            value = value.detach().numpy()
            dist = policy_dist.detach().numpy()

            # Assume traffic distribution for each time period
            random_slot = random.randint(0, num_steps - 1)
            random_slot = steps

            # Dynamic traffic assignment
            # env.traffic_load = traffic[random_slot]
            env.traffic_load = actual_traffic[random_slot]

            # Select an action using the select_action method
            actions = actor_critic.select_action(value, policy_dist, epsilon)

            # Convert the action tensor to a numpy array
            # action = action_tensor.numpy()

            # print('Action check', actions)
            # actions.append(action)

            # Sample an action from the distribution
            # action = np.random.choice(range(1, env.num_locations), env.num_base_stations, replace=False)
            # print('action', action)

            # Traffic load difference with 30%, 50%, 80% Accuracy
            # env.traffic_load = traffic[random_slot] #100% accuracy
            # env.traffic_load = actual_traffic[random_slot]  # dynamic

            new_observation, reward, done, fairness_index, total_traffic, total_energy, _ = env.step(actions)

            # Clip the rewards before using them for updates
            # reward = np.clip(reward, -1, 1)

            # print('new state', new_state)
            # print('reward', reward)

            # Index into policy_dist using action
            log_prob = torch.log(policy_dist.squeeze(0))
            entropy = -np.sum(np.mean(dist) * np.log(dist + 1e-8)) # Add small number to prevent log(0) condition

            rewards.append(reward)
            fairness.append(fairness_index)
            traffics.append(total_traffic)
            energies.append(total_energy)
            values.append(value)
            log_probs.append(log_prob[0])
            entropy_term += entropy
            observation = new_observation

            # Preprocess the observation to create input tensors
            traffic_load_tensor = torch.tensor(new_observation['traffic_load'], dtype=torch.float32)
            traffic_load_tensor = traffic_load_tensor.reshape(env.num_locations)
            energy_consumption_tensor = torch.tensor(new_observation['energy_consumption'], dtype=torch.float32)
            locations_tensor = torch.tensor(new_observation['locations'], dtype=torch.int32)
            traffic_served_tensor = torch.tensor(new_observation['traffic_served'], dtype=torch.float32)
            # signal_strength_tensor = torch.tensor(new_observation['signal_strength'], dtype=torch.float32)
            # interference_tensor = torch.tensor(new_observation['interference'], dtype=torch.float32)
            # noise_tensor = torch.tensor(new_observation['noise'], dtype=torch.float32)
            # historical_data_tensor = torch.tensor(new_observation['historical_data'], dtype=torch.float32)

            # Concatenate the input tensors
            input_tensor = torch.cat([
                traffic_load_tensor,
                energy_consumption_tensor,
                locations_tensor,
                traffic_served_tensor
                # , signal_strength_tensor,
                # interference_tensor,
                # noise_tensor,
                # historical_data_tensor
            ], dim=0)

            if done or steps == num_steps - 1:
                Qval, _ = actor_critic.forward(input_tensor)
                Qval = Qval.detach().numpy()
                all_lengths += steps
                average_lengths = np.mean(all_lengths)

                if episode % 10 == 0:
                    sys.stdout.write("episode: {}, reward: {} \n".format(episode, np.sum(rewards)))
                    sys.stdout.write("episode: {}, traffic: {} \n".format(episode, np.sum(traffics)))
                    sys.stdout.write("episode: {}, energy: {} \n".format(episode, np.sum(energies)))
                    sys.stdout.write("episode: {}, fairness: {} \n".format(episode, np.average(fairness)))
                break

        all_rewards.append(np.sum(rewards))
        all_traffics.append(np.sum(traffics)/1e3) # MB
        all_energies.append(np.sum(energies)/1e3) # kJ
        all_fairness.append(np.average(fairness))

        # compute Q values
        Qvals = np.zeros_like(values)
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + GAMMA * Qval
            Qvals[t] = Qval

        # update actor critic
        values_array = np.array(values)
        values = torch.FloatTensor(values_array)
        Qvals = torch.FloatTensor(Qvals)
        log_probs = torch.stack(log_probs)

        advantage = Qvals - values

        # Normalize advantages
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # Calculate entropy and update entropy coefficient over episodes
        entropy_coeff = initial_entropy_coeff * (entropy_decay_rate ** episode)

        dist = policy_dist.detach().numpy()  # Get policy distribution for agent i
        entropy = -np.sum(np.mean(dist) * np.log(dist + 1e-10))

        # Clipped actor loss
        epsilon_clip = 0.1  # 0.1 Clipping parameter
        ratio = torch.exp(log_probs - log_probs)  # old_log_probs saved from previous policy

        # actor_loss = -torch.min(ratio * advantage, torch.clamp(ratio, 1 - epsilon_clip, 1 + epsilon_clip) * advantage).mean()
        actor_loss = -(log_probs * advantage.detach()).mean() + entropy_coeff * entropy

        # Critic loss with MSE and L2 regularization
        critic_loss = 0.5 * advantage.pow(2).mean()

        # Record losses
        actor_losses.append(actor_loss.item())
        critic_losses.append(critic_loss.item())

        ac_loss = actor_loss + critic_loss
                   #+ entropy_coeff * entropy)

        ac_optimizer.zero_grad()
        ac_loss.backward()
        ac_optimizer.step()

    smoothed_rewards = np.convolve(all_rewards, np.ones(10) / 10, mode='valid')

    print('Average Efficiency:', np.average(all_rewards))
    print('Maximum:', np.max(all_rewards))

    # # Plot the two lines
    plt.plot(all_rewards, label='Energy Efficiencies')
    plt.plot(smoothed_rewards, label='Smoothed Energy Efficiencies')
    # plt.plot(all_energies, label='Energy Consumption')
    # plt.plot(all_traffics, label='Traffic')
    # plt.plot(all_fairness, label='Fairness')

    # # Fill the area between the lines with transparency
    # plt.fill_between(range(len(all_rewards)), all_rewards, smoothed_rewards, color='gray', alpha=0.3,
    #                  label='Fill Between')

    # Set x and y axis values explicitly
    plt.xticks([0, 200, 400, 600, 800, 1000])  # Set specific x-axis values
    plt.yticks([0, 10, 20, 30, 40, 50, 60, 70])  # Set specific y-axis values

    # Add labels and legend
    plt.xlabel('Episode')
    plt.ylabel('Energy Efficiency')
    plt.legend()

    # Show the plot
    plt.show()

    # plt.plot(all_lengths)
    # plt.plot(average_lengths)
    # plt.xlabel('Episode')
    # plt.ylabel('Episode length')
    # plt.show()
    # Export evaluation results to .csv file
    # set the file name and location
    filename = 'evaluation_results_04_ACDRLModel_diurnal.csv'

    # open the file for writing
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)

        # Write the column titles as the first row
        writer.writerow(all_rewards)
        writer.writerow(all_traffics)
        writer.writerow(all_energies)
        writer.writerow(all_fairness)

if __name__ == "__main__":
    env = BaseStationDeploymentEnv(120 , 25)
    a2c(env)

end_time = time.time()
execution_time = end_time - start_time
print(execution_time)