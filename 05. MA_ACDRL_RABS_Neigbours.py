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
entropy_decay_rate = 0.99  # Decay rate for entropy coefficient

# Constants
GAMMA = 0.99
num_steps = 24
max_episodes = 1000

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate):
        super(ActorCritic, self).__init__()

        # LSTM layer to capture temporal dependencies
        self.lstm = nn.LSTM(input_size=num_inputs, hidden_size=hidden_size, batch_first=True)

        # Actor Network
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
            nn.Softmax(dim=-1)
        )

        # Critic Network
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state):
        # LSTM
        lstm_out, _ = self.lstm(state.unsqueeze(0))
        lstm_out = lstm_out.squeeze(0)
        value = self.critic(lstm_out)
        policy_dist = self.actor(lstm_out)

        return value, policy_dist

def get_dynamic_radius(agent_id, observation):

    # Calculate a dynamic radius for a given agent based on real-time conditions.
    # Traffic-based dynamic radius
    traffic_load = observation['traffic_load'][agent_id]
    max_traffic_load = np.max(observation['traffic_load'])
    min_traffic_load = np.min(observation['traffic_load'])

    # Scale traffic to a [0.5, 2.0] range for radius adjustment
    scaled_radius_factor = 0.5 + 1.5 * ((traffic_load - min_traffic_load) / (max_traffic_load - min_traffic_load + 1e-5))
    base_radius = 500  # Default base radius

    return base_radius * scaled_radius_factor

def get_neighbor_state(env, observation, agent_id, neighbors):
    # Extract neighbor-related data fields
    neighbor_traffic_load = torch.tensor([observation['traffic_load'][n] for n in neighbors], dtype=torch.float32)
    neighbor_energy_consumption = torch.tensor([observation['energy_consumption'][n] for n in neighbors],
                                               dtype=torch.float32)
    neighbor_traffic_served = torch.tensor([observation['traffic_served'][n] for n in neighbors],
                                           dtype=torch.float32)
    neighbor_locations = torch.tensor([observation['locations'][n] for n in neighbors],
                                               dtype=torch.float32)

    # Combine into a single tensor
    neighbor_state = torch.cat([
        neighbor_traffic_load,
        neighbor_energy_consumption,
        neighbor_locations,
        neighbor_traffic_served
    ], dim=0)

    return neighbor_state

def pad_neighbor_state(state, max_size, pad_value=0.0):
    if state.size(0) < max_size:
        padding = torch.full((max_size - state.size(0),), pad_value)
        return torch.cat([state, padding], dim=0)

    return state[:max_size]

def multi_agent_a2c(env, learning_rate=learning_rate):
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

    # num_outputs = env.action_space.n
    num_outputs = env.num_locations

    print("Input dimension:", num_inputs)
    print("Output dimension (number of actions):", num_outputs)

    # Multi Agent
    agents = [ActorCritic(num_inputs, num_outputs, hidden_size, learning_rate) for _ in range(env.num_base_stations)]
    optimizers = [optim.Adam(agent.parameters(), lr=learning_rate) for agent in agents]

    all_rewards = []
    all_traffics = []
    all_energies = []
    all_fairness = []

    entropy_term = [[] for _ in range(env.num_base_stations)]

    # Initialize lists to store the losses
    actor_losses = []
    critic_losses = []
    actions_list = []
    replay_buffer = []
    epsilon = 1.0  # Start with high exploration

    episode_traffic = np.load('demand_data_120_1000_diurnal.npy')  # Actual traffic load at each location
    actual_episode_traffic = episode_traffic[:, :, :env.num_locations]

    # Limiting maximum neighbours to reduce noise
    max_neighbors = env.num_base_stations

    for episode in range(max_episodes):
        observation = env.reset()

        # Exploration rate decay
        epsilon = max(0.01, epsilon * epsilon_decay_rate) # Exploration rate decay

        # Learning rate adjustment
        adjusted_lr = max(1e-4, learning_rate * (lr_decay_factor ** (episode // 10)))

        # Initialization
        log_probs_list = [[] for _ in range(env.num_base_stations)]
        values_list = [[] for _ in range(env.num_base_stations)]
        rewards = []
        traffics = []
        energies = []
        fairness = []

        traffic = episode_traffic[episode]
        actual_traffic = actual_episode_traffic[episode]

        for steps in range(num_steps):
            action_probabilities = []

            # Assume traffic distribution for each time period
            random_slot = random.randint(0, num_steps - 1)
            random_slot = steps

            # Dynamic traffic assignment
            env.traffic_load = actual_traffic[random_slot]

            # Initialize action array for each base station
            actions = np.zeros(env.num_base_stations, dtype=int)

            # Track available locations to ensure uniqueness
            available_locations = list(range(env.num_locations))  # Locations are [0, 1, 2, ..., 9]

            # Learning Neighbouring State only, Distributed MA-ACDRL
            for i, agent in enumerate(agents):
                # Calculate dynamic radius if not provided
                radius = get_dynamic_radius(i, observation)
                # print("radius", radius)

                # Get the neighbors for the current agent
                neighbors = env.get_neighbors(i, radius, max_neighbors)  # Use the radius that fits your scenario
                # print('Neighbours for agent', i, 'are', neighbors)

                # Extract the neighbor-specific state
                neighbor_state = get_neighbor_state(env, observation, i, neighbors)
                padded_neighbor_state = pad_neighbor_state(neighbor_state, max_size=num_inputs)

                # Pass the neighbor-specific state to the agent's forward function
                value, policy_dist = agent.forward(padded_neighbor_state)

                if np.random.rand() < epsilon:
                    action = np.random.choice(available_locations)  # Exploration
                else:
                    # Filter the policy distribution to include only available locations
                    available_policy_dist = policy_dist[available_locations]  # Use only available locations
                    best_location_index = torch.argmax(available_policy_dist).item()
                    action = available_locations[best_location_index]
                    # print('Action for RABS', i , 'is', action)

                actions[i] = action
                # print('Action', action)
                available_locations.remove(action)

                action_probabilities.append(policy_dist[action])
                log_probs_list[i].append(torch.log(policy_dist[action]))
                values_list[i].append(value)

            # print('Actions', actions)
            actions_list.append(actions)

            new_state, reward, done, fairness_index, total_traffic, total_energy, _ = env.step(np.array(actions))
            # print('Reward', steps, 'is' , reward)

            rewards.append(reward)
            fairness.append(fairness_index)
            traffics.append(total_traffic)
            energies.append(total_energy)

            # Transition to next state
            new_observation = new_state

            # Store in replay buffer
            # episode_experiences.append((observation, actions, reward, new_observation, done))
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
                traffic_served_tensor,
                # signal_strength_tensor,
                # interference_tensor,
                # noise_tensor,
                # historical_data_tensor
            ], dim=0)

            if done or steps == num_steps - 1:
                break

        # Calculate entropy and update entropy coefficient over episodes
        entropy_coeff = initial_entropy_coeff * (entropy_decay_rate ** episode)

        # Add small number to prevent log(0) condition
        for i in range(env.num_base_stations):
            dist = policy_dist[i].detach().numpy()  # Get policy distribution for agent i
            entropy = -np.sum(np.mean(dist) * np.log(dist + 1e-10))
            entropy_term[i] = entropy

        # Compute Q-values and back propagate for each agent
        for i in range(env.num_base_stations):
            Qvals = []
            Qval = 0

            for reward in reversed(range(len(rewards))):
                Qval = reward + GAMMA * Qval
                Qvals.insert(0, Qval)
            Qvals = torch.tensor(Qvals, dtype=torch.float32)

            values = torch.stack(values_list[i])
            log_probs = torch.stack(log_probs_list[i])
            advantage = Qvals - values.squeeze()

            # Modified with entropy weight:
            actor_loss = -(log_probs * advantage.detach()).mean() + entropy_coeff * entropy_term[i]
            critic_loss = 0.5 * advantage.pow(2).mean()

            loss = actor_loss + critic_loss

            optimizers[i].zero_grad()
            loss.backward()
            optimizers[i].step()
            for param_group in optimizers[i].param_groups:
                param_group['lr'] = adjusted_lr

            # Gradient clipping to stabilize updates
            # torch.nn.utils.clip_grad_norm_(agents[i].parameters(), max_norm=1.0)

        if episode % 10 == 0:
            sys.stdout.write("episode: {}, reward: {} \n".format(episode, np.sum(rewards)))
            sys.stdout.write("episode: {}, traffic: {} \n".format(episode, np.sum(traffics)))
            sys.stdout.write("episode: {}, energy: {} \n".format(episode, np.sum(energies)))
            sys.stdout.write("episode: {}, fairness: {} \n".format(episode, np.average(fairness)))

        all_rewards.append(np.sum(rewards))
        all_traffics.append(np.sum(traffics)/1e3) #MB
        all_energies.append(np.sum(energies)/1e3) #kJ
        all_fairness.append(np.average(fairness))

    # Plot results
    smoothed_rewards = np.convolve(all_rewards, np.ones(10) / 10, mode='valid')

    print('Average Efficiency:', np.average(all_rewards))
    print('Maximum:', np.max(all_rewards))

    # # Plot the two lines
    plt.plot(all_rewards, label='Energy Efficiencies')
    plt.plot(smoothed_rewards, label='Smoothed Energy Efficiencies')
    # plt.plot(all_fairness, label='Fairness')
    # plt.plot(all_energies, label='Energy Consumption')
    # plt.plot(all_traffics, label='Traffic')

    # Set x and y axis values explicitly
    plt.xticks([0, 200, 400, 600, 800, 1000])  # Set specific x-axis values
    plt.yticks([0, 10, 20, 30, 40, 50, 60, 70])  # Set specific y-axis values

    # Add labels and legend
    plt.xlabel('Episode')
    plt.ylabel('Energy Efficiency')
    # plt.ylabel('Fairness')
    # plt.ylabel('Energy Consumption')
    # plt.ylabel('Traffic')

    plt.legend()

    # Show the plot
    plt.show()

    # Export evaluation results to .csv file
    # set the file name and location
    filename = 'evaluation_results_05_MA_ACDRLModel_diurnal.csv'
    # open the file for writing
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the column titles as the first row
        writer.writerow(all_rewards)
        writer.writerow(all_traffics)
        writer.writerow(all_energies)
        writer.writerow(all_fairness)

    # torch.save(agent.state_dict(), "multi_agent_actor_critic_model.pth")

if __name__ == "__main__":
    env = BaseStationDeploymentEnv(120 ,25)
    multi_agent_a2c(env)

end_time = time.time()
execution_time = end_time - start_time
print('Time:', execution_time)