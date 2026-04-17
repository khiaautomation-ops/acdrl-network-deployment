import gym
from gym import spaces
import numpy as np
import torch

class BaseStationDeploymentEnv():
    def __init__(self, num_locations, num_base_stations):
        super(BaseStationDeploymentEnv, self).__init__()

        self.num_locations = num_locations
        self.num_base_stations = num_base_stations

        # Extract the specified number of traffic that fit with parameters i.e., num_locations
        distances_loaded = np.load('locations_coordinates_120.npy')
        self.fly_distances = torch.from_numpy(distances_loaded)

        self.traffic_load = np.zeros(num_locations)
        # print('traffic load',self.traffic_load)
        self.energy_consumption = np.zeros(num_base_stations)  # Energy consumption of RABSs

        self.constraints_violated = np.zeros(num_base_stations)  # Penalty
        self.energy_constraint_penalty = np.zeros(num_base_stations)
        self.capacity_constraint_penalty = np.zeros(num_base_stations)
        self.energy_violation_count = 0
        self.capacity_violation_count = 0

        #print('energy',self.energy_consumption)
        self.locations = np.random.choice(range(1, num_locations), num_base_stations, replace=False)  # Locations served by RABSs
        self.traffic_served = np.zeros(num_base_stations) # Traffic served by RABSs
        # self.signal_strength = np.zeros(num_locations) # Signal strength at locations
        # self.interference = np.zeros(num_locations) # Interference at locations
        # self.noise = np.zeros(num_locations) # Noise at locations
        # self.historical_data = np.zeros(num_base_stations) # Previous locations of RABSs

        # print('Traffic',self.traffic_load)
        self.max_steps = 24
        self.active_energy_rate = 72.38
        self.grasp_energy_rate = 10
        self.sleep_energy_rate = 0
        self.P_fly = 356
        self.v = 30

        # Constraints
        # self.energy_constraints = np.random.uniform(333672, 333672, size=num_base_stations)
        self.energy_constraints = np.random.uniform(100101, 100101, size=num_base_stations) # Assume 30% of battery recharge every step
        self.capacity_constraints = np.random.uniform(7200, 7200, size=num_base_stations)
        self.location_constraints = np.random.choice(num_locations, size=num_base_stations, replace=False)

        # Define the observation space
        self.observation_space = spaces.Dict({
            'traffic_load': spaces.Box(low=0, high=np.inf, shape=(num_locations,), dtype=np.float32),
            'energy_consumption': spaces.Box(low=0, high=np.inf, shape=(num_base_stations,), dtype=np.float32),
            #'locations': spaces.MultiDiscrete([num_locations] * num_base_stations),
            'locations': spaces.Box(low=0, high=num_base_stations, shape=(num_base_stations,), dtype=int),
            'traffic_served':spaces.Box(low=0, high=np.inf, shape=(num_base_stations,), dtype=np.float32),
            # 'signal_strength': spaces.Box(low=0, high=np.inf, shape=(num_locations,), dtype=np.float32),
            # 'interference': spaces.Box(low=0, high=np.inf, shape=(num_locations,), dtype=np.float32),
            # 'noise': spaces.Box(low=0, high=np.inf, shape=(num_locations,), dtype=np.float32),
            # 'historical_data': spaces.Box(low=0, high=np.inf, shape=(num_locations,), dtype=np.float32)
        })

        # Define the action space
        self.action_space = spaces.MultiBinary(num_locations)
        # print('Action Space', self.action_space)

    def step(self, action):
        previous_locations = self.locations.copy()
        # print('Previous',previous_locations)

        # Check occupacy constraints
        unique_action = self.occupacy_constraint(previous_locations, action)
        self.locations = unique_action
        # print('Action Location', unique_action)

        # Update the energy consumption based on the new locations
        self.energy_consumption, self.traffic_served, self.locations = self.calculate_energy_consumption(previous_locations, unique_action)
        # print('traffic served', self.traffic_served)
        # print('energy', self.energy_consumption)

        # Check termination condition
        done = self.is_termination_condition_met()

        # Calculate the reward based on the traffic load and energy consumption
        reward, fairness_index, total_traffic, total_energy = self.calculate_reward()

        # Update the new state with the updated base station locations
        new_state = self._get_observation()
        self.current_step += 1

        return new_state, reward, done , fairness_index, total_traffic, total_energy, {}

    def reset(self):
        self.current_step = 0
        self.reward = 0

        self.traffic_load = np.zeros(self.num_locations)
        # print('traffic load',self.traffic_load)
        self.energy_consumption = np.zeros(self.num_base_stations)  # Energy consumption of RABSs
        self.constraints_violated = np.zeros(self.num_base_stations)  # Penality
        self.energy_constraint_penalty = np.zeros(self.num_base_stations)
        self.capacity_constraint_penalty = np.zeros(self.num_base_stations)
        self.energy_violation_count = 0
        self.capacity_violation_count = 0

        # print('energy',self.energy_consumption)
        # self.locations = np.random.choice(range(1, self.num_locations ), self.num_base_stations, replace=False)  # Locations served by RABSs
        self.traffic_served = np.zeros(self.num_base_stations)  # Traffic served by RABSs
        # self.signal_strength = np.zeros(self.num_locations )  # Signal strength at locations
        # self.interference = np.zeros(self.num_locations )  # Interference at locations
        # self.noise = np.zeros(self.num_locations )  # Noise at locations
        # self.historical_data = self.locations  # Previous locations of RABSs

        # Return the initial state of the environment
        initial_state = self._get_observation()

        return initial_state

    def calculate_energy_consumption(self, previous_locations, new_locations):
        # ---------------------------------------------------------
        # PRACTICALITY UPDATE 1: Interference Modeling
        # Real-world radio constraint: Proximity causes signal degradation.
        # ---------------------------------------------------------
        interference_radius = 200  # meters (Defining "Too Close")
        interference_penalty = 0.8  # 20% throughput loss if interfering

        # Initialize penalty mask (1.0 = no penalty)
        throughput_factors = np.ones(self.num_base_stations)

        # Check for proximity between all pairs of RABS
        for i in range(self.num_base_stations):
            for j in range(i + 1, self.num_base_stations):
                # Calculate distance between RABS i and RABS j
                loc_i = new_locations[i]
                loc_j = new_locations[j]

                # Use the pre-loaded distance matrix
                dist_ij = self.fly_distances[loc_i][loc_j]

                if dist_ij < interference_radius:
                    # Apply penalty to BOTH RABS
                    throughput_factors[i] *= interference_penalty
                    throughput_factors[j] *= interference_penalty

        # ---------------------------------------------------------
        # PRACTICALITY UPDATE 2: Service Latency
        # Real-world flight constraint: Flying takes time, reducing service availability.
        # ---------------------------------------------------------
        time_slot_duration = 3600  # 1 hour in seconds

        for i, (prev, new) in enumerate(zip(previous_locations, new_locations)):
            if prev != new:
                distance = self.fly_distances[prev][new]

                # Calculate flight time (Latency)
                flight_time = distance / self.v

                # Calculate "Service Ratio": Fraction of the time slot remaining for service
                # If flight takes 6 mins, you only have 90% of the hour to serve users.
                service_ratio = max(0, (time_slot_duration - flight_time) / time_slot_duration)

                # Energy Cost (Standard)
                self.energy_consumption[i] = self.active_energy_rate + self.grasp_energy_rate + (
                        self.P_fly * distance / self.v)

                # Traffic Served (Modified by Latency AND Interference)
                max_possible_traffic = self.capacity_constraints[i] * service_ratio
                served_raw = min(max_possible_traffic, self.traffic_load[new])

                # Apply Interference Penalty
                self.traffic_served[i] = served_raw * throughput_factors[i]

                # Fail-Safe: If energy runs out, revert movement
                if (self.energy_consumption[i] >= self.energy_constraints[i]):
                    energy_violation = self.energy_consumption[i] - self.energy_constraints[i]
                    self.energy_constraint_penalty[i] = np.log(energy_violation + 1)
                    self.energy_violation_count += 1

                    # Revert to hover state
                    self.energy_consumption[i] = self.active_energy_rate + self.grasp_energy_rate
                    # Even hovering suffers from interference if neighbor is close
                    self.traffic_served[i] = self.traffic_load[prev] * throughput_factors[i]
                    self.locations[i] = prev
            else:
                # Stationary RABS
                self.energy_consumption[i] = self.active_energy_rate + self.grasp_energy_rate
                # Still suffers from interference penalty
                self.traffic_served[i] = min(self.capacity_constraints[i], self.traffic_load[prev]) * \
                                         throughput_factors[i]

        return self.energy_consumption, self.traffic_served, self.locations

    def calculate_reward(self):
        # Calculate the reward based on the traffic load and energy consumption

        qoc_per_location = self.calculate_coverage_signal_strength()
        # print("QoC values per location:", np.var(qoc_per_location))

        fairness_index = (np.sum(qoc_per_location) ** 2) / (
                self.num_locations * np.sum(qoc_per_location ** 2) + 1e-8
        )
        # print('fairness_index', fairness_index)

        # Fairness index (e.g., Jain's fairness index)
        capacity_fairness_index = (np.sum(self.traffic_served) ** 2) / (
                        self.num_base_stations * np.sum(self.traffic_served ** 2) + 1e-8)
        # print('capacity fairness_index', capacity_fairness_index)

        energy_fairness_index = (np.sum(self.energy_consumption) ** 2) / (
                self.num_base_stations * np.sum(self.energy_consumption ** 2) + 1e-8)
        # print('energy fairness_index', energy_fairness_index)

        # Apply penalties and calculate base reward
        penalty_weight = 0
        total_penalty = penalty_weight * (
                    np.sum(self.energy_constraint_penalty) + np.sum(self.capacity_constraint_penalty))
        # print('penalty', total_penalty)

        # print('traffic check', self.traffic_served)
        # print('energy check', self. energy_consumption)
        base_reward = np.average(self.traffic_served / self.energy_consumption)
        # base_reward = np.sum(self.traffic_served) / np.sum(self.energy_consumption)

        # Ensure the reward does not become overly negative
        fairness_weight = 0
        # self.reward = max(0, base_reward - total_penalty + self.fairness_weight * fairness_index)
        self.reward = max(0, base_reward - total_penalty + fairness_weight * fairness_index)
        # self.reward = fairness_index

        return self.reward, fairness_index, np.sum(self.traffic_served), np.sum(self.energy_consumption)

    def calculate_coverage_signal_strength(self):
        """
        Calculate base station coverage for each location based on signal strength (inverse of distance^2).
        Returns:
            coverage_per_location (np.array): Coverage values for each location.
        """
        coverage_per_location = np.zeros(self.num_locations)

        for loc in range(self.num_locations):
            for bs in range(self.num_base_stations):
                # Calculate signal strength contribution
                distance = self.fly_distances[self.locations[bs], loc]
                signal_strength = 1 / (1 + distance ** 2)  # Inverse-square law for signal strength

                # Contribution is weighted by traffic served by the base station
                coverage_per_location[loc] += signal_strength * self.traffic_served[bs]

        # Normalize coverage per location
        coverage_per_location /= (np.sum(coverage_per_location) + 1e-8)

        return coverage_per_location

    def get_neighbors(self, base_station_id, radius, max_neighbors):
        """
        Finds neighboring base stations based on the distance matrix of locations.

        Parameters:
            base_station_id (int): ID of the base station.
            radius (float): Maximum distance for considering neighbors.
            max_neighbors (int or None): Maximum number of neighbors to return.
                                         If None, returns all neighbors within the radius.

        Returns:
            List[int]: IDs of neighboring base stations sorted by distance (ascending).
        """
        # Get the location served by the current base station
        current_location = self.locations[base_station_id]

        # List to store neighbors and their distances
        neighbors = []

        for other_station_id in range(self.num_base_stations):
            if other_station_id == base_station_id:
                continue  # Skip itself

            # Get the location served by the other base station
            other_location = self.locations[other_station_id]

            # Calculate the distance between the two locations
            distance = self.fly_distances[current_location, other_location]

            # Check if the distance is within the radius
            if distance <= radius:
                neighbors.append((other_station_id, distance))

        # Sort neighbors by distance (ascending)
        neighbors = sorted(neighbors, key=lambda x: x[1])

        # Limit to max_neighbors if specified
        if max_neighbors is not None:
            neighbors = neighbors[:max_neighbors]

        # Return only the neighbor IDs
        return [neighbor_id for neighbor_id, _ in neighbors]

    def is_termination_condition_met(self):
        # Terminate after a fixed number of steps
        return self.current_step >= self.max_steps

    def occupacy_constraint(self, previous_locations, action):
        # Check constraints and penalize if violated
        # Ensure unique actions by replacing duplicates with previous locations
        unique_action = list(set(action))  # Remove duplicates

        # Replace duplicate locations with previous locations
        while len(unique_action) < self.num_base_stations:
            for i in range(len(action)):
                if list(action).count(action[i]) > 1:
                    # Check if the previous location is unique
                    if previous_locations[i] not in action:
                        action[i] = previous_locations[i]  # Replace with previous location if unique
                    else:
                        # If the previous location is not unique, leave the action unchanged
                        # Find the remaining locations that were not assigned
                        remaining_locations = set(range(1, self.num_locations + 1)) - set(unique_action)
                        # Randomly pick one of the unassigned locations and add it to unique_action
                        action[i] = remaining_locations.pop()
            unique_action = list(set(action))  # Check for uniqueness after replacement

        return action

    def _get_observation(self):
        # Return the current observation as the state
        return {
            'traffic_load': self.traffic_load,
            'energy_consumption': self.energy_consumption,
            'locations': self.locations,
            'traffic_served': self.traffic_served,
            # 'signal_strength': self.signal_strength,
            # 'interference': self.interference,
            # 'noise': self.noise,
            # 'historical_data': self.historical_data
        }

# Example usage
#
# env = BaseStationDeploymentEnv(10, 5)
#
# # Initial reset
# initial_observation = env.reset()
#
# action = np.random.choice(range(1, 10), 5, replace=False)
# print('Action', action)
# observation, reward, done, _ = env.step(action)
# print('reward',reward)
#
# action2 = np.random.choice(range(1, 10), 5, replace=False)
# print('Action 2', action2)
# observation, reward, done, _ = env.step(action2)
# print('reward 2',reward)
#
# action3 = np.random.choice(range(1, 10), 5, replace=False)
# print('Action 3', action3)
# observation, reward, done, _ = env.step(action3)
# print('reward 3',reward)
