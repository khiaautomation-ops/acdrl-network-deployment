import random
import math
import torch
import numpy as np

num_locations = 120
# Side length is based on the square root of number of locations (assuming a square area)
side_length = 5 # # side length in kilometers
# Area and site to site distance
# Convert kilometers to meters
side_length_m = side_length * 1000
# Calculate the area of the square
area = side_length_m ** 2
# distance = math.sqrt(area / 120)

def place_randomly(num_locations, area_side_length_m):
    # Initialize the list to store drone coordinates (x, y)
    drone_coordinates = []

    for _ in range(num_locations):
        # Generate random coordinates for each drone
        x_coord = random.uniform(0, area_side_length_m)
        y_coord = random.uniform(0, area_side_length_m)

        # Add the drone coordinates to the list
        drone_coordinates.append((x_coord, y_coord))

    return drone_coordinates

def euclidean_distance(coord1, coord2):
    x1, y1 = coord1
    x2, y2 = coord2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calculate_distances_between_locations(locations_coordinates):
    num_locations = len(locations_coordinates)
    distances = [[0.0 for _ in range(num_locations)] for _ in range(num_locations)]

    for i in range(num_locations):
        for j in range(i + 1, num_locations):
            distance = euclidean_distance(locations_coordinates[i], locations_coordinates[j])
            distances[i][j] = distance
            distances[j][i] = distance

    return distances

location_coordinates = place_randomly(num_locations, side_length_m)
fly_distances = torch.tensor(calculate_distances_between_locations(location_coordinates))

np.save('locations_coordinates_1200.npy', fly_distances)

