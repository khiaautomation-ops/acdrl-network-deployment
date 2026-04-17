import numpy as np

# Load the prediction data
prediction_data = np.load('demand_data_120_1000_high_variance.npy')

# Step 1: Extract shape and basic statistics for scaling
shape = prediction_data.shape
mean_val = prediction_data.mean()
std_val = prediction_data.std()

# Step 2: Generate a sinusoidal pattern across the time dimension
x = np.linspace(0, 2 * np.pi, shape[1])  # 24 points for 24 time periods
sinusoidal_pattern = (np.sin(x)[:, np.newaxis] * std_val * 0.5) + mean_val  # Scaled and shifted

# Repeat the pattern across all locations and episodes
patterned_data = np.tile(sinusoidal_pattern, (shape[0], 1, shape[2]))

# Step 3: Add controlled random noise
noise = np.random.normal(0, std_val * 0.3, shape)  # Noise scaled by 30% of the standard deviation
mixed_data = abs(patterned_data - noise)  # Combine pattern and noise with base structure

# Step 4: Save the new dataset
np.save('mixed_data_120_diurnal_high_variance.npy', mixed_data)