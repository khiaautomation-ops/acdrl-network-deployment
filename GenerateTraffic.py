import numpy as np
import math

class TrafficSimulation:
    def __init__(self, M, sigma):
        self.M = M
        self.sigma = sigma
        self.V = None
        self.RABS = None

    def calculate_V(self, n):
        angle1 = (math.pi / 12) * n + 3.08
        angle2 = (math.pi / 6) * n + 2.08
        angle3 = (math.pi / 4) * n + 1.13

        V = 173.29 + 89.83 * math.sin(angle1) + 52.6 * math.sin(angle2) + 16.68 * math.sin(angle3)
        return V

    def sinusoid_superposition(self, t, *params):
        Vmean = params[0]
        n = (len(params) - 1) // 3
        V = Vmean + sum([params[3 * i + 1] * np.sin(np.pi * params[3 * i + 2] * t + params[3 * i + 3]) for i in range(n)])
        return V

    # def generate_V_m(self, V, sigma):
    #     log_V = np.log(V)
    #     log_V_n = np.random.lognormal(mean=np.log(V) - 0.5 * sigma, sigma=sigma)
    #     return log_V_n

    def generate_V_m(self, V, sigma):
        V = np.clip(V, a_min=1e-3, a_max=None)  # Ensure all values are > 0
        log_V_n = np.random.lognormal(mean=np.log(V) - 0.5 * sigma, sigma=sigma)
        return log_V_n

    # def simulate_traffic(self):
    #     # Generate data to fit the model
    #     t = np.linspace(0, 24, self.M)
    #
    #     # mean traffic volume
    #     Vmean = 173.29
    #     A1, f1, phi1 = 89.83, 1 / 12, 3.08
    #     A2, f2, phi2 = 52.6, 1 / 6, 2.08
    #     A3, f3, phi3 = 16.68, 1 / 4, 1.13
    #     params = [Vmean, A1, f1, phi1, A2, f2, phi2, A3, f3, phi3]
    #
    #     # Define the sinusoid superposition model
    #     self.V = self.sinusoid_superposition(t, *params)
    #
    #     # Generate V_m values
    #     V_m = self.generate_V_m(self.V, self.sigma)
    #
    #     return V_m

    def simulate_traffic(self, profile_type="diurnal"):
        t = np.linspace(0, 24, self.M)

        if profile_type == "diurnal":
            # Existing sinusoidal pattern
            Vmean = 173.29
            A1, f1, phi1 = 89.83, 1 / 12, 3.08
            A2, f2, phi2 = 52.6, 1 / 6, 2.08
            A3, f3, phi3 = 16.68, 1 / 4, 1.13
            params = [Vmean, A1, f1, phi1, A2, f2, phi2, A3, f3, phi3]
            self.V = self.sinusoid_superposition(t, *params)

        elif profile_type == "flat":
            self.V = np.full(self.M, 173.29)

        elif profile_type == "bursty":
            self.V = np.full(self.M, 100.0)
            # Add random bursts at random times
            for _ in range(5):
                burst_time = np.random.randint(0, self.M)
                burst_duration = np.random.randint(5, 15)
                burst_value = np.random.uniform(200, 400)
                end_time = min(burst_time + burst_duration, self.M)
                self.V[burst_time:end_time] += burst_value

        elif profile_type == "high_variance":
            # Exaggerated sinusoidal pattern
            Vmean = 173.29
            A1, f1, phi1 = 120, 1 / 12, 2.5
            A2, f2, phi2 = 70, 1 / 6, 1.5
            A3, f3, phi3 = 25, 1 / 4, 0.5
            params = [Vmean, A1, f1, phi1, A2, f2, phi2, A3, f3, phi3]
            self.V = self.sinusoid_superposition(t, *params)

        else:
            raise ValueError("Unknown traffic profile type")

        # Generate V_m values
        V_m = self.generate_V_m(self.V, self.sigma)
        return V_m

# traffic = TrafficSimulation(M=120, sigma=1.3)
# result = list(np.zeros(24))
#
# for epoch in range(24):
#     result[epoch] = traffic.simulate_traffic()
#
# np.save('demand_data_120.npy', result)

traffic = TrafficSimulation(M=120, sigma=1.3)
profiles = ["diurnal", "flat", "bursty", "high_variance"]

result = {profile: [] for profile in profiles}
for profile in profiles:
    for epoch in range(24):
        result[profile].append(traffic.simulate_traffic(profile_type=profile))

# Save each profile as separate .npy file
for profile in profiles:
    np.save(f'demand_data_120_{profile}.npy', result[profile])
