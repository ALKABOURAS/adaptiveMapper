import numpy as np

class MockIMUSensor:
    def __init__(self, noise_std=0.5):
        self.noise_std = noise_std
        self.time_step = 0

    def read_gyro_z(self):
        # Προσομοίωση κίνησης (Sine wave)
        true_val = np.sin(self.time_step / 10.0) * 10

        # Προσθήκη θορύβου
        noise = np.random.normal(0, self.noise_std)
        noisy_val = true_val + noise

        self.time_step += 1
        return true_val, noisy_val