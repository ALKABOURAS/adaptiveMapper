import numpy as np

class MockIMUSensor:
    def __init__(self, noise_std=0.5):
        self.noise_std = noise_std
        self.time_step = 0

    # --- ΓΙΑ ΤΑ ΠΑΛΙΑ TESTS (1D) ---
    def read_gyro_z(self):
        """
        Επιστρέφει (true_val, noisy_val) για έναν άξονα.
        Χρησιμοποιείται από: run_simulation.py, run_udp_test.py
        """
        true_val = np.sin(self.time_step / 10.0) * 10
        noise = np.random.normal(0, self.noise_std)
        noisy_val = true_val + noise

        self.time_step += 1
        return true_val, noisy_val

    # --- ΓΙΑ ΤΑ ΝΕΑ TESTS (2D) ---
    def read_2d_gyro(self):
        """
        Επιστρέφει (true_x, noisy_x, true_y, noisy_y).
        Χρησιμοποιείται από: run_udp_2d.py
        """
        true_x = np.sin(self.time_step / 10.0) * 10
        true_y = np.cos(self.time_step / 10.0) * 8 # Cosine για να κάνει κύκλο

        noise_x = np.random.normal(0, self.noise_std)
        noise_y = np.random.normal(0, self.noise_std)

        noisy_x = true_x + noise_x
        noisy_y = true_y + noise_y

        self.time_step += 1
        return true_x, noisy_x, true_y, noisy_y