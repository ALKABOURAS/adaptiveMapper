import sys
import os

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Προσθήκη του src path για να βλέπει τα modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime

from src.sensors.mock_imu import MockIMUSensor
from src.filters.kalman import SimpleKalmanFilter
from src.processing.adaptive import AdaptiveMapper

def run_experiment():
    # Setup (Πειράζουμε αυτές τις τιμές για να δούμε διαφορές)
    sensor = MockIMUSensor(noise_std=2.5) # Αρκετός θόρυβος
    kf = SimpleKalmanFilter(process_noise=0.1, measurement_noise=3.0)
    mapper = AdaptiveMapper(threshold=2.5, precision_factor=0.1) # Πολύ αργό στο precision

    raw_data = []
    filtered_data = []
    final_data = []
    ground_truth = []

    # Simulation Loop (150 frames)
    for _ in range(150):
        true_val, noisy_val = sensor.read_gyro_z()

        # 1. Φιλτράρισμα
        clean_val = kf.update(noisy_val)

        # 2. Adaptive Mapping
        mapped_val = mapper.map_input(clean_val)

        raw_data.append(noisy_val)
        filtered_data.append(clean_val)
        final_data.append(mapped_val)
        ground_truth.append(true_val)

    # Plotting
    plt.figure(figsize=(10, 8))

    # Γράφημα 1: Θόρυβος vs Φίλτρο
    plt.subplot(2, 1, 1)
    plt.title("Step 1: Kalman Filter Performance")
    plt.plot(raw_data, label='Raw (Noisy)', color='lightgray')
    plt.plot(ground_truth, label='Ideal Motion', color='green', linestyle='--')
    plt.plot(filtered_data, label='Filtered Output', color='blue', linewidth=2)
    plt.legend()
    plt.grid(True)

    # Γράφημα 2: Adaptive Output
    plt.subplot(2, 1, 2)
    plt.title("Step 2: Adaptive Mapping (Precision Mode)")
    plt.plot(filtered_data, label='Filtered Input', color='blue', alpha=0.3)
    plt.plot(final_data, label='Adaptive Output', color='red', linewidth=2)

    # Ζωγραφίζουμε το Threshold zone
    plt.axhline(y=mapper.threshold, color='orange', linestyle=':', label='Threshold')
    plt.axhline(y=-mapper.threshold, color='orange', linestyle=':')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Save the plot with a timestamp
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'simulation_results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(results_dir, f"simulation_{timestamp}.png"))

    plt.show()

if __name__ == "__main__":
    run_experiment()