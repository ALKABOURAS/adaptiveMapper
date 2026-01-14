import sys
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from datetime import datetime

# Προσθήκη path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.sensors.mock_imu import MockIMUSensor
from src.filters.kalman import SimpleKalmanFilter
from src.processing.adaptive import AdaptiveMapper

def run_experiment():
    # Setup
    sensor = MockIMUSensor(noise_std=2.5)
    kf = SimpleKalmanFilter(process_noise=0.1, measurement_noise=3.0)
    mapper = AdaptiveMapper(threshold=2.5, precision_factor=0.1)

    raw_data = []
    filtered_data = []
    final_data = []
    ground_truth = []

    # Simulation Loop
    for _ in range(150):
        true_val, noisy_val = sensor.read_gyro_z()

        # 1. Filter
        clean_val = kf.update(noisy_val)

        # 2. Adaptive Mapping (ΤΩΡΑ ΔΙΑΒΑΖΟΥΜΕ ΚΑΙ ΤΙΣ 2 ΤΙΜΕΣ)
        mapped_val, mode = mapper.map_input(clean_val) # <--- Η ΔΙΟΡΘΩΣΗ ΕΙΝΑΙ ΕΔΩ

        raw_data.append(noisy_val)
        filtered_data.append(clean_val)
        final_data.append(mapped_val)
        ground_truth.append(true_val)

    # Plotting
    plt.figure(figsize=(10, 8))

    # Γράφημα 1
    plt.subplot(2, 1, 1)
    plt.title("Step 1: Kalman Filter Performance")
    plt.plot(raw_data, label='Raw (Noisy)', color='lightgray')
    plt.plot(ground_truth, label='Ideal Motion', color='green', linestyle='--')
    plt.plot(filtered_data, label='Filtered Output', color='blue', linewidth=2)
    plt.legend()
    plt.grid(True)

    # Γράφημα 2
    plt.subplot(2, 1, 2)
    plt.title("Step 2: Adaptive Mapping (Precision Mode)")
    plt.plot(filtered_data, label='Filtered Input', color='blue', alpha=0.3)
    plt.plot(final_data, label='Adaptive Output', color='red', linewidth=2)
    plt.axhline(y=mapper.threshold, color='orange', linestyle=':', label='Threshold')
    plt.axhline(y=-mapper.threshold, color='orange', linestyle=':')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # --- SAVE LOGIC ---
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'simulation_results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{timestamp}_simulation_graph.png"
    full_path = os.path.join(results_dir, filename)

    plt.savefig(full_path, dpi=300)
    print(f"💾 Graph saved: {full_path}")

    plt.show()

if __name__ == "__main__":
    run_experiment()