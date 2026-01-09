import sys
import os
import time
import csv
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.sensors.mock_imu import MockIMUSensor
from src.filters.kalman import SimpleKalmanFilter
from src.processing.adaptive import AdaptiveMapper
from src.networking.udp_client import UDPSender

def run_live_simulation():
    # Setup
    sensor = MockIMUSensor(noise_std=1.5)
    kf = SimpleKalmanFilter(process_noise=0.1, measurement_noise=2.0)
    mapper = AdaptiveMapper(threshold=2.5, precision_factor=0.1)
    sender = UDPSender(port=5005)

    # --- SETUP CSV LOGGING ---
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'simulation_results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_filename = os.path.join(results_dir, f"{timestamp}_live_data.csv")

    print(f"🚀 Sending Data to Unity... Logging to: {csv_filename}")
    print("Press Ctrl+C to stop.")

    # Ανοίγουμε το αρχείο για γράψιμο
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Γράφουμε τους τίτλους των στηλών
        writer.writerow(["Timestamp", "Raw_Noise", "Filtered", "Final_Adaptive", "Mode"])

        try:
            start_time = time.time()
            while True:
                # 1. Processing Loop
                _, noisy_val = sensor.read_gyro_z()
                clean_val = kf.update(noisy_val)
                final_val, mode = mapper.map_input(clean_val)

                # 2. Send to Unity
                sender.send_data(final_val, 0, 0)

                # 3. Log to CSV
                current_time = time.time() - start_time
                writer.writerow([f"{current_time:.4f}", f"{noisy_val:.4f}", f"{clean_val:.4f}", f"{final_val:.4f}", mode])

                time.sleep(0.016) # ~60 FPS

        except KeyboardInterrupt:
            print("\n🛑 Stopped.")
            print(f"✅ Data saved to {csv_filename}")

if __name__ == "__main__":
    run_live_simulation()