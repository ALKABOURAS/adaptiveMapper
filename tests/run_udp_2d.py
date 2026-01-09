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

def run_2d_simulation():
    sensor = MockIMUSensor(noise_std=1.5)

    # Χρειαζόμαστε ΔΥΟ φίλτρα (ανεξάρτητα states)
    kf_x = SimpleKalmanFilter(process_noise=0.1, measurement_noise=2.0)
    kf_y = SimpleKalmanFilter(process_noise=0.1, measurement_noise=2.0)

    mapper = AdaptiveMapper(threshold=3.0, precision_factor=0.15)
    sender = UDPSender(port=5005)

    print("🚀 Sending 2D Data (Circles) to Unity...")

    try:
        while True:
            # 1. Read 2D Data
            _, noisy_x, _, noisy_y = sensor.read_2d_gyro()

            # 2. Filter Separately
            clean_x = kf_x.update(noisy_x)
            clean_y = kf_y.update(noisy_y)

            # 3. Adapt Together (Vector Math)
            final_x, final_y, mode = mapper.map_2d_input(clean_x, clean_y)

            # 4. Send "X,Y,0"
            sender.send_data(final_x, final_y, 0)

            # Optional Debug Print
            # print(f"Mode: {mode} | X: {final_x:.2f} | Y: {final_y:.2f}")

            time.sleep(0.016)

    except KeyboardInterrupt:
        print("\n🛑 Stopped.")

if __name__ == "__main__":
    run_2d_simulation()