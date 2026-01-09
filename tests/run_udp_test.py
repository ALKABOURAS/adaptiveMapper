import sys
import os
import time

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

    print("🚀 Sending Data to Unity... Press Ctrl+C to stop.")

    try:
        while True:
            # 1. Read
            _, noisy_val = sensor.read_gyro_z()

            # 2. Filter
            clean_val = kf.update(noisy_val)

            # 3. Adapt
            final_val, mode = mapper.map_input(clean_val)

            # 4. Send to Unity (Στέλνουμε το final_val ως X)
            sender.send_data(final_val, 0, 0)

            # Μικρή καθυστέρηση για να προλαβαίνουμε να το δούμε (simulate 60fps)
            time.sleep(0.016)

    except KeyboardInterrupt:
        print("\n🛑 Stopped.")

if __name__ == "__main__":
    run_live_simulation()