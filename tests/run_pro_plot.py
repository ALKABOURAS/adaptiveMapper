import sys
import os
import time
import matplotlib
matplotlib.use('TkAgg') # Για να ανοίγει παράθυρο στα Windows
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.sensors.joycon_driver import JoyConDriver
from src.filters.kalman import SimpleKalmanFilter
from src.processing.adaptive import AdaptiveMapper
from src.networking.udp_client import UDPSender

results_dir = os.path.join(os.path.dirname(__file__), '..', 'simulation_results')
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def run_with_plot():
    # 1. Setup
    # device_type='pro' ή 'right' ανάλογα τι έχεις
    joy = JoyConDriver(device_type='left')
    if not joy.open():
        print("❌ Controller not found")
        return

    # Φίλτρα & Mapper
    kf_x = SimpleKalmanFilter(process_noise=1.0, measurement_noise=10.0)
    kf_y = SimpleKalmanFilter(process_noise=1.0, measurement_noise=10.0)
    mapper = AdaptiveMapper(threshold=10.0, precision_factor=0.6)
    sender = UDPSender(port=5005)

    # Tuning
    SENSITIVITY = 40.0
    INVERT_Y = False
    INVERT_X = False

    cursor_x, cursor_y = 0.0, 0.0

    # --- LISTS ΓΙΑ ΤΟ PLOTTING ---
    history_raw_x = []
    history_filt_x = []
    history_final_x = [] # Μετά το adaptive
    timestamps = []
    start_time = time.time()

    print("🚀 LIVE RECORDING! Move the controller...")
    print("🛑 Press Ctrl+C to STOP and VIEW THE GRAPH.")

    try:
        while True:
            # A. Read
            data = joy.read_imu_dps()
            if not data: continue
            raw_x, raw_y, raw_z = data

            # Mapping (Pro Controller logic)
            # Αν έχεις Joy-Con άλλαξέ τα όπως πριν (π.χ. input_yaw = raw_z)
            input_pitch = raw_y
            input_yaw   = raw_z

            # B. Filter
            clean_yaw = kf_x.update(input_yaw)
            clean_pitch = kf_y.update(input_pitch)

            # C. Adaptive
            final_yaw, final_pitch, mode = mapper.map_2d_input(clean_yaw, clean_pitch)

            # D. Integration
            step_x = (final_yaw / SENSITIVITY)
            step_y = (final_pitch / SENSITIVITY)
            if INVERT_Y: step_y *= -1
            if INVERT_X: step_x *= -1

            cursor_x += step_x
            cursor_y += step_y

            # Limit
            cursor_x = max(-20, min(20, cursor_x))
            cursor_y = max(-10, min(10, cursor_y))

            # Send to Unity
            sender.send_data(cursor_x, cursor_y, 0)

            # --- RECORD DATA (Αποθηκεύουμε τον Yaw άξονα για το γράφημα) ---
            # Καταγράφουμε τι γίνεται στον Χ άξονα (Yaw)
            history_raw_x.append(input_yaw)      # Τι έδωσε ο αισθητήρας
            history_filt_x.append(clean_yaw)     # Τι έβγαλε το Kalman
            # history_final_x.append(final_yaw)  # Τι έβγαλε το Adaptive (προαιρετικό)
            timestamps.append(time.time() - start_time)

            # time.sleep(0.001)

    except KeyboardInterrupt:
        joy.close()
        print("\n📊 Generating Plot... Please wait.")

        # --- PLOTTING CODE in 2axis---
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        # Zoom στα τελευταία 500 frames για λεπτομέρεια
        limit = -500 if len(timestamps) > 500 else 0

        # Plot X Axis
        axs[0].plot(timestamps[limit:], history_raw_x[limit:], color='lightgray', label='Raw X')
        axs[0].plot(timestamps[limit:], history_filt_x[limit:], color='blue', linewidth=2, label='Filtered X')
        axs[0].set_title(f"Horizontal Axis (Yaw) - Noise R={kf_x.R}")
        axs[0].set_ylabel("Velocity (dps)")
        axs[0].legend(loc='upper right')
        axs[0].grid(True)

        # Plot Y Axis
        axs[1].plot(timestamps[limit:], history_raw_x[limit:], color='lightgray', label='Raw Y')
        axs[1].plot(timestamps[limit:], history_filt_x[limit:], color='red', linewidth=2, label='Filtered Y')
        axs[1].set_title(f"Vertical Axis (Pitch) - Noise R={kf_y.R}")
        axs[1].set_xlabel("Time (seconds)")
        axs[1].set_ylabel("Velocity (dps)")
        axs[1].legend(loc='upper right')
        axs[1].grid(True)

        print("💾 Saving plot to 'performance_graph.png'...")
        plt.savefig(os.path.join(results_dir, f"{timestamp}_performance_graph.png"))
        plt.show()

if __name__ == "__main__":
    run_with_plot()