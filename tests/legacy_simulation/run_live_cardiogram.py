import sys
import os
import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from collections import deque

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.sensors.joycon_driver import JoyConDriver
from src.filters.kalman import SimpleKalmanFilter
from src.filters.one_euro import OneEuroFilter # Το νέο μας φίλτρο

def run_cardiogram():
    # 1. Setup Controller
    joy = JoyConDriver(device_type='left') # ή 'pro'
    if not joy.open(): return

    print("⚖️ Auto-calibrating...")
    time.sleep(2)

    # 2. Setup Filters for Comparison
    # Kalman (Σταθερό smoothing)
    kf = SimpleKalmanFilter(process_noise=1.0, measurement_noise=10.0)

    # One Euro (Προσαρμοστικό smoothing)
    # min_cutoff=0.1 -> Πολύ σταθερό όταν είναι ακίνητο
    # beta=5.0       -> Πολύ γρήγορο όταν κινείται
    one_euro = OneEuroFilter(min_cutoff=0.5, beta=10.0, d_cutoff=1.0)

    # 3. Setup Plot
    plt.ion() # Interactive Mode On
    fig, ax = plt.subplots(figsize=(12, 6))

    # Buffer size: Πόσα σημεία θα δείχνει η οθόνη (π.χ. 200 frames)
    maxlen = 200
    x_data = np.arange(maxlen)

    # Buffers δεδομένων
    raw_buffer = deque([0]*maxlen, maxlen=maxlen)
    kf_buffer = deque([0]*maxlen, maxlen=maxlen)
    oe_buffer = deque([0]*maxlen, maxlen=maxlen)

    # Lines setup
    line_raw, = ax.plot(x_data, raw_buffer, color='lightgray', label='Raw (Noisy)', linewidth=1)
    line_kf,  = ax.plot(x_data, kf_buffer, color='blue', label='Kalman (Fixed)', linewidth=2)
    line_oe,  = ax.plot(x_data, oe_buffer, color='green', label='1€ Filter (Adaptive)', linewidth=2)

    ax.set_ylim(-100, 100) # Αρχικά όρια Y
    ax.set_title("Live Sensor Data: Kalman vs 1€ Filter")
    ax.legend(loc='upper right')
    ax.grid(True)

    print("🚀 LIVE CARDIOGRAM STARTED!")
    print("🟢 Green: 1€ Filter (Adaptive)")
    print("🔵 Blue: Kalman (Fixed)")
    print("⚪ Gray: Raw Noise")

    try:
        while True:
            # A. Read Sensor (Yaw axis only for demo)
            data = joy.read_imu_dps()
            if not data: continue
            raw_x, raw_y, raw_z = data

            val_to_plot = raw_z # Ας δούμε τον άξονα Z (Yaw)

            # B. Apply Filters
            val_kf = kf.update(val_to_plot)
            val_oe = one_euro.update(val_to_plot)

            # C. Update Buffers
            raw_buffer.append(val_to_plot)
            kf_buffer.append(val_kf)
            oe_buffer.append(val_oe)

            # D. Update Plot Lines (Fast update)
            line_raw.set_ydata(raw_buffer)
            line_kf.set_ydata(kf_buffer)
            line_oe.set_ydata(oe_buffer)

            # Auto-scale Y axis (για να μη χάνεται η γραμμή)
            current_max = max(50, np.max(np.abs(raw_buffer)))
            ax.set_ylim(-current_max, current_max)

            # Draw
            fig.canvas.draw()
            fig.canvas.flush_events()

            # Δεν βάζουμε sleep εδώ, το plotting τρώει χρόνο

    except KeyboardInterrupt:
        joy.close()
        print("Stopped.")

if __name__ == "__main__":
    run_cardiogram()