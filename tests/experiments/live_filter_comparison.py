import sys
import os
import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # Απαραίτητο για Windows
import matplotlib.pyplot as plt
from collections import deque

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.sensors.joycon_driver import JoyConDriver
from src.filters.kalman import SimpleKalmanFilter
from src.filters.one_euro import OneEuroFilter

def suppress_crosstalk(val_x, val_y, ratio_threshold=4.0, min_activity=10.0):
    """
    Αν ο ένας άξονας είναι πολύ πιο δυνατός από τον άλλον, μειώνει τον αδύναμο.

    ratio_threshold: Πόσες φορές μεγαλύτερος πρέπει να είναι ο κυρίαρχος άξονας.
    min_activity: Ελάχιστη ταχύτητα για να ενεργοποιηθεί η λογική (ώστε να μην χαλάμε τις μικρο-διορθώσεις).
    """
    abs_x = abs(val_x)
    abs_y = abs(val_y)

    # Αν δεν υπάρχει σημαντική κίνηση, μην κάνεις τίποτα (για να δουλεύει το precision)
    if abs_x < min_activity and abs_y < min_activity:
        return val_x, val_y

    # Έλεγχος Κυριαρχίας
    # Αν το X είναι κυρίαρχο (π.χ. 4 φορές μεγαλύτερο από το Y)
    if abs_x > abs_y * ratio_threshold:
        # Θεωρούμε το Y ως θόρυβο/crosstalk και το μειώνουμε δραματικά
        val_y = val_y * 0.1 # Το αφήνουμε ελάχιστα ζωντανό ή το κάνουμε 0.0

    # Αν το Y είναι κυρίαρχο
    elif abs_y > abs_x * ratio_threshold:
        # Θεωρούμε το X ως θόρυβο
        val_x = val_x * 0.1

    return val_x, val_y

def run_2d_cardiogram():
    # 1. SETUP HARDWARE
    # Άλλαξε σε 'left' ή 'right' αν δεν έχεις το Pro τώρα
    joy = JoyConDriver(device_type='left')
    if not joy.open(): return

    print("⚖️ Auto-calibrating... Please stay still for 2 seconds.")
    time.sleep(2)

    # 2. SETUP FILTERS (Σύγκριση)


    # --- YAW AXIS (Left/Right) ---
    kf_yaw = SimpleKalmanFilter(process_noise=1.0, measurement_noise=10.0)
    oe_yaw = OneEuroFilter(min_cutoff=0.5, beta=4.0, d_cutoff=1.0)

    # --- PITCH AXIS (Up/Down) ---
    kf_pitch = SimpleKalmanFilter(process_noise=1.0, measurement_noise=10.0)
    oe_pitch = OneEuroFilter(min_cutoff=0.5, beta=4.0, d_cutoff=1.0)

    # 3. SETUP PLOT
    plt.ion() # Interactive Mode
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    maxlen = 150 # Πόσα frames βλέπουμε στην οθόνη
    x_data = np.arange(maxlen)

    # Buffers - Χρειαζόμαστε 6 buffers (Raw, Kalman, OneEuro για κάθε άξονα)
    # Yaw Buffers
    raw_yaw_buf = deque([0]*maxlen, maxlen=maxlen)
    kf_yaw_buf  = deque([0]*maxlen, maxlen=maxlen)
    oe_yaw_buf  = deque([0]*maxlen, maxlen=maxlen)

    # Pitch Buffers
    raw_pit_buf = deque([0]*maxlen, maxlen=maxlen)
    kf_pit_buf  = deque([0]*maxlen, maxlen=maxlen)
    oe_pit_buf  = deque([0]*maxlen, maxlen=maxlen)

    # Lines - Yaw
    ax_yaw = axs[0]
    l_r_y, = ax_yaw.plot(x_data, raw_yaw_buf, color='lightgray', label='Raw', lw=1)
    l_k_y, = ax_yaw.plot(x_data, kf_yaw_buf, color='blue', label='Kalman (Fixed)', lw=2)
    l_o_y, = ax_yaw.plot(x_data, oe_yaw_buf, color='green', label='1€ (Adaptive)', lw=2)
    ax_yaw.set_title("YAW Axis (Z-Gyro) - Up/Down")
    ax_yaw.set_ylim(-100, 100)
    ax_yaw.legend(loc='upper right')
    ax_yaw.grid(True)

    # Lines - Pitch
    ax_pit = axs[1]
    l_r_p, = ax_pit.plot(x_data, raw_pit_buf, color='lightgray', label='Raw', lw=1)
    l_k_p, = ax_pit.plot(x_data, kf_pit_buf, color='blue', label='Kalman (Fixed)', lw=2)
    l_o_p, = ax_pit.plot(x_data, oe_pit_buf, color='green', label='1€ (Adaptive)', lw=2)
    ax_pit.set_title("PITCH Axis (X-Gyro) - Left/Right")
    ax_pit.set_ylim(-100, 100)
    ax_pit.legend(loc='upper right')
    ax_pit.grid(True)

    print("🚀 LIVE 2-AXIS CARDIOGRAM STARTED!")
    print("-----------------------------------")
    print("🔵 Blue: Kalman (R=10) -> Smooth but Laggier")
    print("🟢 Green: 1€ Filter (beta=4) -> Fast & Smooth")
    print("-----------------------------------")

    try:
        while True:
            # A. Read Sensors
            current_ts = time.time()
            data = joy.read_imu_dps()
            if not data: continue

            raw_x, raw_y, raw_z, is_triggered = data

            # MAPPING (Pro Controller)
            in_yaw   = raw_y
            in_pitch = raw_z # Αν θες να δεις τον Y άξονα, άλλαξε σε raw_y

            # B. Update Filters
            # Yaw
            k_y = kf_yaw.update(in_yaw)
            o_y = oe_yaw.update(in_yaw, current_ts)

            # Pitch
            k_p = kf_pitch.update(in_pitch)
            o_p = oe_pitch.update(in_pitch, current_ts)

            # Crosstalk Suppression
            k_y, k_p = suppress_crosstalk(k_y, k_p, ratio_threshold=3.5, min_activity=15.0)
            o_y, o_p = suppress_crosstalk(o_y, o_p, ratio_threshold=3.5, min_activity=15.0)

            # C. Update Buffers
            raw_yaw_buf.append(in_yaw)
            kf_yaw_buf.append(k_y)
            oe_yaw_buf.append(o_y)

            raw_pit_buf.append(in_pitch)
            kf_pit_buf.append(k_p)
            oe_pit_buf.append(o_p)

            # D. Update Plot Lines
            # Yaw
            l_r_y.set_ydata(raw_yaw_buf)
            l_k_y.set_ydata(kf_yaw_buf)
            l_o_y.set_ydata(oe_yaw_buf)

            # Pitch
            l_r_p.set_ydata(raw_pit_buf)
            l_k_p.set_ydata(kf_pit_buf)
            l_o_p.set_ydata(oe_pit_buf)

            # Auto-Scale Y axis (Dynamic Zoom)
            # Βρίσκουμε το μέγιστο για να προσαρμόσουμε την κλίμακα
            max_y = max(50, np.max(np.abs(raw_yaw_buf)))
            ax_yaw.set_ylim(-max_y, max_y)

            max_p = max(50, np.max(np.abs(raw_pit_buf)))
            ax_pit.set_ylim(-max_p, max_p)

            # Draw
            fig.canvas.draw()
            fig.canvas.flush_events()

            # No sleep needed, plotting is the bottleneck

    except KeyboardInterrupt:
        joy.close()
        print("Stopped.")

if __name__ == "__main__":
    run_2d_cardiogram()