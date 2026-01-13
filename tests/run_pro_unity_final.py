import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.sensors.joycon_driver import JoyConDriver
from src.filters.kalman import SimpleKalmanFilter
from src.processing.adaptive import AdaptiveMapper
from src.networking.udp_client import UDPSender

def run_final_link():
    # 1. Hardware Init
    joy = JoyConDriver(device_type='pro')
    if not joy.open():
        print("❌ Connect Pro Controller first!")
        return

    print("⚖️  Auto-Calibration active. Leave still if drifting occurs.")

    # 2. Pipeline Init
    # Χρησιμοποιούμε Adaptive Mapper για να έχουμε Precision Mode
    mapper = AdaptiveMapper(threshold=5.0, precision_factor=0.5)

    # Φίλτρα Kalman (ρυθμισμένα για Pro Controller που είναι σταθερό)
    kf_x = SimpleKalmanFilter(process_noise=1.0, measurement_noise=2.0)
    kf_y = SimpleKalmanFilter(process_noise=1.0, measurement_noise=2.0)

    sender = UDPSender(port=5005)

    # --- ΡΥΘΜΙΣΕΙΣ (TUNING) ---
    SENSITIVITY = 100 # Όσο πιο μικρό, τόσο πιο γρήγορο.
    INVERT_Y = True    # Αν πηγαίνει ανάποδα το πάνω-κάτω, κάντο False
    INVERT_X = True   # Αν πηγαίνει ανάποδα το αριστερά-δεξιά, κάντο True

    # Θέση Κέρσορα
    cursor_x, cursor_y = 0.0, 0.0

    print("🚀 PRO CONTROLLER LINKED TO UNITY")
    print("---------------------------------")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            # A. Read Sensor (dps)
            data = joy.read_imu_dps()
            if not data: continue

            raw_x, raw_y, raw_z = data

            # B. Axis Mapping (ΔΙΟΡΘΩΣΗ)

            # ΓΙΑ ΝΑ ΚΟΥΝΙΕΤΑΙ ΠΑΝΩ-ΚΑΤΩ ΜΕ TILT (PITCH):
            # Χρησιμοποιούμε το raw_x (ή raw_y ανάλογα τη συσκευή).
            # Αφού τώρα το Tilt Left/Right κουνάει το Y, σημαίνει ότι διαβάζεις λάθος άξονα.

            # Δοκίμασε ΑΥΤΟ τον συνδυασμό:
            input_pitch = raw_y   # <--- Άλλαξε το σε raw_x ή raw_y (δοκίμασε το αντίθετο από ό,τι έχεις)
            input_yaw   = raw_z   # Αυτό το κρατάμε, αφού το αριστερά-δεξιά δουλεύει σωστά

            # ΣΗΜΕΙΩΣΗ: Στα Joy-Cons/Pro Controllers:
            # Gyro X = Pitch (Πάνω/Κάτω)
            # Gyro Y = Roll (Κλίση Αριστερά/Δεξιά)
            # Gyro Z = Yaw (Περιστροφή στο τραπέζι)

            # C. Filtering
            clean_pitch = kf_y.update(input_pitch)
            clean_yaw   = kf_x.update(input_yaw)

            # D. Adaptive Gain (Precision Mode)
            final_yaw, final_pitch, mode = mapper.map_2d_input(clean_yaw, clean_pitch)

            # E. Integration (Velocity -> Position)
            # Εδώ εφαρμόζουμε το Sensitivity
            dt = 0.015 # Περίπου 15ms loop time

            step_x = (final_yaw / SENSITIVITY)
            step_y = (final_pitch / SENSITIVITY)

            if INVERT_Y: step_y *= -1
            if INVERT_X: step_x *= -1

            cursor_x += step_x
            cursor_y += step_y

            # F. Bounds (Για να μη χάνουμε τον κύβο)
            cursor_x = max(-20, min(20, cursor_x))
            cursor_y = max(-10, min(10, cursor_y))

            # G. Send
            sender.send_data(cursor_x, cursor_y, 0)

            # Debug Print (κάθε 10 frames για να μη ζαλιζόμαστε)
            # print(f"Mode: {mode} | X: {cursor_x:.2f} | Y: {cursor_y:.2f}")

            time.sleep(0.001)

    except KeyboardInterrupt:
        joy.close()

if __name__ == "__main__":
    run_final_link()