import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.sensors.joycon_driver import JoyConDriver
# Αντικαθιστούμε το Kalman με το OneEuro
from src.filters.one_euro import OneEuroFilter
from src.processing.adaptive import AdaptiveMapper
from src.networking.udp_client import UDPSender

def run_navigation():
    # 1. Hardware Init
    # Διάλεξε 'pro', 'left', ή 'right'
    joy = JoyConDriver(device_type='left')
    if not joy.open():
        print("Controller not found.")
        return

    print("Auto-Calibration active. Leave still if drifting.")

    # 2. Filters Init (1€ Filter)
    # min_cutoff: 0.5 -> Σταθερότητα όταν είναι ακίνητο
    # beta: 4.0       -> Ταχύτητα όταν κινείται (αν θες πιο γρήγορο, κάντο 10.0)
    oe_filter_yaw   = OneEuroFilter(min_cutoff=0.5, beta=4.0, d_cutoff=1.0)
    oe_filter_pitch = OneEuroFilter(min_cutoff=0.5, beta=4.0, d_cutoff=1.0)

    # 3. Adaptive Mapper (Precision Mode)
    # threshold=8.0: Κάτω από 8 dps ταχύτητα, ρίχνει την ευαισθησία
    mapper = AdaptiveMapper(threshold=8.0, precision_factor=0.4)

    sender = UDPSender(port=5005)

    # --- USER CONFIGURATION (TUNING) ---
    SENSITIVITY = 30.0  # Μικρότερο = Πιο γρήγορο
    INVERT_X = False    # Αν το Αριστερά πάει Δεξιά
    INVERT_Y = False    # Αν το Πάνω πάει Κάτω

    cursor_x, cursor_y = 0.0, 0.0

    print(f"ΝAVIGATION STARTED | Sens:{SENSITIVITY} | InvX:{INVERT_X} | InvY:{INVERT_Y}")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            # A. Read Sensor
            # Περνάμε το χρόνο στο update για να δουλεύει σωστά το 1€ Filter
            current_timestamp = time.time()

            data = joy.read_imu_dps()
            if not data: continue

            raw_x, raw_y, raw_z = data

            # B. Axis Mapping (Pro Controller / Joy-Con differences)
            # Για Pro Controller & Joy-Con (συνήθως):
            # Yaw (Left/Right) = Z axis
            # Pitch (Up/Down)  = X axis (ή Y σε κάποιους κλώνους)

            raw_input_yaw   = raw_z
            raw_input_pitch = raw_y # Αν το πάνω-κάτω είναι λάθος άξονας, δοκίμασε raw_y

            # C. Filtering (One Euro)
            # Περνάμε το timestamp για να υπολογίσει την ταχύτητα
            clean_yaw   = oe_filter_yaw.update(raw_input_yaw, current_timestamp)
            clean_pitch = oe_filter_pitch.update(raw_input_pitch, current_timestamp)

            # D. Adaptive Gain (Precision Mode logic)
            final_yaw, final_pitch, mode = mapper.map_2d_input(clean_yaw, clean_pitch)

            # E. Integration & Sensitivity
            step_x = (final_yaw / SENSITIVITY)
            step_y = (final_pitch / SENSITIVITY)

            # F. Inversions
            if INVERT_X: step_x *= -1
            if INVERT_Y: step_y *= -1

            # G. Update Position
            cursor_x += step_x
            cursor_y += step_y

            # H. Bounds (Περιορισμός οθόνης Unity -20 έως 20)
            cursor_x = max(-22, min(22, cursor_x))
            cursor_y = max(-12, min(12, cursor_y))

            # I. Send to Unity
            sender.send_data(cursor_x, cursor_y, 0)

            # Μικρή αναμονή για να μην καίμε CPU (το 1€ filter διαχειρίζεται το dt αυτόματα)
            # time.sleep(0.001)

    except KeyboardInterrupt:
        joy.close()
        print("\nNavigation Stopped.")

if __name__ == "__main__":
    run_navigation()