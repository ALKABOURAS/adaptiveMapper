import sys
import os
import time

# Προσαρμογή path ανάλογα με το πού έβαλες το αρχείο
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.sensors.joycon_driver import JoyConDriver
from src.filters.one_euro import OneEuroFilter
from src.processing.adaptive import AdaptiveMapper
from src.networking.udp_client import UDPSender

def run_turbo_navigation():
    # 1. Hardware
    # Βάλε 'left' ή 'right' ανάλογα ποιο SL θες να πατάς
    joy = JoyConDriver(device_type='left')
    if not joy.open(): return

    print("⚖Auto-Calibration active...")

    # 2. Filters
    oe_yaw   = OneEuroFilter(min_cutoff=0.1, beta=4.0, d_cutoff=1.0) # Πολύ σταθερό
    oe_pitch = OneEuroFilter(min_cutoff=0.1, beta=4.0, d_cutoff=1.0)

    # 3. Adaptive Mapper (Για τη "Super Low" κατάσταση)
    # Θέλουμε όταν δεν πατάς κουμπί, να είναι πολύ precise.
    mapper = AdaptiveMapper(threshold=5.0, precision_factor=0.3)

    sender = UDPSender(port=5005)

    # --- CONFIGURATION ---
    # Κανονική λειτουργία (Super Low / Precise)
    SENSITIVITY_NORMAL = 60.0 # Πολύ αργό (μεγάλος διαιρέτης)

    # Turbo λειτουργία (Όταν πατάς SL)
    SENSITIVITY_TURBO  = 15.0 # Πολύ γρήγορο (μικρός διαιρέτης)

    cursor_x, cursor_y = 0.0, 0.0

    print("NAVIGATION WITH TURBO (SL BUTTON)")
    print(f"Default: Slow/Precise (Sens: {SENSITIVITY_NORMAL})")
    print(f"Hold SL: TURBO MODE (Sens: {SENSITIVITY_TURBO})")

    try:
        while True:
            # A. Read Sensor + Button
            data = joy.read_imu_dps() # Τώρα επιστρέφει 4 τιμές!
            if not data: continue

            # Unpack 4 values
            gyro_x, gyro_y, gyro_z, is_turbo = data

            # B. Mapping (Joy-Con L logic)
            # Αν έχεις Left JoyCon κάθετα: Yaw=Z, Pitch=Y (ή X ανάλογα το inversion σου)
            input_yaw   = gyro_z
            input_pitch = gyro_y

            # C. Filtering
            # Περνάμε τα πάντα από το φίλτρο για να μην κλωτσάει
            current_ts = time.time()
            clean_yaw   = oe_yaw.update(input_yaw, current_ts)
            clean_pitch = oe_pitch.update(input_pitch, current_ts)

            # D. LOGIC SWITCHING (CLUTCH)
            if is_turbo:
                # TURBO MODE:
                # 1. Αγνοούμε τον Adaptive Mapper (πάντα Fast)
                # 2. Χρησιμοποιούμε Turbo Sensitivity
                final_yaw = clean_yaw
                final_pitch = clean_pitch
                current_sens = SENSITIVITY_TURBO
            else:
                # NORMAL MODE:
                # 1. Χρησιμοποιούμε Adaptive Mapper (Precision)
                # 2. Χρησιμοποιούμε Normal Sensitivity
                final_yaw, final_pitch, _ = mapper.map_2d_input(clean_yaw, clean_pitch)
                current_sens = SENSITIVITY_NORMAL

            # E. Integration
            # Πρόσεξε: Έχεις κάνει internal inversion στον driver, οπότε ίσως
            # δεν χρειάζεσαι -1 εδώ. Δοκίμασε το.
            cursor_x += (final_yaw / current_sens)
            cursor_y += (final_pitch / current_sens) # * -1 αν χρειαστεί

            # F. Bounds
            cursor_x = max(-22, min(22, cursor_x))
            cursor_y = max(-12, min(12, cursor_y))

            sender.send_data(cursor_x, cursor_y, 0)


    except KeyboardInterrupt:
        joy.close()

if __name__ == "__main__":
    run_turbo_navigation()