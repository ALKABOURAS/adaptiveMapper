import sys
import os
import time
import math

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.sensors.joycon_driver import JoyConDriver
from src.filters.one_euro import OneEuroFilter
from src.networking.udp_client import UDPSender
from src.processing.crosstalk import SignalProcessor

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

def run_precision_navigation():
    # 1. Hardware
    joy = JoyConDriver(device_type='left') # Χρησιμοποίησε το L (ZL trigger)
    if not joy.open(): return

    print("Auto-Calibration active...")

    # 2. Filters (One Euro) - Αρχικές τιμές για NORMAL mode
    # Beta: Πόσο "ακούει" την ταχύτητα. Μεγάλο = Γρήγορο.
    # MinCutoff: Πόσο φιλτράρει την ηρεμία. Μικρό = Σταθερό.
    oe_yaw   = OneEuroFilter(min_cutoff=0.8, beta=5.0, d_cutoff=1.0)
    oe_pitch = OneEuroFilter(min_cutoff=0.8, beta=5.0, d_cutoff=1.0)

    sender = UDPSender(port=5005)

    cursor_x, cursor_y = 0.0, 0.0

    print("PRECISION MODE TEST")
    print("Hold ZL/Trigger: SNIPER MODE (Heavy filtering + Speed Limit)")
    print("Release: NORMAL MODE (Fast response)")

    try:
        while True:
            # A. Read
            data = joy.read_imu_dps()
            if not data: continue
            raw_x, raw_y, raw_z, is_precision_btn = data

            # Mapping
            input_yaw   = raw_z
            input_pitch = raw_y

            # B. DYNAMIC TUNING (Το "Αυτί" που ζήτησες)
            if is_precision_btn:
                # --- SNIPER MODE (TUNED) ---
                # Παλιό: 0.01 -> Πολύ βαρύ (προκαλεί ρόμβους)
                # Νέο: 0.1 -> Αρκετό για να κόβει τρέμουλο, αλλά επιτρέπει καμπύλες
                oe_yaw.min_cutoff   = 0.05
                oe_pitch.min_cutoff = 0.05

                # Παλιό: 0.5 -> Πολύ αργή αντίδραση
                # Νέο: 1.0 -> Λίγο πιο ζωντανό
                oe_yaw.beta   = 0.8
                oe_pitch.beta = 0.8

                # Παλιό: 60.0 -> Προκαλεί stuttering
                # Νέο: 40.0 -> Πιο γρήγορο, αλλά ελέγχεται από το Clamping
                CURRENT_SENSITIVITY = 50.0

                # Το Clamping το κρατάμε ίδιο, γιατί δουλεύει καλά για τον "κόφτη"
                MAX_SPEED_LIMIT = 0.5

            else:
                # --- NORMAL MODE ---
                # 1. Επαναφέρουμε το φίλτρο σε "Sport mode"
                oe_yaw.min_cutoff   = 1.0
                oe_pitch.min_cutoff = 1.0
                oe_yaw.beta   = 6.0
                oe_pitch.beta = 6.0

                CURRENT_SENSITIVITY = 25.0 # Γρήγορο
                MAX_SPEED_LIMIT = 5.0      # Ουσιαστικά χωρίς όριο

            # C. Update Filters with NEW parameters
            current_ts = time.time()
            clean_yaw   = oe_yaw.update(input_yaw, current_ts)
            clean_pitch = oe_pitch.update(input_pitch, current_ts)

            # Process Crosstalk
            clean_yaw, clean_pitch = SignalProcessor.suppress_crosstalk(
                clean_yaw, clean_pitch, ratio_threshold=3.5, min_activity=15.0
            )

            # D. Calculate Step
            step_x = clean_yaw / CURRENT_SENSITIVITY
            step_y = clean_pitch / CURRENT_SENSITIVITY

            # E. APPLY CLAMPING (Η λύση για το "2 μέτρα κίνηση")
            # Αν είσαι σε precision mode, το MAX_SPEED_LIMIT είναι 0.1
            # Άρα ακόμα και αν κουνήσεις το χέρι βίαια, το βήμα δεν θα ξεπεράσει το 0.1
            step_x = clamp(step_x, -MAX_SPEED_LIMIT, MAX_SPEED_LIMIT)
            step_y = clamp(step_y, -MAX_SPEED_LIMIT, MAX_SPEED_LIMIT)

            # F. Update Position
            cursor_x += step_x
            cursor_y += step_y

            # Bounds
            cursor_x = max(-22, min(22, cursor_x))
            cursor_y = max(-12, min(12, cursor_y))

            sender.send_data(cursor_x, cursor_y, 0)


    except KeyboardInterrupt:
        joy.close()

if __name__ == "__main__":
    run_precision_navigation()