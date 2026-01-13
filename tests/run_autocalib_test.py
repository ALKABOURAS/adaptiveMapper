import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.sensors.joycon_driver import JoyConDriver
from src.networking.udp_client import UDPSender

def run_test():
    joy = JoyConDriver(is_left=False) # Joy-Con R
    if not joy.open(): return

    sender = UDPSender(port=5005)

    # ΠΡΟΣΟΧΗ: ΔΕΝ καλούμε το joy.calibrate() στην αρχή!
    # Θέλουμε να δούμε το Drift να συμβαίνει.
    print("⚠️  WARNING: Starting WITHOUT calibration.")
    print("➡️  Ο κύβος θα κουνιέται μόνος του (Drift).")
    print("➡️  Άσε το Joy-Con στο τραπέζι και περίμενε 2-3 δευτερόλεπτα...")

    pos_x, pos_y = 0.0, 0.0

    try:
        while True:
            data = joy.read_imu_dps() # Αυτό καλεί το auto-calib internallly
            if not data: continue

            dx, dy, dz = data

            # Απλό integration για να βλέπουμε κίνηση
            pos_x += -dz * 0.1 # Z-gyro -> X-screen
            pos_y += -dy * 0.1 # Y-gyro -> Y-screen

            # Send to Unity
            sender.send_data(pos_x, pos_y, 0)

            # Αν δεις αυτό το μήνυμα στην κονσόλα, πέτυχε!
            # (Το print γίνεται μέσα στον driver, αλλά εδώ βλέπουμε την επίδραση)

            time.sleep(0.01)

    except KeyboardInterrupt:
        joy.close()

if __name__ == "__main__":
    run_test()