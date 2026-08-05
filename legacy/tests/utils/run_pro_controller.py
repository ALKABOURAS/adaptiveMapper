import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from legacy.src.sensors.joycon_driver import JoyConDriver
from legacy.src.networking.udp_client import UDPSender

def run_pro_test():
    # Ζητάμε συγκεκριμένα το 'pro'
    joy = JoyConDriver(device_type='pro')

    if not joy.open():
        print("❌ Δεν βρέθηκε Pro Controller. Βεβαιώσου ότι είναι συνδεδεμένο.")
        print("⚠️  Επίσης: ΚΛΕΙΣΕ ΤΟ STEAM (το Steam κλέβει το Pro Controller).")
        return

    sender = UDPSender(port=5005)

    print("🚀 Pro Controller Connected!")
    print("⚖️  Ακούμπα το κάτω για Auto-Calibration...")

    pos_x, pos_y = 0.0, 0.0

    try:
        while True:
            # Διαβάζει και κάνει auto-calibrate μόνο του
            data = joy.read_imu_dps()
            if not data: continue

            gx, gy, gz = data

            # Απλή πλοήγηση
            # Pro Controller Orientation:
            # Συνήθως το Gyro Z είναι το Yaw (αριστερά/δεξιά)
            # Το Gyro X ή Y είναι το Pitch (πάνω/κάτω) ανάλογα πώς είναι η πλακέτα.
            # Δοκίμασε τα παρακάτω:

            pos_x += -gz * 0.05  # Yaw -> Screen X
            pos_y += -gx * 0.05  # Pitch -> Screen Y (στο Pro ίσως είναι το X αντί για Y)

            sender.send_data(pos_x, pos_y, 0)

            time.sleep(0.01)

    except KeyboardInterrupt:
        joy.close()

if __name__ == "__main__":
    run_pro_test()