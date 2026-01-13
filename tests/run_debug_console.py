import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.sensors.joycon_driver import JoyConDriver

def run_debug():
    # Διάλεξε τι θες να τεστάρεις: 'pro', 'left', ή 'right'
    DEVICE_TYPE = 'pro'

    joy = JoyConDriver(device_type=DEVICE_TYPE)

    if not joy.open():
        print(f"❌ Δεν βρέθηκε {DEVICE_TYPE.upper()} controller.")
        return

    print(f"🚀 {DEVICE_TYPE.upper()} Connected! Printing sensor data...")
    print("-------------------------------------------------------------")
    print("Περίμενε λίγο ακίνητος για να δεις το Bias να αλλάζει (Auto-Calib)")
    print("-------------------------------------------------------------")

    try:
        while True:
            # Διαβάζουμε τα δεδομένα (αυτό τρέχει και το auto-calib στο παρασκήνιο)
            data = joy.read_imu_dps()

            if data:
                gx, gy, gz = data

                # Παίρνουμε και τα Bias για να βλέπουμε πότε ενημερώνονται
                bx = joy.bias_x
                by = joy.bias_y
                bz = joy.bias_z

                # FORMATTING:
                # :6.1f σημαίνει "κράτα 6 θέσεις χώρο, με 1 δεκαδικό".
                # Έτσι τα νούμερα δεν θα χοροπηδάνε δεξιά-αριστερά.

                gyro_str = f"GYRO [dps] | X:{gx:6.1f} | Y:{gy:6.1f} | Z:{gz:6.1f}"
                bias_str = f"BIAS (Offset) | X:{bx:5.1f} | Y:{by:5.1f} | Z:{bz:5.1f}"

                print(f"{gyro_str}   ||   {bias_str}")

            # Λίγο πιο αργό refresh rate για να προλαβαίνει το μάτι (10Hz)
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n🛑 Stopped.")
        joy.close()

if __name__ == "__main__":
    run_debug()