import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.sensors.joycon_driver import JoyConDriver
from src.filters.kalman import SimpleKalmanFilter
from src.processing.adaptive import AdaptiveMapper
from src.networking.udp_client import UDPSender

def run_application():
    joy = JoyConDriver()
    if not joy.open(): return

    # --- ΒΗΜΑ 1: CALIBRATION ---
    print("⚖️  CALIBRATING... ΑΚΟΥΜΠΗΣΕ ΤΟ JOY-CON ΣΤΟ ΤΡΑΠΕΖΙ ΚΑΙ ΜΗΝ ΤΟ ΚΟΥΝΑΣ!")
    time.sleep(1) # Σου δίνω 1 δευτερόλεπτο να το αφήσεις
    joy.calibrate(samples=1000) # Παίρνουμε περισσότερα δείγματα για ακρίβεια

    # --- CONFIGURATION ---
    # Πειράζουμε αυτά τα νούμερα για να φτιάξουμε την αίσθηση
    SENSITIVITY_X = 200  # Μεγαλύτερο νούμερο = Πιο αργή κίνηση
    SENSITIVITY_Y = 200
    DEADZONE = 50       # Αν η τιμή (μετά το bias) είναι < 10, γίνε 0.

    kf_x = SimpleKalmanFilter(process_noise=0.5, measurement_noise=10.0)
    kf_y = SimpleKalmanFilter(process_noise=0.5, measurement_noise=10.0)

    mapper = AdaptiveMapper(threshold=4.0, precision_factor=0.1)
    sender = UDPSender(port=5005)

    # Αρχική θέση (Κέντρο)
    pos_x, pos_y = 0.0, 0.0

    print("🚀 LIVE! (Ctrl+C to stop)")
    print("💡 TIP: Αν φεύγει μόνο του, αύξησε το DEADZONE.")

    try:
        while True:
            data = joy.read_gyro()
            if not data: continue
            rid, gx, gy, gz = data
            if rid != 0x30: continue

            # --- ΒΗΜΑ 2: DEADZONE (Η Λύση στο Drift) ---
            # Αν η κίνηση είναι μικρή (θόρυβος), την μηδενίζουμε
            if abs(gx) < DEADZONE: gx = 0
            if abs(gy) < DEADZONE: gy = 0
            if abs(gz) < DEADZONE: gz = 0

            # --- ΒΗΜΑ 3: MAPPING (Pointer Grip) ---
            # Κρατώντας το Joy-Con(R) όρθιο σαν δείκτη:
            # Στροφή καρπού αριστερά/δεξιά = Gyro Z (Yaw) -> Screen X
            # Στροφή καρπού πάνω/κάτω = Gyro Y (Pitch) -> Screen Y

            # Προσοχή στα πρόσημα (+/-) για να μην πηγαίνει ανάποδα
            input_vel_x = (gz / SENSITIVITY_X) * -1  # Δοκίμασε -1 ή 1
            input_vel_y = (gy / SENSITIVITY_Y) * -1  # Δοκίμασε -1 ή 1

            # --- ΒΗΜΑ 4: PROCESSING ---
            clean_x = kf_x.update(input_vel_x)
            clean_y = kf_y.update(input_vel_y)

            final_vel_x, final_vel_y, mode = mapper.map_2d_input(clean_x, clean_y)

            # --- ΒΗΜΑ 5: INTEGRATION (Velocity -> Position) ---
            pos_x += final_vel_x
            pos_y += final_vel_y

            # --- ΒΗΜΑ 6: BOUNDS (Όρια οθόνης) ---
            # Περιορίζουμε τον κύβο για να μην χάνεται στο άπειρο (-10 έως 10)
            pos_x = max(-20, min(20, pos_x))
            pos_y = max(-10, min(10, pos_y))

            sender.send_data(pos_x, pos_y, 0)

            # Debugging - Ξε-σχολίασε για να δεις τι στέλνεις
            # print(f"In: {gz:4d} | OutX: {final_vel_x:.2f} | PosX: {pos_x:.2f}")

    except KeyboardInterrupt:
        joy.close()

if __name__ == "__main__":
    run_application()