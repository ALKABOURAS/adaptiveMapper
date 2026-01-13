import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.sensors.joycon_driver import JoyConDriver
from src.filters.kalman import SimpleKalmanFilter
from src.networking.udp_client import UDPSender

def run_dual_app():
    # 1. Connect Both Controllers
    joy_L = JoyConDriver(is_left=True)
    joy_R = JoyConDriver(is_left=False)

    connected_L = joy_L.open()
    connected_R = joy_R.open()

    if not connected_L and not connected_R:
        print("❌ Δεν βρέθηκε κανένα Joy-Con!")
        return

    # 2. Calibration (Μόνο το δεξί μας νοιάζει για το Gyro drift)
    if connected_R:
        joy_R.calibrate(samples=500)

    # 3. Setup Filters & Network
    kf_yaw = SimpleKalmanFilter(process_noise=0.1, measurement_noise=5.0)
    kf_pitch = SimpleKalmanFilter(process_noise=0.1, measurement_noise=5.0)
    sender = UDPSender(port=5005)

    # State Variables
    cursor_x, cursor_y = 0.0, 0.0  # Θέση δείκτη (από δεξί χέρι)
    cube_pos_x, cube_pos_z = 0.0, 0.0 # Θέση κύβου στο χώρο (από αριστερό χέρι)

    last_time = time.time()

    print("🚀 DUAL MODE: Left Stick -> Move, Right Gyro -> Aim")

    try:
        while True:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            # --- LEFT HAND (Movement) ---
            if connected_L:
                move_x, move_y = joy_L.read_stick()
                # Deadzone για το stick
                if abs(move_x) < 0.15: move_x = 0
                if abs(move_y) < 0.15: move_y = 0

                # Update Position
                SPEED = 5.0
                cube_pos_x += move_x * SPEED * dt
                cube_pos_z += move_y * SPEED * dt

            # --- RIGHT HAND (Aiming / Rotation) ---
            if connected_R:
                gyro_data = joy_R.read_imu_dps() # Επιστρέφει DPS τώρα!
                if gyro_data:
                    dps_x, dps_y, dps_z = gyro_data

                    # Deadzone (σε μοίρες/δευτερόλεπτο πλέον)
                    if abs(dps_z) < 2.0: dps_z = 0
                    if abs(dps_y) < 2.0: dps_y = 0

                    # Filter
                    filt_z = kf_yaw.update(dps_z)
                    filt_y = kf_pitch.update(dps_y)

                    # Integrate: Degrees = degrees/sec * seconds
                    # (Mapping: Z-axis gyro -> X-axis screen, Y-axis gyro -> Y-axis screen)
                    cursor_x += filt_z * dt * -1.0 # -1 για αντιστροφή
                    cursor_y += filt_y * dt * -1.0

            # --- SEND TO UNITY ---
            # Εδώ πρέπει να αποφασίσουμε τι στέλνουμε.
            # Για το Fitts' Law Test (που είναι 2D pointing), μας νοιάζει το cursor_x/y.
            # Αλλά αν θες να δείξεις "Navigation", στέλνουμε και τα 3.

            # Στέλνουμε: X (Aim), Y (Aim), Z (Movement from Left Hand?)
            # Ή αν θες να δεις το Stick να δουλεύει:
            # X = Cursor X, Y = Cursor Y, Z = Stick Value (για να το δούμε να κουνιέται)

            sender.send_data(cursor_x, cursor_y, cube_pos_x)

            # (Στο Unity θα χρειαστεί να αλλάξουμε λίγο το script αν θέλουμε να δούμε 3D κίνηση)

            # time.sleep(0.005) # Fast loop

    except KeyboardInterrupt:
        joy_L.close()
        joy_R.close()

if __name__ == "__main__":
    run_dual_app()