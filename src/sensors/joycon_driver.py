import hid
import time

class JoyConDriver:
    def __init__(self):
        self.VENDOR_ID = 0x057E
        self.PRODUCT_L = 0x2006
        self.PRODUCT_R = 0x2007
        self.device = None
        self.global_packet_number = 0
        self.bias_x = 0
        self.bias_y = 0
        self.bias_z = 0

    def open(self):
        print("🔍 Scanning for Joy-Cons...")
        for device_info in hid.enumerate(self.VENDOR_ID):
            pid = device_info['product_id']
            if pid == self.PRODUCT_L or pid == self.PRODUCT_R:
                print(f"✅ Found Joy-Con ({'Left' if pid == self.PRODUCT_L else 'Right'})")
                try:
                    self.device = hid.device()
                    self.device.open_path(device_info['path'])
                    self.device.set_nonblocking(True)

                    # Καθαρίζουμε τυχόν σκουπίδια από το buffer
                    self._flush_input()

                    # --- SEQUENCE ΕΝΕΡΓΟΠΟΙΗΣΗΣ ---
                    self._enable_imu_sequence()
                    return True
                except Exception as e:
                    print(f"❌ Failed to open: {e}")
                    return False
        return False

    def _flush_input(self):
        """Αδειάζει το buffer πριν στείλουμε εντολές."""
        for _ in range(10):
            self.device.read(64)

    def _send_command(self, subcommand, argument):
        """
        Στέλνει εντολή με σωστό 'Neutral Rumble' για να μην την αγνοεί το Joy-Con.
        Format: [0x01] [Timer] [Rumble(8 bytes)] [Subcmd] [Arg]
        """
        self.global_packet_number = (self.global_packet_number + 1) % 16

        # Neutral Rumble bytes: x00 x01 x40 x40 (για High/Low bands) x2
        # Αυτό λέει στο Joy-Con "Μην δονείσαι, αλλά άκου την εντολή"
        rumble_data = [0x00, 0x01, 0x40, 0x40, 0x00, 0x01, 0x40, 0x40]

        command = [0x01, self.global_packet_number] \
                  + rumble_data \
                  + [subcommand] \
                  + argument

        self.device.write(bytes(command))
        time.sleep(0.05) # Μικρή καθυστέρηση για να προλάβει να επεξεργαστεί

    def _enable_imu_sequence(self):
        print("⚙️  Waking up Sensors...")

        # 1. Enable IMU (6-Axis Sensor)
        # Subcmd: 0x40, Arg: 0x01 (Enable)
        self._send_command(0x40, [0x01])

        # 2. Set Input Report Mode to Standard Full (0x30)
        # Subcmd: 0x03, Arg: 0x30
        self._send_command(0x03, [0x30])

        print("🚀 Commands Sent. Waiting for response...")
        time.sleep(0.5) # Περιμένουμε λίγο να ξυπνήσει

    def read_gyro(self):
        if not self.device: return None

        # Διαβάζουμε μέχρι 64 bytes (το report 0x30 είναι συνήθως 49 bytes)
        report = self.device.read(64)

        if not report: return None

        # DEBUG: Ας δούμε τι Report στέλνει
        # Αν στέλνει 0x3F (63), είναι ακόμα σε απλό mode (buttons only)
        # Αν στέλνει 0x30 (48), είναι στο σωστό mode
        report_id = report[0]

        if report_id == 0x30:
            # RAW Data Parsing (Little Endian)
            # Bytes 19-24 είναι το 1ο Gyro Sample
            raw_gyro_x = report[19] | (report[20] << 8)
            raw_gyro_y = report[21] | (report[22] << 8)
            raw_gyro_z = report[23] | (report[24] << 8)

            def to_signed(n):
                return n - 65536 if n > 32767 else n

            # Joy-Con Hardware Scaling (χονδρικό calibration για να βγάλει dps)
            # Το coefficient είναι περίπου 0.00061 για degrees/ms ή κάτι παρόμοιο.
            # Εμείς θέλουμε απλά raw values τώρα.
            gx = to_signed(raw_gyro_x)
            gy = to_signed(raw_gyro_y)
            gz = to_signed(raw_gyro_z)
            # --- NEW: Subtract Calibration Bias ---
            final_gx = gx - self.bias_x
            final_gy = gy - self.bias_y
            final_gz = gz - self.bias_z

            # Επιστρέφουμε και το Report ID για debug
            return report_id, gx, gy, gz

        elif report_id == 0x3F:
            # Είναι ακόμα σε Button Mode
            return report_id, 0, 0, 0

        return report_id, 0, 0, 0

    def calibrate(self, samples=500):
        """
        Διαβάζει 500 τιμές ενώ το χειριστήριο είναι ακίνητο
        και υπολογίζει το μέσο σφάλμα (Bias).
        """
        print(f"⚖️  Calibrating... DO NOT MOVE the Joy-Con! ({samples} samples)")

        sum_x, sum_y, sum_z = 0, 0, 0
        count = 0

        while count < samples:
            data = self.read_gyro()
            if data:
                rid, gx, gy, gz = data
                if rid == 0x30:
                    sum_x += gx
                    sum_y += gy
                    sum_z += gz
                    count += 1
            # Μικρή καθυστέρηση για να μην βομβαρδίζουμε
            time.sleep(0.002)

        self.bias_x = sum_x / count
        self.bias_y = sum_y / count
        self.bias_z = sum_z / count

        print(f"✅ Calibration Done! Bias -> X:{self.bias_x:.1f}, Y:{self.bias_y:.1f}, Z:{self.bias_z:.1f}")
        return self.bias_x, self.bias_y, self.bias_z

    def close(self):
        if self.device: self.device.close()

if __name__ == "__main__":
    joy = JoyConDriver()
    if joy.open():
        try:
            print("📡 Monitoring Sensor Data... (Ctrl+C to stop)")
            while True:
                data = joy.read_gyro()
                if data:
                    rid, gx, gy, gz = data

                    if rid == 0x30:
                        # Τυπώνουμε μόνο αν οι τιμές ΔΕΝ είναι μηδέν (ή για debug)
                        print(f"✅ [0x30] Gyro -> X: {gx:5d} | Y: {gy:5d} | Z: {gz:5d}")
                    elif rid == 0x3F:
                        print(f"⚠️ [0x3F] Joy-Con is stuck in Button Mode. Retrying init...")
                        joy._enable_imu_sequence() # Ξαναδοκιμάζουμε να το ξυπνήσουμε
                    else:
                        print(f"❓ Unknown Report ID: {hex(rid)}")

        except KeyboardInterrupt:
            print("\nStopping...")
            joy.close()