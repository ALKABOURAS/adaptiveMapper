import hid
import time
from collections import deque # ΝΕΟ: Για να κρατάμε ιστορικό

class JoyConDriver:
    def __init__(self, device_type='right'):
        """
        device_type: 'left', 'right', ή 'pro'
        """
        self.VENDOR_ID = 0x057E

        # Mapping ονομάτων σε Product IDs
        self.PRODUCT_IDS = {
            'left': 0x2006,
            'right': 0x2007,
            'pro': 0x2009
        }

        # Έλεγχος αν δόθηκε σωστός τύπος
        if device_type not in self.PRODUCT_IDS:
            print(f"⚠️ Unknown device type '{device_type}', defaulting to 'right'")
            device_type = 'right'

        self.target_pid = self.PRODUCT_IDS[device_type]
        self.device_type = device_type

        self.device = None
        self.global_packet_number = 0

        # ... (τα υπόλοιπα variables: bias, history, κλπ παραμένουν ίδια) ...
        self.bias_x = 0
        self.bias_y = 0
        self.bias_z = 0
        self.DPS_FACTOR = 0.06103

        # Auto-calib vars
        self.history_len = 50
        self.gyro_history_x = deque(maxlen=self.history_len)
        self.gyro_history_y = deque(maxlen=self.history_len)
        self.gyro_history_z = deque(maxlen=self.history_len)
        self.still_start_time = None
        self.required_still_time = 2.0

    def open(self):
        print(f"🔍 Scanning for {self.device_type.upper()} Controller (PID: {hex(self.target_pid)})...")
        for device_info in hid.enumerate(self.VENDOR_ID):
            if device_info['product_id'] == self.target_pid:
                print(f"✅ Found Nintendo Device: {self.device_type.upper()}")
                try:
                    self.device = hid.device()
                    self.device.open_path(device_info['path'])
                    self.device.set_nonblocking(True)
                    self._enable_imu_sequence()
                    return True
                except Exception as e:
                    print(f"❌ Failed to open: {e}")
        return False

    def _send_command(self, subcommand, argument):
        self.global_packet_number = (self.global_packet_number + 1) % 16
        rumble_data = [0x00, 0x01, 0x40, 0x40, 0x00, 0x01, 0x40, 0x40]
        command = [0x01, self.global_packet_number] + rumble_data + [subcommand] + argument
        self.device.write(bytes(command))
        time.sleep(0.05)

    def _enable_imu_sequence(self):
        self._send_command(0x40, [0x01])
        self._send_command(0x03, [0x30])
        time.sleep(0.2)

    # --- Η ΚΛΑΣΙΚΗ CALIBRATE (ΧΕΙΡΟΚΙΝΗΤΗ) ---
    def calibrate(self, samples=500):
        print(f"⚖️  Manual Calibration... STAY STILL!")
        sx, sy, sz = 0, 0, 0
        count = 0
        while count < samples:
            data = self._read_raw_dps() # Διαβάζουμε χωρίς αφαίρεση bias
            if data:
                gx, gy, gz = data
                sx += gx; sy += gy; sz += gz
                count += 1
            time.sleep(0.002)

        self.bias_x = sx / count
        self.bias_y = sy / count
        self.bias_z = sz / count
        print(f"✅ Manual Bias Set: {self.bias_x:.2f}, {self.bias_y:.2f}, {self.bias_z:.2f}")

    # --- Η ΝΕΑ AUTO-CALIBRATE LOGIC ---
    def check_auto_calibration(self, raw_dps_x, raw_dps_y, raw_dps_z):
        """
        Καλείται σε κάθε frame. Ελέγχει αν είμαστε ακίνητοι και διορθώνει το Bias.
        Επιστρέφει True αν έγινε recalibration.
        """
        self.gyro_history_x.append(raw_dps_x)
        self.gyro_history_y.append(raw_dps_y)
        self.gyro_history_z.append(raw_dps_z)

        # Πρέπει να γεμίσει το buffer πρώτα
        if len(self.gyro_history_x) < self.history_len:
            return False

        # Υπολογισμός 'Θορύβου' (Max - Min)
        noise_x = max(self.gyro_history_x) - min(self.gyro_history_x)
        noise_y = max(self.gyro_history_y) - min(self.gyro_history_y)
        noise_z = max(self.gyro_history_z) - min(self.gyro_history_z)

        # Όριο θορύβου (Threshold): Αν κουνιέται λιγότερο από 3.0 dps, θεωρείται ακίνητο.
        STABILITY_THRESHOLD = 3.0

        is_stable = (noise_x < STABILITY_THRESHOLD) and \
                    (noise_y < STABILITY_THRESHOLD) and \
                    (noise_z < STABILITY_THRESHOLD)

        if is_stable:
            if self.still_start_time is None:
                self.still_start_time = time.time()
            else:
                # Αν είμαστε σταθεροί για τον απαιτούμενο χρόνο
                if time.time() - self.still_start_time > self.required_still_time:
                    # UPDATING BIAS!
                    self.bias_x = sum(self.gyro_history_x) / self.history_len
                    self.bias_y = sum(self.gyro_history_y) / self.history_len
                    self.bias_z = sum(self.gyro_history_z) / self.history_len

                    self.still_start_time = None # Reset timer
                    self.gyro_history_x.clear() # Clear buffers
                    return True # Ενημερώνουμε ότι έγινε recalibration
        else:
            self.still_start_time = None # Κουνήθηκε, άρα reset το χρονόμετρο

        return False

    def _read_raw_dps(self):
        """Helper function: Διαβάζει DPS χωρίς να αφαιρεί το bias."""
        if not self.device: return None
        report = self.device.read(64)
        if not report or report[0] != 0x30: return None

        raw_gx = report[19] | (report[20] << 8)
        raw_gy = report[21] | (report[22] << 8)
        raw_gz = report[23] | (report[24] << 8)
        def to_signed(n): return n - 65536 if n > 32767 else n

        return (to_signed(raw_gx)*self.DPS_FACTOR,
                to_signed(raw_gy)*self.DPS_FACTOR,
                to_signed(raw_gz)*self.DPS_FACTOR)

    def read_imu_dps(self):
        """Η κύρια συνάρτηση που καλείς. Κάνει ΚΑΙ τον έλεγχο auto-calib."""
        raw_data = self._read_raw_dps()
        if not raw_data: return None

        raw_x, raw_y, raw_z = raw_data

        # 1. Τρέξε τον έλεγχο (παρασκηνιακά)
        was_calibrated = self.check_auto_calibration(raw_x, raw_y, raw_z)

        # 2. Επίστρεψε το διορθωμένο
        final_x = raw_x - self.bias_x
        final_y = raw_y - self.bias_y
        final_z = raw_z - self.bias_z

        if was_calibrated:
            print(f"✨ Auto-Calibrated! New Bias -> X:{self.bias_x:.1f}, Y:{self.bias_y:.1f}")

        return final_x, final_y, final_z

    def close(self):
        if self.device: self.device.close()