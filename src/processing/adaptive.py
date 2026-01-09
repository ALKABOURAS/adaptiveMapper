import math

class AdaptiveMapper:
    def __init__(self, base_gain=1.0, precision_factor=0.2, threshold=2.0):
        self.base_gain = base_gain
        self.precision_factor = precision_factor
        self.threshold = threshold

    # --- ΓΙΑ ΤΑ ΠΑΛΙΑ TESTS (1D) ---
    def map_input(self, filtered_val):
        """
        Δέχεται μία τιμή. Υπολογίζει abs().
        Χρησιμοποιείται από: run_simulation.py
        """
        abs_val = abs(filtered_val)
        status = ""

        if abs_val < self.threshold:
            current_gain = self.base_gain * self.precision_factor
            status = "Precision Mode"
        else:
            current_gain = self.base_gain
            status = "Fast Mode"

        final_result = filtered_val * current_gain
        return final_result, status

    # --- ΓΙΑ ΤΑ ΝΕΑ TESTS (2D) ---
    def map_2d_input(self, val_x, val_y):
        """
        Δέχεται X, Y. Υπολογίζει Magnitude (Πυθαγόρειο).
        Χρησιμοποιείται από: run_udp_2d.py
        """
        # Υπολογισμός συνολικής ταχύτητας (Magnitude)
        magnitude = math.sqrt(val_x**2 + val_y**2)
        status = ""

        if magnitude < self.threshold:
            current_gain = self.base_gain * self.precision_factor
            status = "Precision Mode"
        else:
            current_gain = self.base_gain
            status = "Fast Mode"

        # Εφαρμόζουμε το ΙΔΙΟ gain και στους δύο άξονες
        out_x = val_x * current_gain
        out_y = val_y * current_gain

        return out_x, out_y, status