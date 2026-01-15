# src/filters/crosstalk_suppression.py
import math


class SignalProcessor:
    @staticmethod
    def suppress_crosstalk(val_x, val_y, ratio_threshold=3.0, min_activity=5.0, dampening_factor=0.1):
        """
        Εφαρμόζει Dominant Axis Isolation.

        Args:
            val_x, val_y: Οι τιμές ταχύτητας (dps) μετά το φιλτράρισμα.
            ratio_threshold: Πόσες φορές μεγαλύτερος πρέπει να είναι ο ένας άξονας (default: 3.0).
            min_activity: Κατώφλι ταχύτητας κάτω από το οποίο δεν εφαρμόζεται suppression (default: 5.0).
            dampening_factor: Πόσο 'σκοτώνουμε' τον αδύναμο άξονα (0.0 = τελείως, 0.1 = soft lock).

        Returns:
            (new_x, new_y): Οι επεξεργασμένες τιμές.
        """
        abs_x = abs(val_x)
        abs_y = abs(val_y)

        # 1. Safety check: Αν είμαστε ακίνητοι ή κάνουμε micro-adjustments,
        # ΜΗΝ πειράξεις το σήμα. Θέλουμε ελευθερία στα μικρά movements.
        if abs_x < min_activity and abs_y < min_activity:
            return val_x, val_y

        # 2. Axis Isolation Logic
        if abs_x > abs_y * ratio_threshold:
            # Το X είναι κυρίαρχο -> Μείωσε το Y
            val_y = val_y * dampening_factor

        elif abs_y > abs_x * ratio_threshold:
            # Το Y είναι κυρίαρχο -> Μείωσε το X
            val_x = val_x * dampening_factor

        return val_x, val_y