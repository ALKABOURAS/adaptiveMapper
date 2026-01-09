class AdaptiveMapper:
    def __init__(self, base_gain=1.0, precision_factor=0.2, threshold=2.0):
        """
        threshold: Κάτω από αυτή την τιμή, μπαίνουμε σε 'Precision Mode'.
        precision_factor: Πολλαπλασιαστής για το precision mode (π.χ. 0.2 = 20% ταχύτητα).
        """
        self.base_gain = base_gain
        self.precision_factor = precision_factor
        self.threshold = threshold

    def map_input(self, filtered_val):
        abs_val = abs(filtered_val)
        status = "" # Μεταβλητή για να ξέρουμε σε τι mode είμαστε

        # Λογική Adaptive Gain
        if abs_val < self.threshold:
            # Precision Mode (μικρές κινήσεις -> πολύ χαμηλή ευαισθησία)
            current_gain = self.base_gain * self.precision_factor
            status = "Precision Mode"
        else:
            # Fast Mode (γρήγορες κινήσεις -> κανονική ευαισθησία)
            current_gain = self.base_gain
            status = "Fast Mode"

        final_result = filtered_val * current_gain

        # ΕΠΙΣΤΡΕΦΟΥΜΕ 2 ΤΙΜΕΣ (Την τιμή ΚΑΙ το Status)
        return final_result, status