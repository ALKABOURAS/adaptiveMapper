import math

class AdaptiveMapper:
    """
    Handles the mapping from filtered sensor data to screen coordinates.
    Implements adaptive gain control based on movement velocity.
    """
    def __init__(self, base_gain=1.0, precision_factor=0.2, threshold=2.0):
        """
        Args:
            base_gain (float): Standard sensitivity for fast movements.
            precision_factor (float): Multiplier for precision mode (e.g., 0.2 means 20% speed).
            threshold (float): Velocity threshold below which precision mode is activated.
        """
        self.base_gain = base_gain
        self.precision_factor = precision_factor
        self.threshold = threshold

    def map_2d_input(self, val_x, val_y):
        """
        Applies adaptive gain to 2D vector input.

        Args:
            val_x (float): Filtered velocity X.
            val_y (float): Filtered velocity Y.

        Returns:
            tuple: (final_x, final_y, mode_string)
        """
        # Calculate velocity magnitude (Euclidean norm)
        magnitude = math.sqrt(val_x**2 + val_y**2)
        status = ""

        # Determine gain based on velocity threshold
        if magnitude < self.threshold:
            current_gain = self.base_gain * self.precision_factor
            status = "Precision Mode"
        else:
            current_gain = self.base_gain
            status = "Fast Mode"

        # Apply the calculated gain to both axes to preserve motion direction
        out_x = val_x * current_gain
        out_y = val_y * current_gain

        return out_x, out_y, status