import math
import time

class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        """
        min_cutoff: Ελάχιστη συχνότητα αποκοπής (για αργές κινήσεις).
                    Μικρό νούμερο (π.χ. 0.1) = Πολύ smooth, περισσότερο lag.
        beta:       Συντελεστής ταχύτητας.
                    Μεγάλο νούμερο = Λιγότερο lag στις γρήγορες κινήσεις.
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.x_prev = None
        self.dx_prev = 0
        self.t_prev = None

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def update(self, x, t=None):
        if t is None:
            t = time.time()

        if self.x_prev is None:
            self.dx_prev = 0
            self.x_prev = x
            self.t_prev = t
            return x

        # Υπολογισμός χρονικού βήματος (dt)
        t_e = t - self.t_prev

        # Αν έρχονται δεδομένα υπερβολικά γρήγορα, τα αγνοούμε για αποφυγή διαίρεσης με 0
        if t_e <= 0: return self.x_prev

        # Υπολογισμός ταχύτητας (παράγωγος)
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        # Υπολογισμός του δυναμικού cutoff frequency
        # Εδώ είναι το μυστικό: Cutoff = min + (beta * speed)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        # Τελικό φιλτράρισμα
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        # Save for next frame
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat