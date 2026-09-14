# filters.py
import math
import time

class OneEuroFilter:
    def __init__(self, min_cutoff=0.1, beta=2.0, d_cutoff=1.0):
        self.x = None
        self.dx = 0
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.alpha = 0.0
        self.last_time = 0

    def smoothing_factor(self, te, cutoff):
        r = 2 * math.pi * cutoff * te
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def filter(self, x, timestamp):
        # 1. Calculate time delta (te)
        if self.last_time != 0 and timestamp != self.last_time:
            self.te = timestamp - self.last_time
        else:
            self.te = 0.033 # Default to ~30fps if time delta is missing
        self.last_time = timestamp

        # 2. Initialize if first frame
        if self.x is None:
            self.x = x
            self.dx = 0
            return x

        # 3. Estimate velocity (dx)
        dx_te = (x - self.x) / self.te
        edx_alpha = self.smoothing_factor(self.te, self.d_cutoff)
        self.dx = self.exponential_smoothing(edx_alpha, dx_te, self.dx)

        # 4. Calculate cutoff frequency based on velocity
        # High velocity -> High cutoff -> Less lag
        # Low velocity  -> Low cutoff  -> High smoothing
        cutoff = self.min_cutoff + self.beta * abs(self.dx)
        
        # 5. Filter the signal
        self.alpha = self.smoothing_factor(self.te, cutoff)
        self.x = self.exponential_smoothing(self.alpha, x, self.x)
        return self.x