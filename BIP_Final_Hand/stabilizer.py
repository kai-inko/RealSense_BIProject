import math
import numpy as np
import config

class OneEuroFilter:
    def __init__(self, min_cutoff=0.1, beta=2.0, d_cutoff=1.0):
        self.x = None; self.dx = 0
        self.min_cutoff = min_cutoff; self.beta = beta; self.d_cutoff = d_cutoff
        self.alpha = 0.0; self.last_time = 0

    def smoothing_factor(self, te, cutoff):
        r = 2 * math.pi * cutoff * te
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def filter(self, x, timestamp):
        if self.last_time != 0 and timestamp != self.last_time:
            self.te = timestamp - self.last_time
        else: self.te = 0.033 
        self.last_time = timestamp

        if self.x is None:
            self.x = x; self.dx = 0; return x

        dx_te = (x - self.x) / self.te
        edx_alpha = self.smoothing_factor(self.te, self.d_cutoff)
        self.dx = self.exponential_smoothing(edx_alpha, dx_te, self.dx)

        cutoff = self.min_cutoff + self.beta * abs(self.dx)
        self.alpha = self.smoothing_factor(self.te, cutoff)
        self.x = self.exponential_smoothing(self.alpha, x, self.x)
        return self.x

class HandStabilizer:
    def __init__(self):
        self.filters = [[OneEuroFilter(min_cutoff=config.FILTER_MIN_CUTOFF, beta=config.FILTER_BETA) for _ in range(3)] for _ in range(21)]
        self.last_pos = None

    def process(self, raw_points_np, timestamp):
        smoothed = []
        for i in range(21):
            rx, ry, rz = raw_points_np[i]
            if rz == 0 and self.last_pos is not None:
                rz = self.last_pos[i][2]
            sx = self.filters[i][0].filter(rx, timestamp)
            sy = self.filters[i][1].filter(ry, timestamp)
            sz = self.filters[i][2].filter(rz, timestamp)
            smoothed.append([sx, sy, sz])
        final_pos = np.array(smoothed)
        self.last_pos = final_pos
        return final_pos