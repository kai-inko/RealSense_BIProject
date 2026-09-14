# stabilization.py
import numpy as np
from filters import OneEuroFilter
import config

class HandStabilizer:
    def __init__(self):
        # Create a 21x3 matrix of filters (21 joints, X/Y/Z axes)
        self.filters = [
            [
                OneEuroFilter(min_cutoff=config.FILTER_MIN_CUTOFF, beta=config.FILTER_BETA) 
                for _ in range(3)
            ] 
            for _ in range(21)
        ]
        self.last_pos = None

    def process(self, raw_points_np, timestamp):
        smoothed = []
        for i in range(21):
            rx, ry, rz = raw_points_np[i]
            
            # Safety: If depth drops to 0, hold the last known Z position
            if rz == 0 and self.last_pos is not None:
                rz = self.last_pos[i][2]

            # Filter X, Y, and Z independently
            sx = self.filters[i][0].filter(rx, timestamp)
            sy = self.filters[i][1].filter(ry, timestamp)
            sz = self.filters[i][2].filter(rz, timestamp)
            smoothed.append([sx, sy, sz])
        
        final_pos = np.array(smoothed)
        self.last_pos = final_pos
        return final_pos