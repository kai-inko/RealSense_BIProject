# config.py

# Camera Settings
SHOW_CAM_FEED = True       
CAM_WIDTH = 640 
CAM_HEIGHT = 480 
FPS = 60                   

# Physics & Depth Constants
MAX_DEPTH_METERS = 2.0     
DEPTH_SEARCH_RADIUS = 10   # Radius for spiral search to fill holes

# 1€ Filter Tuning (Jitter vs Lag balance)
FILTER_BETA = 10.0         # Higher = less lag, more jitter
FILTER_MIN_CUTOFF = 0.5    # Higher = more filtering at low speeds

# Visualization Scales (Ursina World Space)
SCALE_X = 30.0
SCALE_Y = 22.0
SCALE_Z = 25.0