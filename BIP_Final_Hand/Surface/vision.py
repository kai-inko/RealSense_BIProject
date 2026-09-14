# vision.py
import time
import math
import threading
import cv2
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs 
import config

class VisionEngine:
    def __init__(self):
        self.running = True
        self.data = {"hands": [], "timestamp": 0}
        self.start_time = time.time()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def get_safe_depth(self, depth_image, x, y):
        h, w = depth_image.shape
        if x < 0 or x >= w or y < 0 or y >= h: return 0
        z = depth_image[y, x]
        if z > 0: return z
        
        # Spiral Neighbor Search to fill holes
        for r in range(1, config.DEPTH_SEARCH_RADIUS + 1):
            y_min, y_max = max(0, y - r), min(h, y + r + 1)
            x_min, x_max = max(0, x - r), min(w, x + r + 1)
            neighborhood = depth_image[y_min:y_max, x_min:x_max]
            valid_pixels = neighborhood[neighborhood > 0]
            if len(valid_pixels) > 0:
                return np.median(valid_pixels)
        return 0

    def run(self):
        pipeline = rs.pipeline()
        rs_config = rs.config()
        
        # Initialize Camera
        try:
            rs_config.enable_stream(rs.stream.depth, config.CAM_WIDTH, config.CAM_HEIGHT, rs.format.z16, config.FPS)
            rs_config.enable_stream(rs.stream.color, config.CAM_WIDTH, config.CAM_HEIGHT, rs.format.bgr8, config.FPS)
            profile = pipeline.start(rs_config)
        except RuntimeError:
            print("(!) Fallback to 30 FPS")
            rs_config.enable_stream(rs.stream.depth, config.CAM_WIDTH, config.CAM_HEIGHT, rs.format.z16, 30)
            rs_config.enable_stream(rs.stream.color, config.CAM_WIDTH, config.CAM_HEIGHT, rs.format.bgr8, 30)
            profile = pipeline.start(rs_config)

        align = rs.align(rs.stream.color)
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        
        # Hardware Filters (On-Camera)
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.holes_fill, 3)
        temporal = rs.temporal_filter()

        mp_hands = mp.solutions.hands
        with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
            while self.running:
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not depth_frame or not color_frame: continue

                # Apply RealSense Filters
                depth_frame = spatial.process(depth_frame)
                depth_frame = temporal.process(depth_frame)
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                # MediaPipe Processing
                rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                rgb_flipped = cv2.flip(rgb, 1)
                results = hands.process(rgb_flipped)
                
                ts = time.time() - self.start_time
                new_data = {"hands": [], "timestamp": ts}
                
                if results.multi_hand_landmarks:
                    for i, lms in enumerate(results.multi_hand_landmarks):
                        # Determine Handedness (Left/Right)
                        label = "Unknown"
                        if i < len(results.multi_handedness):
                            label = results.multi_handedness[i].classification[0].label
                        
                        joint_data = []
                        valid_z = []

                        # Extract 3D Coordinates
                        for lm in lms.landmark:
                            true_x = 1.0 - lm.x
                            px = min(math.floor(true_x * config.CAM_WIDTH), config.CAM_WIDTH - 1)
                            py = min(math.floor(lm.y * config.CAM_HEIGHT), config.CAM_HEIGHT - 1)
                            
                            raw_z = self.get_safe_depth(depth_image, px, py)
                            z_meters = raw_z * depth_scale
                            
                            if 0.1 < z_meters < config.MAX_DEPTH_METERS:
                                valid_z.append(z_meters)
                            
                            joint_data.append([lm.x, lm.y, z_meters])

                        # Outlier Rejection
                        hand_median_depth = np.median(valid_z) if valid_z else 0
                        final_points = []
                        
                        if hand_median_depth > 0:
                            for j in joint_data:
                                x, y, z = j
                                # If a joint is wildly far from the hand center, snap it back
                                if z < 0.1 or abs(z - hand_median_depth) > 0.25:
                                    z = hand_median_depth
                                final_points.append([x, y, z])

                            new_data["hands"].append({
                                "label": label,
                                "points": np.array(final_points, dtype=np.float32)
                            })

                self.data = new_data
                if config.SHOW_CAM_FEED:
                    cv2.imshow("RealSense Feed", cv2.cvtColor(rgb_flipped, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) & 0xFF == 27:
                        self.running = False

        pipeline.stop()