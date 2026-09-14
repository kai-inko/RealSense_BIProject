import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
import math
import time
import threading
import config

class RealSenseCamera:
    def __init__(self):
        self.running = True
        self.data = {"hands": [], "timestamp": 0}
        self.processed_frame = None
        self.start_time = time.time()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def get_safe_depth(self, depth_image, x, y):
        h, w = depth_image.shape
        if x < 0 or x >= w or y < 0 or y >= h: return 0
        z = depth_image[y, x]
        if z > 0: return z
        for r in range(1, 10):
            y_min = max(0, y - r); y_max = min(h, y + r + 1)
            x_min = max(0, x - r); x_max = min(w, x + r + 1)
            valid = depth_image[y_min:y_max, x_min:x_max]
            valid = valid[valid > 0]
            if len(valid) > 0: return np.median(valid)
        return 0

    def run(self):
        pipeline = rs.pipeline()
        rs_config = rs.config()
        # High-Speed config if possible
        try:
            rs_config.enable_stream(rs.stream.depth, config.CAM_WIDTH, config.CAM_HEIGHT, rs.format.z16, config.FPS)
            rs_config.enable_stream(rs.stream.color, config.CAM_WIDTH, config.CAM_HEIGHT, rs.format.bgr8, config.FPS)
            profile = pipeline.start(rs_config)
        except:
            rs_config.enable_stream(rs.stream.depth, config.CAM_WIDTH, config.CAM_HEIGHT, rs.format.z16, 30)
            rs_config.enable_stream(rs.stream.color, config.CAM_WIDTH, config.CAM_HEIGHT, rs.format.bgr8, 30)
            profile = pipeline.start(rs_config)

        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        align = rs.align(rs.stream.color)
        spatial = rs.spatial_filter(); spatial.set_option(rs.option.holes_fill, 3)
        temporal = rs.temporal_filter()

        mp_hands = mp.solutions.hands
        with mp_hands.Hands(max_num_hands=2, model_complexity=0, min_detection_confidence=0.5) as hands:
            while self.running:
                frames = pipeline.wait_for_frames()
                aligned = align.process(frames)
                depth_frame = temporal.process(spatial.process(aligned.get_depth_frame()))
                color_frame = aligned.get_color_frame()
                
                if not depth_frame or not color_frame: continue

                depth_img = np.asanyarray(depth_frame.get_data())
                color_img = np.asanyarray(color_frame.get_data())
                rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
                rgb_flipped = cv2.flip(rgb, 1)
                
                results = hands.process(rgb_flipped)
                ts = time.time() - self.start_time
                new_data = {"hands": [], "timestamp": ts}
                
                if results.multi_hand_landmarks:
                    for i, lms in enumerate(results.multi_hand_landmarks):
                        label = results.multi_handedness[i].classification[0].label if i < len(results.multi_handedness) else "Unknown"
                        
                        raw_data = []
                        valid_z = []
                        for lm in lms.landmark:
                            true_x = 1.0 - lm.x
                            px = min(math.floor(true_x * config.CAM_WIDTH), config.CAM_WIDTH - 1)
                            py = min(math.floor(lm.y * config.CAM_HEIGHT), config.CAM_HEIGHT - 1)
                            z = self.get_safe_depth(depth_img, px, py) * depth_scale
                            if 0.1 < z < 2.5: valid_z.append(z)
                            raw_data.append([true_x, lm.y, z])

                        median_z = np.median(valid_z) if valid_z else 0
                        final_points = []
                        if median_z > 0.1:
                            for x, y, z in raw_data:
                                if z < 0.1 or abs(z - median_z) > 0.20: z = median_z
                                final_points.append([x, y, z])
                            
                            new_data["hands"].append({
                                "label": label,
                                "points": np.array(final_points, dtype=np.float32),
                                "palm_size": np.linalg.norm(np.array(final_points[0]) - np.array(final_points[9]))
                            })

                self.data = new_data
                self.processed_frame = cv2.resize(cv2.cvtColor(rgb_flipped, cv2.COLOR_RGB2BGR), (320, 240))
        pipeline.stop()