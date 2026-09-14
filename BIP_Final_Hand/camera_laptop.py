import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import config

class LaptopCamera:
    def __init__(self):
        self.running = True
        self.data = {"hands": [], "timestamp": 0}
        self.processed_frame = None
        self.start_time = time.time()
        
        # Threading for performance
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        # Open Standard Webcam (Index 0)
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.FPS)

        mp_hands = mp.solutions.hands
        with mp_hands.Hands(max_num_hands=2, model_complexity=0, min_detection_confidence=0.5) as hands:
            while self.running:
                ret, frame = cap.read()
                if not ret: continue
                
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_flipped = cv2.flip(rgb, 1)
                
                results = hands.process(rgb_flipped)
                ts = time.time() - self.start_time
                new_data = {"hands": [], "timestamp": ts}
                
                if results.multi_hand_landmarks:
                    for i, lms in enumerate(results.multi_hand_landmarks):
                        label = results.multi_handedness[i].classification[0].label if i < len(results.multi_handedness) else "Unknown"
                        
                        final_points = []
                        # Simulate Depth using MP Z coordinate
                        # MP Z is relative to wrist. We offset it to make it look like it's 0.5m away.
                        for lm in lms.landmark:
                            true_x = 1.0 - lm.x
                            # Simulated Scale: Scale MP relative Z to meters approx
                            z_sim = 0.5 + (lm.z * 0.5) 
                            final_points.append([true_x, lm.y, z_sim])
                            
                        # Calculate rough palm size
                        p0 = np.array(final_points[0])
                        p9 = np.array(final_points[9])
                        
                        new_data["hands"].append({
                            "label": label,
                            "points": np.array(final_points, dtype=np.float32),
                            "palm_size": np.linalg.norm(p0 - p9)
                        })

                self.data = new_data
                self.processed_frame = cv2.resize(cv2.cvtColor(rgb_flipped, cv2.COLOR_RGB2BGR), (320, 240))

        cap.release()