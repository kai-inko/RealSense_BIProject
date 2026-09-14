import mediapipe as mp
import cv2

class FaceMeshDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process(self, image):
        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        landmarks_3d = []
        if results.multi_face_landmarks:
            h, w, _ = image.shape
            # Extract landmarks for the first face
            for lm in results.multi_face_landmarks[0].landmark:
                # cx, cy are pixels for sampling
                cx, cy = int(lm.x * w), int(lm.y * h)
                # z is the relative depth (scaled roughly same as x)
                landmarks_3d.append((cx, cy, lm.z))
                
        return landmarks_3d