# main.py
from ursina import *
import time
import config
from vision import VisionEngine
from stabilization import HandStabilizer
from visualizer import HandSurface

# 1. Initialize Ursina
app = Ursina(vsync=False)
window.title = "Anatomical Surface Reconstruction"
window.color = color.black
window.show_ursina_splash = False

# 2. Scene Setup
EditorCamera()
camera.position = (0, 0, -20)

vision = VisionEngine()
left_hand = HandSurface(color.azure)
right_hand = HandSurface(color.orange)
left_stab = HandStabilizer()
right_stab = HandStabilizer()

info_text = Text(text="Waiting for RealSense...", position=(-0.85, 0.45), scale=1)

def update():
    data = vision.data
    ts = data["timestamp"]
    
    detected_hands = {h["label"]: h for h in data["hands"]}
    active_count = 0
    
    # Update Right Hand
    if "Right" in detected_hands:
        pts = right_stab.process(detected_hands["Right"]["points"], ts)
        right_hand.enabled = True
        right_hand.update_surface(pts)
        active_count += 1
    else:
        right_hand.enabled = False

    # Update Left Hand
    if "Left" in detected_hands:
        pts = left_stab.process(detected_hands["Left"]["points"], ts)
        left_hand.enabled = True
        left_hand.update_surface(pts)
        active_count += 1
    else:
        left_hand.enabled = False

    info_text.text = f"Surface Tracking: {active_count} Hands | FPS: {int(1/time.dt)}"

def input(key):
    if key == 'q' or key == 'escape':
        vision.running = False
        application.quit()
    if key == 'w':
        # Toggle Wireframe
        left_hand.wireframe_overlay.visible = not left_hand.wireframe_overlay.visible
        right_hand.wireframe_overlay.visible = not right_hand.wireframe_overlay.visible

if __name__ == "__main__":
    app.run()