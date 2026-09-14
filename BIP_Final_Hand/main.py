from ursina import *
import config
import cv2
import sys

from camera_realsense import RealSenseCamera as CameraEngine 

from visualizer import HandVisualizer

app = Ursina(vsync=False)
window.title = "Hand Structure System (Laptop Mode)"
window.color = config.BACKGROUND_COLOR
window.show_ursina_splash = False

EditorCamera()
camera.position = (0, 0, -20)
pivot = Entity()
DirectionalLight(parent=pivot, y=2, z=3, shadows=True, rotation=(45, -45, 0))
AmbientLight(color=color.rgb(60, 60, 60))

vision = CameraEngine()

left_hand = HandVisualizer("Left")
right_hand = HandVisualizer("Right")

status_text = Text(text="System Initialized", position=(-0.85, 0.45), scale=1.2)

def update():
    data = vision.data
    ts = data.get("timestamp", 0)
    vision_hands = {h["label"]: h for h in data["hands"]}
    
    active = 0
    
    if "Left" in vision_hands: 
        left_hand.update(vision_hands["Left"], ts)
        active += 1
    else: 
        left_hand.parent.enabled = False
    
    if "Right" in vision_hands: 
        right_hand.update(vision_hands["Right"], ts)
        active += 1
    else: 
        right_hand.parent.enabled = False
    
    status_text.text = f"Mode: Laptop Camera | Hands: {active}"

    if vision.processed_frame is not None:
        cv2.imshow("Webcam Feed", vision.processed_frame)
        cv2.waitKey(1)

def input(key):
    if key == 'escape':
        vision.running = False
        cv2.destroyAllWindows()
        application.quit()

if __name__ == "__main__":
    app.run()