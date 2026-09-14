import cv2
import threading
import queue
import time

import config
from cam_laptop import LaptopCamera
from cam_realsense import RealSenseCamera
from face_mesh import FaceMeshDetector
from reconstructor import SurfaceReconstructor
from visualizer import Face3DVisualizer

data_queue = queue.Queue(maxsize=1)
running = True

def mouse_callback(event, x, y, flags, param):
    global running
    if event == cv2.EVENT_LBUTTONDOWN:
        if (config.BUTTON_X <= x <= config.BUTTON_X + config.BUTTON_W and 
            config.BUTTON_Y <= y <= config.BUTTON_Y + config.BUTTON_H):
            running = False

def draw_ui(image):
    cv2.rectangle(image, 
                  (config.BUTTON_X, config.BUTTON_Y), 
                  (config.BUTTON_X + config.BUTTON_W, config.BUTTON_Y + config.BUTTON_H), 
                  config.BUTTON_COLOR, -1)
    
    cv2.putText(image, "EXIT", 
                (config.BUTTON_X + 25, config.BUTTON_Y + 28), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.BUTTON_TEXT_COLOR, 2)
    return image

def processing_thread(camera, detector):

    global running
    camera.start()
    
    while running:
        success, color_img, depth_img, intrinsics = camera.get_frames()
        if not success:
            continue

        landmarks = detector.process(color_img)
        mesh_data = None

        if len(landmarks) > 0:
            if config.USE_REALSENSE:
                points_3d, valid_idx = SurfaceReconstructor.generate_point_cloud(
                    landmarks, depth_img, intrinsics
                )
            else:
                points_3d, valid_idx = SurfaceReconstructor.generate_fake_cloud_for_laptop(
                    landmarks, color_img.shape
                )
            
            if points_3d is not None and len(points_3d) > 3:
                mesh_data = (points_3d, landmarks, valid_idx)

        if not data_queue.full():
            data_queue.put((color_img, mesh_data))
            
        time.sleep(0.01)

    camera.stop()

def main():
    global running

    if config.USE_REALSENSE:
        cam = RealSenseCamera()
        print("Initializing RealSense...")
    else:
        cam = LaptopCamera()
        print("Initializing Laptop...")

    detector = FaceMeshDetector()

    vis = Face3DVisualizer()

    cv2.namedWindow("2D Tracking")
    cv2.setMouseCallback("2D Tracking", mouse_callback)

    t = threading.Thread(target=processing_thread, args=(cam, detector))
    t.start()
    print("System Running")

    try:
        while running:
            if not data_queue.empty():
                color_img, mesh_data = data_queue.get()

                vis.update(mesh_data)

                color_img = draw_ui(color_img)
                cv2.imshow("2D Tracking", color_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
                break
                
    except KeyboardInterrupt:
        running = False
    except Exception as e:
        print(f"Error: {e}")
    finally:
        running = False
        t.join()
        vis.close()
        cv2.destroyAllWindows()
        print("Exit.")

if __name__ == "__main__":
    main()