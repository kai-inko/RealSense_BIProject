from abc import ABC, abstractmethod

class CameraBase(ABC):
    @abstractmethod
    def start(self):
        """Start the camera stream."""
        pass

    @abstractmethod
    def get_frames(self):
        """
        Returns a tuple: (success, color_image, depth_image, intrinsics)
        - color_image: RGB array
        - depth_image: Depth array (or None for laptop)
        - intrinsics: Camera intrinsic parameters (for 3D mapping)
        """
        pass

    @abstractmethod
    def stop(self):
        """Release resources."""
        pass