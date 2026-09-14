import pyrealsense2 as rs
import numpy as np
from camera_base import CameraBase

class RealSenseCamera(CameraBase):
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # 1. Enable High Density / High Accuracy
        # (This drastically improves close-range face scanning)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # 2. Define Filters
        self.decimation = rs.decimation_filter()  # Reduces complexity
        self.spatial = rs.spatial_filter()        # Smooths surface, preserves edges
        self.temporal = rs.temporal_filter()      # Smooths over time (reduces jitter)
        self.hole_filling = rs.hole_filling_filter() # Fills black spots

        # Filter Options (Tuned for Face Scanning)
        self.spatial.set_option(rs.option.filter_magnitude, 5)
        self.spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        self.spatial.set_option(rs.option.filter_smooth_delta, 20)
        self.temporal.set_option(rs.option.filter_smooth_alpha, 0.4) # Higher = more stable, more lag
        self.temporal.set_option(rs.option.filter_smooth_delta, 20)

        # Alignment
        self.align = rs.align(rs.stream.color)
        self.intrinsics = None

    def start(self):
        profile = self.pipeline.start(self.config)
        
        # Set "High Density" Preset (4) for better close-range fill
        depth_sensor = profile.get_device().first_depth_sensor()
        if depth_sensor.supports(rs.option.visual_preset):
            depth_sensor.set_option(rs.option.visual_preset, 4) # 4 = High Density

        # Get intrinsics
        color_stream = profile.get_stream(rs.stream.color)
        self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        
        # 3. Apply Filters BEFORE Alignment
        # (Applying filters on raw depth is more accurate)
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            return False, None, None, None

        # Apply chain of filters
        # Note: Decimation lowers resolution. If you need 640x480, skip decimation.
        # depth_frame = self.decimation.process(depth_frame) 
        depth_frame = self.spatial.process(depth_frame)
        depth_frame = self.temporal.process(depth_frame)
        depth_frame = self.hole_filling.process(depth_frame)

        # 4. Align Filtered Depth to Color
        aligned_frames = self.align.process(frames.as_frameset())
        
        aligned_depth = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth or not color_frame:
            return False, None, None, None

        depth_image = np.asanyarray(aligned_depth.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return True, color_image, depth_image, self.intrinsics

    def stop(self):
        self.pipeline.stop()