import pyrealsense2 as rs
import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay

class SurfaceReconstructor:
    _cached_triangles = None
    _cached_count = 0
    _depth_memory = {} 

    @staticmethod
    def generate_point_cloud(landmarks, depth_map, intrinsics):
        """ 
        RealSense Logic with 'Frame Integrity Check'.
        If the face is calculated to be > 25cm deep (physically impossible),
        we discard the frame to prevent explosions.
        """
        if depth_map is None or intrinsics is None:
            return None, None

        h_map, w_map = depth_map.shape
        points_3d = []
        valid_indices = []
        
        # Parameters
        spike_threshold = 0.08  # 8cm limit for inter-frame jumps
        smooth_factor = 0.3     # 30% New Data (Slightly faster tracking)

        temp_z_values = []

        # 1. Calculate Depths & Temporal Smoothing
        for i, pt in enumerate(landmarks):
            u, v = int(pt[0]), int(pt[1])
            
            raw_z = 0.0
            if 2 <= v < h_map - 2 and 2 <= u < w_map - 2:
                window = depth_map[v-1:v+2, u-1:u+2]
                valid_pixels = window[window > 0]
                if len(valid_pixels) > 0:
                    raw_z = np.median(valid_pixels) * 0.001 

            # Temporal Logic
            final_z = raw_z
            if i in SurfaceReconstructor._depth_memory:
                prev_z = SurfaceReconstructor._depth_memory[i]
                
                # If sensor fails (0) or spikes huge (>8cm), hold position
                if raw_z < 0.1 or abs(raw_z - prev_z) > spike_threshold:
                    final_z = prev_z
                else:
                    # Smooth valid movements
                    final_z = (raw_z * smooth_factor) + (prev_z * (1 - smooth_factor))
            else:
                if raw_z < 0.1: continue
                final_z = raw_z

            SurfaceReconstructor._depth_memory[i] = final_z
            
            # Store valid points temporarily
            if 0.1 < final_z < 1.5:
                temp_z_values.append(final_z)
                points_3d.append([u, v, final_z]) # Store u,v,z for now
                valid_indices.append(i)

        # 2. FRAME INTEGRITY CHECK (The Anti-Explosion Guard)
        if not temp_z_values:
            return None, None
            
        z_array = np.array(temp_z_values)
        min_z = np.min(z_array)
        max_z = np.max(z_array)
        
        # A human face is rarely deeper than 20cm (0.20m).
        # If the spread is > 25cm, it means a point has shot off to the background.
        if (max_z - min_z) > 0.25:
            # print("Frame Rejected: Explosion Detected")
            return None, None

        # 3. Final Deprojection (Only if frame is safe)
        final_points = []
        for (u, v, z) in points_3d:
            point = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], z)
            point[0] = -point[0] # Mirror
            point[1] = -point[1] # Flip
            final_points.append(point)
            
        return np.array(final_points), valid_indices

    @staticmethod
    def generate_fake_cloud_for_laptop(landmarks, image_shape):
        """ Laptop Logic """
        h, w = image_shape[:2]
        points = []
        valid_indices = []
        cx, cy = w / 2, h / 2
        for i, (x_px, y_px, z_val) in enumerate(landmarks):
            px = -(x_px - cx) / w  
            py = -(y_px - cy) / w 
            pz = z_val 
            points.append([px, py, pz])
            valid_indices.append(i)
        return np.array(points), valid_indices

    @staticmethod
    def build_mesh(points_3d, landmarks_2d, valid_indices):
        """
        Builds mesh with standard Edge Pruning.
        """
        if points_3d is None or len(points_3d) < 10:
            return None

        # Always Recalculate Topology
        filtered_2d = np.array(landmarks_2d)[valid_indices][:, :2]
        try:
            tri = Delaunay(filtered_2d)
            triangles = tri.simplices.astype(np.int32)
        except:
            return None
        
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(points_3d)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        
        # Edge Pruning (The Web Cutter)
        verts = np.asarray(mesh.vertices)
        tris = np.asarray(mesh.triangles)
        
        if len(tris) > 0:
            v0 = verts[tris[:, 0]]
            v1 = verts[tris[:, 1]]
            v2 = verts[tris[:, 2]]
            
            d01 = np.linalg.norm(v0 - v1, axis=1)
            d12 = np.linalg.norm(v1 - v2, axis=1)
            d20 = np.linalg.norm(v2 - v0, axis=1)
            
            max_len = 0.05 
            valid_mask = (d01 < max_len) & (d12 < max_len) & (d20 < max_len)
            mesh.remove_triangles_by_mask(~valid_mask)
        
        mesh.remove_unreferenced_vertices()
        
        # Visual Polish
        mesh = mesh.subdivide_midpoint(number_of_iterations=1)
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=3, lambda_filter=0.5)
        mesh.compute_vertex_normals()
        
        return mesh

    @staticmethod
    def create_face_outlines(points_3d, valid_indices):
        return None