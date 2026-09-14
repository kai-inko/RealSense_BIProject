import open3d as o3d
import numpy as np
import config
from reconstructor import SurfaceReconstructor

class Face3DVisualizer:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="3D Structure Scan", 
                               width=config.WINDOW_WIDTH, 
                               height=config.WINDOW_HEIGHT)
        
        self.mesh = o3d.geometry.TriangleMesh()
        self.vis.add_geometry(self.mesh)
        
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0.1, 0.1, 0.1])
        opt.mesh_show_back_face = True
        
        self.first_frame = True
        self.prev_vertices = None
        # Alpha 0.6 = Responsive but still smooths out vibration
        self.alpha = 0.6 

    def update(self, mesh_data):
        if mesh_data is None:
            return

        points_3d, landmarks_2d, valid_idx = mesh_data
        temp_mesh = SurfaceReconstructor.build_mesh(points_3d, landmarks_2d, valid_idx)
        
        if temp_mesh:
            current_verts = np.asarray(temp_mesh.vertices)
            
            # Temporal Smoothing (Linear Interpolation)
            if (self.prev_vertices is not None and 
                self.prev_vertices.shape == current_verts.shape):
                smoothed_verts = (self.alpha * current_verts) + ((1 - self.alpha) * self.prev_vertices)
                self.prev_vertices = smoothed_verts
            else:
                smoothed_verts = current_verts
                self.prev_vertices = current_verts
            
            # Update Geometry
            self.mesh.vertices = o3d.utility.Vector3dVector(smoothed_verts)
            self.mesh.triangles = temp_mesh.triangles
            self.mesh.compute_vertex_normals()
            
            # --- STRUCTURE VISUALIZATION ---
            # Map Normals to Color to see surface quality
            normals = np.asarray(self.mesh.vertex_normals)
            colors = (normals + 1) / 2
            
            # Blend with a base "Skin" color to make it look solid
            # (Mix 50% Normal Map, 50% Orange)
            skin_tone = np.array([0.8, 0.5, 0.3])
            final_colors = (colors * 0.5) + (skin_tone * 0.5)
            
            self.mesh.vertex_colors = o3d.utility.Vector3dVector(final_colors)

            self.vis.update_geometry(self.mesh)

        self.vis.poll_events()
        self.vis.update_renderer()
        
        if self.first_frame:
            self.vis.reset_view_point(True)
            self.first_frame = False

    def close(self):
        self.vis.destroy_window()