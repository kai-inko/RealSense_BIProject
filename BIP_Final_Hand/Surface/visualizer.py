# visualizer.py
from ursina import *
import config

# Hand Topology (Indices of vertices to form triangles)
HAND_TRIANGLES = [
    (0, 1, 5), (0, 5, 9), (0, 9, 13), (0, 13, 17), # Palm
    (1, 2, 5), (2, 3, 5), (3, 4, 5),               # Thumb
    (5, 6, 9), (6, 7, 9), (7, 8, 9),               # Index
    (9, 10, 13), (10, 11, 13), (11, 12, 13),       # Middle
    (13, 14, 17), (14, 15, 17), (15, 16, 17),      # Ring
    (17, 18, 19), (17, 19, 20)                     # Pinky
]

class HandSurface(Entity):
    def __init__(self, color_surf, **kwargs):
        super().__init__(**kwargs)
        self.model = Mesh(
            vertices=[Vec3(0,0,0) for _ in range(21)],
            triangles=HAND_TRIANGLES,
            static=False, 
            mode='triangle'
        )
        self.color = color_surf
        self.double_sided = True
        self.alpha = 0.8
        
        # Wireframe overlay
        self.wireframe_overlay = Entity(
            parent=self, 
            model=Mesh(vertices=[], mode='line'), 
            color=color.white, 
            visible=True
        )

    def update_surface(self, points):
        # A. Convert to Ursina World Space
        ursina_verts = []
        for p in points:
            # Scale coordinates for visualization
            x = (p[0] - 0.5) * config.SCALE_X
            y = (0.5 - p[1]) * config.SCALE_Y
            z = p[2] * config.SCALE_Z
            ursina_verts.append(Vec3(x, y, z))

        # B. Update Main Mesh
        self.model.vertices = ursina_verts
        self.model.generate() 

        # C. Update Wireframe
        wf_verts = []
        for t in HAND_TRIANGLES:
            wf_verts.append(ursina_verts[t[0]])
            wf_verts.append(ursina_verts[t[1]])
            wf_verts.append(ursina_verts[t[2]])
            wf_verts.append(ursina_verts[t[0]]) # Close loop
        
        self.wireframe_overlay.model.vertices = wf_verts
        self.wireframe_overlay.model.generate()