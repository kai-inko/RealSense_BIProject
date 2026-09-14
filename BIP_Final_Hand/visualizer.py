from ursina import *
from ursina.shaders import unlit_shader, lit_with_shadows_shader
import numpy as np
import config
from stabilizer import HandStabilizer
import mesh_factory  # Import the new module

class HandVisualizer:
    def __init__(self, side_name):
        self.parent = Entity(name=side_name)
        self.stabilizer = HandStabilizer()
        
        # 1. JOINTS
        self.joints = []
        for i in range(21):
            j = Entity(parent=self.parent, model='sphere', color=config.COLOR_JOINT, 
                       scale=0.2, shader=unlit_shader)
            self.joints.append(j)

        # 2. BONES
        self.bone_pairs = [
            (0,1),(1,2),(2,3),(3,4),       # Thumb
            (0,5),(5,6),(6,7),(7,8),       # Index
            (0,9),(9,10),(10,11),(11,12),  # Middle
            (0,13),(13,14),(14,15),(15,16),# Ring
            (0,17),(17,18),(18,19),(19,20) # Pinky
        ]
        self.bones = []
        for s, e in self.bone_pairs:
            b = Entity(parent=self.parent, model='cube', color=config.COLOR_BONE, 
                       origin_z=-0.5, scale=(1, 1, 1), shader=unlit_shader)
            self.bones.append({'entity': b, 's': s, 'e': e})

        # 3. MESH & WIREFRAME
        self.mesh_entity = Entity(parent=self.parent, model=Mesh(
            vertices=[], triangles=[], static=False, mode='triangle',
        ), color=config.COLOR_SKIN, double_sided=True, shader=lit_with_shadows_shader)

        if config.SHOW_WIREFRAME:
            self.wireframe_entity = Entity(parent=self.parent, model=Mesh(
                vertices=[], mode='line', static=False
            ), color=config.COLOR_WIREFRAME, unlit=True)

    def update(self, hand_data, timestamp):
        # --- 1. PREPARE DATA ---
        points_raw = self.stabilizer.process(hand_data["points"], timestamp)
        
        pts = [Vec3((p[0] - 0.5) * config.SCALE_X, 
                    (0.5 - p[1]) * config.SCALE_Y, 
                    p[2] * config.SCALE_Z) for p in points_raw]
            
        # Global Orientation
        palm_normal = mesh_factory.calculate_palm_normal(pts[0], pts[5], pts[17])

        # --- 2. UPDATE RIGID BODIES (Bones/Joints) ---
        palm_size = hand_data["palm_size"]
        bone_width = palm_size * 2.0
        
        for i, pt in enumerate(pts):
            self.joints[i].position = pt
            self.joints[i].enabled = True
            
        for b in self.bones:
            p_start = pts[b['s']]
            p_end = pts[b['e']]
            b['entity'].position = p_start
            b['entity'].look_at(p_end)
            b['entity'].scale = (bone_width, bone_width, distance(p_start, p_end))

        # --- 3. GENERATE PROCEDURAL MESH ---
        vertices = []
        triangles = []
        v_idx = 0
        
        # A. FINGERS
        finger_chains = [
            [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], 
            [13, 14, 15, 16], [17, 18, 19, 20]
        ]

        for chain in finger_chains:
            for i in range(len(chain) - 1):
                p_curr, p_next = pts[chain[i]], pts[chain[i+1]]
                axis = mesh_factory.get_axis_vector(p_curr, p_next)
                
                # Generate Rings
                r_curr = config.THICKNESS_MAP[chain[i]]
                r_next = config.THICKNESS_MAP[chain[i+1]]
                
                ring1 = mesh_factory.generate_ring(p_curr, axis, r_curr, palm_normal)
                ring2 = mesh_factory.generate_ring(p_next, axis, r_next, palm_normal)
                
                # Stitch
                vertices.extend(ring1 + ring2)
                triangles.extend(mesh_factory.stitch_rings(ring1, ring2, v_idx))
                v_idx += 16 # 8 segments * 2 rings

        # B. PALM
        p_wrist, p_mid = pts[0], pts[9]
        axis_wrist = mesh_factory.get_axis_vector(p_wrist, p_mid)
        ring_wrist = mesh_factory.generate_ring(p_wrist, axis_wrist, config.THICKNESS_MAP[0], palm_normal)
        
        bases = [1, 5, 9, 13, 17]
        for b_idx in bases:
            p_base = pts[b_idx]
            axis_base = mesh_factory.get_axis_vector(p_wrist, p_base)
            ring_base = mesh_factory.generate_ring(p_base, axis_base, config.THICKNESS_MAP[b_idx], palm_normal)
            
            vertices.extend(ring_wrist + ring_base)
            triangles.extend(mesh_factory.stitch_rings(ring_wrist, ring_base, v_idx))
            v_idx += 16

        # C. WEBBING
        for i in range(1, len(bases)):
            idx_a, idx_b = bases[i-1], bases[i]
            pa, pb, pw = pts[idx_a], pts[idx_b], pts[0]
            
            vertices.extend([pw, pa, pb])
            triangles.append((v_idx, v_idx+1, v_idx+2)) # Top
            triangles.append((v_idx+2, v_idx+1, v_idx)) # Bottom
            v_idx += 3

        # Apply to Ursina
        self.mesh_entity.model.vertices = vertices
        self.mesh_entity.model.triangles = triangles
        self.mesh_entity.model.generate()

        if config.SHOW_WIREFRAME:
            wf_verts = []
            for i in range(0, len(triangles), 2): 
                t = triangles[i]
                v1, v2, v3 = vertices[t[0]], vertices[t[1]], vertices[t[2]]
                wf_verts.extend([v1, v2, v2, v3, v3, v1])
            self.wireframe_entity.model.vertices = wf_verts
            self.wireframe_entity.model.generate()

        self.parent.enabled = True