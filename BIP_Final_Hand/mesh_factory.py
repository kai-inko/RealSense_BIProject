import numpy as np
import math
from ursina import Vec3

def calculate_palm_normal(p0, p5, p17):
    """Calculates the normal vector of the palm plane."""
    v1 = np.array([p5.x-p0.x, p5.y-p0.y, p5.z-p0.z])
    v2 = np.array([p17.x-p0.x, p17.y-p0.y, p17.z-p0.z])
    normal = np.cross(v1, v2)
    norm_len = np.linalg.norm(normal)
    return normal / norm_len if norm_len > 0 else np.array([0, 0, 1])

def get_axis_vector(p_curr, p_next):
    """Calculates normalized direction vector between two points."""
    axis = np.array([p_next.x-p_curr.x, p_next.y-p_curr.y, p_next.z-p_curr.z])
    norm_len = np.linalg.norm(axis)
    return axis / norm_len if norm_len > 0 else np.array([0, 1, 0])

def generate_ring(pt, axis, radius, palm_normal, segments=8):
    """Generates a list of vertices forming a ring around a point."""
    # 1. Create Basis Vectors
    t = axis
    if np.linalg.norm(t) == 0: t = np.array([0,1,0])
    
    # Binormal (Right)
    bn = np.cross(t, palm_normal) 
    if np.linalg.norm(bn) < 0.1: bn = np.cross(t, np.array([1,0,0]))
    bn_len = np.linalg.norm(bn)
    if bn_len > 0: bn /= bn_len
    
    # Normal (Up)
    n = np.cross(bn, t)
    n_len = np.linalg.norm(n)
    if n_len > 0: n /= n_len
    
    ring_verts = []
    for i in range(segments):
        theta = (i / segments) * 2 * math.pi
        # Offset vector
        off = (bn * math.cos(theta) + n * math.sin(theta)) * radius
        v = Vec3(pt.x + off[0], pt.y + off[1], pt.z + off[2])
        ring_verts.append(v)
    return ring_verts

def stitch_rings(ring_a, ring_b, start_idx, segments=8):
    """Returns a list of triangle tuples connecting two rings."""
    new_tris = []
    for i in range(segments):
        next_i = (i + 1) % segments
        
        # Indices in global vertex list
        a_curr = start_idx + i
        a_next = start_idx + next_i
        b_curr = start_idx + segments + i
        b_next = start_idx + segments + next_i
        
        # Tri 1
        new_tris.append((a_curr, b_curr, a_next))
        # Tri 2
        new_tris.append((a_next, b_curr, b_next))
        
    return new_tris