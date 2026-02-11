import rerun as rr 
from math import tau 
# tau is the relationship between the circle's circumference to the radius
import numpy as np

from rerun.utilities import build_color_spiral 
from rerun.utilities import bounce_lerp


rr.init("rerun example dna abacus")
rr.spawn()

num_points = 100 
points1, colors1 = build_color_spiral(num_points)
points2, colors2 = build_color_spiral(num_points, angular_offset = tau * 0.5)

rr.log("dna/structure/left", rr.Points3D(points1, colors = colors1, radii = 0.08))
rr.log("dna/structure/left", rr.Points3D(points2, colors = colors2, radii = 0.08))

rr.log(
    "dna/structure/scaffolding",
    rr.LineStrips3D(np.stack((points1, points2), axis=1), colors=[128, 128, 128])
)

offsets = np.random.rand(num_points)
beads = [bounce_lerp(points1[n], points2[n], offsets[n]) for n in range(num_points)]
colors = [[int(bounce_lerp(80, 230, offsets[n] * 2))] for n in range(num_points)]
rr.log(
    "dna/structure/scaffolding/beads",
    rr.Points3D(beads, radii=0.06, colors=np.repeat(colors, 3, axis=-1)),
)


# def build_color_spiral(num_points: int = 100, radius: float = 2.0, angular_velocity: float = 5.0, height: float = 2.0, z_offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
#     t = np.linspace(0, 10, num_points)
    
#     # Calculate positions
#     x = radius * np.cos(angular_velocity * t)
#     y = radius * np.sin(angular_velocity * t)
#     z = z_offset + height * t / 10.0
#     positions = np.stack((x, y, z), axis=-1)
    
#     # Calculate colors (RGB)
#     colors = np.stack((
#         np.sin(t), 
#         np.sin(t + 2), 
#         np.sin(t + 4)
#     ), axis=-1)
#     # Normalize colors to 0-255 uint8
#     colors = ((colors + 1.0) / 2.0 * 255).astype(np.uint8)
    
#     return positions, colors


# def bounce_lerp(v0: float, v1: float, t: float) -> float:
#     """
#     Linearly interpolates between v0 and v1 based on t.
#     The interpolation parameter bounces back and forth between 0 and 1.
#     """
#     # Map t to a 0..1..0 sequence based on integer steps
#     t_norm = t % 2.0
#     if t_norm > 1.0:
#         t_norm = 2.0 - t_norm
    
#     return v0 + (v1 - v0) * t_norm