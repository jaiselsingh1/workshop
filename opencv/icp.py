import numpy as np 
import matplotlib.pyplot as plt
import open3d as o3d 
import open2d as o2d
import copy

demo_pcds = o3d.data.DemoICPPointClouds()
source = o3d.io.read_point_cloud(demo_pcds.paths[0])
target = o3d.io.read_point_cloud(demo_pcds.paths[1])

o3d.visualization.draw_plotly([source], 
                              zoom = 0.455, 
                              front = [-0.4999, -0.1659, -0.8499],
                              lookat = [2.1813, 2.0619, 2.0999],
                              up = [0.1204, -0.9852, 0.1215])

o3d.visualization.draw_plotly([target],
                                  zoom=0.455,
                                  front=[-0.4999, -0.1659, -0.8499],
                                  lookat=[2.1813, 2.0619, 2.0999],
                                  up=[0.1204, -0.9852, 0.1215])

# visualise target point cloud and a source point cloud transformed with an rough initial alignment transformation
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.0706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)

    o3d.visualization.draw_plotly([source_temp, target_temp])

trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                         [-0.139, 0.967, -0.215, 0.7],
                         [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])

# draw_registration_result(source, target, transformation = trans_init)

threshold = 0.02 
print("initial aligment")
evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)
print(evaluation)

# registration through the use of ICP 
print("apply point to point")
reg_p2p = o3d.pipelines.registratation_icp(
    source, target, threshold, trans_init, 
    o3d.pipelines.registration.TransformationEstimationPointToPoint()
)

print(reg_p2p)
print(f"transformation is {reg_p2p.transformation}")
draw_registration_result(source, target, reg_p2p.transformation)