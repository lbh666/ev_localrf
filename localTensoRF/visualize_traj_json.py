import torch, open3d
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from utils.utils import sixD_to_mtx
import trimesh, json
import matplotlib

matplotlib.use('TkAgg')

# load the data first
path = r'transforms.json'
# path_rf = r'transforms_rf.json'
with open(path, 'r') as f:
    transforms = json.load(f)
poses = []
for idx, transform in enumerate(transforms['frames']):
    pose = np.array(transform["transform_matrix"], dtype=np.float32)
    poses.append(pose)


poses = np.array(poses)[:100]
poses = poses.dot(np.array([[1,0,0,0],
                           [0,1,0,0],
                           [0,0,-1,0],
                           [0,0,0,1]]))

WIDTH = 1280
HEIGHT = 720

# Step 1 - Get scene objects
meshFrame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

# Step 2 - Create visualizer object
vizualizer = open3d.visualization.Visualizer()
vizualizer.create_window()
vizualizer.create_window(width=WIDTH, height=HEIGHT)

# Step 3 - Add objects to visualizer
# vizualizer.add_geometry(meshFrame)
# Step 4 - Get camera lines
for pose in poses[:]:
    standardCameraParametersObj  = vizualizer.get_view_control().convert_to_pinhole_camera_parameters()
    cameraLines = open3d.geometry.LineSet.create_camera_visualization(intrinsic=standardCameraParametersObj.intrinsic, 
                                                                      extrinsic=np.linalg.inv(pose),
                                                                      scale=0.1)
    vizualizer.add_geometry(cameraLines)


point_cloud = o3d.geometry.PointCloud()
# point_cloud.points = o3d.utility.Vector3dVector(centers)
# point_cloud.colors = o3d.utility.Vector3dVector(np.array([[0, 1, 0]]).repeat(centers.shape[0],0))
vizualizer.add_geometry(point_cloud)
vizualizer.get_render_option().point_size = 10
vizualizer.run()

