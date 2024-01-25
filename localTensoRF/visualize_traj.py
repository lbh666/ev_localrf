import torch, open3d
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from utils.utils import mtx_to_sixD, sixD_to_mtx
import trimesh
import matplotlib

matplotlib.use('TkAgg')

# load the data first
x = torch.load(r'checkpoints.th', map_location='cuda')
num_imgs, num_rfs = x['state_dict']['blending_weights'].shape
poses = []
idx = 0
one_ = torch.tensor([[0, 0, 0, 1]]).cuda()
while f"r_c2w.{idx}" in x['state_dict'].keys():
    R = sixD_to_mtx(x['state_dict'][f"r_c2w.{idx}"][None])[0]
    T = x['state_dict'][f"t_c2w.{idx}"][..., None]
    tmp_ = torch.cat([R, T], dim = -1)
    poses.append(torch.cat([tmp_, one_], dim = 0))
    idx += 1
poses = torch.stack(poses, dim=0).cpu().numpy()
poses = poses.dot(np.array([[1,0,0,0],
                           [0,1,0,0],
                           [0,0,-1,0],
                           [0,0,0,1]]))
centers = []
for i in range(num_rfs):
    centers.append(-x['state_dict'][f'world2rf.{i}'])
centers = torch.stack(centers, dim=0).cpu().numpy()

WIDTH = 1280
HEIGHT = 720

# Step 1 - Get scene objects
meshFrame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

# Step 2 - Create visualizer object
vizualizer = open3d.visualization.Visualizer()
vizualizer.create_window()
vizualizer.create_window(width=WIDTH, height=HEIGHT)

# Step 3 - Add objects to visualizer
vizualizer.add_geometry(meshFrame)
# Step 4 - Get camera lines
for pose in poses:
    standardCameraParametersObj  = vizualizer.get_view_control().convert_to_pinhole_camera_parameters()
    cameraLines = open3d.geometry.LineSet.create_camera_visualization(intrinsic=standardCameraParametersObj.intrinsic, 
                                                                      extrinsic=np.linalg.inv(pose),
                                                                      scale=0.1)
    vizualizer.add_geometry(cameraLines)

for i, center in enumerate(centers):
    aabb = x['state_dict'][f'tensorfs.{i}.aabb'].cpu().numpy()
    min_, max_ = aabb[0] + centers[i], aabb[1] + centers[i]
    box = open3d.geometry.AxisAlignedBoundingBox(min_, max_)
    box.color = np.array([1,0,0])
    vizualizer.add_geometry(box)

point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(centers)
point_cloud.colors = o3d.utility.Vector3dVector(np.array([[0, 1, 0]]).repeat(centers.shape[0],0))
vizualizer.add_geometry(point_cloud)
vizualizer.get_render_option().point_size = 10
vizualizer.run()

