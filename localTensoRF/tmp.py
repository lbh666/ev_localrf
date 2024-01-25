import torch, open3d
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d


def get_camera_mesh(pose, depth=1):
    vertices = (
        torch.tensor(
            [[-0.5, -0.5, -1], [0.5, -0.5, -1], [0.5, 0.5, -1], [-0.5, 0.5, -1], [0, 0, 0]]
        )
        * depth
    )
    faces = torch.tensor(
        [[0, 1, 2], [0, 2, 3], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]
    )
    vertices = vertices @ pose[:, :3, :3].transpose(-1, -2)
    vertices += pose[:, None, :3, 3]
    # vertices[..., 1:] *= -1 # Axis flip
    wireframe = vertices[:, [0, 1, 2, 3, 0, 4, 1, 2, 4, 3]]
    return vertices, faces, wireframe

def get_bbox(center, aabb):
    '''
    center: (N, 3)
    aabb: (N, 2, 3)
    '''
    N, _ = center.shape
    x_, y_, z_ = aabb[0,:,0], aabb[0,:,1], aabb[0,:,2]
    vertices = torch.stack(torch.meshgrid(x_, y_, z_), dim=-1).reshape(-1,3)[None] # (1, 8, 3)
    vertices += center.reshape(N, 1, 3) # (N, 8, 3)
    wireframe = vertices[:, [0, 1, 3, 2, 0, 4, 5, 1, 3, 7, 6, 4, 5, 7, 6, 2]]
    return vertices, wireframe

def merge_wireframes(wireframe):
    wireframe_merged = [[], [], []]
    for w in wireframe:
        wireframe_merged[0] += [float(n) for n in w[:, 0]]
        wireframe_merged[1] += [float(n) for n in w[:, 1]]
        wireframe_merged[2] += [float(n) for n in w[:, 2]]
    return wireframe_merged


def draw_poses_and_box(poses, colours, centers, aabb):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    centered_poses = poses.clone()
    centered_poses[:, :, 3] -= torch.mean(centered_poses[:, :, 3], dim=0, keepdim=True)

    vertices, faces, wireframe = get_camera_mesh(
        centered_poses, 0.05
    )
    vertices_box, wireframe_box = get_bbox(
        centers, aabb
    )
    center = vertices[:, -1]
    ps = max(torch.max(center).item(), 0.1)
    ms = min(torch.min(center).item(), -0.1)
    ax.set_xlim3d(ms, ps)
    ax.set_ylim3d(ms, ps)
    ax.set_zlim3d(ms, ps)
    wireframe_merged = merge_wireframes(wireframe)
    wireframe_merged_box = merge_wireframes(wireframe_box)
    for c in range(center.shape[0]):
        ax.plot(
            wireframe_merged[0][c * 10 : (c + 1) * 10],
            wireframe_merged[1][c * 10 : (c + 1) * 10],
            wireframe_merged[2][c * 10 : (c + 1) * 10],
            color=colours[c],
        )
    for c in range(centers.shape[0]):
        ax.plot(
            wireframe_merged_box[0][c * 16 : (c + 1) * 16],
            wireframe_merged_box[1][c * 16 : (c + 1) * 16],
            wireframe_merged_box[2][c * 16 : (c + 1) * 16],
            color="C2",
        )
    plt.tight_layout()
    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.show()
    plt.close(fig)
    return img

import json, os
import cv2 as cv

with open(r"F:\docker_shared\lego\raw\transforms.json", 'r') as fp:
    meta = json.load(fp)
poses = []
for frame in meta['frames']:
    poses.append(np.array(frame['transform_matrix']))
poses = np.array(poses).astype(np.float32)
aabb = np.array([[[-1,-1,-1],
                [1,1,1]]])
centers = np.array([[0,0,0]])
colours = ["C1"] * poses.shape[0]
img = draw_poses_and_box(torch.from_numpy(poses).float(), colours, torch.from_numpy(centers), torch.from_numpy(aabb))
cv.imwrite('1.png', img)
# get_bbox(torch.from_numpy(centers), torch.from_numpy(aabb))
