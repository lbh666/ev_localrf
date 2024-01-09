import json, os
import numpy as np
import torch
import matplotlib.pyplot as plt
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
    vertices[..., 1:] *= -1 # Axis flip
    wireframe = vertices[:, [0, 1, 2, 3, 0, 4, 1, 2, 4, 3]]
    return vertices, faces, wireframe
def merge_wireframes(wireframe):
    wireframe_merged = [[], [], []]
    for w in wireframe:
        wireframe_merged[0] += [float(n) for n in w[:, 0]]
        wireframe_merged[1] += [float(n) for n in w[:, 1]]
        wireframe_merged[2] += [float(n) for n in w[:, 2]]
    return wireframe_merged
def draw_poses(poses, colours):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    centered_poses = poses.clone()
    centered_poses[:, :, 3] -= torch.mean(centered_poses[:, :, 3], dim=0, keepdim=True)

    vertices, faces, wireframe = get_camera_mesh(
        centered_poses, 0.05
    )
    center = vertices[:, -1]
    ps = max(torch.max(center).item(), 0.1)
    ms = min(torch.min(center).item(), -0.1)
    ax.set_xlim3d(ms, ps)
    ax.set_ylim3d(ms, ps)
    ax.set_zlim3d(ms, ps)
    wireframe_merged = merge_wireframes(wireframe)
    for c in range(center.shape[0]):
        ax.plot(
            wireframe_merged[0][c * 10 : (c + 1) * 10],
            wireframe_merged[1][c * 10 : (c + 1) * 10],
            wireframe_merged[2][c * 10 : (c + 1) * 10],
            color=colours[c],
        )

    plt.tight_layout()
    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.show()
    plt.close(fig)
    return img # np.zeros([5, 5, 3], dtype=np.uint8)

path = r'transforms.json'
with open(path, 'r') as f:
    transforms = json.load(f)
poses = []
for idx, transform in enumerate(transforms['frames']):
    pose = np.array(transform["transform_matrix"], dtype=np.float32)
    poses.append(pose)

colors =  ["C1"] * len(poses)
poses = torch.from_numpy(np.array(poses))

print(poses.shape)
draw_poses(poses, colors)