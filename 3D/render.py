from pytorch3d.utils import ico_sphere
import torch

import os
import time
import numpy as np
import torch

from pytorch3d.io import IO

from pytorch3d.transforms import axis_angle_to_matrix, euler_angles_to_matrix
from pytorch3d.structures import Pointclouds, join_meshes_as_batch
from tqdm.notebook import trange, tqdm
import matplotlib.pyplot as plt
import warnings
from torch import nn
import time

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, save_obj, IO

# Data structures and functions for rendering
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    AlphaCompositor,
)

from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.vis.plotly_vis import plot_scene


class PointcloudRender(nn.Module):
    def __init__(self,
                 img_size=128,
                 radius=0.0045,
                 points_per_pixel=40,
                 num_points = 4096,
                 ):
        super(PointcloudRender, self).__init__()
        self.raster_settings_silhouette_pointcloud = PointsRasterizationSettings(
            image_size=img_size,
            radius=radius,
            points_per_pixel=points_per_pixel,
        )
        self.num_points = num_points

    def render_silhouette_old(self, meshes, device,
                          dist=2.4,
                          elev=torch.tensor(0),
                          azim=torch.tensor(0)):
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        # Silhouette renderer
        rasterizer = PointsRasterizer(cameras=cameras,
                                      raster_settings=self.raster_settings_silhouette_pointcloud)
        renderer_silhouette_pointcloud = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor(background_color=(1, 1, 1)),
            # compositor=NormWeightedCompositor(background_color=(1, 1, 1)),
        )
        # Sample pointclouds
        points = sample_points_from_meshes(meshes, self.num_points).to(device)
        rgb = torch.zeros(points.size(0), points.size(1), 4).to(device)
        pointclouds = Pointclouds(points=points, features=rgb)
        images = renderer_silhouette_pointcloud(pointclouds)
        return images

    def render_silhouette(self, meshes, device,
                            angles:torch.tensor,
                            dist=2.4,
                            ):
        num_meshes = len(meshes)


        angles = angles.repeat((num_meshes,1))

        R = euler_angles_to_matrix(
            euler_angles=angles,
            convention="XYZ"
        )

        T = torch.zeros([num_meshes, 3])+torch.tensor([0,0,dist])
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        # Silhouette renderer
        rasterizer = PointsRasterizer(cameras=cameras,
                                        raster_settings=self.raster_settings_silhouette_pointcloud)
        renderer_silhouette_pointcloud = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor(background_color=(1, 1, 1)),
            # compositor=NormWeightedCompositor(background_color=(1, 1, 1)),
        )
        # Sample pointclouds
        points = sample_points_from_meshes(meshes, self.num_points).to(device)
        rgb = torch.zeros(points.size(0), points.size(1), 4).to(device)
        pointclouds = Pointclouds(points=points, features=rgb)
        images = renderer_silhouette_pointcloud(pointclouds)
        return images
    


def test_render_old(device):
    mesh = IO().load_mesh(os.path.join(f'data/ModelNet10', f'chair/train/chair_0001.off'), device=device)
    verts = mesh.verts_packed()
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    trg_mesh = mesh.offset_verts_(-center)
    trg_mesh = trg_mesh.scale_verts_((1.0 / float(scale)))

    pr = PointcloudRender()

    num = 10
    start = time.time()
    my_images = pr.render_silhouette_old(trg_mesh.extend(num),
                                     device,
                                     dist=3,
                                     num_points=4500,
                                     elev=torch.linspace(0, 90, num),
                                     azim=torch.linspace(-180, 180, num)
                                        )

    end = time.time()
    print(f'Test rendering {num} pointcloud images runtime {(end - start):>0.4f}s')

def test_render(device):
    mesh = IO().load_mesh(os.path.join(f'data/ModelNet10', f'chair/train/chair_0001.off'), device=device)
    verts = mesh.verts_packed()
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    trg_mesh = mesh.offset_verts_(-center)
    trg_mesh = trg_mesh.scale_verts_((1.0 / float(scale)))

    pr = PointcloudRender()

    num = 10
    start = time.time()

    # euler_angles = torch.
    bias = torch.tensor([np.pi, 0, np.pi]) # only vaild for X-Y-Z order
    R = euler_angles_to_matrix(
        euler_angles=torch.tensor(
            [
            [0, 0, 0],
            [np.pi/2, 0, 0],
            [0, np.pi/2, 0],
            [0, 0, np.pi/2],
            [-np.pi/2, 0, 0],
            [0, -np.pi/2, 0],
            [0, 0, -np.pi/2],
            [np.pi/2, np.pi/2, 0],
            [0, np.pi/2, np.pi/2],
            [np.pi/2, np.pi/2, np.pi/2],]
        ) + bias,
        convention="XYZ"
    )

    my_images = pr.render_silhouette(trg_mesh.extend(num),
                                     device,
                                     R=R,
                                     dist=3,
                                     num_points=4096)

    end = time.time()
    print(f'Test rendering {num} pointcloud images runtime {(end - start):>0.4f}s')

if __name__ == '__main__':
    test_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    test_render_old(test_device)
    test_render(test_device)

