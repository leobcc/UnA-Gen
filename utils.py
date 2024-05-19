import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import gaussian_filter
import trimesh
import torch.nn.functional as F

# Mesh handling functions

def export_mesh(vertices):
    '''Export mesh from vertices to .obj file.'''
    mesh = trimesh.Trimesh(vertices)
    mesh.export('mesh.obj')

# Cooridnates transformations 

def quat_to_rot(q):
    batch_size, _ = q.shape
    q = F.normalize(q, dim=1)
    R = torch.ones((batch_size, 3,3)).cuda()
    qr=q[:,0]
    qi = q[:, 1]
    qj = q[:, 2]
    qk = q[:, 3]
    R[:, 0, 0]=1-2 * (qj**2 + qk**2)
    R[:, 0, 1] = 2 * (qj *qi -qk*qr)
    R[:, 0, 2] = 2 * (qi * qk + qr * qj)
    R[:, 1, 0] = 2 * (qj * qi + qk * qr)
    R[:, 1, 1] = 1-2 * (qi**2 + qk**2)
    R[:, 1, 2] = 2*(qj*qk - qi*qr)
    R[:, 2, 0] = 2 * (qk * qi-qj * qr)
    R[:, 2, 1] = 2 * (qj*qk + qi*qr)
    R[:, 2, 2] = 1-2 * (qi**2 + qj**2)
    return R

def get_camera_loc(pose):
    if pose.shape[1] == 7: #In case of quaternion vector representation
        cam_loc = pose[:, 4:]
        R = quat_to_rot(pose[:,:4])
        p = torch.eye(4).repeat(pose.shape[0],1,1).cuda().float()
        p[:, :3, :3] = R
        p[:, :3, 3] = cam_loc
    else: # In case of pose matrix representation
        cam_loc = pose[:, :3, 3]
        p = pose
    
    return cam_loc

def coordinates_projection_map(intrinsics, points_3d=None, points_2d=None):
    if points_3d is not None:
        # Project from 3D to 2D
        points_3d_homogeneous = torch.cat([points_3d, torch.ones(*points_3d.shape[:-1], 1).cuda()], dim=-1)
        points_2d_homogeneous = torch.bmm(intrinsics, points_3d_homogeneous.transpose(-2, -1)).transpose(-2, -1)
        points_2d = points_2d_homogeneous[..., :2] / points_2d_homogeneous[..., 2:3]
        return points_2d
    elif points_2d is not None:
        # Project from 2D to 3D
        points_2d_homogeneous = torch.cat([points_2d, torch.zeros(*points_2d.shape[:-1], 1).cuda(), torch.ones(*points_2d.shape[:-1], 1).cuda()], dim=-1)
        intrinsics_inv = torch.inverse(intrinsics)
        points_3d_homogeneous = torch.bmm(intrinsics_inv, points_2d_homogeneous.transpose(-2, -1)).transpose(-2, -1)
        points_3d = points_3d_homogeneous[..., :3]
        return points_3d
    else:
        raise ValueError("Either points_3d or points_2d must be provided.")
    
def get_boundaries(image, intrinsics, pose):
    '''Returns the 4 boundary points corresponding to the xy boundaries of the dynamical space.'''
    if pose.shape[1] == 7: #In case of quaternion vector representation
        cam_loc = pose[:, 4:]
        R = quat_to_rot(pose[:,:4])
        p = torch.eye(4).repeat(pose.shape[0],1,1).cuda().float()
        p[:, :3, :3] = R
        p[:, :3, 3] = cam_loc
    else: # In case of pose matrix representation
        cam_loc = pose[:, :3, 3]
        p = pose

    uv = torch.tensor([[0, 0], [0, image.shape[2]], [image.shape[3], 0], [image.shape[3], image.shape[2]]]).cuda().float()
    uv = uv.unsqueeze(0).repeat(pose.shape[0], 1, 1)
    batch_size, num_samples, _ = uv.shape

    depth = torch.ones((batch_size, num_samples)).cuda()
    x_cam = uv[:, :, 0].view(batch_size, -1)
    y_cam = uv[:, :, 1].view(batch_size, -1)
    z_cam = depth.view(batch_size, -1)

    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics)

    # permute for batch matrix product
    pixel_points_cam = pixel_points_cam.permute(0, 2, 1)

    world_coords = torch.bmm(p, pixel_points_cam).permute(0, 2, 1)[:, :, :3]
    ray_dirs = world_coords - cam_loc[:, None, :]
    ray_dirs = F.normalize(ray_dirs, dim=2)

    return ray_dirs, cam_loc

def get_rays(uv, intrinsics, pose):
    '''Returns the normalized rays of the points with given uv cooridnates.'''
    if pose.shape[1] == 7: #In case of quaternion vector representation
        cam_loc = pose[:, 4:]
        R = quat_to_rot(pose[:,:4])
        p = torch.eye(4).repeat(pose.shape[0],1,1).cuda().float()
        p[:, :3, :3] = R
        p[:, :3, 3] = cam_loc
    else: # In case of pose matrix representation
        cam_loc = pose[:, :3, 3]
        p = pose

    #uv = uv.unsqueeze(0).repeat(pose.shape[0], 1, 1)
    batch_size, num_samples, _ = uv.shape

    depth = torch.ones((batch_size, num_samples)).cuda()
    x_cam = uv[:, :, 0].view(batch_size, -1)
    y_cam = uv[:, :, 1].view(batch_size, -1)
    z_cam = depth.view(batch_size, -1)

    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics)

    # permute for batch matrix product
    pixel_points_cam = pixel_points_cam.permute(0, 2, 1)

    world_coords = torch.bmm(p, pixel_points_cam).permute(0, 2, 1)[:, :, :3]
    ray_dirs = world_coords - cam_loc[:, None, :]
    ray_dirs = F.normalize(ray_dirs, dim=2)

    return ray_dirs, cam_loc

def get_uv(world_coords, cam_loc, intrinsics, pose):
    '''Returns the uv coordinates corresponding to the given rays.'''
    world_coords = world_coords.permute(0, 2, 1)
    world_coords = torch.cat([world_coords, torch.ones((world_coords.shape[0], 1, world_coords.shape[2])).cuda()], dim=1)

    # Compute the camera coordinates
    if pose.shape[1] == 7: # In case of quaternion vector representation
        R = quat_to_rot(pose[:,:4])
        p_inv = torch.eye(4).repeat(pose.shape[0],1,1).cuda().float()
        p_inv[:, :3, :3] = R
        p_inv[:, :3, 3] = cam_loc
        p_inv = torch.inverse(p_inv)
    else: # In case of pose matrix representation
        p_inv = torch.inverse(pose)

    cam_coords = torch.bmm(p_inv, world_coords).permute(0, 2, 1)[:, :, :3]

    # Project to the image plane
    uv = project(cam_coords[:, :, 0], cam_coords[:, :, 1], cam_coords[:, :, 2], intrinsics=intrinsics)

    uv[..., 0] = torch.clamp(uv[..., 0], 0, 1080 - 1)
    uv[..., 1] = torch.clamp(uv[..., 1], 0, 1920 - 1)
    exceeded_width_indices = torch.where(uv[..., 0] >= 1080)
    exceeded_height_indices = torch.where(uv[..., 1] >= 1920)

    return uv, exceeded_width_indices, exceeded_height_indices

def project(x, y, z, intrinsics):
    '''Projects 3D points to the image plane.'''
    fx, fy, cx, cy = intrinsics[:, 0, 0], intrinsics[:, 1, 1], intrinsics[:, 0, 2], intrinsics[:, 1, 2]

    fx = fx.unsqueeze(1).expand_as(x)
    fy = fy.unsqueeze(1).expand_as(y)
    cx = cx.unsqueeze(1).expand_as(x)
    cy = cy.unsqueeze(1).expand_as(y)

    u = x * fx / z + cx
    v = y * fy / z + cy

    uv = torch.stack([u, v], dim=2)

    return uv

def lift(x, y, z, intrinsics):
    # parse intrinsics
    intrinsics = intrinsics.cuda()
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    sk = intrinsics[:, 0, 1]

    x_lift = (x - cx.unsqueeze(-1) + cy.unsqueeze(-1)*sk.unsqueeze(-1)/fy.unsqueeze(-1) - sk.unsqueeze(-1)*y/fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z
    y_lift = (y - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z

    # homogeneous
    return torch.stack((x_lift, y_lift, z, torch.ones_like(z).cuda()), dim=-1)


# Unet parts

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_op(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)

        return down, p


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
       x1 = self.up(x1)
       x = torch.cat([x1, x2], 1)
       return self.conv(x)