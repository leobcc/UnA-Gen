import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation as R
import trimesh
import torch.nn.functional as F
import wandb

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
    
def get_cam(pose):
    '''Returns the 4 boundary points corresponding to the xy boundaries of the dynamical space.'''
    if pose.shape[1] == 7: #In case of quaternion vector representation
        cam_loc = pose[:, 4:]
    else: # In case of pose matrix representation
        cam_loc = pose[:, :3, 3]

    return cam_loc

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

def get_uv(world_coords, intrinsics, pose):
    '''Returns the uv coordinates corresponding to the given rays.'''
    world_coords = world_coords.permute(0, 2, 1)
    world_coords = torch.cat([world_coords, torch.ones((world_coords.shape[0], 1, world_coords.shape[2])).cuda()], dim=1)

    # Compute the camera coordinates
    if pose.shape[1] == 7: #In case of quaternion vector representation
        cam_loc = pose[:, 4:]
        R = quat_to_rot(pose[:,:4])
        p = torch.eye(4).repeat(pose.shape[0],1,1).cuda().float()
        p[:, :3, :3] = R
        p[:, :3, 3] = cam_loc
    else: # In case of pose matrix representation
        cam_loc = pose[:, :3, 3]
        p = pose

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

    return uv

import torch

def undistort_points(distorted_coords, K, D, max_iterations=10, tolerance=1e-6):
    """
    Remove distortion from a set of 2D points without explicit loops.
    
    :param distorted_coords: (batch_size, n_points, 2) distorted 2D coordinates (u, v) in pixel space
    :param K: (batch_size, 3, 3) intrinsic matrices
    :param D: (batch_size, 5) distortion coefficients [k1, k2, p1, p2, k3]
    :param max_iterations: Maximum number of iterations for the undistortion process
    :param tolerance: Convergence tolerance for iterative correction
    :return: (batch_size, n_points, 2) undistorted 2D coordinates
    """
    batch_size, n_points, _ = distorted_coords.shape

    # Extract intrinsic parameters
    fx = K[:, 0, 0].unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1)
    fy = K[:, 1, 1].unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1)
    cx = K[:, 0, 2].unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1)
    cy = K[:, 1, 2].unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1)

    # Normalize the distorted coordinates using the intrinsic parameters
    x_distorted = (distorted_coords[..., 0:1] - cx) / fx  # (batch_size, n_points, 1)
    y_distorted = (distorted_coords[..., 1:2] - cy) / fy  # (batch_size, n_points, 1)

    # Initialize the undistorted coordinates as the distorted ones (initial guess)
    x_undistorted = x_distorted.clone()
    y_undistorted = y_distorted.clone()

    # Distortion coefficients
    k1 = D[:, 0].unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1)
    k2 = D[:, 1].unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1)
    p1 = D[:, 2].unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1)
    p2 = D[:, 3].unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1)
    k3 = D[:, 4].unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1)

    for _ in range(max_iterations):
        r2 = x_undistorted ** 2 + y_undistorted ** 2  # (batch_size, n_points, 1)

        # Radial distortion
        radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3

        # Tangential distortion
        x_tangential = 2 * p1 * x_undistorted * y_undistorted + p2 * (r2 + 2 * x_undistorted ** 2)
        y_tangential = p1 * (r2 + 2 * y_undistorted ** 2) + 2 * p2 * x_undistorted * y_undistorted

        # Updated undistorted coordinates
        x_new = (x_distorted - x_tangential) / radial
        y_new = (y_distorted - y_tangential) / radial

        # Check for convergence using torch's built-in functionality
        if torch.max(torch.abs(x_new - x_undistorted)) < tolerance and torch.max(torch.abs(y_new - y_undistorted)) < tolerance:
            break

        x_undistorted = x_new
        y_undistorted = y_new

    # Convert back to pixel coordinates using intrinsic parameters
    u_undistorted = x_undistorted * fx + cx  # (batch_size, n_points, 1)
    v_undistorted = y_undistorted * fy + cy  # (batch_size, n_points, 1)

    return torch.cat([u_undistorted, v_undistorted], dim=-1)  # (batch_size, n_points, 2)


def world_to_pixel_batch(world_points, K_batch, RT_batch):
    """
    Transforms a batch of 3D world points into 2D image space coordinates using batched camera parameters.

    Args:
        world_points (torch.Tensor): 3D world points of shape (batch_size, n_points, 3).
        K_batch (torch.Tensor): Intrinsic matrices of shape (batch_size, 3, 3).
        RT_batch (torch.Tensor): Extrinsic matrices of shape (batch_size, 4, 4).

    Returns:
        uv_coords (torch.Tensor): 2D pixel coordinates of shape (batch_size, n_points, 2).
    """
    # Ensure all inputs are float32
    world_points = world_points.float()
    K_batch = K_batch.float()
    RT_batch = RT_batch.float()
    
    batch_size, n_points, _ = world_points.shape
    
    # Convert world points to homogeneous coordinates by adding a '1' at the end
    ones = torch.ones((batch_size, n_points, 1), device=world_points.device, dtype=world_points.dtype)
    world_points_hom = torch.cat([world_points, ones], dim=-1)  # Shape: (batch_size, n_points, 4)

    # Apply the RT matrix to transform world coordinates to camera coordinates
    camera_points_hom = torch.bmm(world_points_hom, RT_batch.transpose(1, 2))  # Shape: (batch_size, n_points, 4)
    
    # Extract X_c, Y_c, Z_c from camera coordinates
    X_c = camera_points_hom[:, :, 0]
    Y_c = camera_points_hom[:, :, 1]
    Z_c = camera_points_hom[:, :, 2]

    # Apply the intrinsic matrix K to get image coordinates in homogeneous form
    camera_points = camera_points_hom[:, :, :3].transpose(1, 2)  # Shape: (batch_size, 3, n_points)
    pixel_coords_hom = torch.bmm(K_batch, camera_points).transpose(1, 2)  # Shape: (batch_size, n_points, 3)

    # Convert from homogeneous to (u, v) by dividing by the third coordinate (w)
    u = pixel_coords_hom[:, :, 0] / pixel_coords_hom[:, :, 2]
    v = pixel_coords_hom[:, :, 1] / pixel_coords_hom[:, :, 2]

    # Stack (u, v) into the final output of shape (batch_size, n_points, 2)
    uv_coords = torch.stack((u, v), dim=-1)  # Shape: (batch_size, n_points, 2)

    return uv_coords

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

# --------------

def axis_angle_to_rotation_matrix(axis_angle_batch):
    """
    Converts batched axis-angle representations to rotation matrices.
    
    Args:
        axis_angle_batch (tensor): Batched axis-angle representation of shape [batch_size, 3].
        
    Returns:
        rotation_matrix_batch (tensor): Batched rotation matrices of shape [batch_size, 3, 3].
    """
    batch_size = axis_angle_batch.size(0)
    rotation_matrices = torch.zeros((batch_size, 3, 3), device=axis_angle_batch.device)

    # Compute rotation matrices
    for i in range(batch_size):
        r = R.from_rotvec(axis_angle_batch[i].detach().cpu().numpy())  # Convert to rotation object
        rotation_matrices[i] = torch.tensor(r.as_matrix(), device=axis_angle_batch.device)  # Convert to tensor
    
    return rotation_matrices

def get_global_transformation(smpl_pose_batch, smpl_trans_batch, scale_batch):
    """
    Computes global transformation matrices and translations for batched SMPL pose and translation.
    
    Args:
        smpl_pose_batch (tensor): Batched SMPL pose parameters of shape [batch_size, 72].
        smpl_trans_batch (tensor): Batched SMPL translation vectors of shape [batch_size, 3].
        scale_batch (tensor): Batched global scale factors of shape [batch_size, 1].
    
    Returns:
        global_rotation_batch (tensor): Batched global rotation matrices of shape [batch_size, 3, 3].
        global_translation_batch (tensor): Batched global translation vectors of shape [batch_size, 3].
        global_scale_batch (tensor): Batched global scale factors of shape [batch_size, 1].
    """
    global_pose_batch = smpl_pose_batch[:, :3]  # The first 3 values are the global rotation in axis-angle
    global_rotation_batch = axis_angle_to_rotation_matrix(global_pose_batch)  # Convert to rotation matrix
    global_translation_batch = smpl_trans_batch  # Global translation vector
    global_scale_batch = scale_batch  # Scale factor

    return global_rotation_batch, global_translation_batch, global_scale_batch


# --------------

def stable_softmax(logits, dim):
    max_vals, _ = logits.max(dim=dim, keepdim=True)
    exps = torch.exp(logits - max_vals)
    sum_exps = exps.sum(dim=dim, keepdim=True)
    return exps / sum_exps

def print_grad(grad):
    wandb.log({"grad": grad.mean().item()})


# Matrix refinement

def upsample_matrix(matrix, scale_factor, threshold=0.5):
    matrix = matrix.unsqueeze(0).unsqueeze(0)  # Add two dimensions for batch and channel
    upsampled_matrix = F.interpolate(matrix, scale_factor=scale_factor, mode='trilinear', align_corners=False)
    upsampled_matrix = upsampled_matrix.squeeze(0).squeeze(0)  # Remove the added dimensions

    upsampled_matrix = (upsampled_matrix > threshold).float()

    return upsampled_matrix


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
    
class PixelShuffle3d(nn.Module):
    '''
    This class is a 3d version of pixelshuffle.
    '''
    def __init__(self, scale):
        '''
        :param scale: upsample scale
        '''
        super().__init__()
        self.scale = scale

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // self.scale ** 3

        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale

        input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_depth, in_height, in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)

class PixelUnshuffle3d(nn.Module):
    '''
    This class is a 3D version of pixel unshuffle, the inverse operation of PixelShuffle3d.
    '''
    def __init__(self, scale):
        '''
        :param scale: downsample scale
        '''
        super().__init__()
        self.scale = scale

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()

        # Ensure that the depth, height, and width are divisible by the scale factor
        assert in_depth % self.scale == 0, f"Incompatible depth {in_depth} for scale {self.scale}"
        assert in_height % self.scale == 0, f"Incompatible height {in_height} for scale {self.scale}"
        assert in_width % self.scale == 0, f"Incompatible width {in_width} for scale {self.scale}"

        out_depth = in_depth // self.scale
        out_height = in_height // self.scale
        out_width = in_width // self.scale
        nOut = channels * self.scale ** 3

        # Reshape the input to merge the spatial dimensions into the channel dimension
        input_view = input.view(batch_size, channels, out_depth, self.scale, out_height, self.scale, out_width, self.scale)
        output = input_view.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)