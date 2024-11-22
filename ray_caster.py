# This class is used to retrieve values by ray casting
# It is used by the UnA_Gen model to retrieve values both for the training and the evaluation

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import torch.nn.functional as F
from utils import get_rays, get_uv
import wandb
from torchvision.utils import make_grid
import time

# render_values_at_rays(dynamical_voxels_world_coo, occupancy_field, rgb_field, image, depth_image)

class RayCaster:
    def __init__(self, model, dynamical_voxels_world_coo, voxels_uv, occupancy_field, cum_of, rgb_field, image, depth_image=None, device='cuda'):
        self.opt = model.opt
        self.batch_size = model.batch_size
        self.mapping_dim = model.mapping_dim
        self.smpl_params = model.smpl_params
        self.intrinsics = model.intrinsics
        self.pose = model.pose
        self.img_size = model.img_size
        self.image_size = model.image_size
        self.original_size = model.original_size
        self.voxels_uv = voxels_uv
        #self.min_x = model.min_x
        #self.max_x = model.max_x
        #self.min_y = model.min_y
        #self.max_y = model.max_y
        self.cam_loc = model.cam_loc
        #self.scale = model.scale   # Some of this parameters can be removed when removing visualization
        #self.trans = model.trans
        #self.standard_depth_n = model.standard_depth_n
        self.visualize_stats = model.visualize_stats
        self.visualize_voxels = model.visualize_voxels

        self.dynamical_voxels_world_coo = dynamical_voxels_world_coo.to(device)
        self.occupancy_field = occupancy_field.to(device)
        self.cum_of = cum_of.to(device)
        self.rgb_field = rgb_field.to(device)
        self.image = image.to(device)
        self.depth_image = depth_image.to(device)
        self.device = device
        self.mode = None

        #self.shadow_field = model.shadow_field   # TODO: remove this, used only for visualization

    def render_values_at_rays(self, mode):
        if mode == 'training':
            self.mode = mode
            assert self.depth_image != None, "Depth image is required for training mode"
            dynamical_voxels_world_coo = self.dynamical_voxels_world_coo  
            
            voxels_uv = self.voxels_uv
            #voxels_uv = get_uv(dynamical_voxels_world_coo, self.intrinsics, self.pose)
            #voxels_uv[..., 0] = (voxels_uv[..., 0] - self.min_x) / (self.max_x - self.min_x) * self.image_size[1]
            #voxels_uv[..., 1] = (voxels_uv[..., 1] - self.min_y) / (self.max_y - self.min_y) * self.image_size[0]
            mask_out_of_range_x = (voxels_uv[..., 0] < 0) | (voxels_uv[..., 0] >= self.image_size[1].item())
            mask_out_of_range_y = (voxels_uv[..., 1] < 0) | (voxels_uv[..., 1] >= self.image_size[0].item())
            self.occupancy_field[mask_out_of_range_x | mask_out_of_range_y] = 0
            self.rgb_field[mask_out_of_range_x | mask_out_of_range_y] = 0
            voxels_uv[..., 0] = torch.clamp(voxels_uv[..., 0], 0, self.image_size[1].item()-1)
            voxels_uv[..., 1] = torch.clamp(voxels_uv[..., 1], 0, self.image_size[0].item()-1)
            #voxels_uv[..., 0] = voxels_uv[..., 0] / self.original_size[1] * self.image_size[1] - self.min_x
            #voxels_uv[..., 1] = voxels_uv[..., 1] / self.original_size[0] * self.image_size[0] - self.min_y
            #voxels_ov[exceeded_width_indices] = 0   # voxels backprojecting outside the image have their opacity put to 0
            #voxels_ov[exceeded_height_indices] = 0
            voxels_ov = self.occupancy_field
            voxels_rgb = self.rgb_field
            
            # ---------------------------------------------------------------------------------------------------------------------------------------
            if self.visualize_stats:
                with torch.no_grad():
                    occupancy_map = (voxels_ov > self.opt['occupancy_threshold']).float() 
                    if torch.count_nonzero(occupancy_map)==0:
                        occupancy_map[:, 0, 0] = 1  # This is to avoid the case where there are no occupied voxels
                    self.visualize_voxels((dynamical_voxels_world_coo * occupancy_map)[0], output_file='occupied_voxels.png', world=True)
                    try:
                        occupied_canonical_voxels = occupancy_map[0].view(self.matrix_mapping.shape)
                        occupied_canonical_voxels_coo = self.voxel_mapping(occupied_canonical_voxels)
                        self.visualize_voxels(occupied_canonical_voxels_coo, output_file='occupied_canonical_voxels.png', world=False)
                    except:
                        pass
            # ---------------------------------------------------------------------------------------------------------------------------------------

            if self.opt['mask_pruning']:
                occupied_pixels_mask = self.render_mask_pruning(dynamical_voxels_world_coo, voxels_ov, voxels_uv, self.image, self.depth_image)

            if self.opt['voxel_splatting']:
                original_rgb_values, rendered_rgb_values = self.render_voxel_splatting(dynamical_voxels_world_coo, voxels_ov, voxels_rgb, voxels_uv, self.image, self.depth_image)

            if self.opt['n_training_rays'] > 0:
                if self.opt['ray_cast_rgb'] and not self.opt['ray_cast_depth']:
                    original_rgb_values_rays, rendered_rgb_values_rays = self.render_rays(dynamical_voxels_world_coo, voxels_ov, voxels_rgb, voxels_uv, self.image, self.depth_image)
                elif self.opt['ray_cast_depth'] and not self.opt['ray_cast_rgb']:
                    depth_values, estimated_depth_values = self.render_rays(dynamical_voxels_world_coo, voxels_ov, voxels_rgb, voxels_uv, self.image, self.depth_image)
                elif self.opt['ray_cast_depth'] and self.opt['ray_cast_rgb']:
                    original_rgb_values_rays, rendered_rgb_values_rays, depth_values, estimated_depth_values = self.render_rays(dynamical_voxels_world_coo, voxels_ov, voxels_rgb, voxels_uv, self.image, self.depth_image)

            training_values = {}
            if self.opt['voxel_splatting']:
                training_values['original_rgb_values'] = original_rgb_values
                training_values['rendered_rgb_values'] = rendered_rgb_values  
            if self.opt['ray_cast_rgb']:
                training_values['original_rgb_values_rays'] = original_rgb_values_rays
                training_values['rendered_rgb_values_rays'] = rendered_rgb_values_rays
            if self.opt['ray_cast_depth']:
                training_values['depth_values'] = depth_values
                training_values['estimated_depth_values'] = estimated_depth_values
            if self.opt['mask_pruning']:
                training_values['occupied_pixels_mask'] = occupied_pixels_mask

            return training_values
        
        elif mode == 'render_image':
            self.mode = mode
            rendered_image = torch.zeros_like(self.image)
            dynamical_voxels_world_coo = self.dynamical_voxels_world_coo  
            voxels_ov = self.occupancy_field
            voxels_rgb = self.rgb_field

            voxels_uv = get_uv(dynamical_voxels_world_coo, self.intrinsics, self.pose)
            voxels_uv[..., 0] = (voxels_uv[..., 0] - self.min_x) / (self.max_x - self.min_x) * self.image_size[1]
            voxels_uv[..., 1] = (voxels_uv[..., 1] - self.min_y) / (self.max_y - self.min_y) * self.image_size[0]
            voxels_uv[..., 0] = torch.clamp(voxels_uv[..., 0], 0, self.image_size[1].item()-1)
            voxels_uv[..., 1] = torch.clamp(voxels_uv[..., 1], 0, self.image_size[0].item()-1)

            u = torch.arange(self.image_size[1].item(), device=self.device)
            v = torch.arange(self.image_size[0].item(), device=self.device)
            u, v = torch.meshgrid(u, v)
            rays_uv_all = torch.stack((u, v), dim=-1).reshape(-1, 2)

            n_pixels = self.opt['n_training_rays']   # Should be able to handle this with the memory
            n_patches = rays_uv_all.shape[0] // n_pixels
            for i in range(n_patches):
                print("ETA:", round(i/n_patches*100), "%", end='\r')
                if (i+1)*n_pixels > rays_uv_all.shape[0]:
                    rays_uv = rays_uv_all[i*n_pixels:]
                else:
                    rays_uv = rays_uv_all[i*n_pixels:(i+1)*n_pixels]
                original_rgb_values_rays, rendered_rgb_values_rays = self.render_rays(dynamical_voxels_world_coo, voxels_ov, voxels_rgb, voxels_uv, self.image, self.depth_image, rays_uv)
               
                batch_indices = torch.arange(rendered_image.shape[0]).view(-1, 1)
                rendered_image[batch_indices, :, rays_uv[:, 1], rays_uv[:, 0]] = rendered_rgb_values_rays

            return rendered_image


    def render_mask_pruning(self, dynamical_voxels_world_coo, voxels_ov, voxels_uv, image, depth_image):
        height, width = self.img_size
        occupied_pixels_mask = torch.zeros(self.batch_size, dynamical_voxels_world_coo.shape[1], 1, device=image.device, requires_grad=False)
        
        # Get the indices for the batch and color dimensions
        batch_indices = torch.arange(image.shape[0]).view(-1, 1, 1).to(voxels_uv.device)
        rgb_indices = torch.arange(image.shape[1]).view(1, -1, 1).to(voxels_uv.device)

        # Get the indices for the height and width dimensions # TODO: This will have to interpolate from values to be more precise
        v_int = voxels_uv[..., 1].long().unsqueeze(1)
        u_int = voxels_uv[..., 0].long().unsqueeze(1)
        v_frac = (voxels_uv[..., 1].unsqueeze(1) - v_int.float())
        u_frac = (voxels_uv[..., 0].unsqueeze(1) - u_int.float())

        # Use advanced indexing to select the values from the original image
        occupied_pixels_00 = image[batch_indices, rgb_indices, v_int, u_int]
        occupied_pixels_01 = image[batch_indices, rgb_indices, v_int, torch.clamp(u_int + 1, 0, width - 1)]
        occupied_pixels_10 = image[batch_indices, rgb_indices, torch.clamp(v_int + 1, 0, height - 1), u_int]
        occupied_pixels_11 = image[batch_indices, rgb_indices, torch.clamp(v_int + 1, 0, height - 1), torch.clamp(u_int + 1, 0, width - 1)]

        occupied_pixels = (occupied_pixels_00 * (1 - v_frac) * (1 - u_frac) + \
                            occupied_pixels_01 * (1 - v_frac) * u_frac + \
                            occupied_pixels_10 * v_frac * (1 - u_frac) + \
                            occupied_pixels_11 * v_frac * u_frac)

        # Now `selected_values` is a tensor of shape (1, 32768, 3)
        occupied_pixels_gray = occupied_pixels.mean(dim=1, keepdim=True).permute(0, 2, 1)

        # Threshold the grayscale values to get a binary mask
        occupied_pixels_mask = (occupied_pixels_gray > 0.01).float()

        if self.visualize_stats:
            try:
                wandb.log({"Occupied pixels mask": [wandb.Image(occupied_pixels_mask[0].detach().view(64, -1).cpu().numpy(), mode='L')]})
            except:
                wandb.log({"Occupied pixels mask": [wandb.Image(occupied_pixels_mask[0].detach().cpu().numpy(), mode='L')]})

        if self.visualize_stats:
            test_image = torch.zeros_like(image).cuda()
            test_image[batch_indices, :, v_int, u_int] = torch.ones(3).cuda()
            wandb.log({"Test image": [wandb.Image(test_image[0].detach().cpu().numpy().transpose(1, 2, 0))]})

        # Correction term: avoid opacity correction of in-shape voxels
        if self.opt['correct']:   # If true, the loss is applied only for out of mask samples
            occupied_pixels_mask = voxels_ov.detach() * occupied_pixels_mask   # TODO: figure out what's best

        return occupied_pixels_mask
    
    def render_voxel_splatting(self, dynamical_voxels_world_coo, voxels_ov, voxels_rgb, voxels_uv, image, depth_image):
        height, width = self.image.shape[2:]
        original_rgb_values = torch.zeros(self.batch_size, dynamical_voxels_world_coo.shape[1], 3, device=image.device, requires_grad=False)
        rendered_rgb_values = torch.zeros(self.batch_size, dynamical_voxels_world_coo.shape[1], 3, device=image.device, requires_grad=True)
        
        # Convert to long and get fractional part
        x_int, y_int = voxels_uv[..., 0].long(), voxels_uv[..., 1].long()
        x_frac, y_frac = voxels_uv[..., 0] - x_int.float(), voxels_uv[..., 1] - y_int.float()

        batch_idx = torch.arange(self.batch_size).unsqueeze(1).to(y_int.device)  

        # -----------
        test_image = torch.zeros_like(image).cuda()
        test_image[batch_idx, :, y_int[0], x_int[0]] = torch.ones(3).cuda()
        if self.visualize_stats:
            wandb.log({"Test image": [wandb.Image(test_image[0].detach().cpu().numpy().transpose(1, 2, 0))]})
            wandb.log({"Original image": [wandb.Image(image[0].detach().cpu().numpy().transpose(1, 2, 0))]})
        # -----------

        # Get the neighboring pixels
        top_left = image[batch_idx, :, y_int, x_int]
        top_right = image[batch_idx, :, y_int, torch.clamp(x_int + 1, 0, width - 1)]
        bottom_left = image[batch_idx, :, torch.clamp(y_int + 1, 0, height - 1), x_int]
        bottom_right = image[batch_idx, :, torch.clamp(y_int + 1, 0, height - 1),  torch.clamp(x_int + 1, 0, width - 1)]

        # Perform bilinear interpolation
        original_rgb_values = (top_left * ((1 - y_frac) * (1 - x_frac)).unsqueeze(-1).repeat(1,1,3) + \
                            top_right * ((1 - y_frac) * x_frac).unsqueeze(-1).repeat(1,1,3) + \
                            bottom_left * (y_frac * (1 - x_frac)).unsqueeze(-1).repeat(1,1,3) + \
                            bottom_right * (y_frac * x_frac).unsqueeze(-1).repeat(1,1,3))
                    
        # Compute rendered_rgb_values
        #occupancy_map = (voxels_ov > self.opt['occupancy_threshold']).float() * ((1 - self.cum_of) > 0).float()
        occupancy_map = (voxels_ov > self.opt['occupancy_threshold']).float() 
        rendered_rgb_values = voxels_rgb * occupancy_map.detach()
        #rendered_rgb_values = voxels_rgb
        #rendered_rgb_values = voxels_rgb * self.shadow_field.detach()   # Train only the rgb field
        #rendered_rgb_values = occupied_voxels_ov * (occupied_voxels_rgb.clone().detach())  

        original_rgb_values = original_rgb_values * occupancy_map.detach()   # Suppress rgb values corresponding to non occupied voxels
                                                                             # Only train for occupied voxels rgb values
        
        # Compute pdeth-based correction term 
        depth_all = torch.norm(dynamical_voxels_world_coo - self.cam_loc.unsqueeze(1), dim=-1)
        depth_all = (depth_all - depth_all.min(dim=1, keepdim=True)[0]) / (depth_all.max(dim=1, keepdim=True)[0] - depth_all.min(dim=1, keepdim=True)[0])
        weights = ((1 - depth_all)).unsqueeze(-1).expand_as(rendered_rgb_values)

        #original_rgb_values = original_rgb_values + (rendered_rgb_values.clone().detach() - original_rgb_values) * (1 - weights.clone().detach())   # Correction term for further voxels
                                                                                                                                                    # Train only for values more in front

        if self.visualize_stats:
            with torch.no_grad():
                rgb_values_differences = abs(original_rgb_values - rendered_rgb_values)
                strings_height = 256
                try:
                    res = voxels_ov.shape[1] % strings_height
                    if res != 0:
                        zeros = torch.zeros(self.batch_size, strings_height - res, 3, device=image.device, requires_grad=False)
                        original_rgb_values = torch.cat((original_rgb_values, zeros), dim=1)
                        rendered_rgb_values = torch.cat((rendered_rgb_values, zeros), dim=1)
                        rgb_values_differences = torch.cat((rgb_values_differences, zeros), dim=1)
                    original_values = original_rgb_values.view(self.batch_size, 3, strings_height, -1)
                    rendered_values = rendered_rgb_values.view(self.batch_size, 3, strings_height, -1)
                    differences_values = rgb_values_differences.view(self.batch_size, 3, strings_height, -1)
                except:
                    original_values = original_rgb_values.view(self.batch_size, 3, 1, -1)
                    rendered_values = rendered_rgb_values.view(self.batch_size, 3, 1, -1)
                    differences_values = rgb_values_differences.view(self.batch_size, 3, 1, -1)
                images = torch.cat((original_values, rendered_values, differences_values))
                grid = make_grid(images, nrow=self.batch_size)  # Arrange the images in a 3xbtach_size grid
                wandb.log({'original_vs_rendered_rgb_values (vsplat)': [wandb.Image(grid)]})

        return original_rgb_values, rendered_rgb_values

    def render_rays(self, dynamical_voxels_world_coo, voxels_ov, voxels_rgb, voxels_uv, image, depth_image, rays_uv=None):
        if rays_uv is None:
            n_rays = self.opt['n_training_rays']
            height, width = self.img_size

            if self.opt['train_on_non_black']:
                non_black_indices = torch.nonzero(image.sum(dim=1) > 0)
                indices_by_batch = [non_black_indices[non_black_indices[:, 0] == i] for i in range(image.shape[0])]
                selected_indices = [indices[torch.randperm(indices.shape[0])[:n_rays], 1:] for indices in indices_by_batch]
                selected_indices = torch.stack(selected_indices)[..., [1, 0]]
            else:
                height, width = image.shape[-2:]
                y_indices = torch.randint(height, (self.batch_size, n_rays), device='cuda')  
                x_indices = torch.randint(width, (self.batch_size, n_rays), device='cuda') 
                selected_indices = torch.stack((x_indices, y_indices), dim=-1)

        else:
            height, width = self.img_size
            n_rays = rays_uv.shape[0]   # (n_rays, 2)
            selected_indices = rays_uv.unsqueeze(0)   # (1, n_rays, 2)

        if self.opt['ray_cast_rgb']:
            original_rgb_values_rays = torch.zeros(self.batch_size, n_rays, 3, device=image.device, requires_grad=False)
            rendered_rgb_values_rays = torch.zeros(self.batch_size, n_rays, 3, device=image.device, requires_grad=True)

            # Expand dimensions for broadcasting
            selected_indices_exp = selected_indices.unsqueeze(2)   # (batch, n_rays, 1, 2)
            occupied_voxels_uv_exp = voxels_uv.unsqueeze(1)   # (batch, 1, n_voxels, 2)                   

            if self.opt['nearest_voxels'] == -1:   # This is when controlling canonical space
                # Calculate Euclidean distance
                distances = torch.norm(selected_indices_exp - occupied_voxels_uv_exp, dim=-1).unsqueeze(-1)   # (batch, n_rays, n_voxels)

                # Create a mask for distances less than the threshold
                if self.opt['closeness_threshold'] > 0:
                    threshold = self.opt['closeness_threshold']
                    mask = (distances < threshold)
                elif self.opt['closeness_threshold'] < 0:   # k-nearest_neighbors
                    k = -self.opt['closeness_threshold']
                    _, indices = distances.topk(k, dim=2, largest=False, sorted=True)
                    mask = torch.zeros_like(distances)
                    mask = mask.scatter(2, indices, 1).bool()
                if self.visualize_stats:
                    close_voxels_number = torch.sum(mask, dim=2, keepdim=True)
                    close_voxels_number_0 = close_voxels_number[0].detach().cpu().numpy()
                    wandb.log({"Number of close voxels histogram:": wandb.Histogram(close_voxels_number_0)})

                # Apply the mask to get the distances and indices of the voxels within the threshold
                #distances = distances / threshold 

                if self.visualize_stats:
                    fig = plt.figure()
                    mask_0_0 = mask[0, 0].detach().squeeze(-1).cpu().numpy()
                    selected_voxels_uv = voxels_uv[0].detach().cpu().numpy()
                    selected_voxels_uv = selected_voxels_uv[mask_0_0]
                    plt.scatter(selected_voxels_uv[:, 0], selected_voxels_uv[:, 1], c='r', s=10)
                    wandb.log({"Close voxels": [wandb.Image(fig)]})
                    plt.close()

                # Expand occupied_voxels_ov to match the shape of mask
                occupied_voxels_ov_expanded = voxels_ov.unsqueeze(1).expand(-1, mask.shape[1], -1, -1)   # Opacity 0 for far voxels
                occupied_voxels_rgb_expanded = voxels_rgb.unsqueeze(1).expand(-1, mask.shape[1], -1, -1)   # RGB 0 for far voxels

                # Apply the mask to the tensors
                #occupied_voxels_ov_selected = torch.where(mask, occupied_voxels_ov_expanded, torch.zeros_like(occupied_voxels_ov_expanded))   # Put ones for the formula, it nullifies non considered voxels' contributes
                #occupied_voxels_rgb_selected = torch.where(mask, occupied_voxels_rgb_expanded, torch.zeros_like(occupied_voxels_rgb_expanded))
                occupied_voxels_ov_selected = mask * occupied_voxels_ov_expanded   # Put ones for the formula, it nullifies non considered voxels' contributes
                occupied_voxels_rgb_selected = mask * occupied_voxels_rgb_expanded

                eps = 1e-6
                depth_all = torch.norm(dynamical_voxels_world_coo - self.cam_loc.unsqueeze(1), dim=-1)
                depth_all = (depth_all - depth_all.min(dim=1, keepdim=True)[0]) / (depth_all.max(dim=1, keepdim=True)[0] - depth_all.min(dim=1, keepdim=True)[0])
                depth_all = depth_all.unsqueeze(1).unsqueeze(-1).expand_as(mask)
                #depth = torch.where(mask, depth_all, torch.zeros_like(depth_all))   # far voxels are suppress to normalize
                #depth = (depth - depth.min(dim=2, keepdim=True)[0]) / (depth.max(dim=2, keepdim=True)[0] - depth.min(dim=2, keepdim=True)[0] + eps)
                #depth = torch.where(mask, depth, torch.ones_like(depth))   # far voxels have distance 1
                depth = mask * depth_all   # far voxels are suppress to normalize
                depth = (depth - depth.min(dim=2, keepdim=True)[0]) / (depth.max(dim=2, keepdim=True)[0] - depth.min(dim=2, keepdim=True)[0] + eps)
                depth = (~mask) + depth   # far voxels have distance 1
                # NOTE: Here the depth values are normalized between the voxel concerning the specific rays, while depth_all is normalized between all voxels

                '''
                weights = (1 - depth)**3 * occupied_voxels_ov_selected *(1 - distances)

                rendered_rgb_values_rays = torch.sum(weights * occupied_voxels_rgb_selected, dim=2) / (torch.sum(weights, dim=2) + eps)
                '''

                # Sort the depth tensor along the n_voxels dimension and get the indices
                sorted_depth, indices = depth.sort(dim=2, descending=False)
                sorted_depth_all = depth_all.gather(2, indices)

                # Use the indices to reorder the occupied_voxels_ov_selected tensor
                sorted_occupied_voxels_ov_selected = occupied_voxels_ov_selected.gather(2, indices)
                sorted_occupied_voxels_rgb_selected = occupied_voxels_rgb_selected.gather(2, indices.expand_as(occupied_voxels_rgb_selected))
                sorted_voxels_distances = distances.gather(2, indices)
                sorted_mask = mask.gather(2, indices)

                sorted_voxels_distances_selected = sorted_voxels_distances * sorted_mask
                if self.opt['closeness_threshold'] > 0:
                    sorted_voxels_distances_selected = sorted_voxels_distances_selected  / threshold   # TODO: check if this works well
                else:
                    sorted_voxels_distances_selected = (sorted_voxels_distances_selected - sorted_voxels_distances_selected.min(dim=2, keepdim=True)[0]) / (sorted_voxels_distances_selected.max(dim=2, keepdim=True)[0] - sorted_voxels_distances_selected.min(dim=2, keepdim=True)[0] + eps)
                sorted_voxels_distances_selected = (~sorted_mask) + sorted_voxels_distances_selected   # far voxels have distance 1

                samples_distances = torch.cat([sorted_depth[:, :, 1:, :] - sorted_depth[:, :, :-1, :], torch.zeros(self.batch_size, n_rays, 1, 1, device=sorted_depth.device)], dim=2)
                norm_dis = ((1 + sorted_depth) * (1 + sorted_voxels_distances_selected) - 1) / 3
                #norm_dis = (norm_dis - norm_dis.min(dim=2, keepdim=True)[0]) / (norm_dis.max(dim=2, keepdim=True)[0] - norm_dis.min(dim=2, keepdim=True)[0] + eps)
                w = sorted_occupied_voxels_ov_selected * (1 - norm_dis)
                #w = sorted_occupied_voxels_ov_selected * (1 - sorted_depth)
                cum_w = torch.cumsum(w, dim=2) - w
                weights = torch.exp(-cum_w)*(1-torch.exp(-w))  
                #weights = weights / (torch.sum(weights, dim=2, keepdim=True)+eps)

                #wandb.log({"cum_of histogram:": wandb.Histogram(self.cum_of[0].detach().cpu().numpy())})
                #wandb.log({"norm_dis histogram:": wandb.Histogram(norm_dis[0].detach().cpu().numpy())})

                #sorted_cum_of = (self.cum_of.unsqueeze(1).repeat(1, n_rays, 1, 1)).gather(2, indices)
                #weights = (1+(1 - (sorted_cum_of > 0.5).float())) * (1 + sorted_occupied_voxels_ov_selected) * (1+(1 - norm_dis)) * sorted_mask / 8
                #weights = weights / (torch.sum(weights, dim=2, keepdim=True)+eps)

                #w_s = torch.sum(weights, dim=2, keepdim=True)
                #weights = weights + (1 - sorted_voxels_distances_selected)   # This is to give more importance to the voxels closer to the rays
                #weights = weights / torch.sum(weights, dim=2, keepdim=True) * w_s

                w_mask = (w != 0)

                #sorted_closest_voxel_distances = torch.where(sorted_mask, sorted_closest_voxel_distances, torch.zeros_like(sorted_closest_voxel_distances))   # for normalization
                #sorted_closest_voxel_distances = sorted_closest_voxel_distances / torch.sum(sorted_closest_voxel_distances, dim=2, keepdim=True)[0]
                #sorted_closest_voxel_distances = torch.where(w_mask, sorted_closest_voxel_distances, torch.ones_like(sorted_closest_voxel_distances))   # far voxels have distance 1
                #sorted_closest_voxel_distances = torch.where(sorted_mask, sorted_closest_voxel_distances, torch.ones_like(sorted_closest_voxel_distances))   # far voxels have distance 1
                #sorted_closest_voxel_distances = torch.where(w_mask, sorted_closest_voxel_distances, torch.ones_like(sorted_closest_voxel_distances))   # far voxels have distance 1
                '''
                sorted_closest_voxel_distances = sorted_mask * sorted_closest_voxel_distances   # for normalization
                sorted_closest_voxel_distances = (~w_mask) + sorted_closest_voxel_distances   # far voxels have distance 1
                sorted_closest_voxel_distances = sorted_closest_voxel_distances / torch.sum(sorted_closest_voxel_distances, dim=2, keepdim=True)[0]
                sorted_closest_voxel_distances = (~sorted_mask) + sorted_closest_voxel_distances   # far voxels have distance 1
                sorted_closest_voxel_distances = (~w_mask) + sorted_closest_voxel_distances   # far voxels have distance 1
                '''

                if self.visualize_stats:
                    weights_0 = weights[0, :10].detach().view(10, -1).cpu().numpy()
                    wandb.log({"weights for first 10 rays histogram:": wandb.Histogram(weights_0)})

                eps = 1e-6
                # Compute rendered_rgb_values_rays
                exp = 1
                #rendered_rgb_values_rays = torch.sum(occupied_voxels_rgb_selected * ((1 - closest_voxel_distances)**exp), dim=2) / (torch.sum((1 - closest_voxel_distances)**exp, dim=2) + eps)
                #rendered_rgb_values_rays = torch.sum(weights * occupied_voxels_ov_selected * occupied_voxels_rgb_selected * ((1 - closest_voxel_distances)**exp), dim=2) / (torch.sum(weights, dim=2) + torch.sum((1 - closest_voxel_distances)**exp, dim=2) + eps)
                
                #rendered_rgb_values_rays = torch.sum(weights * sorted_occupied_voxels_rgb_selected * ((1 - sorted_closest_voxel_distances)**exp), dim=2) / (torch.sum((1 - sorted_closest_voxel_distances)**exp, dim=2) + eps)
                rendered_rgb_values_rays = torch.sum(weights * sorted_occupied_voxels_rgb_selected, dim=2)

            if self.opt['nearest_voxels'] == -2:   # This is when controlling canonical space
                # Calculate Euclidean distance
                distances = torch.norm(selected_indices_exp - occupied_voxels_uv_exp, dim=-1).unsqueeze(-1)   # (batch, n_rays, n_voxels)

                # Create a mask for distances less than the threshold
                if self.opt['closeness_threshold'] > 0:
                    threshold = self.opt['closeness_threshold']
                    mask = (distances < threshold)
                elif self.opt['closeness_threshold'] < 0:   # k-nearest_neighbors
                    k = -self.opt['closeness_threshold']
                    _, indices = distances.topk(k, dim=2, largest=False, sorted=True)
                    mask = torch.zeros_like(distances)
                    mask = mask.scatter(2, indices, 1).bool()
                if self.visualize_stats:
                    close_voxels_number = torch.sum(mask, dim=2, keepdim=True)
                    close_voxels_number_0 = close_voxels_number[0].detach().cpu().numpy()
                    wandb.log({"Number of close voxels histogram:": wandb.Histogram(close_voxels_number_0)})

                # Apply the mask to get the distances and indices of the voxels within the threshold
                #distances = distances / threshold 

                if self.visualize_stats:
                    fig = plt.figure()
                    mask_0_0 = mask[0, 0].detach().squeeze(-1).cpu().numpy()
                    selected_voxels_uv = voxels_uv[0].detach().cpu().numpy()
                    selected_voxels_uv = selected_voxels_uv[mask_0_0]
                    plt.scatter(selected_voxels_uv[:, 0], selected_voxels_uv[:, 1], c='r', s=10)
                    wandb.log({"Close voxels": [wandb.Image(fig)]})
                    plt.close()

                # Expand occupied_voxels_ov to match the shape of mask
                occupied_voxels_ov_expanded = voxels_ov.unsqueeze(1).expand(-1, mask.shape[1], -1, -1)   # Opacity 0 for far voxels
                occupied_voxels_rgb_expanded = voxels_rgb.unsqueeze(1).expand(-1, mask.shape[1], -1, -1)   # RGB 0 for far voxels

                # Apply the mask to the tensors
                #occupied_voxels_ov_selected = torch.where(mask, occupied_voxels_ov_expanded, torch.zeros_like(occupied_voxels_ov_expanded))   # Put ones for the formula, it nullifies non considered voxels' contributes
                #occupied_voxels_rgb_selected = torch.where(mask, occupied_voxels_rgb_expanded, torch.zeros_like(occupied_voxels_rgb_expanded))
                occupied_voxels_ov_selected = mask * occupied_voxels_ov_expanded   # Put ones for the formula, it nullifies non considered voxels' contributes
                occupied_voxels_rgb_selected = mask * occupied_voxels_rgb_expanded

                eps = 1e-6
                depth_all = torch.norm(dynamical_voxels_world_coo - self.cam_loc.unsqueeze(1), dim=-1)
                depth_all = (depth_all - depth_all.min(dim=1, keepdim=True)[0]) / (depth_all.max(dim=1, keepdim=True)[0] - depth_all.min(dim=1, keepdim=True)[0])
                depth_all = depth_all.unsqueeze(1).unsqueeze(-1).expand_as(mask)
                #depth = torch.where(mask, depth_all, torch.zeros_like(depth_all))   # far voxels are suppress to normalize
                #depth = (depth - depth.min(dim=2, keepdim=True)[0]) / (depth.max(dim=2, keepdim=True)[0] - depth.min(dim=2, keepdim=True)[0] + eps)
                #depth = torch.where(mask, depth, torch.ones_like(depth))   # far voxels have distance 1
                depth = mask * depth_all   # far voxels are suppress to normalize
                depth = (depth - depth.min(dim=2, keepdim=True)[0]) / (depth.max(dim=2, keepdim=True)[0] - depth.min(dim=2, keepdim=True)[0] + eps)
                depth = (~mask) + depth   # far voxels have distance 1
                # NOTE: Here the depth values are normalized between the voxel concerning the specific rays, while depth_all is normalized between all voxels

                voxels_distances_selected = distances * mask
                if self.opt['closeness_threshold'] > 0:
                    voxels_distances_selected = voxels_distances_selected  / threshold   # TODO: check if this works well
                else:
                    voxels_distances_selected = (voxels_distances_selected - voxels_distances_selected.min(dim=2, keepdim=True)[0]) / (voxels_distances_selected.max(dim=2, keepdim=True)[0] - voxels_distances_selected.min(dim=2, keepdim=True)[0] + eps)
                voxels_distances_selected = (~mask) + voxels_distances_selected   # far voxels have distance 1

                #wandb.log({"cum_of histogram:": wandb.Histogram(self.cum_of[0].detach().cpu().numpy())})
                #wandb.log({"norm_dis histogram:": wandb.Histogram(norm_dis[0].detach().cpu().numpy())})

                cum_of = (self.cum_of.unsqueeze(1).repeat(1, n_rays, 1, 1))
                #selected_active_voxels = (((1 - cum_of) + occupied_voxels_ov_selected)/2 > 0).float()   # front_voxels, should be pushed to 1 to contiribute
                selected_active_voxels = (1 - cum_of) * occupied_voxels_ov_selected
                weights = selected_active_voxels * (1 - voxels_distances_selected)
                #weights = (1+(1 - (cum_of > 0.5).float())) * (1 + sorted_occupied_voxels_ov_selected) * (1+(1 - norm_dis)) * sorted_mask / 8
                weights = weights / (torch.sum(weights, dim=2, keepdim=True)+eps)

                if self.visualize_stats:
                    weights_0 = weights[0, :10].detach().view(10, -1).cpu().numpy()
                    wandb.log({"weights for first 10 rays histogram:": wandb.Histogram(weights_0)})

                eps = 1e-6
                # Compute rendered_rgb_values_rays
                exp = 1
                #rendered_rgb_values_rays = torch.sum(occupied_voxels_rgb_selected * ((1 - closest_voxel_distances)**exp), dim=2) / (torch.sum((1 - closest_voxel_distances)**exp, dim=2) + eps)
                #rendered_rgb_values_rays = torch.sum(weights * occupied_voxels_ov_selected * occupied_voxels_rgb_selected * ((1 - closest_voxel_distances)**exp), dim=2) / (torch.sum(weights, dim=2) + torch.sum((1 - closest_voxel_distances)**exp, dim=2) + eps)
                
                #rendered_rgb_values_rays = torch.sum(weights * sorted_occupied_voxels_rgb_selected * ((1 - sorted_closest_voxel_distances)**exp), dim=2) / (torch.sum((1 - sorted_closest_voxel_distances)**exp, dim=2) + eps)
                rendered_rgb_values_rays = torch.sum(weights * occupied_voxels_rgb_selected, dim=2)
               
            u, v = selected_indices[..., 0], selected_indices[..., 1]
            batch_idx = torch.arange(self.batch_size).unsqueeze(1).to(dynamical_voxels_world_coo.device)
            original_rgb_values_rays = image[batch_idx, :, v, u]

            if self.visualize_stats and self.opt['n_training_rays'] > 0:
                with torch.no_grad():
                    rgb_values_differences_add = abs(original_rgb_values_rays - rendered_rgb_values_rays)
                    strings_height = 64
                    try:
                        original_values = original_rgb_values_rays.view(self.batch_size, 3, strings_height, -1)
                        rendered_values = rendered_rgb_values_rays.view(self.batch_size, 3, strings_height, -1)
                        differences_values = rgb_values_differences_add.view(self.batch_size, 3, strings_height, -1)
                    except:
                        original_values = original_rgb_values_rays.view(self.batch_size, 3, 1, -1)
                        rendered_values = rendered_rgb_values_rays.view(self.batch_size, 3, 1, -1)
                        differences_values = rgb_values_differences_add.view(self.batch_size, 3, 1, -1)
                    images = torch.cat((original_values[:, :, :, :], rendered_values[:, :, :, :], differences_values[:, :, :, :]))
                    grid = make_grid(images, nrow=self.batch_size)  # Arrange the images in a 3xbtach_size grid
                    wandb.log({'original_vs_rendered_rgb_values (rays)': [wandb.Image(grid)]})

        if self.opt['ray_cast_depth'] and self.mode == 'training':
            estimated_depth_values = torch.zeros(self.batch_size, n_rays, device=image.device, requires_grad=False)
            depth_values = torch.zeros(self.batch_size, n_rays, device=image.device, requires_grad=True)

            # Find the position of the maximum weight for each batch and each ray
            #max_weight_indices = weights.argmax(dim=2)
            # Apply softmax to the weights
            #softmax_weights = F.softmax(weights, dim=2)
            
            if self.opt['nearest_voxels'] == -1:
                w_1 = weights 

                sorted_occupied_voxels_ov = occupied_voxels_ov_selected.gather(2, indices)
                occupied_voxels_mask = sorted_occupied_voxels_ov > self.opt['occupancy_threshold']
                sorted_depth_all = sorted_depth_all * occupied_voxels_mask
                sorted_depth_all = (sorted_depth_all - sorted_depth_all.min(dim=2, keepdim=True)[0]) / (sorted_depth_all.max(dim=2, keepdim=True)[0] - sorted_depth_all.min(dim=2, keepdim=True)[0] + 1e-6)
                sorted_depth_all = (~occupied_voxels_mask) + sorted_depth_all   # far voxels have distance 1   

                # Compute weighted sum of depth_values
                weighted_depth_values = w_1 * (1 - sorted_depth_all)   # TODO: this could probably be weights without softmax
                depth_values = weighted_depth_values.sum(dim=2).squeeze(-1)

            if self.opt['nearest_voxels'] == -2:
                active_voxels_depths = depth_all * selected_active_voxels
                depth_values = active_voxels_depths.mean(dim=2).squeeze(-1)

            # Get the depth values
            #min_val = depth_image.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
            #max_val = depth_image.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]

            # Normalize depth_image between 0 and 1
            #depth_image = (depth_image - min_val) / (max_val - min_val)
            uv = selected_indices_exp.float()
            x_int, y_int = uv[..., 0].long(), uv[..., 1].long()
            x_frac, y_frac = uv[..., 0] - x_int.float(), uv[..., 1] - y_int.float()
            batch_idx = batch_idx.unsqueeze(1).expand_as(x_int)
            top_left = depth_image[batch_idx, y_int, x_int]
            top_right = depth_image[batch_idx, y_int, torch.clamp(x_int + 1, 0, width - 1)]
            bottom_left = depth_image[batch_idx, torch.clamp(y_int + 1, 0, height - 1), x_int]
            bottom_right = depth_image[batch_idx, torch.clamp(y_int + 1, 0, height - 1),  torch.clamp(x_int + 1, 0, width - 1)]

            # Perform bilinear interpolation
            estimated_depth_values = (top_left * ((1 - y_frac) * (1 - x_frac)) + \
                                top_right * ((1 - y_frac) * x_frac) + \
                                bottom_left * (y_frac * (1 - x_frac)) + \
                                bottom_right * (y_frac * x_frac)).squeeze(-1)

            if self.visualize_stats:
                with torch.no_grad():
                    string_height = 64
                    try: 
                        original_values = estimated_depth_values.view(self.batch_size, string_height, -1)
                        rendered_values = depth_values.view(self.batch_size, string_height, -1)
                        differences_values = abs(depth_values - estimated_depth_values).view(self.batch_size, string_height, -1)
                    except:
                        original_values = estimated_depth_values.view(self.batch_size, 1, -1)
                        rendered_values = depth_values.view(self.batch_size, 1, -1)
                        differences_values = abs(depth_values - estimated_depth_values).view(self.batch_size, 1, -1)
                    images = torch.cat((original_values, rendered_values, differences_values)).unsqueeze(1)
                    grid = make_grid(images, nrow=self.batch_size)
                    wandb.log({'depth_values (rays)': [wandb.Image(grid, mode='L')]})

        # ---------------------------------------------------------------------------------------------------------------------------------------
        if self.mode == 'training' and self.visualize_stats:
            with torch.no_grad():
                try:
                    batch_idx = torch.arange(self.batch_size).unsqueeze(1).to(image.device)

                    # Compute images
                    rays_uv_image = torch.zeros(self.batch_size, 3, height, width, device=image.device, requires_grad=False)
                    rays_uv_image[batch_idx, :, selected_indices[:, :, 1], selected_indices[:, :, 0]] = torch.ones(3).cuda()
                    y_int = voxels_uv[..., 1].long().unsqueeze(1)
                    x_int = voxels_uv[..., 0].long().unsqueeze(1)
                    voxels_uv_image = torch.zeros(self.batch_size, 3, height, width, device=image.device, requires_grad=False)
                    voxels_uv_image[batch_idx, :, y_int, x_int] = torch.ones(3).cuda()
                    occupancy_map = (voxels_ov > self.opt['occupancy_threshold']).float()
                    masked_voxels = voxels_uv * occupancy_map
                    y_int = masked_voxels[..., 1].long().unsqueeze(1)
                    x_int = masked_voxels[..., 0].long().unsqueeze(1)
                    occupied_voxels_uv_image = torch.zeros(self.batch_size, 3, height, width, device=image.device, requires_grad=False)
                    occupied_voxels_uv_image[batch_idx, :, y_int, x_int] = torch.ones(3).cuda()
                    occupied_voxels_uv_image_rgb = torch.zeros(self.batch_size, 3, height, width, device=image.device, requires_grad=False)
                    occupied_voxels_uv_image_rgb[batch_idx, :, y_int, x_int] = voxels_rgb[batch_idx, :, :]
                    #voxels_rgb_image = torch.zeros(self.batch_size, 3, height, width, device=image.device, requires_grad=False)
                    #voxels_rgb_image[batch_idx, :, y_int, x_int] = voxels_rgb[batch_idx, :, :]/self.shadow_field[batch_idx, :, :].squeeze(1).expand_as(voxels_rgb)
                    #shadow_image = torch.zeros(self.batch_size, 3, height, width, device=image.device, requires_grad=False)
                    #shadow_image[batch_idx, :, y_int, x_int] = self.shadow_field[batch_idx, :, :].squeeze(1).expand_as(voxels_rgb)
                    depth_image_from_values = torch.zeros(self.batch_size, 3, height, width, device=image.device, requires_grad=False)  
                    depth_image_from_values[batch_idx, :, selected_indices[:, :, 1], selected_indices[:, :, 0]] = depth_values.unsqueeze(-1).repeat(1, 1, 3)
                    #stacked_images = torch.cat((image[0], rays_uv_image[0], voxels_uv_image[0], 
                    #                            occupied_voxels_uv_image[0], occupied_voxels_uv_image_rgb[0], voxels_rgb_image[0], 
                    #                            shadow_image[0], depth_image_from_values[0], depth_image[0].unsqueeze(0).repeat(3, 1, 1)), dim=-1)  
                    stacked_images = torch.cat((image[0], rays_uv_image[0], voxels_uv_image[0], 
                                                occupied_voxels_uv_image[0], occupied_voxels_uv_image_rgb[0], 
                                                depth_image_from_values[0], depth_image[0].unsqueeze(0).repeat(3, 1, 1)), dim=-1)  
                    wandb.log({"Stacked images": [wandb.Image(stacked_images.detach().cpu().numpy().transpose(1, 2, 0))]})
                except:
                    pass
        # -----------------------------------------------------------------------------------------------------------------------------------
            
        if self.mode == 'render_image':
            return original_rgb_values_rays, rendered_rgb_values_rays

        if self.opt['ray_cast_rgb'] and not self.opt['ray_cast_depth']:
            return original_rgb_values_rays, rendered_rgb_values_rays
        elif self.opt['ray_cast_depth'] and not self.opt['ray_cast_rgb']:
            return depth_values, estimated_depth_values
        elif self.opt['ray_cast_rgb'] and self.opt['ray_cast_depth']:
            return original_rgb_values_rays, rendered_rgb_values_rays, depth_values, estimated_depth_values