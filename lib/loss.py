import torch
import torch.nn.functional as F
import wandb
import math

def loss_c(confs, outputs, phase=0):
    """
    Compute the loss given the outputs and the inputs.
    """
    if confs['rendering_loss']:   # Main loss term, training rays
        rendering_loss = F.l1_loss(outputs['rendered_rgb_values_add'], outputs['original_rgb_values_add'], reduction='mean') 
        wandb.log({'rendering_loss': rendering_loss})
    
    regularization = 0.0
    exp = 1

    if confs['segmentation_loss']:
        #segmentation_loss = F.l1_loss(outputs['occupancy_field'], outputs['occupied_pixels_mask'], reduction='mean')
        #regularization += segmentation_loss * confs['segmentation_loss_weight']
        #wandb.log({'segmentation_loss': segmentation_loss})

        ray_opacity = outputs['occupancy_field_t'].mean(dim=1)
        inner_rays = ray_opacity * (1 - outputs['outliers_mask'].detach())
        outlier_rays = ray_opacity[outputs['outliers_mask'].bool()]

        segmentation_loss_p = F.l1_loss(inner_rays, torch.ones_like(inner_rays), reduction='mean')
        segmentation_loss_n = F.l1_loss(outlier_rays, torch.zeros_like(outlier_rays), reduction='mean')
        segmentation_loss = segmentation_loss_n
        #segmentation_loss = segmentation_loss_n
        #segmentation_loss = F.l1_loss(ray_opacity, (1 - outputs['outliers_mask'].detach()), reduction='mean')
        stacked_images = torch.cat((ray_opacity.clone().detach()[0], (1 - outputs['outliers_mask'].detach()).clone().detach()[0]), dim=1)
        wandb.log({'segmentation_loss full': [wandb.Image(stacked_images.clone().detach().cpu().numpy(), mode='L')]})

        #segmentation_loss = F.l1_loss(outputs['occupancy_field_t'], (1 - outputs['outliers_mask'].unsqueeze(1).repeat(1, outputs['occupancy_field_t'].shape[1], 1, 1)), reduction='mean')
        #segmentation_loss = F.l1_loss(ray_opacity, (1 - outputs['outliers_mask']), reduction='mean')
        regularization += segmentation_loss * confs['segmentation_loss_weight']
        wandb.log({'segmentation_loss': segmentation_loss})


    if confs['canonical_consistency_loss'] and outputs['epoch'] > 0:
        activity_occupancy_b = outputs['activity_occupancy'].unsqueeze(0).expand_as(outputs['occupancy_field'])
        #activity_ocuupancy_b = outputs['prev_ov'].unsqueeze(0).expand_as(outputs['occupancy_field'])
        #canonical_consistency_loss = F.mse_loss(outputs['occupancy_field'], activity_ocuupancy_b, reduction='mean')
        #incr_weight = confs['canonical_consistency_loss_weight']*(math.cos(outputs['epoch']/20*math.pi) + 1)

        incr_weight = confs['canonical_consistency_loss_weight'] * (1 - math.exp(-outputs['epoch']/100))
        #incr_weight = confs['canonical_consistency_loss_weight']

        #incr_weight = confs['canonical_consistency_loss_weight'] * max(outputs['epoch']/100, 0)
        #regularization += canonical_consistency_loss * incr_weight
        #wandb.log({'canonical_consistency_loss': canonical_consistency_loss})

        activity_occupancy_rgb_b = outputs['activity_occupancy_rgb'].unsqueeze(0).expand_as(outputs['rgb_field'])
        #activity_occupancy_rgb_b = outputs['prev_rgb'].unsqueeze(0).expand_as(outputs['rgb_field'])
        #cc_rgb_loss = F.mse_loss(outputs['rgb_field'], activity_occupancy_rgb_b, reduction='mean')
        #regularization += cc_rgb_loss * incr_weight
        #wandb.log({'cc_rgb_loss': cc_rgb_loss})

        ao_mask_p = (activity_occupancy_b > 0.6).squeeze(-1)
        ao_mask_n = (activity_occupancy_b < 0.4).squeeze(-1)
        ao_mask = ao_mask_p | ao_mask_n
        if ao_mask.sum() != 0:
            cc_of_weak_loss = 0
            cc_rgb_weak_loss = 0
            if ao_mask_p.sum() != 0:
                p_of_cano = F.mse_loss(outputs['occupancy_field'][ao_mask_p], 
                                            torch.ones_like(outputs['occupancy_field'][ao_mask_p]), reduction='mean') 
                wandb.log({'p_of_cano': p_of_cano})
                cc_of_weak_loss += p_of_cano**exp

                if cc_of_weak_loss < 0.05:
                    phase = 1

            if ao_mask_n.sum() != 0:
                n_of_cano = F.mse_loss(outputs['occupancy_field'][ao_mask_n], 
                                            torch.zeros_like(outputs['occupancy_field'][ao_mask_n]), reduction='mean') 
                wandb.log({'n_of_cano': n_of_cano})
                cc_of_weak_loss += n_of_cano**exp
            if outputs['epoch'] > 0:   # Every 10 epochs, the loss is shut off to allow the model to explore other solutions
                regularization += cc_of_weak_loss * incr_weight
                wandb.log({'cc_of_weak_loss': cc_of_weak_loss})
              
            if ao_mask_p.sum() != 0:
                cc_rgb_weak_loss = F.mse_loss(outputs['rgb_field'][ao_mask_p], 
                                            activity_occupancy_rgb_b[ao_mask_p], reduction='mean') 
            if outputs['epoch'] > 0:   # Every 10 epochs, the loss is shut off to allow the model to explore other solutions
                regularization += cc_rgb_weak_loss * incr_weight
                wandb.log({'cc_rgb_weak_loss': cc_rgb_weak_loss})
    elif confs['canonical_consistency_loss'] and outputs['epoch'] == 0:
        cc_of_weak_loss = 1
        cc_rgb_weak_loss = 1
        regularization += cc_of_weak_loss * confs['canonical_consistency_loss_weight']
        wandb.log({'cc_of_weak_loss': cc_of_weak_loss})
        regularization += cc_rgb_weak_loss * confs['canonical_consistency_loss_weight']
        wandb.log({'cc_rgb_weak_loss': cc_rgb_weak_loss})                

        '''
        cc_of_weak_loss = F.mse_loss(torch.gather(outputs['occupancy_field'], 1, random_indices.unsqueeze(-1)), 
                                     torch.gather(activity_occupancy_b, 1, random_indices.unsqueeze(-1)), reduction='mean')
        regularization += cc_of_weak_loss * incr_weight
        wandb.log({'cc_of_weak_loss': cc_of_weak_loss})

        cc_rgb_weak_loss = F.mse_loss(torch.gather(outputs['rgb_field'], 1, random_indices.unsqueeze(-1)), 
                                     torch.gather(activity_occupancy_rgb_b, 1, random_indices.unsqueeze(-1)), reduction='mean')
        regularization += cc_rgb_weak_loss * incr_weight
        wandb.log({'cc_rgb_weak_loss': cc_rgb_weak_loss})
        '''

    
    if confs['splatting_loss']:
        #splatting_loss = F.mse_loss(outputs['rendered_rgb_values'], outputs['original_rgb_values'], reduction='mean') 
        #                #F.l1_loss(outputs['rgb_field']*(outputs['shadow_field'].detach().expand_as(outputs['rgb_field'])), outputs['original_rgb_values'], reduction='mean')
        #regularization += splatting_loss * confs['splatting_loss_weight']
        #wandb.log({'splatting_loss': splatting_loss})

        rasterize_loss = F.l1_loss(outputs['rasterized_image'], outputs['image'], reduction='mean') 
                        #F.l1_loss(outputs['rgb_field']*(outputs['shadow_field'].detach().expand_as(outputs['rgb_field'])), outputs['original_rgb_values'], reduction='mean')
        regularization += rasterize_loss**exp * confs['splatting_loss_weight']
        wandb.log({'rasterize_loss': rasterize_loss})
        
        #occupied_voxels = (outputs['occupancy_field_t'].detach().unsqueeze(1).expand_as(outputs['rgb_field_t']) > 0).bool()
        back_proj_loss = F.mse_loss(outputs['rgb_field_t'].mean(dim=2), outputs['image'], reduction='mean') 
        #                #F.l1_loss(outputs['rgb_field']*(outputs['shadow_field'].detach().expand_as(outputs['rgb_field'])), outputs['original_rgb_values'], reduction='mean')
        regularization += back_proj_loss * confs['splatting_loss_weight'] / 10
        wandb.log({'back_proj_loss': back_proj_loss})  

        #back_proj_loss = F.mse_loss(outputs['voxels_rgb_t']*(outputs['occupancy_field_t'].unsqueeze(1).expand_as(outputs['voxels_rgb_t'])), outputs['image'].unsqueeze(2).expand_as(outputs['voxels_rgb_t']), reduction='mean') 
        #back_proj_loss = F.mse_loss(outputs['voxels_rgb_t']*(outputs['occupancy_field_t'].unsqueeze(1).expand_as(outputs['voxels_rgb_t'])), 
        #                            outputs['image'].unsqueeze(2).expand_as(outputs['voxels_rgb_t']*(outputs['occupancy_field_t'].unsqueeze(1).expand_as(outputs['voxels_rgb_t']) > 0.5)), reduction='mean') 
        
        '''this is useful only when we do not rasterize'''
        #back_proj_loss = F.mse_loss(outputs['voxels_rgb_t'], 
        #                            outputs['image'].unsqueeze(2).repeat(1, 1, outputs['voxels_rgb_t'].shape[2], 1, 1), reduction='mean') 
        #                #F.l1_loss(outputs['rgb_field']*(outputs['shadow_field'].detach().expand_as(outputs['rgb_field'])), outputs['original_rgb_values'], reduction='mean')
        #regularization += back_proj_loss * confs['splatting_loss_weight'] 
        #wandb.log({'back_proj_loss': back_proj_loss})  

    if confs['depth_loss']:
        #depth_loss = F.l1_loss(outputs['depth_values'], outputs['estimated_depth_values'], reduction='mean')
        #regularization += depth_loss * confs['depth_loss_weight']
        #wandb.log({'depth_loss': depth_loss})

        #depth_img_loss = F.l1_loss(outputs['of_dpt'], outputs['depth_image'], reduction='mean')
        #regularization += depth_img_loss**exp * confs['depth_loss_weight'] 
        #wandb.log({'depth_img_loss': depth_img_loss})

        depth_img_e_loss = F.l1_loss(outputs['of_dpt_e'], outputs['depth_image'], reduction='mean')
        regularization += depth_img_e_loss**exp * confs['depth_loss_weight'] 
        wandb.log({'depth_img_e_loss': depth_img_e_loss})

        if depth_img_e_loss < 0.02 and phase >= 1:
            phase = 2

    if confs['soft_diff_loss']:
        soft_diff_loss = F.mse_loss(outputs['of_diff'], torch.zeros_like(outputs['of_diff']))
        regularization += soft_diff_loss * confs['soft_diff_loss_weight']
        wandb.log({'soft_diff_loss': soft_diff_loss})

    if confs['ray_opacity_loss']:
        ray_opacity_loss = F.l1_loss(outputs['ray_opacity'], torch.ones_like(outputs['ray_opacity']))
        regularization += ray_opacity_loss * confs['ray_opacity_loss_weight']
        wandb.log({'ray_opacity_loss': ray_opacity_loss})

    if confs['eikonal_loss']:
        gradient_norm = outputs['occupancy_field_grad'].norm(2, dim=1)
        eikonal_loss = ((gradient_norm - 1) ** 2).mean()
        regularization += eikonal_loss * confs['eikonal_loss_weight']
        wandb.log({'eikonal_loss': eikonal_loss})

    if confs['binary_cross_entropy']:
        binary_cross_entropy_loss = torch.mean((outputs['occupancy_field_t'] - 0)**2 * (outputs['occupancy_field_t'] - 1)**2)
        regularization += binary_cross_entropy_loss * confs['binary_cross_entropy_weight']  
        wandb.log({'binary_cross_entropy_loss': binary_cross_entropy_loss})

        #ray_opacity = outputs['occupancy_field_t'].sum(dim=1) / outputs['occupancy_field_t'].shape[1]
        #binary_cross_entropy_loss = torch.mean((ray_opacity - 0)**2 * (ray_opacity - 1)**2)
        #regularization += binary_cross_entropy_loss * confs['binary_cross_entropy_weight']  
        #wandb.log({'binary_cross_entropy_loss': binary_cross_entropy_loss})

        #batch_size, depth_res, height, width = outputs['occupancy_field_t'].shape
        #padded_tensor = outputs['occupancy_field_t'].unsqueeze(1)
        #kernel = torch.ones((1, 1, 3, 3, 3), device=outputs['occupancy_field_t'].device) 
        #ns = F.conv3d(padded_tensor, kernel, padding=1)/ 27
        #binary_cross_entropy_loss_f = torch.mean((ns - 0)**2 * (ns - 1)**2)
        #regularization += binary_cross_entropy_loss_f * confs['binary_cross_entropy_weight']  
        #wandb.log({'binary_cross_entropy_loss_f': binary_cross_entropy_loss_f})

        #binary_cross_entropy_loss_softmin = torch.mean((outputs['softmin'] - 0)**2 * (outputs['softmin'] - 1)**2)
        #regularization += binary_cross_entropy_loss_softmin * confs['binary_cross_entropy_weight']  
        #wandb.log({'binary_cross_entropy_loss_softmin': binary_cross_entropy_loss_softmin})

        #binary_cross_entropy_loss_ray_opacity = torch.mean((outputs['ray_opacity'] - 0)**2 * (outputs['ray_opacity'] - 1)**2)
        #regularization += binary_cross_entropy_loss_ray_opacity * confs['binary_cross_entropy_weight']  
        #wandb.log({'binary_cross_entropy_loss_ray_opacity': binary_cross_entropy_loss_ray_opacity})

        #binary_cross_entropy_loss_cao = torch.mean((outputs['cao'] - 0)**2 * (outputs['cao'] - 1)**2)
        #regularization += binary_cross_entropy_loss_cao * confs['binary_cross_entropy_weight']  
        #wandb.log({'binary_cross_entropy_loss_cao': binary_cross_entropy_loss_cao})

        #binary_cross_entropy_loss = torch.mean((outputs['cum_of'] - 0)**2 * (outputs['cum_of'] - 1)**2)
        #regularization += binary_cross_entropy_loss * confs['binary_cross_entropy_weight']  
        #wandb.log({'binary_cross_entropy_loss': binary_cross_entropy_loss})

    if confs['occupancy_loss']:
        occupancy_loss = F.mse_loss(outputs['occupancy_field'], torch.ones_like(outputs['occupancy_field']))
        regularization += occupancy_loss * confs['occupancy_loss_weight']
        wandb.log({'occupancy_loss': occupancy_loss})

    if confs['t_loss']:
        p_ratio = outputs['mask_cano_coo'].sum() / outputs['mask_cano_coo'].numel()
        if outputs['mask_cano_coo'].sum() != 0:
            occupancy_field_t_p = outputs['occupancy_field_t'][outputs['mask_cano_coo'].bool().squeeze(1).detach()]
            t_loss_p = F.l1_loss(occupancy_field_t_p, torch.ones_like(occupancy_field_t_p), reduction='mean')
        else:
            t_loss_p = 0.5
        
        if (1-outputs['mask_cano_coo']).sum() != 0:
            occupancy_field_t_n = outputs['occupancy_field_t'][~outputs['mask_cano_coo'].bool().squeeze(1).detach()]
            t_loss_n = F.l1_loss(occupancy_field_t_n, torch.zeros_like(occupancy_field_t_n), reduction='mean')
        else:
            t_loss_n = 0.5
        
        #t_loss = (1 - p_ratio) * t_loss_p + p_ratio * t_loss_n
        t_loss = t_loss_p + t_loss_n
        #t_loss = F.l1_loss(outputs['occupancy_field_t'], outputs['mask_cano_coo'].float(), reduction='mean')
        regularization += t_loss * confs['t_loss_weight']
        wandb.log({'t_loss': t_loss})

    if confs['t_loss_e']:
        p_ratio = outputs['mask_cano_coo'].sum() / outputs['mask_cano_coo'].numel()
        mask_cano_coo_e = outputs['mask_cano_coo'].squeeze(1) * (1 - outputs['outliers_mask'].unsqueeze(1).expand_as(outputs['mask_cano_coo'].squeeze(1)))
        #occupancy_field_t_p = outputs['occupancy_field_t'][mask_cano_coo_e.bool()]
        #occupancy_field_t_n = outputs['occupancy_field_t'][~mask_cano_coo_e.bool()]
        #t_loss_p = F.l1_loss(occupancy_field_t_p, torch.ones_like(occupancy_field_t_p), reduction='mean')
        #t_loss_n = F.l1_loss(occupancy_field_t_n, torch.zeros_like(occupancy_field_t_n), reduction='mean')
        #t_loss = (1 - p_ratio) * t_loss_p + p_ratio * t_loss_n
        #t_loss = t_loss_p + t_loss_n
        if outputs['mask_cano_coo'].sum() != 0:
            occupancy_field_t_p = outputs['occupancy_field_t'][mask_cano_coo_e.bool().squeeze(1).detach()]
            t_loss_p = F.l1_loss(occupancy_field_t_p, torch.ones_like(occupancy_field_t_p), reduction='mean')
        else:
            t_loss_p = 0.5
        
        if (1-outputs['mask_cano_coo']).sum() != 0:
            occupancy_field_t_n = outputs['occupancy_field_t'][~mask_cano_coo_e.bool().squeeze(1).detach()]
            t_loss_n = F.l1_loss(occupancy_field_t_n, torch.zeros_like(occupancy_field_t_n), reduction='mean')
        else:
            t_loss_n = 0.5
        
        #t_loss = (1 - p_ratio) * t_loss_p + p_ratio * t_loss_n
        t_loss = t_loss_p + t_loss_n
        #t_loss = F.l1_loss(outputs['occupancy_field_t'], mask_cano_coo_e, reduction='mean')
        incr_weight = confs['canonical_consistency_loss_weight'] * (math.exp(-outputs['epoch']/100))
        regularization += t_loss**exp * confs['t_loss_weight']
        wandb.log({'t_loss': t_loss})

    if confs['tv_loss']:
        tv_loss = tv_loss_3d(outputs['occupancy_field_t'])
        regularization += tv_loss * confs['tv_loss_weight']
        wandb.log({'tv_loss': tv_loss})

    # Add the regularization term to the original loss
    if confs['rendering_loss']:
        loss = rendering_loss + regularization
    else:
        loss = regularization
    
    return loss, phase

def tv_loss_3d(voxels):
    """
    Calculate Total Variation (TV) loss for a 3D voxel grid.
    
    Args:
        voxels (torch.Tensor): A 3D tensor of shape (batch_size, depth, height, width)
                               containing predicted occupancy values.

    Returns:
        torch.Tensor: The total variation loss.
    """
    # Difference along x-axis (depth direction)
    diff_x = torch.abs(voxels[:, 1:, :, :] - voxels[:, :-1, :, :])
    # Difference along y-axis (height direction)
    diff_y = torch.abs(voxels[:, :, 1:, :] - voxels[:, :, :-1, :])
    # Difference along z-axis (width direction)
    diff_z = torch.abs(voxels[:, :, :, 1:] - voxels[:, :, :, :-1])

    # Sum all differences
    loss = (torch.sum(diff_x) + torch.sum(diff_y) + torch.sum(diff_z)) / (voxels[0].numel()*3)

    return loss