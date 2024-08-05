import torch
import torch.nn.functional as F
import wandb
import math

def loss_c(confs, outputs):
    """
    Compute the loss given the outputs and the inputs.
    """
    if confs['rendering_loss']:   # Main loss term, training rays
        rendering_loss = F.l1_loss(outputs['rendered_rgb_values_add'], outputs['original_rgb_values_add'], reduction='mean')
        wandb.log({'rendering_loss': rendering_loss})
    
    regularization = 0.0

    if confs['canonical_consistency_loss']:
        activity_occupancy_b = outputs['activity_occupancy'].unsqueeze(0).expand_as(outputs['occupancy_field'])
        #activity_ocuupancy_b = outputs['prev_ov'].unsqueeze(0).expand_as(outputs['occupancy_field'])
        #canonical_consistency_loss = F.mse_loss(outputs['occupancy_field'], activity_ocuupancy_b, reduction='mean')
        #incr_weight = confs['canonical_consistency_loss_weight']*(math.cos(outputs['epoch']/20*math.pi) + 1)
        incr_weight = confs['canonical_consistency_loss_weight'] * (1 - math.exp(-outputs['epoch']/100))
        #regularization += canonical_consistency_loss * incr_weight
        #wandb.log({'canonical_consistency_loss': canonical_consistency_loss})

        activity_occupancy_rgb_b = outputs['activity_occupancy_rgb'].unsqueeze(0).expand_as(outputs['rgb_field'])
        #activity_occupancy_rgb_b = outputs['prev_rgb'].unsqueeze(0).expand_as(outputs['rgb_field'])
        #cc_rgb_loss = F.mse_loss(outputs['rgb_field'], activity_occupancy_rgb_b, reduction='mean')
        #regularization += cc_rgb_loss * incr_weight
        #wandb.log({'cc_rgb_loss': cc_rgb_loss})

        n_voxels = min(outputs['occupancy_field'].shape[1] * (outputs['epoch']+1) // 10, outputs['occupancy_field'].shape[1])
        random_indices = torch.randint(0, outputs['occupancy_field'].shape[1], (outputs['occupancy_field'].shape[0], n_voxels), device=outputs['occupancy_field'].device)
        ao_mask_p = (activity_occupancy_b > 0.9).squeeze(-1)
        ao_mask_n = (activity_occupancy_b < 0.1).squeeze(-1)
        ao_mask = ao_mask_p | ao_mask_n
        if ao_mask.sum() != 0:
            cc_of_weak_loss = 0
            cc_rgb_weak_loss = 0
            if ao_mask_p.sum() != 0:
                cc_of_weak_loss =+ F.mse_loss(outputs['occupancy_field'][ao_mask_p], 
                                            torch.ones_like(outputs['occupancy_field'][ao_mask_p]), reduction='mean') 
            if ao_mask_n.sum() != 0:
                cc_of_weak_loss =+ F.mse_loss(outputs['occupancy_field'][ao_mask_n], 
                                            torch.zeros_like(outputs['occupancy_field'][ao_mask_n]), reduction='mean')
            if outputs['epoch'] > 0:   # Every 10 epochs, the loss is shut off to allow the model to explore other solutions
                regularization += cc_of_weak_loss * incr_weight 
                wandb.log({'cc_of_weak_loss': cc_of_weak_loss})

            if ao_mask_p.sum() != 0:
                cc_rgb_weak_loss = F.mse_loss(outputs['rgb_field'][ao_mask_p], 
                                            activity_occupancy_rgb_b[ao_mask_p], reduction='mean')
            if outputs['epoch'] > 0:   # Every 10 epochs, the loss is shut off to allow the model to explore other solutions
                regularization += cc_rgb_weak_loss * incr_weight
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

        rasterize_loss = F.mse_loss(outputs['rasterized_image'], outputs['image'], reduction='mean') 
                        #F.l1_loss(outputs['rgb_field']*(outputs['shadow_field'].detach().expand_as(outputs['rgb_field'])), outputs['original_rgb_values'], reduction='mean')
        regularization += rasterize_loss * confs['splatting_loss_weight']
        wandb.log({'rasterize_loss': rasterize_loss})
        
        #back_proj_loss = F.mse_loss(outputs['rgb_field_t']*(outputs['occupancy_field_t'].unsqueeze(1).expand_as(outputs['rgb_field_t'])), outputs['image'].unsqueeze(2).expand_as(outputs['rgb_field_t']), reduction='mean') 
        #                #F.l1_loss(outputs['rgb_field']*(outputs['shadow_field'].detach().expand_as(outputs['rgb_field'])), outputs['original_rgb_values'], reduction='mean')
        #regularization += back_proj_loss * confs['splatting_loss_weight']
        #wandb.log({'back_proj_loss': back_proj_loss})  

        #back_proj_loss = F.mse_loss(outputs['voxels_rgb_t']*(outputs['occupancy_field_t'].unsqueeze(1).expand_as(outputs['voxels_rgb_t'])), outputs['image'].unsqueeze(2).expand_as(outputs['voxels_rgb_t']), reduction='mean') 
        back_proj_loss = F.mse_loss(outputs['voxels_rgb_t']*(outputs['occupancy_field_t'].unsqueeze(1).expand_as(outputs['voxels_rgb_t'])), 
                                    outputs['image'].unsqueeze(2).expand_as(outputs['voxels_rgb_t']*(outputs['occupancy_field_t'].unsqueeze(1).expand_as(outputs['voxels_rgb_t']) > 0.5)), reduction='mean') 
                        #F.l1_loss(outputs['rgb_field']*(outputs['shadow_field'].detach().expand_as(outputs['rgb_field'])), outputs['original_rgb_values'], reduction='mean')
        regularization += back_proj_loss * confs['splatting_loss_weight'] 
        wandb.log({'back_proj_loss': back_proj_loss})  

    if confs['segmentation_loss']:
        #segmentation_loss = F.l1_loss(outputs['occupancy_field'], outputs['occupied_pixels_mask'], reduction='mean')
        #regularization += segmentation_loss * confs['segmentation_loss_weight']
        #wandb.log({'segmentation_loss': segmentation_loss})
        
        ray_opacity = outputs['occupancy_field_t'].sum(dim=1) / outputs['occupancy_field_t'].shape[1]
        segmentation_loss = F.l1_loss(ray_opacity, (1 - outputs['outliers_mask']), reduction='mean')
        regularization += segmentation_loss * confs['segmentation_loss_weight']
        wandb.log({'segmentation_loss': segmentation_loss})

    if confs['depth_loss']:
        #depth_loss = F.l1_loss(outputs['depth_values'], outputs['estimated_depth_values'], reduction='mean')
        #regularization += depth_loss * confs['depth_loss_weight']
        #wandb.log({'depth_loss': depth_loss})

        depth_img_loss = F.l1_loss(outputs['of_dpt'], outputs['depth_image'], reduction='mean')
        regularization += depth_img_loss * confs['depth_loss_weight']
        wandb.log({'depth_img_loss': depth_img_loss})

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
        #binary_cross_entropy_loss = torch.mean((ns - 0)**2 * (ns - 1)**2)
        #regularization += binary_cross_entropy_loss * confs['binary_cross_entropy_weight']  
        #wandb.log({'binary_cross_entropy_loss': binary_cross_entropy_loss})

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

    # Add the regularization term to the original loss
    if confs['rendering_loss']:
        loss = rendering_loss + regularization
    else:
        loss = regularization
    
    return loss