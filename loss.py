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
        activity_ocuupancy_b = outputs['activity_occupancy'].unsqueeze(0)
        #activity_ocuupancy_b = outputs['prev_ov'].unsqueeze(0).expand_as(outputs['occupancy_field'])
        canonical_consistency_loss = F.mse_loss(outputs['occupancy_field'], activity_ocuupancy_b, reduction='mean')
        #incr_weight = confs['canonical_consistency_loss_weight']*(math.cos(outputs['epoch']/20*math.pi) + 1)
        if outputs['epoch'] < 20:
            incr_weight = 0
        else:
            incr_weight = confs['canonical_consistency_loss_weight']
        regularization += canonical_consistency_loss * incr_weight
        wandb.log({'canonical_consistency_loss': canonical_consistency_loss})

        activity_ocuupancy_rgb_b = outputs['activity_occupancy_rgb'].unsqueeze(0)
        #activity_ocuupancy_rgb_b = outputs['prev_rgb'].unsqueeze(0).expand_as(outputs['rgb_field'])
        cc_rgb_loss = F.mse_loss(outputs['rgb_field'], activity_ocuupancy_rgb_b, reduction='mean')
        regularization += cc_rgb_loss * incr_weight
        wandb.log({'cc_rgb_loss': cc_rgb_loss})
    
    if confs['splatting_loss']:
        splatting_loss = F.mse_loss(outputs['rendered_rgb_values'], outputs['original_rgb_values'], reduction='mean') 
                        #F.l1_loss(outputs['rgb_field']*(outputs['shadow_field'].detach().expand_as(outputs['rgb_field'])), outputs['original_rgb_values'], reduction='mean')
        regularization += splatting_loss * confs['splatting_loss_weight']
        wandb.log({'splatting_loss': splatting_loss})

    if confs['segmentation_loss']:
        segmentation_loss = F.l1_loss(outputs['occupancy_field'], outputs['occupied_pixels_mask'], reduction='mean')
        regularization += segmentation_loss * confs['segmentation_loss_weight']
        wandb.log({'segmentation_loss': segmentation_loss})

    if confs['depth_loss']:
        depth_loss = F.l1_loss(outputs['depth_values'], outputs['estimated_depth_values'], reduction='mean')
        regularization += depth_loss * confs['depth_loss_weight']
        wandb.log({'depth_loss': depth_loss})

        depth_img_loss = F.l1_loss(outputs['of_dpt'], outputs['depth_image'], reduction='mean')
        regularization += depth_img_loss * confs['depth_loss_weight']
        wandb.log({'depth_img_loss': depth_img_loss})

    if confs['binary_cross_entropy']:
        binary_cross_entropy_loss = torch.mean((outputs['occupancy_field'] - 0)**2 * (outputs['occupancy_field'] - 1)**2)
        regularization += binary_cross_entropy_loss * confs['binary_cross_entropy_weight']  
        wandb.log({'binary_cross_entropy_loss': binary_cross_entropy_loss})

        #binary_cross_entropy_loss = torch.mean((outputs['activity_occupancy'] - 0)**2 * (outputs['activity_occupancy'] - 1)**2)
        #regularization += binary_cross_entropy_loss * confs['binary_cross_entropy_weight']  
        #wandb.log({'binary_cross_entropy_loss': binary_cross_entropy_loss})

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