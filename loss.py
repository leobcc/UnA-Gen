import torch
import torch.nn.functional as F

def loss_c(confs, outputs):
    """
    Compute the loss given the outputs and the inputs.
    """
    rendering_loss = F.l1_loss(outputs['original_rgb_values'], outputs['rendered_rgb_values'])
    regularization = 0.0

    if confs['binary_cross_entropy']:
        binary_cross_entropy_loss = torch.mean((outputs['occupancy_field'] - 0)**2 * (outputs['occupancy_field'] - 1)**2)
        regularization += binary_cross_entropy_loss * confs['binary_cross_entropy_weight']

    # Add the regularization term to the original loss
    loss = rendering_loss + regularization
    
    return loss