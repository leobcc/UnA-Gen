import torch
import torch.nn.functional as F

def loss_c(rendered_image, original_image):
    """
    Compute the loss given the outputs and the inputs.
    """
    loss = F.mse_loss(rendered_image, original_image)
    
    return loss