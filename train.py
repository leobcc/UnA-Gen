import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import una_gen_dataset
from loss import loss_c
from model import UnaGenModel
from utils import *  
import yaml
import time
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt
from torch import autograd
import wandb

def train():
    confs_file = "/UnA-Gen/confs.yaml"
    with open(confs_file, 'r') as file:
        confs = yaml.safe_load(file)

    wandb.init(
        project="una-gen",

        config=confs
    )

    learning_rate = confs['learning_rate']
    batch_size = confs['batch_size']
    num_epochs = confs['num_epochs']
    continue_training = confs['continue_training']

    transform = transforms.Compose([
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset = una_gen_dataset(data_dir="/UnA-Gen/data/data", split='train', frame_skip=confs['frame_skip'], transform=None)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=confs['shuffle'], num_workers=confs['num_workers'])

    model = UnaGenModel(confs['model'], in_channels=3, features=128) 
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('----------------------------------------------')
    print('Total parameters: {:.2f}M'.format(num_params / 1e6))
    print('Trainable parameters: {:.2f}M'.format(num_trainable_params / 1e6))
    print('----------------------------------------------')

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if model.visualize_stats:
        print('Visualizing stats')
        os.makedirs('outputs/stats', exist_ok=True)

    if continue_training and os.path.exists('outputs/last.pth'):
        model.load_state_dict(torch.load('outputs/last.pth'))

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f'Model moved to {device}')

    torch.autograd.set_detect_anomaly(True)
    
    for epoch in range(num_epochs):
        print(f'Training Epoch [{epoch+1}/{num_epochs}]', end='\r')
        model.train()
        total_loss = 0

        for batch_idx, inputs in enumerate(dataloader):
            #inputs = inputs.to(device)   # inputs are a dictionary, they are moved to cuda later
            t0 = time.time()

            inputs['masked_image'] = inputs['masked_image'].to(device)

            if confs['model']['depth_refinement'] and epoch % confs['model']['refinement_epochs'] == 0:
                print(f'Refinement epoch: {epoch}, ETA: {batch_idx}/{len(dataloader)}', end='\r')
                model.refinement(inputs)

                opt_t0 = time.time()
                opt_t1 = time.time()
            else:
                outputs = model(inputs)
            
                loss = loss_c(confs['loss'], outputs)
                total_loss += loss.item()

                opt_t0 = time.time()
                loss.backward()
                #print("Gradient of scale:", model.scale.grad)
                #print("Gradient of standard depth n:", model.standard_depth_n.grad)
                optimizer.step()
                optimizer.zero_grad()
                opt_t1 = time.time()
            
                t1 = time.time()

                if model.visualize_stats:                    
                    wandb.log({"loss": loss.item()})
                    wandb.log({"batch_time": t1-t0})
                    wandb.log({"optimization_time": opt_t1-opt_t0})
                    print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}, Time: {t1-t0:.4f}', end='\r')

                    if confs['model']['render_full_image'] and epoch != 0 and epoch % confs['model']['render_full_image_every_n_epochs'] == 0:
                        with torch.no_grad():
                            print('Rendering images --------------------------')
                            ren_t0 = time.time()
                            rendered_image = model.render_image(outputs['dynamical_voxels_coo'], outputs['occupancy_field'], outputs['rgb_field'], inputs['masked_image'])
                            ren_t1 = time.time()
                            print(f'Rendering took {ren_t1-ren_t0} seconds')
                            
                            wandb.log({"rendered_image": [wandb.Image(rendered_image[0].cpu().detach().numpy().transpose(1, 2, 0))]})
            
        # On refinement epoch end: prune the matrix mapping
        if confs['model']['depth_refinement'] and epoch % confs['model']['refinement_epochs'] == 0:  
            print("model.mapping_prob_density max:", model.mapping_prob_density.max())
            print("model.mapping_prob_density min:", model.mapping_prob_density.min())
            model.mapping_prob_density = model.mapping_prob_density / (len(dataloader)*batch_size)
            print("model.mapping_prob_density max:", model.mapping_prob_density.max())
            print("model.mapping_prob_density min:", model.mapping_prob_density.min())
            threshold = 0.2
            model.matrix_mapping = (model.mapping_prob_density > threshold).float()

        torch.save(model.state_dict(), 'outputs/last.pth')

        average_loss = total_loss / len(dataloader)
        print(f'Training Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}')

    torch.save(model.state_dict(), 'outputs/last.pth')

if __name__ == "__main__":
    train()
