import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from lib.dataset import una_gen_dataset
from lib.loss import loss_c
from model import UnaGenModel
from lib.utils import *  
import yaml
import time
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt
from torch import autograd
import wandb
from skimage.metrics import structural_similarity as ssim

def train():
    confs_file = "/home/lbocchi/UnA-Gen/confs.yaml"
    with open(confs_file, 'r') as file:
        confs = yaml.safe_load(file)

    os.environ['WANDB_DIR'] = '/home/lbocchi/UnA-Gen/data/wandb'
    wandb.require("core")
    wandb.init(
        project="una-gen",
        config=confs,
        dir='/home/lbocchi/UnA-Gen/data/wandb'
    )

    learning_rate = confs['learning_rate']
    batch_size = confs['batch_size']
    num_epochs = confs['num_epochs']
    continue_training = confs['continue_training']

    transform = transforms.Compose([
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset = una_gen_dataset(data_dir="/home/lbocchi/UnA-Gen/data/data", split='train', frame_skip=confs['frame_skip'], image_size=(confs['image_dim'], confs['image_dim']), transform=None)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=confs['shuffle'], num_workers=confs['num_workers'])

    os.makedirs('outputs', exist_ok=True)

    model = UnaGenModel(confs['model'], in_channels=3, features=confs['model']['features']) 
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('----------------------------------------------')
    print('Total parameters: {:.2f}M'.format(num_params / 1e6))
    print('Trainable parameters: {:.2f}M'.format(num_trainable_params / 1e6))
    print('----------------------------------------------')

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if continue_training and os.path.exists('outputs/last.pth'):
        model.load_state_dict(torch.load('outputs/last.pth'))

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f'Model moved to {device}')
    
    for epoch in range(num_epochs):
        print(f'Training Epoch [{epoch+1}/{num_epochs}]', end='\r')
        model.train()
        total_loss = 0
        phase = 0

        for batch_idx, inputs in enumerate(dataloader):
            t0 = time.time()

            inputs['masked_image'] = inputs['masked_image'].to(device)

            inputs['epoch'] = epoch
            inputs['batch_idx'] = batch_idx
            inputs['num_samples'] = len(dataloader)
            outputs = model(inputs)

            loss, phase = loss_c(confs['loss'], outputs, phase=phase)
            total_loss += loss.item()

            opt_t0 = time.time()
            loss.backward()
            #print("Gradient of scale:", model.scale.grad)
            #print("Gradient of standard depth n:", model.standard_depth_n.grad)
            optimizer.step()
            optimizer.zero_grad()
            opt_t1 = time.time()
        
            t1 = time.time()

            wandb.log({"loss": loss.item()})
            wandb.log({"batch_time": t1-t0})
            wandb.log({"optimization_time": opt_t1-opt_t0})
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}, Time: {t1-t0:.4f}', end='\r')

            if confs['model']['render_full_image'] and epoch != 0 and epoch % confs['model']['render_full_image_every_n_epochs'] == 0 and batch_idx == 0:
                with torch.no_grad():
                    print("Generating mesh --------------------------")
                    mesh_t0 = time.time()
                    mesh = model.generate_mesh(outputs['dynamical_voxels_coo'], outputs['occupancy_field'], outputs['rgb_field'])
                    mesh_t1 = time.time()
                    print(f'Mesh generation took {mesh_t1-mesh_t0} seconds')
                    
                    print('Rendering images --------------------------')
                    ren_t0 = time.time()
                    rendered_image = model.render_image(outputs['dynamical_voxels_coo'], outputs['occupancy_field'], outputs['rgb_field'], inputs['masked_image'], inputs['depth_image'])
                    ren_t1 = time.time()
                    print(f'Rendering took {ren_t1-ren_t0} seconds')
                    
                    ssim_value = ssim(rendered_image[0].cpu().detach().numpy().transpose(1, 2, 0), inputs['masked_image'][0].cpu().detach().numpy().transpose(1, 2, 0), multichannel=True)
                    wandb.log({"rendered_image": [wandb.Image(rendered_image[0].cpu().detach().numpy().transpose(1, 2, 0))]})
                    wandb.log({"ssim": ssim_value})

        if phase == 2:
            model.opt['phase_push'] = True

        if (epoch+1) % 10 == 0:
            try:
                outputs_folder = os.path.join('outputs', 'train', inputs['metadata']['subject'][0], 'cano')
                os.makedirs(outputs_folder, exist_ok=True)
                mesh = model.generate_mesh(outputs['dynamical_voxels_coo'], outputs['occupancy_field'], outputs['rgb_field'], outputs_folder, epoch, mode='cano')
                
                outputs_folder = os.path.join('outputs', 'train', inputs['metadata']['subject'][0], 'ao_cano')
                os.makedirs(outputs_folder, exist_ok=True)
                mesh = model.generate_mesh(None, model.activity_occupancy, model.activity_occupancy_rgb, outputs_folder, epoch, mode='ao_cano')
                
                outputs_folder = os.path.join('outputs', 'train', inputs['metadata']['subject'][0], 'dynamical_ao')
                os.makedirs(outputs_folder, exist_ok=True)
                mesh = model.generate_mesh(None, model.activity_occupancy, model.activity_occupancy_rgb, outputs_folder, epoch, mode='dynamical_ao')
                
                outputs_folder = os.path.join('outputs', 'train', inputs['metadata']['subject'][0], 'dynamical')
                os.makedirs(outputs_folder, exist_ok=True)
                mesh = model.generate_mesh(outputs['dynamical_voxels_coo'], outputs['occupancy_field_t'], outputs['rgb_field_t'], outputs_folder, epoch, mode='dynamical_pc') 
            except:
                print('Error generatiing mesh')
        torch.save(model.state_dict(), 'outputs/last.pth')

        average_loss = total_loss / len(dataloader)
        print(f'Training Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}')

    torch.save(model.state_dict(), 'outputs/last.pth')

if __name__ == "__main__":
    train()
