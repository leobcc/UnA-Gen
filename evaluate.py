import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from lib.dataset import una_gen_dataset
from lib.loss import loss_c
from model import UnaGenModel
from utils import *  
import yaml
import time
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt
from torch import autograd
import wandb
from skimage.metrics import structural_similarity as ssim

def evaluate():
    confs_file = "/home/lbocchi/UnA-Gen/confs.yaml" 
    with open(confs_file, 'r') as file:
        confs = yaml.safe_load(file)

    wandb.init(
        project="una-gen-eval",
        config=confs
    )

    batch_size = confs['batch_size']

    transform = transforms.Compose([
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset = una_gen_dataset(data_dir="/home/lbocchi/UnA-Gen/data/data", split='train', frame_skip=confs['frame_skip'], image_size=(confs['image_dim'], confs['image_dim']), transform=None)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=confs['shuffle'], num_workers=confs['num_workers'])

    confs['model']['closeness_threshold'] = -4   # Enforce closeness threshold (negative values for k-nearest neighbors)
    confs['shuffle'] = False
    model = UnaGenModel(confs['model'], in_channels=3, features=confs['model']['features']) 
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('----------------------------------------------')
    print('Total parameters: {:.2f}M'.format(num_params / 1e6))
    print('Trainable parameters: {:.2f}M'.format(num_trainable_params / 1e6))
    print('----------------------------------------------')

    if os.path.exists('/home/lbocchi/outputs/last.pth'):
        print("Loading model from last checkpoint")
        model.load_state_dict(torch.load('/home/lbocchi/outputs/last_1.pth'))

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f'Model moved to {device}')

    model.eval()
    model.mapping_dim = 256
    model.matrix_mapping = model.initialize_matrix_mapping(model.mapping_dim, 0.5).to(device)
    with torch.no_grad():
        # Pass through the whole video to retain the activity occupancy canonical shape
        print("Starting evaluation...")
        print("Inferencing thorugh the whole video to output the canonical shape.")
        for batch_idx, inputs in enumerate(dataloader):
            print("ETA:", (len(dataloader)-batch_idx), "batches left", "|" , round(batch_idx/len(dataloader)*100), "%", "|", end="\r")
            inputs['masked_image'] = inputs['masked_image'].to(device)

            inputs['epoch'] = 0
            inputs['batch_idx'] = batch_idx
            inputs['num_samples'] = len(dataloader)
            outputs = model(inputs)

            # Generate dynamical mesh
            outputs_folder = os.path.join('outputs', 'eval', inputs['metadata']['subject'][0], 'canonical_from_frames')
            os.makedirs(outputs_folder, exist_ok=True)
            mesh = model.generate_mesh(outputs['dynamical_voxels_coo'], outputs['occupancy_field'], outputs['rgb_field'], outputs_folder, batch_idx, mode='cano')
            
        print("Generating canonical mesh ----------------")
        outputs_folder = os.path.join('outputs', 'eval', inputs['metadata']['subject'][0], 'canonical', 'activity_occupancy')
        os.makedirs(outputs_folder, exist_ok=True)
        mesh_t0 = time.time()
        #activity_occupancy_b = model.activity_occupancy.unsqueeze(0).expand_as(outputs['occupancy_field'])
        mesh = model.generate_mesh(None, model.activity_occupancy, model.activity_occupancy_rgb, outputs_folder, inputs['frame_id'], mode='ao_cano') 
        mesh_t1 = time.time()
        print(f'Mesh generation took {mesh_t1-mesh_t0} seconds')

        for batch_idx, inputs in enumerate(dataloader):
            t0 = time.time()

            outputs_folder = os.path.join('outputs', 'eval', inputs['metadata']['subject'][0])
            os.makedirs(outputs_folder, exist_ok=True)

            inputs['masked_image'] = inputs['masked_image'].to(device)

            inputs['epoch'] = 0
            inputs['batch_idx'] = batch_idx
            inputs['num_samples'] = len(dataloader)
            outputs = model(inputs)
        
            t1 = time.time()

            print("Evaluation time: ", t1-t0)
            wandb.log({"evaluation time": t1-t0})

            print("Generating mesh --------------------------")
            mesh_t0 = time.time()
            mesh = model.generate_mesh(outputs['dynamical_voxels_coo'], outputs['occupancy_field'], outputs['rgb_field'], outputs_folder, inputs['frame_id'])
            mesh_t1 = time.time()
            print(f'Mesh generation took {mesh_t1-mesh_t0} seconds')
            
            print('Rendering images --------------------------')
            ren_t0 = time.time()
            rendered_image = model.render_image(outputs['dynamical_voxels_coo'], outputs['occupancy_field'], outputs['rgb_field'], inputs['masked_image'], inputs['depth_image'])
            ren_t1 = time.time()
            print(f'Rendering took {ren_t1-ren_t0} seconds')
            output_path = os.path.join(outputs_folder, f'{inputs["frame_id"][0]}.png')
            save_image(rendered_image, output_path)
            wandb.log({"rendered_image": [wandb.Image(output_path)]})
            
            ssim_value = ssim(rendered_image[0].cpu().detach().numpy().transpose(1, 2, 0), inputs['masked_image'][0].cpu().detach().numpy().transpose(1, 2, 0), multichannel=True, channel_axis=2, data_range=1.0)
            wandb.log({"rendered_image": [wandb.Image(rendered_image[0].cpu().detach().numpy().transpose(1, 2, 0))]})
            wandb.log({"ssim": ssim_value})
                   

if __name__ == "__main__":
    evaluate()
