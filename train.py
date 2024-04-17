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

def train():
    confs_file = "/UnA-Gen/confs.yaml"
    with open(confs_file, 'r') as file:
        confs = yaml.safe_load(file)
    learning_rate = confs['learning_rate']
    batch_size = confs['batch_size']
    num_epochs = confs['num_epochs']
    
    dataset = una_gen_dataset(data_dir="/UnA-Gen/data/data", split='train', transform=None)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = UnaGenModel(in_channels=3, features=128, mmap_dim=1024, mmap_res=256, num_classes=1) 

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f'Model moved to {device}')

    # Create a matrix mapping
    matrix_mapping = initialize_matrix_mapping(32, 0.5, device=device)

    for epoch in range(num_epochs):
        print(f'Training Epoch [{epoch+1}/{num_epochs}]')
        model.train()
        total_loss = 0

        for batch_idx, inputs in enumerate(dataloader):
            #inputs = inputs.to(device)   # inputs are a dictionary, they are moved to cuda later

            inputs['masked_image'] = inputs['masked_image'].to(device)

            outputs = model(inputs, matrix_mapping)

            loss = loss_c(outputs['rendered_image'], inputs['masked_image'])
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch_idx % 1 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

        print(f'Training Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}')

    torch.save(model.state_dict(), 'your_model.pth')

if __name__ == "__main__":
    train()
