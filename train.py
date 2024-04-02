import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import una_gen_dataset
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

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f'Model moved to {device}')

    # Create a matrix mapping
    matrix_mapping = initialize_matrix_mapping(128, 0.5, device=device)
    print("matrix_mapping shape:", matrix_mapping.shape)
    print("matrix_mapping:", matrix_mapping)
    print("matrix_mapping number of ones:", torch.sum(matrix_mapping).item())

    for epoch in range(num_epochs):
        print(f'Training Epoch [{epoch+1}/{num_epochs}]')
        model.train()
        total_loss = 0

        for batch_idx, inputs in enumerate(dataloader):
            #inputs = inputs.to(device)   # inputs are a dictionary, they are move to cuda later

            outputs = model(inputs, matrix_mapping)
            # outputs projection

            loss = criterion(outputs, inputs)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 1 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

        print(f'Training Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}')

    torch.save(model.state_dict(), 'your_model.pth')

if __name__ == "__main__":
    train()
