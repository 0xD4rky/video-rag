import torch
import torch.nn as nn

from model import *
from dataload import *
from trainer import *


class SRCNN_config:
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_fn: str = 'mse'
    optimizer: str = 'adam'
    save_path: str = 'srcnn_model.pth'
    plot_interval: int = 1
    
config = SRCNN_config(
        batch_size=32,
        learning_rate=0.001,
        num_epochs=100,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        loss_fn='mse',
        optimizer='adam',
        save_path='srcnn_model.pth',
        plot_interval=5
    )


transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

train_dataset = Data(root_dir='path/to/data', split='train', transform=transform)
val_dataset = Data(root_dir='path/to/data', split='val', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

model = SRCNN()