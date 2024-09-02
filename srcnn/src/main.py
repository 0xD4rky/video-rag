import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from model import *
from dataload import *
from fit import *

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataload import Data
from model import SRCNN
import wandb

os.environ["WANDB_API_KEY"] = "e8691a82f043359cea3223425001813d243c5a45"

def main():
    # Set up configuration
    config = Config()

    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = Data(root_dir=r'C:\Users\DELL\cv_stack\srcnn\data\train', transform=transform)
    val_dataset = Data(root_dir=r'C:\Users\DELL\cv_stack\srcnn\data\val', transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    # Create model
    model = SRCNN()

    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, config)

    # Start training
    trainer.train()

    print("Training completed!")

    # Optional: Load the best model and run evaluation
    trainer.load_model()
    val_loss, val_psnr = trainer.evaluate()
    print(f"Final Validation Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f}")

    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    main()