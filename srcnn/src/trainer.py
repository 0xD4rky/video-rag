import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt



class Trainer:
    
    def __init__(self, model, train_loader, val_loader, config: SRCNN_config):
        
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.criterion = self._get_loss_fn()
        self.optimizer = self._get_optimizer()
        
        self.train_losses = []
        self.val_losses = []
        self.train_psnrs = []
        self.val_psnrs = []
        
    
    def _get_loss_fn(self):
        if self.config.loss_fn.lower() == 'mse':
            return nn.MSELoss()
        elif self.config.loss_fn.lower() == 'l1':
            return nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss function: {self.config.loss_fn}")

    def _get_optimizer(self):
        if self.config.optimizer.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        elif self.config.optimizer.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.config.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
        
    
    def train(self):
        
        for epoch in range(self.config.num_epochs):
            self.model.train()
            train_loss = 0
            train_psnr = 0
            
            with tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}", unit="batch") as tepoch:
                
                for lr_imgs, hr_imgs in tepoch:
                    lr_imgs, hr_imgs = lr_imgs.to(self.config.device), hr_imgs.to(self.config.device)
                    
                    self.optimizer.zero_grad()
                    outputs = self.model(lr_imgs)
                    loss = self.criterion(outputs, hr_imgs)
                    loss.backward()
                    self.optimizer.step()
                    
                    train_loss += loss.item()
                    train_psnr += self.calculate_psnr(outputs, hr_imgs)
                    
                    tepoch.set_postfix(loss=loss.item())
                    
            avg_train_loss = train_loss / len(self.train_loader)
            avg_train_psnr = train_psnr / len(self.train_loader)
            
            self.train_losses.append(avg_train_loss)
            self.train_psnrs.append(avg_train_psnr)
            
            val_loss, val_psnr = self.evaluate()
            self.val_losses.append(val_loss)
            self.val_psnrs.append(val_psnr)
            
            print(f"Epoch {epoch+1}/{self.config.num_epochs}")
            print(f"Train Loss: {avg_train_loss:.4f}, Train PSNR: {avg_train_psnr:.2f}")
            print(f"Val Loss: {val_loss:.4f}, Val PSNR: {val_psnr:.2f}")
            
            if (epoch + 1) % self.config.plot_interval == 0:
                self.plot_results()
                
        self.save_model()
    
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        val_loss = 0
        val_psnr = 0
        
        with torch.no_grad():
            for lr_imgs, hr_imgs in self.val_loader:
                lr_imgs, hr_imgs = lr_imgs.to(self.config.device), hr_imgs.to(self.config.device)
                outputs = self.model(lr_imgs)
                loss = self.criterion(outputs, hr_imgs)
                val_loss += loss.item()
                val_psnr += self.calculate_psnr(outputs, hr_imgs)
        
        return val_loss / len(self.val_loader), val_psnr / len(self.val_loader)
        
    @staticmethod
    def calculate_psnr(img1,img2):
        mse = nn.MSE()(img1,img2)
        return 10*torch.log10(1/mse).item()
    
    def plot_results(self):
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.legend()
        plt.title('Loss')
        
        plt.subplot(2, 2, 2)
        plt.plot(self.train_psnrs, label='Train PSNR')
        plt.plot(self.val_psnrs, label='Val PSNR')
        plt.legend()
        plt.title('PSNR')
        
        plt.subplot(2, 2, 3)
        lr_img = next(iter(self.val_loader))[0][0].cpu()
        sr_img = self.model(lr_img.unsqueeze(0).to(self.config.device)).squeeze(0).cpu()
        hr_img = next(iter(self.val_loader))[1][0].cpu()
        img_grid = make_grid([lr_img, sr_img, hr_img], nrow=3)
        plt.imshow(img_grid.permute(1, 2, 0))
        plt.title('LR / SR / HR')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('training_results.png')
        plt.close()
    
    def save_model(self):
        torch.save(self.model.state_dict(), self.config.save_path)
    
    def load_model(self):
        self.model.load_state_dict(torch.load(self.config.save_path))