import os
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class Data(Dataset):
    def __init__(self, root_dir, transform = None):
        
        self.root_dir = root_dir
        self.transform = transform
        
        self.high_res = os.path.join(self.root_dir, 'high_res')
        self.low_res = os.path.join(self.root_dir, 'low_res')
        
        if self.transform :
            self.high_res = self.transform(self.high_res)
            self.low_res = self.transform(self.low_res)
        
    def __len__(self):
        return len(self.high_res)
    
    def __getitem__(self,idx):
        
        high_res_path = os.path.join(self.high_res, self.high_res[idx])
        low_res_path = os.path.join(self.low_res, self.low_res[idx])
        
        high_res_images = Image.open(high_res_path).convert('RGB')
        low_res_images = Image.open(low_res_path).convert('RGB')
        
        if self.transform :
            high_res_images = self.transform(high_res_images)
            low_res_images = self.transform(low_res_images)
            
        