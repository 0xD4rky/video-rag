import os
from PIL import Image
from torch.utils.data import Dataset

class Data(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        self.high_res_dir = os.path.join(self.root_dir, 'high_res')
        self.low_res_dir = os.path.join(self.root_dir, 'low_res')
        
        self.image_files = os.listdir(self.high_res_dir)
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        high_res_path = os.path.join(self.high_res_dir, img_name)
        low_res_path = os.path.join(self.low_res_dir, img_name)
        
        high_res_image = Image.open(high_res_path).convert("RGB")
        low_res_image = Image.open(low_res_path).convert("RGB")
        
        if self.transform:
            high_res_image = self.transform(high_res_image)
            low_res_image = self.transform(low_res_image)
            
        return low_res_image, high_res_image