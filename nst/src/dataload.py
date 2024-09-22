import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision.models import vgg19, VGG19_Weights
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader

import os
import numpy as np
import matplotlib.pyplot as plt
import copy
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device) # setting default device as gpu


img_size = 256 if torch.cuda.is_available() else 128

loader = transforms.Compose([
    transforms.Resize((img_size,img_size)),
    transforms.ToTensor()
])

def img_loader(img_path):
    img = Image.open(img_path)
    img = loader(img).unsqueeze(0)
    return img.to(device, torch.float)


content = img_loader(r'/home/dell/Documents/cv_stack/nst/data/content.jpg')
art = img_loader(r'/home/dell/Documents/cv_stack/nst/data/art.jpeg')

assert art.size() == content.size() 
# assert is a debugging tool in python, raises an Assertion error if condition is not matched'

unloader = transforms.ToPILImage()
plt.ion()

def imshow(tensor, title = None):

    image = tensor.cpu().clone() # we clone the tensor to not do changes in it
    image = image.squeeze()
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(10)

# plt.figure()
# imshow(art, title = 'Style')
# plt.figure()
# imshow(content, title = 'Content')