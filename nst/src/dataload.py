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

img_size = 256 if torch.cuda.is_available() else 128

