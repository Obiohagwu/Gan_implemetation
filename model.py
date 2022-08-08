import torch 
import torch.nn as nn 
import torchvision 
from torch.utils.data import DataLoader 
from torch.utils.data import Dataset 
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np 
import math


class Generator(nn.Module):
    def __init__(self, dimension_z, image_dimension):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(dimension_z, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, image_dimension),
            nn.Tanh() # this normalizes inputs to range [-1,1] so output also be [-1, 1]
        )
    def forward(self, x):
        return self.generator(x)

class Discriminator(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.discriminator(x)

# Model Hyperparameters
lr =3e4 
epochs = 50
device = "cuda" if torch.cuda.is_available() else "cpu"
dimension_z = 64
image_dimension = 28*28*1
batch_size = 32

generator = Generator(dimension_z, image_dimension).to(device)
discriminator = Discriminator(image_dimension).to(device) 
#fixed_znoise = will add later. weird
#transforms = will add later. weird



def test():
    print("Crude Testing!")



if __name__ == "__main__":
    test()