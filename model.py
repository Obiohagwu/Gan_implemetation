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
fixed_znoise = torch.randn((batch_size, dimension_z)).to(device)
transfomation = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),]
)

generator_optimizer = optim.Adam(discriminator.parameters(), lr=lr)
discriminator_optimizer = optim.Adam(generator.parameters(), lr=lr)
dataset = datasets.MNIST(root="data", download=True, transform=transformations)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
loss_criterion = nn.BCELoss()

iteration = 0
for epoch in range(epochs):
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        #Now we train the discriminator. Refering to paper
        # We want to get the max over this distribution log(D(x)) + log(1-D(G(Z)))
        admul_noise = torch.randn(batch_size, dimension_z).to(device)
        forged = generator(admul_noise)
        ground_discriminator = discriminator(real).view(-1)
        ground_discriminatorLoss = loss_criterion(ground_discriminator, torch.ones_like(ground_discriminator))
        discriminator_forged = discriminator(forged).view(-1)
        



def test():
    print("Crude Testing!")



if __name__ == "__main__":
    test()