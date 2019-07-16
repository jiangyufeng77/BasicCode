import torch.nn as nn
import torch
import numpy as np

class Generator(nn.Module):
    def __init__(self, latent_dim, channels):
        super(Generator, self).__init__()

        # first linear layer
        self.fc1 = nn.Linear(latent_dim, 384)
        # Transposed Convolution 2
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(384, 192, 4, 1, 0, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )
        # Transposed Convolution 3
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(192, 96, 4, 2, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(96, 48, 4, 2, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.tconv5 = nn.Sequential(
            nn.ConvTranspose2d(48, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise):
        noise = noise.view(-1, 100)
        fc1 = self.fc1(noise)
        fc1 = fc1.view(-1, 384, 1, 1)
        tconv2 = self.tconv2(fc1)
        tconv3 = self.tconv3(tconv2)
        tconv4 = self.tconv4(tconv3)
        tconv5 = self.tconv5(tconv4)
        output = tconv5
        return output


class Discriminator(nn.Module):
    def __init__(self, classes, channels):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(channels, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True)
        )

        self.adv = nn.Sequential(
            nn.Linear(512*4*4, 1),
            nn.Sigmoid()
        )

        self.cla = nn.Sequential(
            nn.Linear(512*4*4, classes+1),
            nn.Softmax()
        )

    def forward(self, image):
        output = self.model(image)
        output = output.view(output.size(0), -1) # output.size(0) = 128
        # print(output.shape)
        output_adv = self.adv(output)
        output_class = self.cla(output)
        return output_adv, output_class