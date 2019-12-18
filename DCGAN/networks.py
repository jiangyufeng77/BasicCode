import torch.nn as nn
import os
import torch
import numpy as np
from knn import dense_knn_matrix

class Generator(nn.Module):
    def __init__(self, latent_dim, channels):
        super(Generator, self).__init__()
        self.convT1 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 1024, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024, 0.8),
            nn.ReLU()
        )

        self.convT2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512, 0.8),
            nn.ReLU(inplace=True)
        )

        self.convT3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256, 0.8),
            nn.ReLU(inplace=True)
        )

        self.convT4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128, 0.8),
            nn.ReLU(inplace=True)
        )

        self.convT5 = nn.Sequential(
            nn.ConvTranspose2d(128, channels, 3, 1, 1, bias=False),
            nn.Tanh()
        )


    def forward(self, inputs):
        output1 = self.convT1(inputs)
        output2 = self.convT2(output1)
        output3 = self.convT3(output2)
        output4 = self.convT4(output3)
        output5 = self.convT5(output4)
        return output5


class Discriminator(nn.Module):
    def __init__(self, channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # channels * 64 * 64->64*32*32
            nn.Conv2d(channels, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 64*32*32->128*16*16
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            # 128*16*16->256*8*8
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            # 256*8*8->512*4*4
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            # 512 * 4 * 4 -> 1*1*1
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(opt.channels, 0.8),
            nn.Sigmoid()
        )


    def forward(self, image):

        output = self.model(image)
        output = output.view(-1, 1).squeeze(1)
        return output
