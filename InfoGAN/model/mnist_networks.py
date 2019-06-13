import torch.nn as nn
import torch
import numpy as np

class G_mnist(nn.Module):
    def __init__(self, n_classes, channels):
        super(G_mnist, self).__init__()

        self.label_emb = nn.Embedding(n_classes, n_classes)

        # self.model_linear = nn.Sequential(
        #     nn.Linear(74, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU()
        # )

        self.model_convT = nn.Sequential(
            nn.ConvTranspose2d(74, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),

            nn.ConvTranspose2d(1024, 128, kernel_size=7, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels, code):
        input = torch.cat((noise, self.label_emb(labels), code), -1)
        input = input.view(input.shape[0], 74, 1, 1)
        # output = self.model_linear(input)
        output = self.model_convT(input)

        return output


class D_mnist(nn.Module):
    def __init__(self, channels):
        super(D_mnist, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1, bias=False), # 28->14
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False), # 14->7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(128, 1024, kernel_size=7, stride=1, padding=0, bias=False), # 7->1
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, True)
        )

    def forward(self, image):
        output = self.model(image)
        return output

class D_net(nn.Module):
    def __init__(self, channels):
        super(D_net, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1024, channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        gan = self.model(x).view(-1, 1).squeeze(1)
        return gan

class Q_net(nn.Module):
    def __init__(self, n_classes, code_dim):
        super(Q_net, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1024, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.conv_class = nn.Sequential(
            nn.Conv2d(128, n_classes, 1),
            nn.Softmax()
        )
        self.conv_mu = nn.Sequential(
            nn.Conv2d(128, code_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        x = self.model(img)

        classes = self.conv_class(x)
        classes = classes.view(img.shape[0], 10)
        latent_code = self.conv_mu(x)
        latent_code = latent_code.view(img.shape[0], 2)

        return classes, latent_code