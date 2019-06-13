import torch.nn as nn
import torch

class G_SVHN(nn.Module):
    def __init__(self, n_classes, channels):
        super(G_SVHN, self).__init__()

        self.label_emb = nn.Embedding(n_classes, n_classes)

        self.model = nn.Sequential(
            nn.ConvTranspose2d(168, 448, kernel_size=2, stride=1, bias=False), # 1->2
            nn.BatchNorm2d(448),
            nn.ReLU(),

            nn.ConvTranspose2d(448, 256, kernel_size=4, stride=2, padding=1, bias=False), # 2->4
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False), # 4->8
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False), # 8->16
            nn.ReLU(),

            nn.ConvTranspose2d(64, channels, kernel_size=4, stride=2, padding=1, bias=False), # 16->32
            nn.Tanh()
        )

    def forward(self, noise, label, code):
        input = torch.cat((noise, self.label_emb(label), code), -1)
        input = input.view(input.shape[0], 168, 1, 1)
        output = self.model(input)

        return output

class D_SVHN(nn.Module):
    def __init__(self, channels):
        super(D_SVHN, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1, bias=False), # 32->16
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False), # 16->8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False), # 8->4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, img):
        output = self.model(img)

        return output

class D_net(nn.Module):
    def __init__(self):
        super(D_net, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=4),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.model(x)

        return output

class Q_net(nn.Module):
    def __init__(self, n_classes, code_dim):
        super(Q_net, self).__init__()

        self.model_Q = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=4, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.class_label = nn.Sequential(
            nn.Conv2d(128, n_classes, kernel_size=1),
            nn.Softmax()
        )

        self.latent_code = nn.Sequential(
            nn.Conv2d(128, code_dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, img):
        output_Q = self.model_Q(img)

        classes = self.class_label(output_Q)
        classes = classes.view(img.shape[0], 40)
        latent_code = self.latent_code(output_Q)
        latent_code = latent_code.view(img.shape[0], 4)

        return classes, latent_code