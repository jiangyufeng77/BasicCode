import torch.nn as nn
import torch
import numpy as np


class Generator_128(nn.Module):
    def __init__(self, n_classes, latent_dim, channels, ngpu):
        super(Generator_128, self).__init__()
        self.ngpu = ngpu
        # self.emb = nn.Embedding(n_classes, n_classes)
        self.fc = nn.Linear(latent_dim, 768) # the origin: 784
        # self.relu = nn.ReLU()
        self.convT1 = nn.Sequential(
            nn.ConvTranspose2d(768, 384, 5, 2, 0, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(True),
        )

        self.convT2 = nn.Sequential(
            nn.ConvTranspose2d(384, 256, 5, 2, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        self.convT3 = nn.Sequential(
            nn.ConvTranspose2d(256, 192, 5, 2, 0, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )

        self.convT4 = nn.Sequential(
            nn.ConvTranspose2d(192, 64, 5, 2, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.convT5 = nn.Sequential(
            nn.ConvTranspose2d(64, channels, 8, 2, 0, bias=False),
            nn.Tanh()
         )

    def forward(self, noise):
        # input = torch.cat((self.emb(labels), noise), -1)
        if isinstance(noise.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            input = noise.view(-1, 110)
            fc = nn.parallel.data_parallel(self.fc, input, range(self.ngpu))
            fc = fc.view(-1, 768, 1, 1)
            convT1 = nn.parallel.data_parallel(self.convT1, fc, range(self.ngpu))
            convT2 = nn.parallel.data_parallel(self.convT2, convT1, range(self.ngpu))
            convT3 = nn.parallel.data_parallel(self.convT3, convT2, range(self.ngpu))
            convT4 = nn.parallel.data_parallel(self.convT4, convT3, range(self.ngpu))
            convT5 = nn.parallel.data_parallel(self.convT5, convT4, range(self.ngpu))
            output = convT5
        else:
            input = noise.view(-1, 110)
            fc = self.fc(input)
            fc = fc.view(-1, 768, 1, 1)
            convT1 = self.convT1(fc)
            convT2 = self.convT2(convT1)
            convT3 = self.convT3(convT2)
            convT4 = self.convT4(convT3)
            convT5 = self.convT5(convT4)
            output = convT5
        return output


class Discriminator_128(nn.Module):
    def __init__(self, channels, ngpu):
        super(Discriminator_128, self).__init__()
        self.ngpu = ngpu

        self.convT1 = nn.Sequential(
            nn.Conv2d(channels, 16, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )

        self.convT2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )

        self.convT3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )

        self.convT4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )

        self.convT5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )

        self.convT6 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )

        self.model_gan = nn.Sequential(
            nn.Linear(512 * 13 * 13, 1, bias=False),
            nn.Sigmoid()
        )

        self.model_label = nn.Sequential(
            nn.Linear(512 * 13 * 13, 10, bias=False),
            nn.Softmax()
        )

    def forward(self, img):
        if isinstance(img.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            convT1 = nn.parallel.data_parallel(self.convT1, img, range(self.ngpu))
            convT2 = nn.parallel.data_parallel(self.convT2, convT1, range(self.ngpu))
            convT3 = nn.parallel.data_parallel(self.convT3, convT2, range(self.ngpu))
            convT4 = nn.parallel.data_parallel(self.convT4, convT3, range(self.ngpu))
            convT5 = nn.parallel.data_parallel(self.convT5, convT4, range(self.ngpu))
            convT6 = nn.parallel.data_parallel(self.convT6, convT5, range(self.ngpu))
            output = convT6.view(-1, 13*13*512)
            output_gan = nn.parallel.data_parallel(self.model_gan, output, range(self.ngpu))
            output_label = nn.parallel.data_parallel(self.model_label, output, range(self.ngpu))

        else:
            convT1 = self.convT1(img)
            convT2 = self.convT2(convT1)
            convT3 = self.convT3(convT2)
            convT4 = self.convT4(convT3)
            convT5 = self.convT5(convT4)
            convT6 = self.convT6(convT5)
            output = convT6.view(-1, 13*13*512)
            output_gan = self.model_gan(output).view(-1, 1).squeeze(1)
            output_label = self.model_label(output)

        return output_gan, output_label

class Generator(nn.Module):
    def __init__(self, n_classes, latent_dim, channels):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(n_classes, n_classes)

        self.fc = nn.Linear(latent_dim+n_classes, 384)

        self.convT1 = nn.Sequential(
            nn.ConvTranspose2d(384, 192, kernel_size=5, stride=2, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(192)
        )

        self.convT2 = nn.Sequential(
            nn.ConvTranspose2d(192, 96, kernel_size=5, stride=2, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(96)
        )

        self.convT3 = nn.Sequential(
            nn.ConvTranspose2d(96, channels, kernel_size=8, stride=2, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, label):
        gen_in = torch.cat((noise, self.label_emb(label)), -1)
        # print(noise.shape)
        # print(self.label_emb(label).shape)
        output = self.fc(gen_in)
        output = output.view(output.size(0), -1, 1, 1)
        output = self.convT1(output)
        output = self.convT2(output)
        output = self.convT3(output)
        # print(gen_img.shape)
        # gen_img = gen_img.view(gen_img.size(0), *imgs_shape)
        # print(gen_img.shape)
        return output


class Discriminator(nn.Module):
    def __init__(self, channels):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2, True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Dropout(0.5),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Dropout(0.5),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Dropout(0.5),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Dropout(0.5),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Dropout(0.5),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True)
        )

        self.model_CGAN = nn.Sequential(
            nn.Linear(512*4*4, 10),
            # nn.Linear(512, 10)
            nn.Softmax()
        )

        self.model_GAN = nn.Sequential(
            nn.Linear(512*4*4, 1),
            nn.Sigmoid()
        )

    def forward(self, image):
        # img_x = image.view(image.size(0), -1, ) # torch.size([64, 784])
        output = self.conv1(image) # torch.size([64, 512])
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        model1_output = self.conv6(output)

        dis_gan = self.model_GAN(model1_output.view(model1_output.size(0), -1))
        # print(dis_gan.shape)
        # print(model1_output.shape)
        dis_cgan = self.model_CGAN(model1_output.view(model1_output.size(0), -1))
        return dis_gan, dis_cgan

