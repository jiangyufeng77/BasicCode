import torch.nn as nn
import torch
import numpy as np

class Generator(nn.Module):
    def __init__(self, n_classes, latent_dim, img_shapes):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(n_classes, n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + n_classes, 128, False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shapes))),
            nn.Tanh()
        )
    def forward(self, noise, label):
        gen_in = torch.cat((noise, self.label_emb(label)), -1)
        # print(noise.shape)
        # print(self.label_emb(label).shape)
        gen_img = self.model(gen_in)
        # print(gen_img.shape)
        # gen_img = gen_img.view(gen_img.size(0), *imgs_shape)
        # print(gen_img.shape)
        return gen_img


class Discriminator(nn.Module):
    def __init__(self, n_classes, img_shape):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(n_classes, n_classes)

        self.model1 = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.4),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.4)
        )

        self.model_CGAN = nn.Sequential(
            nn.Linear(512 + n_classes, 10),
            # nn.Linear(512, 10)
            nn.Softmax()
        )

        self.model_GAN = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, image, label):
        img_x = image.view(image.size(0), -1) # torch.size([64, 784])
        model1_output = self.model1(img_x) # torch.size([64, 512])

        dis_gan = self.model_GAN(model1_output)
        # print(dis_gan.shape)
        # print(model1_output.shape)

        img_cgan = torch.cat((model1_output, self.label_embedding(label)), -1)
        dis_cgan = self.model_CGAN(img_cgan)
        return dis_gan, dis_cgan