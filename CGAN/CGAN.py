import argparse
import os
import torch

import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import datasets
from tensorboardX import SummaryWriter
from networks import Generator, Discriminator


os.makedirs('images_CIFAR10', exist_ok=True)
os.makedirs('images_CIFAR10/checkpoints', exist_ok=True)
os.makedirs('images_CIFAR10/model', exist_ok=True)
os.makedirs('images_CIFAR10/test', exist_ok=True)
#
parse = argparse.ArgumentParser()
parse.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parse.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parse.add_argument('--channels', type=int, default=3, help='number of image channels')
parse.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parse.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parse.add_argument('--n_classes', type=int, default=10, help='number of classes for dataset')
parse.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parse.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parse.add_argument('--n_epoch', type=int, default=200, help='number of epochs of training')
parse.add_argument('--n_row', type=int, default=10, help='the now of labels')
parse.add_argument('--gpu_ids', type=int, default=0, help='enables cuda')
opt = parse.parse_args()
print(opt)

writer = SummaryWriter('images_CIFAR10/log')

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

os.makedirs('data/CIFAR10', exist_ok=True)
dataloader = DataLoader(datasets.CIFAR10('data/CIFAR10', train=True,
                                       transform=transforms.Compose([
                                           transforms.Resize(opt.img_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                       ]), download=True),
                        batch_size=opt.batch_size, shuffle=True)

adversarial_loss = nn.MSELoss()

generator = Generator(opt.n_classes, opt.latent_dim, img_shape)
discriminator = Discriminator(opt.n_classes, img_shape)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    torch.cuda.set_device(opt.gpu_ids)

print(generator, discriminator)

optimizer_G = torch.optim.Adam(generator.parameters(), lr = opt.lr, betas = (opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = opt.lr, betas = (opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


for epoch in range (opt.n_epoch):
    for i, (image, label) in enumerate(dataloader):

        batch_size = image.shape[0]

        real_image = Variable(image.type(FloatTensor))
        real_label = Variable(label.type(LongTensor))

        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad = False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad = False)

        # Train Generator

        optimizer_G.zero_grad()

        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

        gen_img = generator(z, gen_labels)

        G_loss = adversarial_loss(discriminator(gen_img, gen_labels), valid)

        G_loss.backward()
        optimizer_G.step()

        # Train Discriminator

        optimizer_D.zero_grad()

        dis_real = adversarial_loss(discriminator(real_image, real_label), valid)

        dis_fake = adversarial_loss(discriminator(gen_img.detach(), gen_labels), fake)

        D_loss = (dis_real + dis_fake) / 2

        D_loss.backward()
        optimizer_D.step()

        print("[Epoch: %d/%d] [Batch: %d/%d] [D loss: %.3f] [G loss: %.3f]"
              % (epoch, opt.n_epoch, i, len(dataloader), D_loss.item(), G_loss.item()))

    if epoch % 1 == 0:
        noise = Variable(FloatTensor(np.random.normal(0, 1, (opt.n_row ** 2, opt.latent_dim))))
        labels = np.array([num for _ in range(opt.n_row) for num in range(opt.n_row)])
        labels = Variable(LongTensor(labels))
        gen_image = generator(noise, labels)
        gen_img = gen_image.view(gen_image.size(0), *img_shape)
        save_image(gen_img.data, 'images_CIFAR10/checkpoints/%d_generated_image.png' % (epoch), nrow=opt.n_row, normalize=True)
        torch.save(generator.state_dict(), 'images_CIFAR10/model/%d_generated_image.pkl' % (epoch))
        writer.add_scalar('Train/G_loss', G_loss.data[0], epoch)
        writer.add_scalar('Train/D_loss', D_loss.data[0], epoch)