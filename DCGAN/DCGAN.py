import argparse
import os
import torch
import sys

import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import datasets
# from tensorboardX import SummaryWriter
from networks import Generator, Discriminator
from logger import Logger

parse = argparse.ArgumentParser()
parse.add_argument('--batch_size', type=int, default=512, help='size of the batches')
parse.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parse.add_argument('--channels', type=int, default=3, help='number of image channels')
parse.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parse.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parse.add_argument('--n_classes', type=int, default=10, help='number of classes for dataset')
parse.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parse.add_argument('--img_size', type=int, default=64, help='size of each image dimension')
parse.add_argument('--n_epoch', type=int, default=200, help='number of epochs of training')
parse.add_argument('--n_row', type=int, default=10, help='the row of labels')
parse.add_argument('--gpu_ids', type=int, default=0, help='enables cuda')
parse.add_argument('--outf', type=str, default='images_LSUN', help='models are saved here')
parse.add_argument('--dataroot', type=str, default='/home/ouc/data/work/datasets/unzip/lsun', help='path to datasets')
parse.add_argument('--dataset', type=str, default='LSUN', help='choose the dataset, e.g. mnist, CIFAR10, imagenet, LSUN')
opt = parse.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

def save_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

save_path = os.path.join(opt.outf, 'checkpoints')
model_path = os.path.join(opt.outf, 'model')
# log_dir = os.path.join(opt.outf, 'log')
save_dir(save_path)
save_dir(model_path)
# save_dir(log_dir)

if opt.dataset =='mnist':
    train_loader = DataLoader(datasets.MNIST(opt.dataroot, train=True,
                                               transform=transforms.Compose([
                                                   transforms.Resize(opt.img_size),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                               ]), download=False),
                              batch_size=opt.batch_size, shuffle=True)
elif opt.dataset == 'CIFAR10':
    train_loader = DataLoader(datasets.CIFAR10(opt.dataroot, train=True,
                                       transform=transforms.Compose([
                                           transforms.Resize(opt.img_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                       ]), download=False),
                        batch_size=opt.batch_size, shuffle=True)
elif opt.dataset == 'imagenet':
    transform = transforms.Compose([
        transforms.Resize(opt.img_size),
        transforms.CenterCrop(opt.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    dataset = datasets.ImageFolder(opt.dataroot, transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

elif opt.dataset == 'LSUN':
    train_loader = DataLoader(datasets.LSUN(opt.dataroot, classes=['bedroom_train'],
                                            transform=transforms.Compose([
                                                transforms.Resize(opt.img_size),
                                                transforms.CenterCrop(opt.img_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])),
                              batch_size=opt.batch_size, shuffle=True)
# adversarial_loss = nn.MSELoss()
adversarial_loss = nn.BCELoss()

generator = Generator(opt.latent_dim, opt.channels)
discriminator = Discriminator(opt.channels)

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
    for i, (image, _)in enumerate(train_loader):

        batch_size = image.shape[0]

        real_image = Variable(image.type(FloatTensor))
        # print(real_image.shape)

        valid = Variable(FloatTensor(batch_size, 1, 1, 1).fill_(1.0), requires_grad = False)
        fake = Variable(FloatTensor(batch_size, 1, 1, 1).fill_(0.0), requires_grad = False)
        # print(valid.shape) #torch.size([512, 3, 1, 1])

        # Train Generator

        optimizer_G.zero_grad()

        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim, 1, 1))))
        # print(z)
        # print(z.shape)

        gen_img = generator(z)
        # print(gen_img.shape) # torch.size([512, 3, 64, 64])

        G_loss = adversarial_loss(discriminator(gen_img), valid)
        # print(discriminator(gen_img).shape)

        G_loss.backward()
        optimizer_G.step()

        # Train Discriminator

        optimizer_D.zero_grad()


        D_real = adversarial_loss(discriminator(real_image), valid)

        D_fake = adversarial_loss(discriminator(gen_img.detach()), fake)

        D_loss = D_real + D_fake

        D_loss.backward()
        optimizer_D.step()

        print("[Epoch: %d/%d] [Batch: %d/%d] [D loss: %.3f] [G loss: %.3f]"
                % (epoch, opt.n_epoch, i, len(train_loader), D_loss.item(), G_loss.item()))

    if epoch % 1 == 0:
        noise = Variable(FloatTensor(np.random.normal(0, 1, (opt.n_row ** 2, opt.latent_dim, 1, 1))))
        gen_image = generator(noise)
        gen_img = gen_image.view(gen_image.size(0), *img_shape)
        save_image(gen_img.data, '%s/%d_generated_image.png' % (save_path, epoch), nrow=opt.n_row, normalize=True)
        torch.save(generator.state_dict(), '%s/%d_generated_image.pkl' % (model_path, epoch))

        # info = {
        #     'G_loss': G_loss.item(),
        #     'D_loss': D_loss.item(),
        # }
        # for tag, value in info.items():
        #     logger.scalar_summary(tag, value, epoch+1)
        #
        # info = {
        #     'fake_images': gen_image.view(img_shape[0], 1, 28, 28)[:10].cpu().detach().numpy(),
        #     'real_images': image.view(img_shape[0], 1, 28, 28)[:10].cpu().numpy()
        # }
        # for tag, images in info.items():
        #     logger.image_summary(tag, images, epoch+1)