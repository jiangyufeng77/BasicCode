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

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.00005, help='adam: learning rate')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_classes', type=int, default=10, help='number of classes for dataset')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=64, help='size of each image dimension')
parser.add_argument('--n_epoch', type=int, default=200, help='number of epochs of training')
parser.add_argument('--n_row', type=int, default=10, help='the row of labels')
parser.add_argument('--gpu_ids', type=int, default=0, help='enables cuda')
parser.add_argument('--outf', type=str, default='images_CIFAR10_noBN_detach', help='models are saved here')
parser.add_argument('--dataroot', type=str, default='../../data/CIFAR10', help='path to datasets')
parser.add_argument('--dataset', type=str, default='CIFAR10', help='choose the dataset, e.g. mnist, CIFAR10, imagenet, LSUN')
parser.add_argument("--n", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip", type=float, default=0.01, help="lower and upper clip value for disc. weights")
opt = parser.parse_args()
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

generator = Generator(opt.latent_dim, opt.channels)
discriminator = Discriminator(opt.channels)

one = torch.cuda.FloatTensor([1])
mone = torch.cuda.FloatTensor(one * -1)

if cuda:
    generator.cuda()
    discriminator.cuda()
    # one.cuda()
    # mone.cuda()
    torch.cuda.set_device(opt.gpu_ids)

print(generator, discriminator)

optimizer_G = torch.optim.RMSprop(generator.parameters(), lr = opt.lr)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr = opt.lr)

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


for epoch in range (opt.n_epoch):
    for i, (image, _)in enumerate(train_loader):

        batch_size = image.shape[0]

        real_image = Variable(image.type(FloatTensor))
        # print(real_image.shape)

        # Train Discriminator

        for p in discriminator.parameters():
            p.data.clamp_(-opt.clip, opt.clip)

        optimizer_D.zero_grad()
        # D_real = torch.mean(discriminator(real_image))
        D_real = discriminator(real_image)
        D_real.backward(one)

        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim, 1, 1))))
        gen_img = generator(z)
        # D_fake = torch.mean(discriminator(gen_img))
        D_fake = discriminator(gen_img.detach())
        D_fake.backward(mone) # mone

        D_loss = D_real - D_fake

        optimizer_D.step()

        # Train Generator
        if i % opt.n == 0:
            optimizer_G.zero_grad()

            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim, 1, 1))))
            # print(z)
            # print(z.shape)

            gen_img = generator(z)
            # print(gen_img.shape) # torch.size([512, 3, 64, 64])

            # G_loss = torch.mean(discriminator(gen_img))
            G_loss = discriminator(gen_img)
            # print(discriminator(gen_img).shape)

            G_loss.backward(one)
            optimizer_G.step()

            print("[Epoch: %d/%d] [Batch: %d/%d] [D loss: %.3f] [G loss: %.3f]"
                    % (epoch, opt.n_epoch, i, len(train_loader), D_loss.item(), G_loss.item())) #

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