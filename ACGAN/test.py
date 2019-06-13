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
from networks import Generator_128, Discriminator
from logger import Logger
from folder import ImageFolder

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_classes', type=int, default=10, help='number of classes for dataset')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=128, help='size of each image dimension')
parser.add_argument('--n_epoch', type=int, default=200, help='number of epochs of training')
parser.add_argument('--n_row', type=int, default=1, help='the row of labels')
parser.add_argument('--gpu_ids', type=int, default=0, help='enables cuda')
parser.add_argument('--outf', type=str, default='images_imagenet_train', help='models are saved here')
parser.add_argument('--dataroot', type=str, default='../../data/Imagenet_test', help='path to datasets')
parser.add_argument('--dataset', type=str, default='imagenet', help='choose the dataset, e.g. mnist, CIFAR10, imagenet')
parser.add_argument('--model_dir', type=str, default='images_imagenet_train/model/499_generated_image.pkl', help='save models here')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

def save_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

save_path = os.path.join(opt.outf, 'result')
real_path = os.path.join(save_path, 'real')
fake_path = os.path.join(save_path, 'fake')
# model_path = os.path.join(opt.outf, 'model')
# log_dir = os.path.join(opt.outf, 'log')
save_dir(real_path)
save_dir(fake_path)
# save_dir(log_dir)

if opt.dataset == 'mnist':
    test_loader = DataLoader(datasets.MNIST(opt.dataroot, train=False,
                                               transform=transforms.Compose([
                                                   transforms.Resize(opt.img_size),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                               ]), download=False),
                              batch_size=opt.batch_size, shuffle=False)
elif opt.dataset == 'CIFAR10':
    test_loader = DataLoader(datasets.CIFAR10(opt.dataroot, train=False,
                                       transform=transforms.Compose([
                                           transforms.Resize(opt.img_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                       ]), download=False),
                        batch_size=opt.batch_size, shuffle=False)
elif opt.dataset == 'imagenet':
    # folder dataset
    dataset = ImageFolder(
        root=opt.dataroot,
        transform=transforms.Compose([
            transforms.Scale(opt.img_size),
            transforms.CenterCrop(opt.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
        classes_idx=(0, 10))
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                             shuffle=False)

def main():

    img_shape = (opt.channels, opt.img_size, opt.img_size)

    cuda = True if torch.cuda.is_available() else False

    generator = Generator_128(opt.n_classes, opt.latent_dim, opt.channels, opt.ngpu)

    print(generator)

    if cuda:
        generator = generator.cuda()
        torch.cuda.set_device(opt.gpu_ids)

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    generator.eval()
    generator.load_state_dict(torch.load(opt.model_dir))
    for i, (image, labels) in enumerate(test_loader):
        z = Variable(FloatTensor(np.random.normal(loc=0.0, scale=1.0, size=(opt.batch_size, opt.latent_dim))))
        real_image = Variable(image.type(FloatTensor))
        # labels = Variable(labels.type(LongTensor))
        gen_label = Variable(LongTensor(np.random.randint(0, opt.n_classes, opt.batch_size)))
        gen_image = generator(z, gen_label)
        # gen_image = generator(z)

        print("[Batch %d/%d]" % (i, len(test_loader)))

        gen_image = gen_image.view(gen_image.size(0), *img_shape)
        real_images = real_image.view(real_image.size(0), *img_shape)
        save_image(gen_image.data, '%s/%d_generated_image.png' % (fake_path, i), nrow=opt.n_row, normalize=True)
        save_image(real_images.data, '%s/%d_real_image.png' % (real_path, i), nrow=opt.n_row, normalize=True)

if __name__ == '__main__':
    main()