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
# from tensorboardX import SummaryWriter
from model.mnist_networks import G_mnist
from model.SVHN_network import G_SVHN

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--lr_D', type=float, default=0.0002, help='adam: learning rate for D')
parser.add_argument('--lr_G', type=float, default=0.001, help='adam, learning rate for G')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_classes', type=int, default=10, help='number of classes for dataset')
parser.add_argument('--code_dim', type=int, default=2, help='dimensionality of the continuous latent code')
parser.add_argument('--latent_dim', type=int, default=62, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
parser.add_argument('--n_row', type=int, default=10, help='the row of labels')
parser.add_argument('--gpu_ids', type=int, default=0, help='enables cuda')
parser.add_argument('--outf', type=str, default='images_MNIST', help='models are saved here')
parser.add_argument('--dataroot', type=str, default='/home/ouc/data1/niuwenjie/celebA_train', help='path to datasets')
parser.add_argument('--dataset', type=str, default='mnist', help='choose the dataset, e.g. mnist, CIFAR10, imagenet, LSUN, SVHN')
parser.add_argument('--model_dir', type=str, default='images_MNIST/model/99_generated_image.pkl', help='save models here')
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
    transform = transforms.Compose([
        transforms.Resize(opt.img_size),
        transforms.CenterCrop(opt.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    dataset = datasets.ImageFolder(opt.dataroot, transform)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)

elif opt.dataset == 'LSUN':
    test_loader = DataLoader(datasets.LSUN(opt.dataroot, classes=['bedroom_train'],
                                            transform=transforms.Compose([
                                                transforms.Resize(opt.img_size),
                                                transforms.CenterCrop(opt.img_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])),
                              batch_size=opt.batch_size, shuffle=False)

elif opt.dataset == 'SVHN':
    test_loader = DataLoader(datasets.SVHN(opt.dataroot, split='test',
                                            transform=transforms.Compose([
                                                transforms.Resize(opt.img_size),
                                                transforms.CenterCrop(opt.img_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]), download=True),
                              batch_size=opt.batch_size, shuffle=False)

def main():

    img_shape = (opt.channels, opt.img_size, opt.img_size)

    cuda = True if torch.cuda.is_available() else False

    generator = G_mnist(opt.n_classes, opt.channels)
    # generator = G_SVHN(opt.n_classes, opt.channels)
    print(generator)

    if cuda:
        generator = generator.cuda()
        torch.cuda.set_device(opt.gpu_ids)

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    generator.eval()
    generator.load_state_dict(torch.load(opt.model_dir))
    for i, (image, labels) in enumerate(test_loader):
        # z = Variable(FloatTensor(np.random.normal(loc=0.0, scale=1.0, size=(opt.batch_size, opt.latent_dim))))
        # label = Variable(labels.type(LongTensor))
        # code = Variable(FloatTensor(np.random.uniform(-1, 1, (opt.batch_size, opt.code_dim))))
        # gen_image = generator(z, label, code)

        # gen_imgs = []
        # for a in range(10):
        #     for j in range(10):
        #         z = Variable(FloatTensor(np.random.normal(loc=0.0, scale=1.0, size=(opt.batch_size, opt.latent_dim))))
        #         label = Variable(labels.type(LongTensor))
        #         code = np.random.uniform(-2, 2, (opt.batch_size, opt.code_dim))
        #         code[0][0] = code[0][0] + j * 0.01
        #         code = Variable(FloatTensor(code))
        #         gen_img = generator(z, label, code)
        #         gen_imgs.append(gen_img)
        #
        # gen_image = torch.cat(gen_imgs, 0)
        gen_imgs = []
        zero = np.zeros((opt.n_row ** 2, 1))
        # zero = Variable(FloatTensor(np.random.normal(0, 1, (opt.n_row ** 2, 1))))
        z = Variable(FloatTensor(np.random.normal(loc=0.0, scale=1.0, size=(opt.n_row ** 2, opt.latent_dim))))
        labels = np.array([num for _ in range(opt.n_row) for num in range(opt.n_row)])
        # labels = np.array([num for num in range(opt.n_row)])
        labels = Variable(LongTensor(labels))
        code = np.repeat(np.linspace(-2, 2, opt.n_row)[:, np.newaxis], opt.n_row, 0)
        # code = Variable(FloatTensor(np.linspace(-2, 2, opt.n_row)))
        code = Variable(FloatTensor(np.concatenate((code, zero), -1)))
        gen_imgs = generator(z, labels, code)

        real_image = Variable(image.type(FloatTensor))

        print("[Batch %d/%d]" % (i, len(test_loader)))

        gen_image = gen_imgs.view(gen_imgs.size(0), *img_shape)
        real_images = real_image.view(real_image.size(0), *img_shape)
        # save_image(sample1.data, '%s/%d_sample1_image.png' % (fake_path, i), nrow=opt.n_row, normalize=True)
        # save_image(sample2.data, '%s/%d_sample2_image.png' % (fake_path, i), nrow=opt.n_row, normalize=True)
        save_image(gen_image.data, '%s/%d_generated_image.png' % (fake_path, i), nrow=opt.n_row, normalize=True)
        save_image(real_images.data, '%s/%d_real_image.png' % (real_path, i), nrow=opt.n_row, normalize=True)

if __name__ == '__main__':
    main()