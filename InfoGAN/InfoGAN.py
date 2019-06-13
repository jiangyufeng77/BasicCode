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
from model.mnist_networks import G_mnist, D_mnist, D_net, Q_net
# from model.SVHN_network import G_SVHN, D_SVHN, D_net, Q_net
from model.CelebA_network import G_CelebA, D_CelebA, D_net, Q_net

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128, help='size of the batches')
parser.add_argument('--lr_D', type=float, default=0.0002, help='adam: learning rate for D')
parser.add_argument('--lr_G', type=float, default=0.001, help='adam, learning rate for G')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_classes', type=int, default=10, help='number of classes for dataset')
parser.add_argument('--code_dim', type=int, default=2, help='dimensionality of the continuous latent code')
parser.add_argument('--latent_dim', type=int, default=62, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--n_epoch', type=int, default=200, help='number of epochs of training')
parser.add_argument('--n_row', type=int, default=10, help='the row of labels')
parser.add_argument('--gpu_ids', type=int, default=0, help='enables cuda')
parser.add_argument('--outf', type=str, default='images_MNIST', help='models are saved here')
parser.add_argument('--dataroot', type=str, default='../../data/mnist', help='path to datasets')
parser.add_argument('--dataset', type=str, default='mnist', help='choose the dataset, e.g. mnist, CIFAR10, imagenet, LSUN, SVHN, CelebA')
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
                                               ]), download=True),
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

elif opt.dataset == 'SVHN':
    train_loader = DataLoader(datasets.SVHN(opt.dataroot, split='train',
                                            transform=transforms.Compose([
                                                transforms.Resize(opt.img_size),
                                                transforms.CenterCrop(opt.img_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]), download=True),
                              batch_size=opt.batch_size, shuffle=True)

elif opt.dataset == 'CelebA':
    transform = transforms.Compose([
        transforms.Resize(opt.img_size),
        transforms.CenterCrop(opt.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    dataset = datasets.ImageFolder(opt.dataroot, transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
# adversarial_loss = nn.MSELoss()
adversarial_loss = nn.BCELoss()
class_loss = nn.CrossEntropyLoss()
continuous_loss = nn.MSELoss()

generator = G_mnist(opt.n_classes, opt.channels)
# generator = G_SVHN(opt.n_classes, opt.channels)
# generator = G_CelebA(opt.n_classes, opt.channels)
discriminator = D_mnist(opt.channels)
# discriminator = D_SVHN(opt.channels)
# discriminator = D_CelebA(opt.channels)
D_Net = D_net()
Q_Net = Q_net(opt.n_classes, opt.code_dim)

if cuda:
    generator.cuda()
    discriminator.cuda()
    D_Net.cuda()
    Q_Net.cuda()
    adversarial_loss.cuda()
    class_loss.cuda()
    continuous_loss.cuda()
    torch.cuda.set_device(opt.gpu_ids)

print(generator, discriminator)

optimizer_G = torch.optim.Adam([{'params': generator.parameters()}, {'params':Q_Net.parameters()}], lr = opt.lr_G, betas = (opt.b1, opt.b2))
optimizer_D = torch.optim.Adam([{'params': discriminator.parameters()}, {'params': D_Net.parameters()}], lr = opt.lr_D, betas = (opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

for epoch in range (opt.n_epoch):
    for i, (image, label)in enumerate(train_loader):

        batch_size = image.shape[0]

        real_image = Variable(image.type(FloatTensor))
        real_label = Variable(label.type(LongTensor))

        valid = Variable(FloatTensor(batch_size).fill_(1.0), requires_grad = False)
        fake = Variable(FloatTensor(batch_size).fill_(0.0), requires_grad = False)

        # Train Discriminator and NetD

        optimizer_D.zero_grad()

        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_label = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))
        con_code = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, opt.code_dim))))
        # print(z)
        # print(z.shape)

        # train D with fake
        gen_img = generator(z, gen_label, con_code)
        # gen_img = generator(z, gen_label)

        output_D_fake = discriminator(gen_img.detach())
        Dis_fake = D_Net(output_D_fake)
        loss_fake = adversarial_loss(Dis_fake, fake)

        # train D with real

        output_D_real = discriminator(real_image)
        Dis_real = D_Net(output_D_real)
        loss_real = adversarial_loss(Dis_real, valid)

        D_loss = loss_fake + loss_real
        D_loss.backward()
        optimizer_D.step()

        # Train Generator and NetQ
        optimizer_G.zero_grad()

        output_D_G = discriminator(gen_img)
        Dis_fake_G = D_Net(output_D_G)
        loss_fake_G = adversarial_loss(Dis_fake_G, valid)

        class_label, latent_code = Q_Net(output_D_G)
        # class_label= Q_Net(output_D_G)
        Q_loss = class_loss(class_label, gen_label) + 0.1 * continuous_loss(latent_code, con_code)
        # Q_loss = class_loss(class_label, gen_label)
        G_loss = loss_fake_G + Q_loss

        G_loss.backward()
        optimizer_G.step()

        print("[Epoch: %d/%d] [Batch: %d/%d] [D loss: %.3f] [G loss: %.3f]"
                % (epoch, opt.n_epoch, i, len(train_loader), D_loss.item(), G_loss.item()))

    if epoch % 1 == 0:
        noise = Variable(FloatTensor(np.random.normal(0, 1, (opt.n_row ** 2, opt.latent_dim))))
        labels = np.array([num for _ in range(opt.n_row) for num in range(opt.n_row)])
        labels = Variable(LongTensor(labels))
        code = Variable(FloatTensor(np.zeros((opt.n_row ** 2, opt.code_dim))))
        gen_image = generator(noise, labels, code)
        # gen_image = generator(noise, labels)
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