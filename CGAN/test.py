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
# from CGAN import Generator

os.makedirs('images/test/real', exist_ok=True)
os.makedirs('images/test/fake', exist_ok=True)

parse = argparse.ArgumentParser()
parse.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parse.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parse.add_argument('--channels', type=int, default=1, help='number of image channels')
parse.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parse.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parse.add_argument('--n_classes', type=int, default=10, help='number of classes for dataset')
parse.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parse.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
parse.add_argument('--n_epoch', type=int, default=200, help='number of epochs of training')
parse.add_argument('--n_row', type=int, default=10, help='the now of labels')
parse.add_argument('--model_dir', type=str, default='images/model/199_generated_image.pkl', help='save models here')
opt = parse.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim + opt.n_classes, 128, False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
    def forward(self, noise, label):
        gen_in = torch.cat((noise, self.label_emb(label)), -1)
        # print(noise.shape)
        # print(self.label_emb(label).shape)
        gen_img = self.model(gen_in)
        # print(gen_img.shape)
        gen_img = gen_img.view(gen_img.size(0), *img_shape)
        # print(gen_img.shape)
        return gen_img

generator = Generator()

if cuda:
    generator = generator.cuda()

os.makedirs('data/mnist', exist_ok=True)
dataloader = DataLoader(datasets.MNIST('data/mnist', train=False,
                                       transform=transforms.Compose([
                                           transforms.Resize(opt.img_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                       ]), download=True),
                        batch_size=opt.batch_size, shuffle=True)

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

generator.eval()
generator.load_state_dict(torch.load(opt.model_dir))

for i, (image, labels) in enumerate(dataloader):
    batch_size = image.shape[0]
    z = Variable(FloatTensor(np.random.normal(loc=0.0, scale=1.0, size=(opt.batch_size, opt.latent_dim))))
    real_image = Variable(image.type(FloatTensor))
    labels = Variable(labels.type(LongTensor))
    gen_image = generator(z, labels)

    print("[Batch %d/%d]" % (i, len(dataloader)))

    real_images = real_image.view(real_image.size(0), *img_shape)
    save_image(gen_image.data, 'images/test/fake/%d_generated_image.png' % (i), nrow=1, normalize=True)
    save_image(real_image.data, 'images/test/real/%d_real_image.png' % i, nrow=1, normalize=True)