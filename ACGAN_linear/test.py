import argparse
import os
import torch

import numpy as np
import torchvision.transforms as transforms

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import datasets
from networks import Generator

os.makedirs('images_4_nocat_MSE_ls_lc/test/real', exist_ok=True)
os.makedirs('images_4_nocat_MSE_ls_lc/test/fake', exist_ok=True)

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
parse.add_argument('--model_dir', type=str, default='images_4_nocat_MSE_ls_lc/model/199_generated_image.pkl', help='save models here')
parse.add_argument('--gpu_ids', type=int, default=0, help='enables cuda')

def main():
    opt = parse.parse_args()
    print(opt)

    img_shape = (opt.channels, opt.img_size, opt.img_size)

    cuda = True if torch.cuda.is_available() else False

    generator = Generator(opt.n_classes, opt.latent_dim, img_shape)

    print(generator)

    if cuda:
        generator = generator.cuda()
        torch.cuda.set_device(opt.gpu_ids)

    os.makedirs('data/mnist', exist_ok=True)
    dataloader = DataLoader(datasets.MNIST('data/mnist', train=False,
                                           transform=transforms.Compose([
                                               transforms.Resize(opt.img_size),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                           ]), download=True),
                            batch_size=opt.batch_size, shuffle=False)

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

        gen_image = gen_image.view(gen_image.size(0), *img_shape)
        real_images = real_image.view(real_image.size(0), *img_shape)
        save_image(gen_image.data, 'images_4_nocat_MSE_ls_lc/test/fake/%d_generated_image.png' % (i), nrow=1, normalize=True)
        save_image(real_images.data, 'images_4_nocat_MSE_ls_lc/test/real/%d_real_image.png' % i, nrow=1, normalize=True)

if __name__ == '__main__':
    main()