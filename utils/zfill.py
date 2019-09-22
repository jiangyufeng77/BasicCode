import os

path = r'/media/ouc/4T_A/jiang/pytorch_GAN/pix2pixHD/datasets/cityscapes/test_label'
for file in os.listdir(path):
    name = file.split('.')[0]
    os.rename(os.path.join(path, file), os.path.join(path, '%05d' % int(name) + ".png"))