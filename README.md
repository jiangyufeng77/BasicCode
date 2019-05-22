# Basic Code
这里写的是基于GAN的经典论文的代码复现，目前有CGAN。

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Installation
- 从http://pytorch.org中安装PyTorch 0.4, torchvision以及其它必要的安装包。
- 安装python libraries [visdom](https://github.com/facebookresearch/visdom) 和 [dominate](https://github.com/Knio/dominate)。
```bash
pip install visdom dominate
```
- 也可以通过以下命令安装：
```bash
pip install -r requirements.txt
```
- 对于Conda用户，可以通过给定的environment.yml创建虚拟环境:
conda env create -f environment.yml

## CGAN
CGAN的代码主要参照[eriklindernoren/PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py)，在MNIST数据集和CIFAR10数据集上进行测试，并用[FID](https://github.com/mseitzer/pytorch-fid)指标评价。

|Datasets                              |FID                                 |
|:------------------------------------:|------------------------------------|
|MNIST                                 |24.176845                           |
|CIFAR10                               |295.323599                          |
