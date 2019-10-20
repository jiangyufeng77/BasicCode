import argparse
import os
import models
from util import util
import  numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d0','--dir0', type=str, default='/media/ouc/4T_A/jiang/multi_domain_model/stargan/data/celeba_split')
# parser.add_argument('-d1','--dir1', type=str, default='./imgs/ex_dir1')
parser.add_argument('-o','--out', type=str, default='/media/ouc/4T_A/jiang/multi_domain_model/stargan/data/celeba_image_dirs_LPIPS.txt')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()

## Initializing the model
model = models.PerceptualLoss(model='net-lin',	net='alex',use_gpu=opt.use_gpu)
dir_mean = []
dir_std = []
# crawl directories
f = open(opt.out,'w')

files = os.listdir(opt.dir0)
# files1 = os.listdir(opt.dir1)

# dir0 存的都是原图， dir1是生成的各种假图
print(files)
for i, file in enumerate(files):
    dir1 = opt.dir0 + '/' + files[i]
    dir1_list = os.listdir(dir1)

    for j, file1 in enumerate(files):
        if j != i:
            dir2 = opt.dir0 + '/' + files[j]
            dir2_list = os.listdir(dir2)

            for k in dir1_list:
                img0 = util.im2tensor(util.load_image(os.path.join(dir1, k)))
                if (opt.use_gpu):
                    img0 = img0.cuda()
                dis = []

                for p in dir2_list:
                    img1 = util.im2tensor(util.load_image(os.path.join(dir2, p)))
                    if (opt.use_gpu):
                        img1 = img1.cuda()

                    dist01 = model.forward(img0, img1).item()
                    # print('%s vs. %s Distance: %.3f'%(k, p, dist01))
                    dis.append(dist01)

                img0_mean = np.mean(dis)
                img0_std = np.std(dis)
                dir_mean.append(img0_mean)
                dir_std.append(img0_std)
                print('%s vs. %s mean: %.3f std: %.3f \n' % (k, file1, img0_mean, img0_std))

            dir_mean_total = np.mean(dir_mean)
            dir_std_total = np.mean(dir_std)
            print('%s to %s mean: %.3f std: %.3f \n' % (files[i], files[j], dir_mean_total, dir_std_total))
            f.writelines('%s to %s mean: %.3f std: %.3f \n' % (files[i], files[j], dir_mean_total, dir_std_total))
f.close()

# img0 = util.im2tensor(util.load_image(os.path.join(opt.dir0,file))) # RGB image from [-1,1]
# img1 = util.im2tensor(util.load_image(os.path.join(opt.dir1,file)))

# 		# Compute distance
# 		dist01 = model.forward(img0,img1).item()
# 		print('%s: %.3f'%(file,dist01))
#
# 		f.writelines('%s: %.6f\n'%(file,dist01))  # 将每一对图片的LPIPS distance 写到txt文件里
#
#
# f.close()
