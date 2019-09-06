import numpy as np
from imageio import imread, imsave
import cv2
import os
import torch
from torch.autograd import Variable


directory_name1 = "/media/ouc/4T_A/jiang/DRPAN/DRPAN_more_proposal/checkpoints_gradient/test/fake1"
directory_name2 = "/media/ouc/4T_A/jiang/DRPAN/DRPAN_more_proposal/checkpoints_gradient/test/fake_labelIDs"
dir = os.listdir(directory_name1)
# result_name = "labelIDs"

# use_gpu = torch.cuda.is_available()
sum = 0
for filename in dir:
    img = imread(directory_name1 + "/" + filename)
    # print(img.shape)
    # img = Variable(torch.cuda.FloatTensor(img))
    sum += 1
    print(filename)
    print(sum)
    labels = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [111, 74, 0], [81, 0, 81], [128, 64, 128],
              [244, 35, 232], [250, 170, 160], [230, 150, 140], [70, 70, 70], [102, 102, 156], [190, 153, 153],
              [180, 165, 180], [150, 100, 100], [150, 120, 90], [153, 153, 153], [153, 153, 153], [250, 170, 30],
              [220, 220, 0], [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142],
              [0, 0, 70], [0, 60, 100], [0, 0, 90], [0, 0, 110], [0, 80, 100], [0, 0, 230], [119, 11, 32], [0, 0, 142]]
    # labels = Variable(torch.cuda.FloatTensor(labels))

    h, w = img.shape[0], img.shape[1]
    label_img = np.zeros((h, w, 1))

    # print(h, w)
    for row in range(h):
        for col in range(w):
            pixel = img[row][col]
            # print(pixel.shape)
            _ = pixel - labels

            new = np.argmin(np.sum(np.abs(_), axis=1))
            # new = np.sqrt(np.sum(np.square(_), axis=1))
            label_img[row][col] = new
            cv2.imwrite(directory_name2 + "/" + filename, label_img)









#
# img = imread('/media/ouc/4T_A/jiang/DRPAN/1.png')
# print(img.shape)
# # print(type(img))
#
# labels = [[ 0, 0, 0], [ 0, 0, 0], [ 0, 0, 0], [ 0, 0, 0], [ 0, 0, 0], [111, 74, 0], [ 81, 0, 81], [128, 64,128], [244, 35,232], [250,170,160], [230,150,140], [ 70, 70, 70], [102,102,156], [190,153,153], [180,165,180], [150,100,100], [150,120, 90], [153,153,153], [153,153,153], [250,170, 30], [220,220, 0], [107,142, 35], [152,251,152], [ 70,130,180], [220, 20, 60], [255, 0, 0], [ 0, 0,142], [ 0, 0, 70], [ 0, 60,100], [ 0, 0, 90], [ 0, 0,110], [ 0, 80,100], [ 0, 0,230], [119, 11, 32], [ 0, 0,142]]
#
# h, w = img.shape[0], img.shape[1]
# label_img = np.zeros((h, w, 1))
#
# # print(h, w)
# for row in range(h):
#     for col in range(w):
#         pixel = img[row][col]
#         # print(pixel.shape)
#         _ = pixel - labels
#
#         new = np.argmin(np.sum(np.abs(_), axis=1))
#     # new = np.sqrt(np.sum(np.square(_), axis=1))
#         label_img[row][col] = new
#
# # imsave('labelimage.jpg', np.float32(label_img))
# cv2.imwrite('/media/ouc/4T_A/jiang/DRPAN/label1024.jpg', label_img)