import os, cv2

# source dir pixvgg
datadir = '/home/jiang/DRPAN/DRPAN_remove_Reviser_and_Proposal/checkpoints_labels2real/test/fake'
# target dir
fcn_dir = '/media/ouc/4T_A/jiang/pytorch_GAN/pytorch-CycleGAN-and-pix2pix/datasets/cityscapesScripts/leftImg8bit_trainvaltest/leftImg8bit/eva'
# renamed dir
rename_dir = '/home/jiang/DRPAN/DRPAN_remove_Reviser_and_Proposal/checkpoints_labels2real/test/fake1'
imgsdir = os.listdir(datadir)
fcndir = os.listdir(fcn_dir)
fcndir.sort()
imgsdir.sort()  # for correct order:   imgsdir.sort(key=lambda x:int(x[:-4]))
print(imgsdir)
# print(fcndir)
for i, img in enumerate(imgsdir):
    img_dir = datadir + '/' + img
    image = cv2.imread(img_dir)
    # image = cv2.resize(image, (256, 256))
    # imgname = '0010' + str(i+1).rjust(4,'0')+'.png'
    imgname = str(i+1) + '.jpg'
    imgname = fcndir[i]
    cv2.imwrite('%s/%s' % (rename_dir, imgname), image)
