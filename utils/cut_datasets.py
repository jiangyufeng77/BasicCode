from PIL import Image
import os
datasets_path = '/media/ouc/4T_A/datasets/edges2shoes/train'
outA_dir = '/media/ouc/4T_A/datasets/edges2shoes/trainA'
outB_dir = '/media/ouc/4T_A/datasets/edges2shoes/trainB'
os.makedirs(outA_dir, exist_ok=True)
os.makedirs(outB_dir, exist_ok=True)
images = os.listdir(datasets_path)
for i, img in enumerate(images):
    AB = Image.open(datasets_path + '/' + img)
    filename = os.path.splitext(img)[0]
    filetype = os.path.splitext(img)[1]
    w, h = AB.size
    w2 = int(w / 2)
    A = AB.crop((0, 0, w2, h))
    B = AB.crop((w2, 0, w, h))
    A.save(outA_dir + '/' + filename + '_A' + filetype)
    B.save(outB_dir + '/' + filename + '_B' + filetype)
