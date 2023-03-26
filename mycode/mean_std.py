import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import imageio
from tqdm import *

dirpath = '/datassd4t/zhipengz/mydataset/train_val' # the path of dataset
seqs = ['s00', 's02', 's03', 's04', 's06', 's07', 's09', 's10']

R_channel = 0
G_channel = 0
B_channel = 0
img_num = 0
for seq in seqs:
    for data_sep in ['query', 'database']:
        filepath = os.path.join(dirpath, seq, data_sep, 'images')
        pathDir = os.listdir(filepath)
        for idx in tqdm(range(len(pathDir))):
            filename = pathDir[idx]
            img = imageio.imread(os.path.join(filepath, filename))
            img = img/255.0
            img_num = img_num + 1
            R_channel = R_channel + np.sum(img[:,:,0])
            G_channel = G_channel + np.sum(img[:,:,1])
            B_channel = B_channel + np.sum(img[:,:,2])
    print(seq + 'done!')

img_size  = img_num * 1400 * 2800
R_mean = R_channel / img_size
G_mean = G_channel / img_size
B_mean = B_channel / img_size
print("R_mean is %f, G_mean is %f, B_mean is %f" % (R_mean, G_mean, B_mean))

R_channel = 0
G_channel = 0
B_channel = 0
for seq in seqs:
    for data_sep in ['query', 'database']:
        filepath = os.path.join(dirpath, seq, data_sep, 'images')
        pathDir = os.listdir(filepath)
        for idx in tqdm(range(len(pathDir))):
            filename = pathDir[idx]
            img = imageio.imread(os.path.join(filepath, filename))
            img = img/255.0
            R_channel = R_channel + np.sum((img[:,:,0] - R_mean)**2)
            G_channel = G_channel + np.sum((img[:,:,1] - G_mean)**2)
            B_channel = B_channel + np.sum((img[:,:,2] - B_mean)**2)

R_var = (R_channel / img_size)**0.5
G_var = (G_channel / img_size)**0.5
B_var = (B_channel / img_size)**0.5


print("R_var is %f, G_var is %f, B_var is %f" % (R_var, G_var, B_var))