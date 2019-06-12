import numpy as np
import sys
import torch
if sys.platform == "darwin":
    import matplotlib
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from ar import Corr
from skimage import color
from skimage import io
from boxx import *

def read_grey(path):
    return color.rgb2gray(io.imread('pics/p0.png'))

use_cuda = False
# if torch.has_cudnn:
#     use_cuda = True

img0 = read_grey("pics/p0.png")
img1 = read_grey("pics/p1.png")
img2 = read_grey("pics/p2.png")

print("img0", img0[0,0])
loga(img0)

print("img1")
loga(img1)

R = np.load("pics/R.npz")
R = R["arr_0"]

img0 = R[0]
img1 = R[1]
img2 = R[2]

# print(img.shape)
# h = 320
# w = 355
# from PIL import Image
# img0 = Image.open('pics/p0.png"').convert('LA')
# img1 = io.imread("pics/p1.png")
# img2 = io.imread("pics/p2.png")
print("img0", img0.shape)
h = img0.shape[0]
w = img0.shape[1]

img0 = np.reshape(img0, (1, 1, h, w))
img1 = np.reshape(img1, (1, 1, h, w))
img2 = np.reshape(img2, (1, 1, h, w))
img0 = torch.from_numpy(img0)
img1 = torch.from_numpy(img1)
img2 = torch.from_numpy(img2)
if use_cuda:
    img0 = img0.cuda()
    img1 = img1.cuda()
    img2 = img2.cuda()

R_thr = -10
mask_R0 = img0 >= R_thr
mask_R1 = img1 >= R_thr
mask_R2 = img2 >= R_thr
mask_R = mask_R0 * mask_R1 * mask_R2
mask_R = mask_R[0,0].float()
# mask_R = torch.from_numpy(mask_R.astype(np.int))
# R = mask_R

# print(R.shape)
# # print(320 * 355 * 5 - np.sum(R==-15))
# plt.imshow(R[0])

corr_module = Corr(window_size=9, sigma=3)
if use_cuda:
    corr_module = corr_module.cuda()
img3 = corr_module(img0, img1, img2, mask_R)


### print ###
# %debug
# print(img3.shape)
loga(img1)
loga(img2)
loga(img3)
plt.imshow(img1[0,0])
plt.show()
plt.imshow(img2[0,0])
plt.show()
plt.imshow(img3[0,0].float())
plt.show()

