import numpy as np
import sys
sys.path.append('./')
sys.path.append('./fft')

import torch
if sys.platform == "darwin":
    import matplotlib
    matplotlib.use('TkAgg')
import matplotlib.pyplot as #plt
from ar import Corr
from skimage import color
from skimage import io
from boxx import *
from myfft.fft import fft_decompose, fft_recompose

def read_grey(path):
    return color.rgb2gray(io.imread('pics/p0.png'))


use_cuda = False
# if torch.has_cudnn:
#     use_cuda = True


R = np.load("pics/R.npz")
R = R["arr_0"]

img0 = R[0]
img1 = R[1]
img2 = R[2]

# assert R.shape[0] == 3

def ar(R):
   # imgs = []
    # for i in range(R.shape[0]):
    # build image
    img0 = R[0]
    img1 = R[1]
    img2 = R[2]

    print("R.shape", R.shape)
    n_levels = 8
    R_thr = -10
    h = R.shape[1]
    w = R.shape[2]
    print("R[0]")
    #loga(R[0])
    mask_R0 = R[0] >= R_thr
    mask_R1 = R[1] >= R_thr
    mask_R2 = R[2] >= R_thr
    mask_R = mask_R0 * mask_R1 * mask_R2 * 1.0
    mask_R = torch.from_numpy(mask_R).float()


    # res = fft_decompose(R, ar_order=2, n_cascade_levels=n_levels, R_thr=R_thr)
    out_imgs = []
    print(">>> run fft...")
    # for each level

    # for each image
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

    # load model
    corr_module = Corr(allones=True)
    if use_cuda:
        corr_module = corr_module.cuda()
    print("img0", img0.shape)
    print("img1", img1.shape)
    print("img2", img2.shape)
    img3 = corr_module(img0, img1, img2, mask_R)

    #loga(img1)
    #loga(img2)
    #loga(img3)

    #plt.imshow(img1[0,0])
    #plt.show()
    print(img3.shape)
    
    plt.imshow(img2[0,0])
    plt.show()
    print(img3.shape)
    
    #plt.imshow(img3[0,0])
    #plt.show()
    print(img3.shape)
   
    #plt.show()


def fft_ar(R):
    # imgs = []
    # for i in range(R.shape[0]):
    print("R.shape", R.shape)
    n_levels = 8
    R_thr = -10
    h = R.shape[1]
    w = R.shape[2]
    print("R[0]")
    # loga(R[0])
    mask_R0 = R[0] >= R_thr
    mask_R1 = R[1] >= R_thr
    mask_R2 = R[2] >= R_thr
    mask_R = mask_R0 * mask_R1 * mask_R2
    mask_R = mask_R[0,0].astype(np.float32)

    res = fft_decompose(R, ar_order=2, n_cascade_levels=n_levels, R_thr=R_thr)
    out_imgs = []
    print(">>> run fft...")
    # for each level
    for i in range(0, n_levels):
        img0 = res[0]["cascade_levels"][i]
        img1 = res[1]["cascade_levels"][i]
        img2 = res[2]["cascade_levels"][i]
        # for each image
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
 
        # load model
        corr_module = Corr(allones=True)
        if use_cuda:
            corr_module = corr_module.cuda()
        print("img0", img0.shape)
        print("img1", img1.shape)
        print("img2", img2.shape)
        img3 = corr_module(img0, img1, img2, mask_R)
        #plt.imshow(img3[0,0])
        #plt.show()
        print(img3.shape)
        out_imgs.append(img3[0,0])
    out = fft_recompose(out_imgs)
    print("out.shape", out.shape)
    #plt.imshow(out)
    #plt.show()

# fft_ar(R)
ar(R)


raise ValueError()

# res = fft_decompose(R, ar_order=2, n_cascade_levels=8, R_thr=-10)
# what(res)


# print(">>> ori img0")
# loga(img0)
# print(res[0].keys())
# for i in range(8):
#     img = res[0]["cascade_levels"][i]
#     plt.imshow(img)
#     plt.show()
# R = res[0]["cascade_levels"]
# print("res[0].keys()", res[0].keys())



# print(">>> out_img")
# loga(out_img)
# plt.imshow(out_img)
# raise ValueError()

print("img0", img0.shape)
h = img0.shape[0]
w = img0.shape[1]




### patch level ###
# corr_module = Corr(window_size=9, sigma=3)
 
### image level ###




#loga(img1)
#loga(img2)
#loga(img3)
#plt.imshow(img1[0,0])
plt.show()
#plt.imshow(img2[0,0])
#plt.show()
#plt.imshow(img3[0,0].float())
#plt.show()

