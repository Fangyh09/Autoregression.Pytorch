# -*- coding: utf-8 -*-

import numpy as np
import sys
sys.path.append('./')
sys.path.append('./fft')

import torch
if sys.platform == "darwin":
    import matplotlib
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from ar_level import Corr
from skimage import color
from skimage import io
from boxx import *
from myfft.fft import fft_decompose, fft_recompose
from matplotlib import cm, colors, gridspec
from vis.vis_radar import vis_radar


def read_grey(path):
    return color.rgb2gray(io.imread('pics/p0.png'))


use_cuda = False
# if torch.has_cudnn:
#     use_cuda = True


R = np.load("pics/R.npz")
R = R["arr_0"]

print("R")
loga(R[0])
# raise ValueError()
img0 = R[0]
img1 = R[1]
img2 = R[2]



def ar_predict(img0, img1, img2):
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

    ### image level ###
    corr_module = Corr(image_level=True)
    if use_cuda:
        corr_module = corr_module.cuda()
    img3 = corr_module(img0, img1, img2, mask_R)

    plt.imshow(img3[0,0].float())
    plt.show()


def fft_ar(R):
    # load model
    # corr_module = Corr(image_level=True)
    corr_module = Corr(window_size=3, sigma=1)

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
    print(">>> mask_R.shape", mask_R.shape)
    # mask_R = mask_R[0,0].astype(np.float32)
    # attention here
    mask_R = mask_R.astype(np.int)
    mask_R = torch.from_numpy(mask_R)
    print(">>> mask_R type", type(mask_R))
    print(mask_R)


    res = fft_decompose(R, ar_order=2, n_cascade_levels=n_levels, R_thr=R_thr)
    print("R.shape", R.shape)
    print("res.shape", res[0]["cascade_levels"].shape)
    # origin_img = fft_recompose(res[0]["cascade_levels"])
    # plt.imshow(origin_img)
    # plt.show()
    # raise ValueError()

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


        if use_cuda:
            corr_module = corr_module.cuda()
      
        print(">>> diff")
        print(torch.sum(torch.abs(img0 - img1)))
        # print(">>> maskR")
        # loga(mask_R)
        # raise ValueError()
        # print(">>> loga img0, img1, img2")
        # loga(img0)
        # loga(img1)
        # loga(img2)
        # print("pre mask_R")
        # loga(mask_R)
        img3 = corr_module(img0, img1, img2, mask_R)
        # print("after mask_R")
        # loga(mask_R)
        print("img3", img3.shape)
        print(img3.shape)
        out_imgs.append(img3[0,0].data.numpy())
        # loga(img3) #shape:(1, 1, 320, 355) type:(float32 of torch.Tensor) max: -5.1484, min: -15.0, mean: -14.286
        # raise ValueError()
    for i in range(8):
        print("current i = ", i)
        print(out_imgs[i][56, 192])
        # loga(out_imgs[i])

    out = fft_recompose(out_imgs)
    print("out,", out[56, 192])
    print("R[0]")
    # loga(R[0])
    vis_radar(R[0], "R0.png")
    print("R[1]")
    # loga(R[1])
    vis_radar(R[1], "R1.png")
    print("R[2]")
    # loga(R[2])
    vis_radar(R[2], "R2.png")

    vis_radar(out, "out.png")

    print("out.shape", out.shape)
    print("mask_R.shape", mask_R.shape)
    # loga(mask_R)
    # raise ValueError()
    print("type out", type(out))
    mask_R = mask_R.data.numpy()
    print("type mask_R", type(mask_R))

    assert type(out) == type(mask_R)
    out[mask_R <= 0.5] = -15

    # loga(out)
    vis_radar(out)
    plt.imshow(out)
    plt.show()
    print("out.shape", out.shape)
    #out, -8.524121


fft_ar(R)