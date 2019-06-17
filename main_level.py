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
from vis.vis_radar import vis_radar


def read_grey(path):
    return color.rgb2gray(io.imread('pics/p0.png'))


use_cuda = False
# SAVE_IMG = False
SAVE_IMG = True
# if torch.has_cudnn:
#     use_cuda = True


R = np.load("pics/R.npz")
R = R["arr_0"]

img0 = R[0]
img1 = R[1]
img2 = R[2]

res = fft_decompose(R, ar_order=2, n_cascade_levels=8, R_thr=-10)

print(">>> ori img0")
#loga(img0)
print(res[0].keys())
for i in range(8):
    img = res[0]["cascade_levels"][i]
    #plt.imshow(img)
    #plt.show()
R = res[0]["cascade_levels"]
print("res[0].keys()", res[0].keys())

out_img = fft_recompose(R)

print(">>> out_img")
# loga(out_img)


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

### patch level ###
corr_module = Corr(window_size=3, sigma=1)

### image level ###
# corr_module = Corr(image_level=True)
if use_cuda:
    corr_module = corr_module.cuda()
img3 = corr_module(img0, img1, img2, mask_R)


img0_data = img0[0,0].data.numpy()
img1_data = img1[0,0].data.numpy()
img2_data = img2[0,0].data.numpy()
img3_data = img3[0,0].data.numpy()

loga(img0_data)
loga(img1_data)
loga(img2_data)
loga(img3_data)

if SAVE_IMG:
    vis_radar(img0_data, "nofft_R0.png")
    vis_radar(img1_data, "nofft_R1.png")
    vis_radar(img2_data, "nofft_R2.png")
    vis_radar(img3_data, "nofft_R3.png")
else:
    vis_radar(img0_data)
    vis_radar(img1_data)
    vis_radar(img2_data)
    vis_radar(img3_data)

plt.imshow(img3[0,0].float())
plt.show()








