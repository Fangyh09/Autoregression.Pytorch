import numpy as np
import sys
sys.path.append('./')
sys.path.append('./fft')

import torch
if sys.platform == "darwin":
    import matplotlib
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from ar_image import Corr
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

# res = fft_decompose(R, ar_order=2, n_cascade_levels=8, R_thr=-10)
# what(res)




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