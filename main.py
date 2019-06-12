import numpy as np
import torch
import matplotlib.pyplot as plt
from ar import Corr

h = 320
w = 355
img0 = np.reshape(R[0], (1,1,h,w))
img1 = np.reshape(R[1], (1,1,h,w))
img2 = np.reshape(R[2], (1,1,h,w))
img0 = torch.from_numpy(img0).cuda()
img1 = torch.from_numpy(img1).cuda()
img2 = torch.from_numpy(img2).cuda()

R_thr = -10
mask_R0 = img0 >= R_thr
mask_R1 = img1 >= R_thr
mask_R2 = img2 >= R_thr
mask_R = mask_R0 * mask_R1 * mask_R2
mask_R = mask_R[0,0].float()
# mask_R = torch.from_numpy(mask_R.astype(np.int))
# R = mask_R

print(R.shape)
# print(320 * 355 * 5 - np.sum(R==-15))
plt.imshow(R[0])

corr_module = Corr(window_size=9, sigma=3)
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

