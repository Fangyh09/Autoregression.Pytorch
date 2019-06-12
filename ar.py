## Corr class
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
from boxx import *

def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in
         range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel, sigma=1.5, allones=False):
    print("window_size", window_size)
    if allones:
        return torch.ones((1, 1, window_size, window_size))
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous())
    # print(">>> windows")
    # print(window)
    # print(window.shape)
    # raise ValueError()
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    EX = F.conv2d(img1, window, padding=window_size // 2,
                         groups=channel)
    EY = F.conv2d(img2, window, padding=window_size // 2,
                         groups=channel)
    EX2 = F.conv2d(img1 * img1, window, padding=window_size // 2,
                         groups=channel)
    EY2 = F.conv2d(img2 * img2, window, padding=window_size // 2,
                         groups=channel)
    EXY = F.conv2d(img1 * img2, window, padding=window_size // 2,
                         groups=channel)
    
    corr = (EXY - EX*EY) / torch.sqrt(EX2 - EX*EX) / torch.sqrt(EY2 - EY*EY)
    corr_ = corr[0,0]

    return corr


class Corr(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, sigma=1.5, allones=False):
        super(Corr, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.sigma = sigma
        self.window = create_window(window_size, self.channel, sigma=self.sigma, allones=allones)

    def _adjust_lag2_corrcoef2(self, gamma_2, gamma_1):
        gamma_2 = max(gamma_2, 2 * gamma_1 * gamma_2 - 1)
        gamma_2 = max(gamma_2, (3 * gamma_1 ** 2 - 2 + 2 * (
                    1 - gamma_1 ** 2) ** 1.5) / gamma_1 ** 2)
        return gamma_2, gamma_1

    def get_phi(self, gamma2, gamma1, mask_R):
        assert (1 - gamma1[0,0][mask_R > 0] > 0).all() 
        gamma1[0,0][mask_R <= 0] = 0
        phi1 = (gamma1 * gamma1 * (1 - gamma2)) / (1 - gamma1 * gamma1)
        phi2 = (gamma2 - gamma1 * gamma1) / (1 - gamma1 * gamma1)
        phi0 = 1 - phi2 * gamma2 - phi1 * gamma1
        phi0 = torch.clamp(phi0, min=0)
        phi0 = torch.sqrt(phi0)
        return phi2, phi1, phi0
    
    def get_u_sigma(self, img):
        u = torch.mean(img)
        u2 = torch.mean(img * img)
        sigma = torch.sqrt(u2 - u * u)
        return u, sigma
    

    def forward(self, _img1, _img2, _img3, mask_R, minval=-100):
        (_, channel, _, _) = _img1.size()

        window = create_window(self.window_size, channel, sigma=self.sigma)

        if _img1.is_cuda:
            window = window.cuda(_img1.get_device())
        window = window.type_as(_img1)
        h = _img1.shape[2]
        w = _img1.shape[3]
        
        self.window = window
        self.channel = channel

        u1, sigma1 = self.get_u_sigma(_img1[0,0][mask_R > 0])
        u2, sigma2 = self.get_u_sigma(_img2[0,0][mask_R > 0])
        u3, sigma3 = self.get_u_sigma(_img3[0,0][mask_R > 0])
        img1 = (_img1 - u1) / sigma1
        img2 = (_img2 - u2) / sigma2
        img3 = (_img3 - u3) / sigma3

        out_img = torch.zeros((1, 1, h, w))
        gamma2 = _ssim(img1, img3, window, self.window_size, channel,
                     self.size_average)
        gamma1 = _ssim(img2, img3, window, self.window_size, channel,
                      self.size_average)
        print(">>>gamma2")
        loga(gamma2[0,0][mask_R > 0])
        print(">>>gamma1")
        loga(gamma1[0,0][mask_R > 0])
#         gamma2, gamma1 = self._adjust_lag2_corrcoef2(gamma2, gamma1)
        phi2, phi1, phi0 = self.get_phi(gamma2, gamma1, mask_R)
        print(">>>phi2")
        loga(phi2[0,0][mask_R > 0])
        print(">>>phi1")
        loga(phi1[0,0][mask_R > 0])

        out = img1 * phi2 + img2 * phi1
        out = u3 + out * sigma3
        out_img[0,0][mask_R > 0] = out[0,0][mask_R > 0].float()

        out_img[0,0][mask_R <= 0] = -15 #torch.mean(_img1[0,0][mask_R <= 0]).float()
        # if out.is_cuda():
        #     out_img = out_img.cuda()
        return out_img
