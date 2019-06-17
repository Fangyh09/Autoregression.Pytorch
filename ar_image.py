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


def create_image_window(window_size, channel):
    assert channel == 1
    return torch.ones((1, channel, window_size[0], window_size[1]))

def create_patch_window(window_size, channel, sigma=1.5):
    """
    if image_level: window_size = [10,20]
    else: window_size = 10
    """
    print("window_size", window_size)
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous())
    # print(">>> windows")
    # print(window)
    # print(window.shape)
    # raise ValueError()
    return window

def _ssim_patch(img1, img2, window, window_size, channel, size_average=True):
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

def _ssim_image(img1, img2, window, window_size, channel, size_average=True):
    EX = F.conv2d(img1, window, groups=channel)
    EY = F.conv2d(img2, window, groups=channel)
    EX2 = F.conv2d(img1 * img1, window, groups=channel)
    EY2 = F.conv2d(img2 * img2, window, groups=channel)
    EXY = F.conv2d(img1 * img2, window, groups=channel)

    var_list = [EX, EY, EX2, EY2, EXY] ##adl
    for idx,x in enumerate(var_list): ##adl
        var_names = "EX, EY, EX2, EY2, EXY".split(",") ##adl
        cur_name = var_names[idx] ##adl
        print("-------print "+ cur_name + "------") ##adl
        import alog ##adl
        from pprint import pprint ##adl
        alog.info(cur_name) ##adl
        print(">>> type(x) = ", type(x)) ##adl
        if hasattr(x, "shape"): ##adl
            print(">>> " + cur_name + ".shape", x.shape) ##adl
        if type(x) is list: ##adl
            print(">>> len(" + cur_name + ") = ", len(x)) ##adl
            pprint(x) ##adl
        else: ##adl
            pprint(x) ##adl
            pass ##adl
        print("------------------------\n") ##adl

    var_list = [img1, img1*img1, img2, img2*img2, img1*img2] ##adl
    for idx,x in enumerate(var_list): ##adl
        var_names = "img1, img1*img1, img2, img2*img2, img1*img2".split(",") ##adl
        cur_name = var_names[idx] ##adl
        print("-------print "+ cur_name + "------") ##adl
        import alog ##adl
        from pprint import pprint ##adl
        alog.info(cur_name) ##adl
        print(">>> type(x) = ", type(x)) ##adl
        if hasattr(x, "shape"): ##adl
            print(">>> " + cur_name + ".shape", x.shape) ##adl
        if type(x) is list: ##adl
            print(">>> len(" + cur_name + ") = ", len(x)) ##adl
            pprint(x) ##adl
        else: ##adl
            pprint(x) ##adl
            pass ##adl
        print("------------------------\n") ##adl
    
    corr = (EXY - EX*EY) / torch.sqrt(EX2 - EX*EX) / torch.sqrt(EY2 - EY*EY)
    corr_ = corr[0,0]
    var_list = [corr_] ##adl
    for idx,x in enumerate(var_list): ##adl
        var_names = "corr_".split(",") ##adl
        cur_name = var_names[idx] ##adl
        print("-------print "+ cur_name + "------") ##adl
        import alog ##adl
        from pprint import pprint ##adl
        alog.info(cur_name) ##adl
        print(">>> type(x) = ", type(x)) ##adl
        if hasattr(x, "shape"): ##adl
            print(">>> " + cur_name + ".shape", x.shape) ##adl
        if type(x) is list: ##adl
            print(">>> len(" + cur_name + ") = ", len(x)) ##adl
            pprint(x) ##adl
        else: ##adl
            pprint(x) ##adl
            pass ##adl
        print("------------------------\n") ##adl
    # raise ValueError()
    return corr 



class Corr(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, sigma=1.5, image_level=False):
        super(Corr, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.sigma = sigma
        # self.window = create_window(window_size, self.channel, sigma=self.sigma, image_level=image_level)
        # assert image_level 
        self.image_level = image_level

    def _adjust_lag2_corrcoef2(self, gamma_2, gamma_1):
        gamma_2 = max(gamma_2, 2 * gamma_1 * gamma_2 - 1)
        gamma_2 = max(gamma_2, (3 * gamma_1 ** 2 - 2 + 2 * (
                    1 - gamma_1 ** 2) ** 1.5) / gamma_1 ** 2)
        return gamma_2, gamma_1

    def get_phi(self, gamma2, gamma1, mask_R):
        # if gamma1[0,0].shape() == mask_R.shape():
        #     assert (1 - gamma1[0,0][mask_R > 0] > 0).all() 
        # gamma1[0,0][mask_R <= 0] = 0
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
        if self.image_level:
            _ssim = _ssim_image
        else:
            _ssim = _ssim_patch

        (_, channel, _, _) = _img1.size()
        
        if self.image_level:
            window = create_image_window([_img1.size()[-2], _img1.size()[-1]], channel)
        else:
            window = create_patch_window(self.window_size, channel, sigma=self.sigma)

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

        var_list = [img1, img2, img3] ##adl
        for idx,x in enumerate(var_list): ##adl
            var_names = "img1, img2, img3".split(",") ##adl
            cur_name = var_names[idx] ##adl
            print("-------print "+ cur_name + "------") ##adl
            import alog ##adl
            from pprint import pprint ##adl
            alog.info(cur_name) ##adl
            print(">>> type(x) = ", type(x)) ##adl
            if hasattr(x, "shape"): ##adl
                print(">>> " + cur_name + ".shape", x.shape) ##adl
            if type(x) is list: ##adl
                print(">>> len(" + cur_name + ") = ", len(x)) ##adl
                pprint(x) ##adl
            else: ##adl
                pprint(x) ##adl
                pass ##adl
            print("------------------------\n") ##adl
        var_list = [window] ##adl
        for idx,x in enumerate(var_list): ##adl
            var_names = "window".split(",") ##adl
            cur_name = var_names[idx] ##adl
            print("-------print "+ cur_name + "------") ##adl
            import alog ##adl
            from pprint import pprint ##adl
            alog.info(cur_name) ##adl
            print(">>> type(x) = ", type(x)) ##adl
            if hasattr(x, "shape"): ##adl
                print(">>> " + cur_name + ".shape", x.shape) ##adl
            if type(x) is list: ##adl
                print(">>> len(" + cur_name + ") = ", len(x)) ##adl
                pprint(x) ##adl
            else: ##adl
                pprint(x) ##adl
                pass ##adl
            print("------------------------\n") ##adl
        
        # raise ValueError()
        out_img = torch.zeros((1, 1, h, w))
        img1_seg = img1.clone()
        img2_seg = img2.clone()
        img3_seg = img3.clone()
        img1_seg[0,0][mask_R <= 0] = 0
        img2_seg[0,0][mask_R <= 0] = 0
        img3_seg[0,0][mask_R <= 0] = 0

        gamma2 = _ssim(img1_seg, img3_seg, window, self.window_size, channel,
                     self.size_average)
        gamma1 = _ssim(img2_seg, img3_seg, window, self.window_size, channel,
                      self.size_average)
        print(">>>gamma2", gamma2)
        # loga(gamma2[0,0][mask_R > 0])
        # loga(gamma2)
        print(">>>gamma1", gamma1)
        # loga(gamma1[0, 0][mask_R > 0])
        #loga(gamma1)
        # raise ValueError()
     
        
#         gamma2, gamma1 = self._adjust_lag2_corrcoef2(gamma2, gamma1)
        phi2, phi1, phi0 = self.get_phi(gamma2, gamma1, mask_R)
        print(">>>phi2")
        print(phi2)
        # loga(phi2[0,0])
        print(">>>phi1")
        print(phi1)
        # loga(phi1[0,0])
        # raise ValueError()
        
        print("img2.shape", img2.shape)
        print("img3.shape", img3.shape)
        print("phi2.shape", phi2.shape)
        outx = 0
        outy = 0
        print("img2[0,0]", img2[0,0,outx,outy])
        print("img3[0,0]", img3[0,0,outx,outy])
        out = img2 * phi2 + img3 * phi1
        print("out[0,0]", out[0,0,outx,outy])
        out = u3 + out * sigma3

        # sleep(3)

        print(">>> new values")
        # loga(out[0,0][mask_R > 0].float())
    
        out_img[0,0][mask_R > 0.5] = out[0,0][mask_R > 0.5].float()

        # for i in range(50, 200):
        #     for j in range(50, 200):
        #         if mask_R[i, j] > 0:
        #             print(">>>i", i)
        #             print(">>>j", j)
        #             print(mask_R[i,j])
        #             raise ValueError()
        # raise ValueError()                
        # out_img[0,0][mask_R <= 0] = -15 / 8.0 #torch.mean(_img1[0,0][mask_R <= 0]).float()
        assert type(out_img) == type(mask_R)
        out_img[0,0][mask_R <= 0.5] = -15 #torch.mean(_img1[0,0][mask_R <= 0]).float()
        # out_img[0,0][mask_R <= 0.5] = -15 
        # check mask_R
        # 很大值(异常值)的时候, 紫色
        
        # out_img[0,0][mask_R > 0] = 3
        # out_img[0,0][mask_R <= 0] = 3
        import matplotlib.pyplot as plt
        # plt.imshow(out_img[0,0])
        # plt.show()
        # raise ValueError()
        # if out.is_cuda():
        #     out_img = out_img.cuda()
        return out_img
