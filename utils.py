import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

def minmaxscaler(x):
    x_min = x.min()
    x_max = x.max()

    return (x-x_min)/(x_max-x_min)

def psnr(a, b, max_val):
    '''
    Compute Peak Signal-to-Noise Ratio
    a: torch.tensor
    b: torch.tensor
    max_val: maximum value of image

    returns: single value
    '''
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    mse = torch.mean(((a-b)**2).flatten(start_dim=1,end_dim=-1), dim=1)
    v = 20*torch.log10(torch.tensor(max_val, dtype=torch.float32)) - 10*torch.log10(mse)
    return v

def psnr_shift(a,b, max_val, shift=1, shift_dir=''):
    '''
    Compute Peak Signal-to-Noise Ratio, mean value of PSNR shifting in 4~8 directions
    a: torch.tensor
    b: torch.tensor
    max_val: maximum value of image
    shift: number of pixels to shift

    returns: mean of psnr
    '''
    # ----------
    # directions
    #
    # ----------
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    # 1. original
    mse = torch.mean(((a-b)**2).flatten(start_dim=1,end_dim=-1), dim=1)
    v_o = 20*torch.log10(torch.tensor(max_val, dtype=torch.float32)) - 10*torch.log10(mse)

    mse_r = torch.mean(((a[...,:,shift:]-b[...,:,:-shift])**2).flatten(start_dim=1,end_dim=-1), dim=1)
    v_r = 20*torch.log10(torch.tensor(max_val, dtype=torch.float32)) - 10*torch.log10(mse_r)

    mse_l = torch.mean(((a[...,:,:-shift]-b[...,:,shift:])**2).flatten(start_dim=1,end_dim=-1), dim=1)
    v_l = 20*torch.log10(torch.tensor(max_val, dtype=torch.float32)) - 10*torch.log10(mse_l)

    mse_u = torch.mean(((a[...,:-shift,:]-b[...,shift:,:])**2).flatten(start_dim=1,end_dim=-1), dim=1)
    v_u = 20*torch.log10(torch.tensor(max_val, dtype=torch.float32)) - 10*torch.log10(mse_u)

    mse_d = torch.mean(((a[...,shift:,:]-b[...,:-shift,:])**2).flatten(start_dim=1,end_dim=-1), dim=1)
    v_d = 20*torch.log10(torch.tensor(max_val, dtype=torch.float32)) - 10*torch.log10(mse_d)

    v = torch.mean(torch.stack([v_o,v_r,v_l,v_u,v_d],dim=0),dim=0)
    # v_ = torch.stack([v_o,v_r,v_l,v_u,v_d],dim=0)
    return v

def g_function(w_size, sigma):
    g = torch.tensor([np.exp(-(x - w_size//2)**2/float(2*sigma**2)) for x in range(w_size)])
    return g/g.sum()

def create_window(w_size, c):
    _1D_window = g_function(w_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.tensor(_2D_window.expand(c, 1, w_size, w_size).contiguous())
    return window

def _ssim(img1, img2, window, w_size, c, size_average = True):
    mu1 = F.conv2d(img1, window, padding = w_size//2, groups = c)
    mu2 = F.conv2d(img2, window, padding = w_size//2, groups = c)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = w_size//2, groups = c) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = w_size//2, groups = c) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = w_size//2, groups = c) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, w_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.w_size = w_size
        self.size_average = size_average
        self.c = 1
        self.window = create_window(w_size, self.c)

    def forward(self, img1, img2):
        (_, c, _, _) = img1.size()

        if c == self.c and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.w_size, c)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.c = c


        return _ssim(img1, img2, window, self.w_size, c, self.size_average)

def ssim(img1, img2, w_size = 11, size_average = True):
    (_, c, _, _) = img1.size()
    window = create_window(w_size, c)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, w_size, c, size_average)

def filter_state_dict(state_dict):
    state_dict_ = {}
    for key, value in state_dict.items():
        key_ = '.'.join(key.split('.')[1:])
        state_dict_[key_] = value
    return state_dict_



# def filter_state_dict(checkpoint_name):
#     state_dict = torch.load('saved_models/%s.pth'%checkpoint_name)
#     state_dict_ = {}
#     for key, value in state_dict.items():
#         key_ = '.'.join(key.split('.')[1:])
#         print(key_)
#         state_dict_[key_] = value
#
#     torch.save(state_dict_, 'saved_models/%s_.pth'%checkpoint_name)
