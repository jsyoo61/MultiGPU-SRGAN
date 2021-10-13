# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision.utils import save_image, make_grid

from datasets import *
from models import *

from utils import *
from tools.tools import Timer, AverageMeter, tdict
import os
import pdb
import argparse

os.makedirs("images_test", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--sample_interval", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--checkpoint_name", type=str, default='default')
parser.add_argument("--grayscale", type=eval, default='True', help="grayscale or RGB")
parser.add_argument('--cuda', type=int, default=0, help='gpu number')
opt = parser.parse_args()
print(opt)
print('Grayscale: %s'%opt.grayscale)

cuda = torch.cuda.is_available()
torch.cuda.set_device(opt.cuda)

generator = GeneratorResNet()
generator.cuda()
generator.load_state_dict(torch.load('saved_models/generator_%s.pth' % opt.checkpoint_name))

dataloader = DataLoader(ImageDataset('../../data/img_align_celeba_eval', hr_shape=(256,256)),
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=8,
)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

psnr_gen_list = []
psnr_lr_list = []
ssim_gen_list = []
ssim_lr_list = []

iter_timer = Timer()
iter_time_meter = AverageMeter()

for i, imgs in enumerate(dataloader):
    with torch.no_grad():
        iter_timer.start()

        imgs_lr = imgs['lr'].type(Tensor)
        imgs_hr = imgs['hr'].type(Tensor)
        imgs_hr_raw = imgs['hr_raw'].type(Tensor)

        gen_hr = generator(imgs_lr)

        imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
        imgs_lr = minmaxscaler(imgs_lr)
        imgs_hr = minmaxscaler(imgs_hr)
        gen_hr = minmaxscaler(gen_hr)

        if opt.grayscale:
            gen_hr_gray = gen_hr.mean(dim=1)[:,None,:,:]
            imgs_lr_gray = imgs_lr.mean(dim=1)[:,None,:,:]
            imgs_hr_raw_gray = imgs_hr_raw.mean(dim=1)[:,None,:,:]

            psnr_gen = psnr(gen_hr_gray, imgs_hr_raw_gray, max_val=1)
            psnr_lr = psnr(imgs_lr_gray, imgs_hr_raw_gray, max_val=1)
            ssim_gen = ssim(gen_hr_gray, imgs_hr_raw_gray, size_average=False)
            ssim_lr = ssim(imgs_lr_gray, imgs_hr_raw_gray, size_average=False)
        else:
            psnr_gen = psnr(gen_hr, imgs_hr_raw, max_val=1)
            psnr_lr = psnr(imgs_lr, imgs_hr_raw, max_val=1)
            ssim_gen = ssim(gen_hr, imgs_hr_raw, size_average=False)
            ssim_lr = ssim(imgs_lr, imgs_hr_raw, size_average=False)

        psnr_gen_list.append(psnr_gen)
        psnr_lr_list.append(psnr_lr)
        ssim_gen_list.append(ssim_gen)
        ssim_lr_list.append(ssim_lr)

        iter_time_meter.update(iter_timer.stop())
        print('[Batch %d/%d] time for iteration: %.4f (%.4f) (sum: %.4f)'%(i, len(dataloader), iter_time_meter.val, iter_time_meter.avg, iter_time_meter.sum))
        batches_done = i
        if batches_done % opt.sample_interval == 0:
            print('%10s %10s %10s'%('','PSNR','SSIM'))
            print('%10s %10.4f %10.4f'%('generator',psnr_gen.mean().item(),ssim_gen.mean().item()))
            print('%10s %10.4f %10.4f'%('low res',psnr_lr.mean().item(),ssim_lr.mean().item()))

            # Save image grid with upsampled inputs and SRGAN outputs
            gen_hr = make_grid(gen_hr[:4], nrow=1, normalize=True)
            imgs_lr = make_grid(imgs_lr[:4], nrow=1, normalize=True)
            imgs_hr_raw = make_grid(imgs_hr_raw[:4], nrow=1, normalize=True)
            img_grid = torch.cat((imgs_hr_raw, imgs_lr, gen_hr), -1)
            save_image(img_grid, "images_test/%d.png" % batches_done, normalize=False)

psnr_gen = torch.cat(psnr_gen_list).mean().item()
psnr_lr = torch.cat(psnr_lr_list).mean().item()
ssim_gen = torch.cat(ssim_gen_list).mean().item()
ssim_lr = torch.cat(ssim_lr_list).mean().item()
print('-'*32)
print('Grayscale: %s'%opt.grayscale)
print('%10s %10s %10s'%('','PSNR','SSIM'))
print('%10s %10.4f %10.4f'%('generator',psnr_gen,ssim_gen))
print('%10s %10.4f %10.4f'%('low res',psnr_lr,ssim_lr))
print('%10s %10.4f %10.4f'%('time', iter_time_meter.avg, iter_time_meter.sum))

'''
205.9635 seconds
'''
