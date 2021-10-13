# %%
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from models import *
from datasets import *

from tools.tools import tdict, Timer, append, AverageMeter
from utils import *

cuda = torch.cuda.is_available()

# %%
opt = tdict()
opt.n_epochs=10
opt.dataset_name='img_align_celeba'
opt.batch_size=4
opt.batch_m=4
opt.lr=0.0002
opt.b1=0.5
opt.b2=0.999
opt.n_cpu=8
opt.hr_height=256
opt.hr_width=256
opt.channels=3
opt.validation_interval=100
opt.checkpoint_name='original'
print(opt)

# %%
hr_shape = (opt.hr_height, opt.hr_width)

# Create model
generator = GeneratorResNet()
discriminator = Discriminator(input_shape=(opt.channels, *hr_shape))
feature_extractor = FeatureExtractor()
discriminator_output_shape = discriminator.output_shape
print('number of parameters: %s'%sum(p.numel() for p in generator.parameters()))
print('number of parameters: %s'%sum(p.numel() for p in discriminator.parameters()))

# Do not train feature extractor
feature_extractor.eval()

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_content = torch.nn.L1Loss()

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    feature_extractor = feature_extractor.cuda()
    criterion_GAN = criterion_GAN.cuda()
    criterion_content = criterion_content.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, hr_shape=hr_shape),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

global_timer = Timer()
epoch_timer = Timer()
iter_timer = Timer()
iter_time_meter = AverageMeter()

# %% Training
global_timer.start()
for epoch in range(opt.n_epochs):
    epoch_timer.start()
    for i, imgs in enumerate(dataloader):
        if i % opt.batch_m == 0:
            iter_timer.start()
        # Configure model input
        imgs_lr = imgs["lr"].type(Tensor)
        imgs_hr = imgs["hr"].type(Tensor)

        # Adversarial ground truths
        valid = torch.ones((imgs_lr.size(0), *discriminator_output_shape)).type(Tensor)
        fake = torch.zeros((imgs_lr.size(0), *discriminator_output_shape)).type(Tensor)

        if i % opt.batch_m == 0:
            print('zero_grad_G')
            optimizer_G.zero_grad()

        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)

        # Adversarial loss
        loss_GAN = criterion_GAN(discriminator(gen_hr), valid)

        # Content loss
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr)
        loss_content = criterion_content(gen_features, real_features.detach())

        # Total loss
        loss_G = loss_content + 1e-3 * loss_GAN

        loss_G = loss_G / opt.batch_m
        loss_G.backward()
        if (i+1) % opt.batch_m == 0:
            print('step_G')
            optimizer_G.step()

        if i % opt.batch_m == 0:
            print('zero_grad_D')
            optimizer_D.zero_grad()

        # Loss of real and fake images
        loss_real = criterion_GAN(discriminator(imgs_hr), valid)
        loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        loss_D = loss_D / opt.batch_m
        loss_D.backward()
        if (i+1) % opt.batch_m == 0:
            print('step_D')
            optimizer_D.step()


        # Logging
        if (i+1) % opt.batch_m == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), loss_D.item()*opt.batch_m, loss_G.item()*opt.batch_m)
            )
            iter_time_meter.update(iter_timer.stop())
            print('time for iteration: %s (%s)'%(iter_time_meter.val, iter_time_meter.avg))

        if i % opt.validation_interval == 0:
            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
            imgs_hr_raw = imgs['hr_raw'].type(Tensor)
            with torch.no_grad():
                print('[psnr] (imgs_lr):%s, (gen_hr):%s'%(psnr(minmaxscaler(imgs_lr), imgs_hr_raw, max_val=1).mean().item(), psnr(minmaxscaler(gen_hr), imgs_hr_raw, max_val=1).mean().item()))

            imgs_hr_raw = make_grid(imgs_hr_raw, nrow=1, normalize=True)
            gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
            imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
            img_grid = torch.cat((imgs_hr_raw, imgs_lr, gen_hr), -1)
            save_image(img_grid, "images/%d.png" % i, normalize=False)

    print('Elapsed_time for epoch(%s): %s'%epoch_timer.stop())

elapsed_time = global_timer.stop()
print('Elapsed_time for training: %s'%str(elapsed_time))
append(str(elapsed_time), 'elapsed_time.txt')
print('Average time per iteration: %s'%str(iter_time_meter.avg))
torch.save(generator.state_dict(), "saved_models/generator_%s.pth" % opt.checkpoint_name)
torch.save(discriminator.state_dict(), "saved_models/discriminator_%s.pth" % opt.checkpoint_name)

'''
batch_size 32, batch_m=4 -> 128 batch
lr = 0.0002
checkpoint_name = original

                 PSNR       SSIM
 generator    22.8740     0.7806
   low res    26.2965     0.7801

time per iteration: 3.997
'''
