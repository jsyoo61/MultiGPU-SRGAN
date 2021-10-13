import os
import argparse
from models import *
from datasets import *
from tools.tools import Timer, AverageMeter, tdict, write
import socket
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

def main():

    os.makedirs("images_test_distributed", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_interval", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--checkpoint_name", type=str, default='default')
    parser.add_argument("--grayscale", type=eval, default='True', help="grayscale or RGB")

    parser.add_argument('-n', '--nodes', default=1, type=int)
    parser.add_argument('-g', '--gpus', default=1, type=int)
    parser.add_argument('-nr', '--nr', default=0, type=int)
    parser.add_argument('--master_addr', default=str(socket.gethostbyname(socket.gethostname())), type=str, help='master ip address')
    parser.add_argument('--master_port', default='8888', type=str, help='master port')
    args = parser.parse_args()

    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = '20000'
    print(args)
    mp.spawn(train, nprocs=args.gpus, args=(args,))

def train(gpu, args):
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    torch.manual_seed(0)

    generator = GeneratorResNet()

    torch.cuda.set_device(gpu)
    generator.cuda(gpu)
    generator.load_state_dict(filter_state_dict(torch.load('saved_models/generator_%s.pth' % args.checkpoint_name)))

    # Wrap the model
    generator = nn.parallel.DistributedDataParallel(generator, device_ids=[gpu])

    # Dataloader
    dataset = ImageDataset("../../data/img_align_celeba_eval", hr_shape=(256, 256))
    sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=sampler)

    torch.autograd.set_detect_anomaly(True)
    total_step = len(loader)

    psnr_gen_list = []
    psnr_lr_list = []
    ssim_gen_list = []
    ssim_lr_list = []

    iter_timer = Timer()
    iter_time_meter = AverageMeter()

    for i, imgs in enumerate(loader):
        with torch.no_grad():
            iter_timer.start()

            imgs_lr = imgs["lr"].cuda(non_blocking=True)
            imgs_hr = imgs["hr"].cuda(non_blocking=True)
            imgs_hr_raw = imgs['hr_raw'].cuda(non_blocking=True)

            gen_hr = generator(imgs_lr)

            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
            imgs_lr = minmaxscaler(imgs_lr)
            imgs_hr = minmaxscaler(imgs_hr)
            gen_hr = minmaxscaler(gen_hr)

            if args.grayscale:
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
            if gpu==0 :
                batches_done = i
                print('[Batch %d/%d] time for iteration: %.4f (%.4f) (sum: %.4f)'%(i, len(loader), iter_time_meter.val, iter_time_meter.avg, iter_time_meter.sum))

                if batches_done % args.sample_interval == 0:
                    print('%10s %10s %10s'%('','PSNR','SSIM'))
                    print('%10s %10.4f %10.4f'%('generator',psnr_gen.mean().item(),ssim_gen.mean().item()))
                    print('%10s %10.4f %10.4f'%('low res',psnr_lr.mean().item(),ssim_lr.mean().item()))

                    # Save image grid with upsampled inputs and SRGAN outputs
                    gen_hr = make_grid(gen_hr[:4], nrow=1, normalize=True)
                    imgs_lr = make_grid(imgs_lr[:4], nrow=1, normalize=True)
                    imgs_hr_raw = make_grid(imgs_hr_raw[:4], nrow=1, normalize=True)
                    img_grid = torch.cat((imgs_hr_raw, imgs_lr, gen_hr), -1)
                    save_image(img_grid, "images_test_distributed/%d.png" % batches_done, normalize=False)

    psnr_gen = torch.cat(psnr_gen_list).mean().item()
    psnr_lr = torch.cat(psnr_lr_list).mean().item()
    ssim_gen = torch.cat(ssim_gen_list).mean().item()
    ssim_lr = torch.cat(ssim_lr_list).mean().item()

    write(' '.join(list(map(str, [psnr_gen, psnr_lr, ssim_gen, ssim_lr, iter_time_meter.sum]))), str(rank)+'txt')

if __name__ == '__main__':
    main()

'''
49.4598 seconds
'''
