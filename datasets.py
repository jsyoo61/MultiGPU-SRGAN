import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import glob


# Normalization parameters for pre-trained PyTorch model
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


class ImageDataset(Dataset):
    def __init__(self, root, hr_shape):
        hr_height, hr_width = hr_shape

        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height // 4, hr_height // 4), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.hr_transform_raw = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                transforms.ToTensor(),
            ]
        )
        self.hr_transform = transforms.Normalize(mean,std)

        self.files = sorted(glob.glob(root + "/*.*"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        print(index)
        img = Image.open(self.files[index % len(self.files)])
        img_lr = self.lr_transform(img)
        img_hr_raw = self.hr_transform_raw(img)
        img_hr = self.hr_transform(img_hr_raw)

        return {"lr": img_lr, "hr": img_hr, 'hr_raw': img_hr_raw}
