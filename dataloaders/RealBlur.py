from typing import Tuple
import cv2
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset

class RealBlurDataset(Dataset):
    def __init__(self, train=True, width=256, height=256, augmentation=None) -> None:
        super().__init__()

        self.train = train
        self.width = width
        self.height = height

        self.data_files: Tuple[str, str] = []
        with open(f'/data/jerryshan/RealBlur/RealBlur_J_{"train" if train else "test"}_list.txt', 'r') as f:
            for line in f.read().split('\n'):
                self.data_files.append(tuple(line.split()[:2]))
        
        self.aug = augmentation or transforms.ToTensor()
        
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, index):
        gt = cv2.cvtColor(cv2.imread('datasets/RealBlur/' + self.data_files[index][0]), cv2.COLOR_BGR2RGB)
        blur = cv2.cvtColor(cv2.imread('datasets/RealBlur/' + self.data_files[index][1]), cv2.COLOR_BGR2RGB)
        gt = gt.astype(np.float32) / 255
        blur = blur.astype(np.float32) / 255

        if self.train:
            # Crop identical size images for batch operations during training
            r = np.random.randint(0, gt.shape[0] - self.height)
            c = np.random.randint(0, gt.shape[1] - self.width)
        else:
            r = (gt.shape[0] - self.height) // 2
            c = (gt.shape[1] - self.width) // 2
        gt = gt[r:r + self.height, c:c + self.width]
        blur = blur[r:r + self.height, c:c + self.width]

        if self.train:
            data = self.aug( np.concatenate([blur, gt], axis=2))
        else:
            data = transforms.ToTensor()(np.concatenate([blur, gt], axis=2))
        

        return data[:3], data[3:]