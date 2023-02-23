from typing import Tuple
import cv2
import numpy as np
from torch.utils.data import Dataset

class RealBlurDataset(Dataset):
    def __init__(self, train=True, width=640, height=720) -> None:
        super().__init__()

        self.train = train
        self.width = width
        self.height = height

        self.data_files: Tuple[str, str] = []
        with open(f'datasets/RealBlur/RealBlur_J_{"train" if train else "test"}_list.txt', 'r') as f:
            for line in f.read().split('\n'):
                self.data_files.append(tuple(line.split()[:2]))
        
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
            gt = gt[r:r + self.height, c:c + self.width]
            blur = blur[r:r + self.height, c:c + self.width]

        return np.transpose(blur, (2, 0, 1)), np.transpose(gt, (2, 0, 1))