import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os

class ColorizationDataset(Dataset):
    def __init__(self, train: bool, width=224, height=224, dataset_root="/data/jerryshan/VOCdevkit/VOC2012") -> None:
        super().__init__()

        self.dataset_root = dataset_root
        self.train = train
        self.width = width
        self.height = height
        with open(f'{self.dataset_root}/ImageSets/Main/{"train" if train else "val"}.txt') as f:
            self.filenames = f.read().splitlines()
    
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        original = cv2.cvtColor(cv2.imread(f"{self.dataset_root}/JPEGImages/{self.filenames[index]}.jpg"), cv2.COLOR_BGR2RGB)
        # assert original.shape[0] >= self.height and original.shape[1] >= self.width, f"{original.shape} too small"
        original = original.astype(np.float32) / 255
        if self.train:
            # Crop identical size images for batch operations during training
            r = np.random.randint(0, max(1, original.shape[0] - self.height))
            c = np.random.randint(0, max(1, original.shape[1] - self.width))
        else:
            r = max(0, (original.shape[0] - self.height) // 2)
            c = max(0, (original.shape[1] - self.width) // 2)
        
        original = original[r:r + self.height, c:c + self.width]
        original = cv2.resize(original, (self.height, self.width))
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        return gray.reshape((1, *gray.shape)).repeat(3, axis=0), original.transpose((2, 0, 1))
    
