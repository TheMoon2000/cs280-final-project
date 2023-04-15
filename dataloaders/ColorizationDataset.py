import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import glob
import os

class ColorizationDataset(Dataset):
    def __init__(self, train: bool, width=256, height=256) -> None:
        super().__init__()

        self.train = train
        self.width = width
        self.height = height
        self.filenames = glob.glob('/data/jerryshan/imagenet_val/*/*.JPEG')
    
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        original = cv2.cvtColor(cv2.imread(self.filenames[index]), cv2.COLOR_BGR2RGB) # (H, W, 3)
        assert original.shape[0] >= self.height, original.shape[1] >= self.width
        original = original.astype(np.float32) / 255
        
        if self.train:
            # Crop identical size images for batch operations during training
            r = np.random.randint(0, original.shape[0] - self.height)
            c = np.random.randint(0, original.shape[1] - self.width)
        else:
            r = (original.shape[0] - self.height) // 2
            c = (original.shape[1] - self.width) // 2
        
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        original = original[r:r + self.height, c:c + self.width]
        gray = gray[r:r + self.height, c:c + self.width]

        return gray.reshape((1, *gray.shape)), original.transpose((2, 0, 1))
    
