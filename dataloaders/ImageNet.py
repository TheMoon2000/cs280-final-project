import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import glob

class ImageNet(Dataset):
    def __init__(self) -> None:
        super().__init__()

        self.filenames = glob.glob('/Users/sagarasanghavi/Desktop/Photos/*.jpg')
    
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        original = cv2.imread(self.filenames[index]) # (H, W, 3)
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        return gray[None], original.transpose((2, 0, 1))
    
