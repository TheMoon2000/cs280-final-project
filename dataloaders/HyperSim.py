import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import pandas as pd
import torch.nn as nn
import torch
import h5py
import skimage.transform

class HyperSimDataset(Dataset):
    def __init__(self, train: bool, width=224, height=224, dataset_root="/data/datasets/hypersim") -> None:
        super().__init__()

        self.dataset_root = dataset_root
        self.train = train
        self.width = width
        self.height = height
        self.samples = pd.read_csv('split.csv')
        self.samples = self.samples[self.samples['included_in_public_release'] & (self.samples['split_partition_name'] == ('train' if train else 'val'))].drop(['included_in_public_release', 'split_partition_name'], axis=1)

    
    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, index):
        scene_name,camera_name,frame_id,*_ = self.samples.iloc[index]
        img = cv2.cvtColor(cv2.imread(f"{self.dataset_root}/{scene_name}/images/scene_{camera_name}_final_preview/frame.{frame_id:04d}.color.jpg"), cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255

        with h5py.File(f"{self.dataset_root}/{scene_name}/images/scene_{camera_name}_geometry_hdf5/frame.{frame_id:04d}.semantic.hdf5") as f:
            label = np.array(f['dataset'])
            label[label != -1] -= 1

        if self.train:
            # Crop identical size images for batch operations during training
            r = np.random.randint(0, max(1, img.shape[0] - self.height * 2))
            c = np.random.randint(0, max(1, img.shape[1] - self.width * 2))
        else:
            r = max(0, (img.shape[0] - self.height * 2) // 2)
            c = max(0, (img.shape[1] - self.width * 2) // 2)
        
        img = img[r:r + self.height * 2, c:c + self.width * 2]
        label = label[r:r + self.height * 2, c:c + self.width * 2].astype(np.int64)

        img = skimage.transform.resize(img, (self.width, self.height), anti_aliasing=True)
        label = skimage.transform.resize(label, (self.width, self.height), anti_aliasing=False, order=0)

        return img.transpose((2, 0, 1)), label
    
class CrossEntropy2DLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.ce = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred = torch.flatten(pred.permute(0, 2, 3, 1), 0, 2)
        target = target.view(-1)
        return self.ce(pred, target)