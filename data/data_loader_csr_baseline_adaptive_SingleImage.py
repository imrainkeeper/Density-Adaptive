import os
import random
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import h5py
import cv2


class SingleImageDataset(Dataset):
    def __init__(self, img_path, gt_path, point_map_path, transform=None):
        self.img_path = img_path
        self.gt_path = gt_path
        self.point_map_path = point_map_path
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, index):
        assert index < 1, 'index range error'
        img = Image.open(self.img_path).convert('RGB')
        img = img.resize((1024, 1024), Image.ANTIALIAS)

        img_temp = img.copy()
        mini_img = img_temp.resize((128, 128), Image.ANTIALIAS)

        point_map_file = h5py.File(self.point_map_path, 'r')
        point_map = np.asarray(point_map_file['point_map'])

        gt_file = h5py.File(self.gt_path, 'r')
        gt_density_map = np.asarray(gt_file['density'])

        original_gt_sum = np.sum(gt_density_map)
        gt_density_map = cv2.resize(gt_density_map, (128, 128), interpolation=cv2.INTER_AREA)
        current_gt_sum = np.sum(gt_density_map)
        gt_density_map = gt_density_map * (original_gt_sum / current_gt_sum)

        # gt_density_map = gt_density_map.reshape((1, gt_density_map.shape[0], gt_density_map.shape[1]))
        # gt_density_map = gt_density_map.astype(np.float32, copy=False)

        if img is None:
            print('Unable to read image %s, Exiting ...', img_path)
            exit(0)
        if self.transform is not None:
            img = self.transform(img)
            mini_img = self.transform(mini_img)

        return img, mini_img, gt_density_map, point_map
