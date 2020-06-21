import os
import random
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import h5py
import cv2


class ImageDataset(Dataset):
    def __init__(self, img_dir, gt_dir, point_map_dir, train=False, transform=None):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.point_map_dir = point_map_dir
        self.img_names = [img_name for img_name in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, img_name))]
        self.train = train

        if self.train:        # 训练时数据扩增
            # self.img_names = self.img_names * 4
            pass
        if self.train:
            random.shuffle(self.img_names)
        self.transform = transform
        assert len(self.img_names) > 0

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        assert index < len(self.img_names), 'index range error'
        img_path = os.path.join(self.img_dir, self.img_names[index])
        img = Image.open(img_path).convert('RGB')
        img = img.resize((512, 512), Image.ANTIALIAS)

        img_temp = img.copy()
        mini_img = img_temp.resize((128, 128), Image.ANTIALIAS)

        point_map_path = os.path.join(self.point_map_dir, self.img_names[index].replace('.jpg', '_point_map.h5'))
        point_map_file = h5py.File(point_map_path, 'r')
        point_map = np.asarray(point_map_file['point_map'])

        gt_path = os.path.join(self.gt_dir, self.img_names[index].replace('.jpg', '.h5'))
        gt_file = h5py.File(gt_path, 'r')
        gt_density_map = np.asarray(gt_file['density'])

        original_gt_sum = np.sum(gt_density_map)
        gt_density_map = cv2.resize(gt_density_map, (128, 128), interpolation=cv2.INTER_AREA)
        current_gt_sum = np.sum(gt_density_map)
        gt_density_map = gt_density_map * (original_gt_sum / current_gt_sum)

        gt_density_map = gt_density_map.reshape((1, gt_density_map.shape[0], gt_density_map.shape[1]))
        gt_density_map = gt_density_map.astype(np.float32, copy=False)

        if img is None:
            print('Unable to read image %s, Exiting ...', img_path)
            exit(0)
        if self.transform is not None:
            img = self.transform(img)
            mini_img = self.transform(mini_img)

        return img, mini_img, gt_density_map, point_map
