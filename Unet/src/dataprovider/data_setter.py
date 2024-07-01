import os
from glob import glob
import numpy as np
from PIL import Image
import cv2
import scipy.io as sio
import torch
from torchvision import transforms

from base.base_data_setter import BaseDataSetter
from Unet.src.utils import palette


class CocoStuff10k(BaseDataSetter):
    def __init__(self, warp_image = True, **kwargs):
        self.warp_image = warp_image
        self.num_classes = 182
        self.palette = palette.COCO_palette
        self.__dict__.update(kwargs)
        self._set_files()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(self.mean, self.std)

    def _set_files(self):
        if self.split in ['train', 'test', 'all']:
            file_list = os.path.join(self.root, 'imageLists', self.split + '.txt')
            self.files = [name.rstrip() for name in tuple(open(file_list, "r"))]
        else:
            raise ValueError(f"Invalid split name {self.split} choose one of [train, test, all]")

    def _load_data(self, index):
        image_id = self.files[index]
        image_path = os.path.join(self.root, 'images', image_id + '.jpg')
        label_path = os.path.join(self.root, 'annotations', image_id + '.mat')
        image = np.asarray(Image.open(image_path), dtype=np.float32)
        label = sio.loadmat(label_path)['S']
        label -= 1  # unlabeled (0 -> -1)
        label[label == -1] = 255
        if self.warp_image:
            image = cv2.resize(image, (513, 513), interpolation=cv2.INTER_LINEAR)
            label = np.asarray(Image.fromarray(label).resize((513, 513), resample=Image.NEAREST))
        if len(image.shape) == 2: # 이미지 shape 안맞아서 추가
            image = np.repeat(image[:,:,np.newaxis],3,-1)
        return image, label, image_id

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image, label, image_id = self._load_data(index)
        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        image = Image.fromarray(np.uint8(image))
        if self.return_id:
            return self.normalize(self.to_tensor(image)), label, image_id
        return self.normalize(self.to_tensor(image)), label

class CocoStuff164k(BaseDataSetter):
    def __init__(self, **kwargs):
        self.num_classes = 182
        self.palette = palette.COCO_palette
        self.__dict__.update(kwargs)
        self._set_files()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(self.mean, self.std)

    def _set_files(self):
        if self.split in ['train2017', 'val2017']:
            file_list = sorted(glob(os.path.join(self.root, 'images', self.split + '/*.jpg')))
            self.files = [os.path.basename(f).split('.')[0] for f in file_list]
        else:
            raise ValueError(f"Invalid split name {self.split}, either train2017 or val2017")

    def _load_data(self, index):
        image_id = self.files[index]
        image_path = os.path.join(self.root, 'images', self.split, image_id + '.jpg')
        label_path = os.path.join(self.root, 'annotations', self.split, image_id + '.png')
        image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        return image, label, image_id

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image, label, image_id = self._load_data(index)
        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        image = Image.fromarray(np.uint8(image))
        if self.return_id:
            return self.normalize(self.to_tensor(image)), label, image_id
        return self.normalize(self.to_tensor(image)), label
