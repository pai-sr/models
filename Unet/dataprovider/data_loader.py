import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from base.base_data_loader import BaseDataLoader
from Unet.dataprovider.data_setter import CocoStuff10k, CocoStuff164k

class VisionDataLoader(BaseDataLoader):
    def __init__(self, dataset, batch_size, num_workers=1, val_split=0.0, shuffle=True, *args, **kwargs):
        self.dataset = dataset
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.nbr_examples = len(self.dataset)
        if self.val_split:
            self.train_sampler, self.val_sampler = self._split_sampler(self.val_split)
        else:
            self.train_sampler, self.val_sampler = None, None

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'num_workers': self.num_workers,
            'pin_memory': True
        }
        super(VisionDataLoader, self).__init__(sampler=self.train_sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        self.shuffle = False

        split_indx = int(self.nbr_examples * split)
        np.random.seed(0)

        indxs = np.arange(self.nbr_examples)
        np.random.shuffle(indxs)
        train_indxs = indxs[split_indx:]
        val_indxs = indxs[:split_indx]
        self.nbr_examples = len(train_indxs)

        train_sampler = SubsetRandomSampler(train_indxs)
        val_sampler = SubsetRandomSampler(val_indxs)
        return train_sampler, val_sampler

    def get_val_loader(self):
        if self.val_sampler is None:
            return None
        # self.init_kwargs['batch_size'] = 1
        return DataLoader(sampler=self.val_sampler, **self.init_kwargs)

class COCO(VisionDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1, partition = 'CocoStuff164k',
                    shuffle=False, flip=False, rotate=False, blur= False, augment=False, val_split= None, return_id=False, val=False):

        self.MEAN = [0.43931922, 0.41310471, 0.37480941]
        self.STD = [0.24272706, 0.23649098, 0.23429529]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val
        }

        if partition == 'CocoStuff10k':
            self.dataset = CocoStuff10k(**kwargs)
        elif partition == 'CocoStuff164k':
            self.dataset = CocoStuff164k(**kwargs)
        else:
            raise ValueError(f"Please choose either CocoStuff10k / CocoStuff164k")

        super(VisionDataLoader, self).__init__(dataset=self.dataset,
                                               batch_size=batch_size,
                                               shuffle=shuffle,
                                               num_workers=num_workers)