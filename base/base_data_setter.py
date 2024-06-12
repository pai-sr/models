from torch.utils.data import Dataset
from abc import abstractmethod

class BaseDataSetter(Dataset):
    def __init__(self, *args, **kwargs):
        super(BaseDataSetter, self).__init__(*args, **kwargs)

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError