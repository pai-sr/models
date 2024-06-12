from torch.utils.data import DataLoader
from abc import abstractmethod

class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, *args, **kwargs):
        super(BaseDataLoader, self).__init__(*args, **kwargs)
