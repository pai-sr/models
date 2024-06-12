from base.base_data_setter import BaseDataSetter
from torchvision.datasets import MNIST

class MnistDataSetter(BaseDataSetter):
    def __init__(self, *args, **kwargs):
        self.mnist = MNIST(*args, **kwargs)

    def __getitem__(self, index):
        return self.mnist.__getitem__(index)

    def __len__(self):
        return len(self.mnist.data)