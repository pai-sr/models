import torch.nn as nn
from abc import abstractmethod

class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self):
        """
        Forward pass logic
        :return: Model output
        """
        raise NotImplementedError