import torch
import torch.nn as nn
from base.base_model import BaseModel

class GloVeModel(BaseModel):
    def __init__(self, embedding_size, context_size, vocab_size, min_occurrence=1, x_max=100, alpha=3/4):
        super(GloVeModel, self).__init__()

        self.embedding_size = embedding_size
        if isinstance(context_size, tuple):
            self.left_context, self.right_context = context_size
        if isinstance(context_size, int):
            self.left_context = self.right_context = context_size
        else:
            raise ValueError(
                "'context_size' should be an int or a tuple of two ints")
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.min_occurrance = min_occurrence
        self.x_max = x_max

        self._focal_embeddings = nn.Embedding(self.vocab_size, self.embedding_size).type(torch.float64)
        self._context_embeddings = nn.Embedding(self.vocab_size, self.embedding_size).type(torch.float64)
        self._focal_bias = nn.Embedding(self.vocab_size, 1).type(torch.float64)
        self._context_bias = nn.Embedding(self.vocab_size, 1).type(torch.float64)

        for params in self.parameters():
            nn.init.uniform_(params, a=-1, b=1)

    def forward(self, x):
        i_, j_, counts = x

        focal_embed = self._focal_embeddings(i_)
        context_embed = self._context_embeddings(j_)
        focal_bias = self._focal_bias(i_)
        context_bias = self._context_bias(j_)

        return (focal_embed, context_embed, focal_bias, context_bias, counts)