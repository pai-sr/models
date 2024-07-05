import numpy as np
import pandas as pd

from base.base_data_setter import BaseDataSetter
from LLama3.src.utils.tokenizer import Tokenizer

class BoolQDataSetter(BaseDataSetter):
    train_dir = 'data/train-00000-of-00001.parquet'
    valid_dir = 'data/validation-00000-of-00001.parquet'
    root_dir = "hf://datasets/google/boolq/"

    def __init__(self, tokenizer_path, seq_len, valid=False):
        super().__init__()
        self.tokenizer_path = tokenizer_path
        self._load_data(valid)
        self._tokenize(seq_len)

    def _load_data(self, valid):
        if not valid:
            self.data = pd.read_parquet(self.root_dir + self.train_dir)
        else:
            self.data = pd.read_parquet(self.root_dir + self.valid_dir)

    def _tokenize(self, seq_len):
        tokenizer = Tokenizer(self.tokenizer_path)
        self.data['question'] = self.data['question'].apply(lambda x: tokenizer.encode(x, bos=True, eos=True, seq_len=seq_len))
        self.data['answer'] = self.data['answer'].apply(lambda x: int(x))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (np.array(self.data.iloc[idx]['question']),
                 np.expand_dims(np.array(self.data.iloc[idx]['answer']), axis=0))
