from base.base_data_setter import BaseDataSetter
from GloVe.src.utils.preprocess import make_word2idx, make_corpus, make_cooccurrence_matrix

class GloVeDataSetter(BaseDataSetter):
    def __init__(self, left_size, right_size, min_occurrence, data_path):
        self.left_size = left_size
        self.right_size = right_size
        self.min_occurrence = min_occurrence
        self.vocab_size, self._coocurrence_matrix = self.set_coocurrence_matrix(data_path)

        super(GloVeDataSetter, self).__init__()

    def __getitem__(self, index):
        return self._coocurrence_matrix[index]

    def __len__(self):
        return len(self._coocurrence_matrix)

    def set_coocurrence_matrix(self, data_path):
        with open(data_path, "r") as f:
            doc = f.readlines()

        word2idx, vocab_size = make_word2idx(doc)
        corpus = make_corpus(doc, word2idx)
        cooccurrence_matrix = make_cooccurrence_matrix(
            corpus, self.left_size, self.right_size, vocab_size, self.min_occurrence
        )

        return vocab_size, cooccurrence_matrix