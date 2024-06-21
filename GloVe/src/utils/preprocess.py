from collections import defaultdict, Counter
from typing import List

def make_word2idx(doc: List[str] = None):
    vocab_size = 0
    word2idx = defaultdict(int)

    if doc is None:
        return

    tokens = set()
    for line in doc:
        tokens.update(line)

    for token in sorted(tokens):
        if token not in word2idx:
            word2idx[token] = vocab_size
            vocab_size += 1

    return word2idx, vocab_size

def make_corpus(doc: List[str], word2idx):
    corpus = [[word2idx[word] for word in line if word in word2idx] for line in doc]
    return corpus

def _window(region, start_index, end_index):
    last_index = len(region) + 1
    selected_tokens = region[max(start_index, 0):min(end_index, last_index) + 1]
    return selected_tokens

def _context_windows(region, left_size, right_size):
    for i, word in enumerate(region):
        start_index = i - left_size
        end_index = i + right_size
        left_context = _window(region, start_index, i - 1)
        right_context = _window(region, i + 1, end_index)
        yield (left_context, word, right_context)


def make_cooccurrence_matrix(corpus, left_size, right_size, vocab_size, min_occurrence):
    # co-occurence count matrix
    word_counts = Counter()
    cooccurrence_counts = defaultdict(float)
    for region in corpus:
        word_counts.update(region)
        for left_context, word, right_context in _context_windows(region, left_size, right_size):
            for i, context_word in enumerate(left_context[::-1]):
                # add (1 / distance from focal word) for this pair
                cooccurrence_counts[(word, context_word)] += 1 / (i + 1)
            for i, context_word in enumerate(right_context):
                cooccurrence_counts[(word, context_word)] += 1 / (i + 1)
    if len(cooccurrence_counts) == 0:
        raise ValueError(
            "No coccurrences in corpus, Did you try to reuse a generator?")

    # get words bag information
    tokens = [word for word, count in
              word_counts.most_common(vocab_size) if count >= min_occurrence]
    coocurrence_matrix = [(words[0], words[1], count)
                          for words, count in cooccurrence_counts.items()
                          if words[0] in tokens and words[1] in tokens]
    return coocurrence_matrix