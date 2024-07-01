import os
from pathlib import Path
from typing import (
    AbstractSet,
    cast,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Sequence,
    Union,
)

import tiktoken
from tiktoken.load import load_tiktoken_bpe


class Tokenizer:
    """ Tokenizing and encoding/decoding text using the Tiktoken tokenizer """

    special_tokens: Dict[str, int]
    # number of reserved special tokens
    num_reserved_special_tokens = 256
    # regex pattern for splitting the text
    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

    def __init__(self, model_path: str):
        assert os.path.isfile(model_path), model_path

        # loading existing tiktoken model
        mergeable_ranks = load_tiktoken_bpe(model_path)
        # number of base tokens = number of tokens in the existing tiktoken model
        num_base_tokens = len(mergeable_ranks)
        # list of some special tokens we will add to the tokenizer
        special_tokens = [
                             "<|begin_of_text|>",
                             "<|end_of_text|>",
                             "<|reserved_special_token_0|>",
                             "<|reserved_special_token_1|>",
                             "<|reserved_special_token_2|>",
                             "<|reserved_special_token_3|>",
                             "<|start_header_id|>",
                             "<|end_header_id|>",
                             "<|reserved_special_token_4|>",
                             "<|eot_id|>",  # end of turn
                         ] + [
                             f"<|reserved_special_token_{i}|>"
                             for i in range(5, self.num_reserved_special_tokens - 5)
                         ]
        # creating a dictionary of special tokens mentioned above
        self.special_tokens = {token: num_base_tokens + i for i, token in enumerate(special_tokens)}

        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )

        # vocabulary size
        self.n_words: int = self.model.n_vocab

        # BOS / EOS token IDs
        self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self.eos_id: int = self.special_tokens["<|end_of_text|>"]
        self.pad_id: int = -1
        self.stop_tokens = {
            self.special_tokens["<|end_of_text|>"],
            self.special_tokens["<|eot_id|>"],
        }

    def encode(
            self,
            s: str,
            *,
            bos: bool,
            eos: bool,
            allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
            disallowed_special: Union[Literal["all"], Collection[str]] = (),
    ) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.
            allowed_tokens ("all"|set[str]): allowed special tokens in string
            disallowed_tokens ("all"|set[str]): special tokens that raise an error when in string

        Returns:
            list[int]: A list of token IDs.

        By default, setting disallowed_special=() encodes a string by ignoring
        special tokens. Specifically:
        - Setting `disallowed_special` to () will cause all text corresponding
          to special tokens to be encoded as natural text (insteading of raising
          an error).
        - Setting `allowed_special` to "all" will treat all text corresponding
          to special tokens to be encoded as special tokens.
        """

        assert type(s) is str, "input must be string"

        # the tiktoken tokenizer can handle <=400k chars without pyo3_runtime.PanicException
        TIKTOKEN_MAX_ENCODE_CHARS = 400_000

        # max number of consecutive whitespace characters in a substring
        MAX_NUM_WHITESPACES_CHARS = 25_000

        # iterating over subsequences and splitting if we exceed the limit of max consecutive non-whitespace or whitespace characters
        substrs = (
            substr
            for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
            for substr in self._split_whitespaces_or_nonwhitespaces(
            s[i: i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NUM_WHITESPACES_CHARS
        )
        )
        # list of token ids
        t: List[int] = []
        for substr in substrs:
            t.extend(
                self.model.encode(
                    substr,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )

        # prepending the beginning-of-sequence token
        if bos:
            t.insert(0, self.bos_id)

        # appending the end-of-sequence token
        if eos:
            t.append(self.eos_id)
        return t

    def decode(self, t: Sequence[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        # Typecast is safe here. Tiktoken doesn't do anything list-related with the sequence.
        return self.model.decode(cast(List[int], t))

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(
            s: str, max_consecutive_slice_len: int
    ) -> Iterator[str]:
        """
        Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
        consecutive whitespaces or consecutive non-whitespaces.
        """
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        slice_start = 0

        for i in range(len(s)):
            is_now_space = s[i].isspace()

            if current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > max_consecutive_slice_len:
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        yield s[slice_start:]