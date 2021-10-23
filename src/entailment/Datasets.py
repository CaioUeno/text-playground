from typing import Any, Callable, List, Optional, Tuple

from torch import tensor
from torch.utils.data import Dataset


class TorchEntailmentTextDataset(Dataset):

    """
    Simple Dataset object for text (pytorch based).
    Reference: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files

    Arguments:
        pairs: list of tuples (text, text);
        tokenizer: tokenizer function;
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        similarities: List[float],
        tokenizer: Optional[Callable[[str], Any]] = None,
    ):

        self.pairs = pairs
        self.similarities = similarities
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        (sentence_a, sentence_b), similarity = self.pairs[idx], self.similarities[idx]

        if self.tokenizer:
            # not necessarily text (a string) anymore
            # just to avoid a new variable declaration
            # and some checks
            sentence_a = tensor(self.tokenizer(sentence_a))
            sentence_b = tensor(self.tokenizer(sentence_b))

        return (sentence_a, sentence_b), similarity
