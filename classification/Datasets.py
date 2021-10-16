from torch.utils.data import Dataset
from torch import tensor
from tensorflow.keras.utils import Sequence
import numpy as np


class TorchSimpleTextDataset(Dataset):

    """
    Simple Dataset object for text (pytorch based).
    Reference: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files

    Arguments:
        texts: list of texts;
        labels: list of labels;
        tokenizer: tokenizer function;
        target_transform: function to apply on raw label;
    """

    def __init__(
        self, texts: list, labels: list, tokenizer=None, target_transform=None
    ):

        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.target_transform = target_transform

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        text, label = self.texts[idx], self.labels[idx]

        if self.tokenizer:
            text = tensor(self.tokenizer.encode(text))

        if self.target_transform:
            label = self.target_transform(label)

        return text, label


class TFSimpleTextDataset(Sequence):

    """
    Simple Dataset object for text (tensorflow based).
    Reference: https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence

    Arguments:
        texts: list of texts;
        labels: list of labels;
        tokenizer: tokenizer function;
        target_transform: function to apply on raw label;
    """

    def __init__(
        self,
        texts: list,
        labels: list,
        batch_size: int = 32,
        tokenizer=None,
        target_transform=None,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.target_transform = target_transform
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.texts) / self.batch_size))

    def __getitem__(self, idx):

        start_idx = idx * self.batch_size
        end_idx = start_idx + self.batch_size

        batch_texts = self.texts[start_idx:end_idx]
        batch_labels = self.labels[start_idx:end_idx]

        if self.tokenizer:
            batch_texts = [self.tokenizer.encode(text) for text in batch_texts]

        if self.target_transform:
            batch_labels = [self.target_transform(label) for label in batch_labels]

        return np.array(batch_texts), np.array(batch_labels)

class SimpleTextDataset(Dataset):

    def __init__(self, tokenized_text, labels=None):
        
        self.tokenized_text = tokenized_text
        self.labels = labels

    def __getitem__(self, idx):

        item = {key: tensor(val[idx]) for key, val in self.tokenized_text.items()}
        
        if self.labels:
            item["labels"] = tensor(self.labels[idx])
        
        return item

    def __len__(self):
        return len(self.tokenized_text["input_ids"])