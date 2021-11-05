from typing import Callable, List, Optional, Union

# import numpy as np
# from tensorflow.keras.utils import Sequence
from torch import tensor
from torch.utils.data import Dataset


class TorchTextDataset(Dataset):

    """
    Dataset class for text (pytorch based).
    Reference: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files

    Arguments:
        texts: list of texts;
        labels: list of labels;
        tokenizer: tokenizer function;
        target_transform: function to apply on raw label;
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[Union[int, List[int]]] = None,
        tokenizer: Optional[Callable[[str], dict]] = None,
        target_transform: Optional[Callable[[Union[int, List[int]]], tensor]] = None,
    ):

        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.target_transform = target_transform

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        text = self.texts[idx]
        text_transformed = self.tokenizer(text, "pytorch") if self.tokenizer else text

        # labels are optional because
        # you may want to simply classify unlabeled texts
        if not self.labels:
            return text_transformed

        else:
            label = self.labels[idx]
            label_transformed = (
                self.target_transform(label) if self.target_transform else label
            )

            return text_transformed, label_transformed


# class TFSimpleTextDataset(Sequence):

#     """
#     Simple Dataset object for text (tensorflow based).
#     Reference: https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence

#     Arguments:
#         texts: list of texts;
#         labels: list of labels;
#         tokenizer: tokenizer function;
#         target_transform: function to apply on raw label;
#     """

#     def __init__(
#         self,
#         texts: list,
#         labels: list,
#         batch_size: int = 32,
#         tokenizer=None,
#         target_transform=None,
#     ):
#         self.texts = texts
#         self.labels = labels
#         self.tokenizer = tokenizer
#         self.target_transform = target_transform
#         self.batch_size = batch_size

#     def __len__(self):
#         return int(np.ceil(len(self.texts) / self.batch_size))

#     def __getitem__(self, idx):

#         start_idx = idx * self.batch_size
#         end_idx = start_idx + self.batch_size

#         batch_texts = self.texts[start_idx:end_idx]
#         batch_labels = self.labels[start_idx:end_idx]

#         if self.tokenizer:
#             batch_texts = [self.tokenizer.encode(text) for text in batch_texts]

#         if self.target_transform:
#             batch_labels = [self.target_transform(label) for label in batch_labels]

#         return np.array(batch_texts), np.array(batch_labels)


class HFSimpleTextDataset(Dataset):

    """
    Simple Dataset object for text (hugging face based model especifically).
    Reference: https://huggingface.co/course/chapter3/3?fw=pt

    Arguments:
        tokenized_texts: dict containing necessary keys for hugging face model (input_ids, token_type_ids and attention_mask);
        labels: list of labels;
    """

    def __init__(
        self,
        tokenized_texts: dict,
        labels: Optional[List[int]] = None,
    ):

        self.tokenized_texts = tokenized_texts
        self.labels = labels

    def __len__(self):
        return len(self.tokenized_texts["input_ids"])

    def __getitem__(self, idx):

        # retrieve an item
        item = {key: tensor(val[idx]) for key, val in self.tokenized_texts.items()}
        if self.labels:
            item["labels"] = tensor(self.labels[idx])

        return item
