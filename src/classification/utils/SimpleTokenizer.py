from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from torch import tensor
import numpy as np


class SimpleTokenizer(object):

    """
    Simple sentence tokenizer using hugginface tokenizers library.
    Reference: https://huggingface.co/docs/tokenizers/python/latest/quicktour.html

    Arguments:
        max_length: max sentence length;
        unk_token: token for unknown words;
        special_tokens: list of other special tokens [padding, mask, ...] (unk_token included);
        only_ids: whether to return only words ids or ids and masks.
    """

    def __init__(
        self,
        max_length: int = 32,
        unk_token: str = "[UNK]",
        special_tokens: list = ["[UNK]", "[BGN]", "[END]", "[PAD]", "[MASK]"],
    ):

        self.tokenizer = Tokenizer(BPE(unk_token=unk_token))
        self.tokenizer.pre_tokenizer = Whitespace()
        self.special_tokens = special_tokens
        self.max_length = max_length

    def fit(
        self,
        texts: list,
        padding: bool = True,
        truncation: bool = True,
        post_processing: bool = False,
    ):

        """
        Train the tokenizer based on a training text set.

        Arguments:
            texts: list of texts (sentences);
            padding: whether to enable padding or not (padding token passed on initialization);
            truncation: whether to enable truncation or not (maximum sequence length passed on initialization);
            post_processing: whether to use bert-like post processing (beginning-of-sentence token and end-of-sentence token);
        """

        self.tokenizer.train_from_iterator(
            texts,
            BpeTrainer(
                special_tokens=self.special_tokens,
            ),
        )

        if padding:
            self.tokenizer.enable_padding(
                direction="right",
                pad_id=self.tokenizer.token_to_id("[PAD]"),
                pad_token="[PAD]",
                length=self.max_length,
            )

        if truncation:
            self.tokenizer.enable_truncation(self.max_length)

        if post_processing:
            self.tokenizer.post_processor = TemplateProcessing(
                single="[BGN] $A [END]",
                special_tokens=[
                    ("[BGN]", self.tokenizer.token_to_id("[BGN]")),
                    ("[END]", self.tokenizer.token_to_id("[END]")),
                ],
            )

    def encode(self, text: str, type: str):

        """
        Tokenize and encode a single text (sentence).

        Arguments:
            text: single sentence;
            type: array type [numpy or pytorch].

        Returns:
            tokenized_text: dict with tokens' ids, positions and masks .
        """

        encoded_text = self.tokenizer.encode(text)

        tokenized_text = {
            "ids": encoded_text.ids,
            "positions": [pos for pos, _ in enumerate(encoded_text.ids)],
            "attention_mask": encoded_text.attention_mask,
        }

        if type == "numpy":
            tokenized_text = np.stack(
                [np.array(value) for _, value in tokenized_text.items()]
            )

        elif type == "pytorch":
            tokenized_text = {
                key: tensor(value) for key, value in tokenized_text.items()
            }

        return tokenized_text

    def token_to_id(self, token):
        return self.tokenizer.token_to_id(token)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()
