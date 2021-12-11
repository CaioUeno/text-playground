import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Concatenate, Dense, Embedding
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()


class BiLSTMClassifier(Model):

    """
    Text Classifier using Bidirectional LSTM layers.

    Arguments:
        vocab_size: vocabulary's size;
        max_length: maximum length of a sentence;
        embedding_dim: size of word embeddings;
        num_layers: number of stacked LSTM layers;
        padding_idx: index of padding [PAD] token;
        num_classes: number of classes.
    """

    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        embedding_dim: int,
        num_layers: int,
        padding_idx: int,
        num_classes: int,
    ):

        super(BiLSTMClassifier, self).__init__()

        # configs
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        # LSTM configs
        self.num_layers = num_layers
        self.hidden_size = 64  # it is changeable
        self.dropout = 0.2 if self.num_layers > 1 else 0

        self.num_classes = num_classes

        # layers
        # Word Embedding layer
        self.word_embedding = Embedding(
            self.vocab_size,
            self.embedding_dim,
            input_length=self.max_length,
            mask_zero=False,
        )

        # Positional Embedding layer - same configs as word embedding layer
        self.pos_embedding = Embedding(
            self.vocab_size,
            self.embedding_dim,
            input_length=self.max_length,
            mask_zero=False,
        )

        # Foward LSTM layers list
        self.forward_lstm_list = [
            LSTM(
                self.hidden_size,
                dropout=self.dropout,
                return_sequences=True,
                return_state=True,
                go_backwards=False,
            )
            for _ in range(self.num_layers)
        ]

        # Backward LSTM layers list
        self.backward_lstm_list = [
            LSTM(
                self.hidden_size,
                dropout=self.dropout,
                return_sequences=True,
                return_state=True,
                go_backwards=True,
            )
            for _ in range(self.num_layers)
        ]

        # Concatenate layer
        self.concat = Concatenate(axis=1)

        # Dense layer
        self.dense = Dense(self.num_classes, activation="softmax")

    def call(self, x: np.array, training=False):

        # unpack x
        ids, positions, attention_mask = (
            x[:, 0, :],
            x[:, 1, :],
            x[:, 2, :],
        )

        word_embeddings = self.word_embedding(ids)
        pos_embeddings = self.pos_embedding(positions)

        # sum word embedding and positional embeddings
        summed_embeddings = word_embeddings + pos_embeddings

        # foward lstm loop
        forward_lstm_output, _, _ = self.forward_lstm_list[0](
            summed_embeddings, mask=attention_mask.astype(bool), training=training
        )
        for idx, lstm_layer in enumerate(self.forward_lstm_list):
            if idx == 0:
                continue
            forward_lstm_output, _, _ = lstm_layer(
                forward_lstm_output, mask=attention_mask.astype(bool), training=training
            )

        # backward lstm loop
        backward_lstm_output, _, _ = self.backward_lstm_list[0](
            summed_embeddings, mask=attention_mask.astype(bool), training=training
        )
        for idx, lstm_layer in enumerate(self.backward_lstm_list):
            if idx == 0:
                continue
            backward_lstm_output, _, _ = lstm_layer(
                backward_lstm_output,
                mask=attention_mask.astype(bool),
                training=training,
            )

        bidirectional_lstm_output = self.concat(
            [forward_lstm_output, backward_lstm_output]
        )

        # return last memory state (batch_size, states, features)
        return self.dense(bidirectional_lstm_output)[:, -1, :]
