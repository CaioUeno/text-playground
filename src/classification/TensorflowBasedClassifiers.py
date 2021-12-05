from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, LSTM, Concatenate, Dense


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

        # Foward LSTM
        self.forward_lstm = LSTM(
            self.hidden_size,
            dropout=self.dropout,
            return_sequences=True,
            go_backwards=False,
        )

        # Backward LSTM
        self.forward_lstm = LSTM(
            self.hidden_size,
            dropout=self.dropout,
            return_sequences=True,
            go_backwards=False,
        )

        # Concatenate layer
        self.concat = Concatenate(axis=1)

        # Dense layer
        self.dense = Dense(self.num_classes, activation="softmax")

    def call(self, x: dict, training=False):

        # unpack x and put on correct device
        ids, positions, attention_mask = (
            x["ids"],
            x["positions"],
            x["attention_mask"],
        )

        word_embeddings = self.word_embedding(ids)
        pos_embeddings = self.pos_embedding(positions)

        # sum word embedding and positional embeddings
        summed_embeddings = word_embeddings + pos_embeddings

        # foward lstm loop
        for i in range(self.num_layers):

            # forward_lstm_output = self.forward_lstm(
            #     summed_embeddings if i == 0 else forward_lstm_output
            # )

            if i == 0:
                forward_lstm_output = self.forward_lstm(
                    summed_embeddings, mask=attention_mask, training=training
                )

            else:
                forward_lstm_output = self.forward_lstm(
                    forward_lstm_output, mask=attention_mask, training=training
                )

        # backward lstm loop
        for i in range(self.num_layers):

            # whole_seq_output = self.forward_lstm(
            #     summed_embeddings if i == 0 else whole_seq_output
            # )

            if i == 0:
                backward_lstm_output = self.forward_lstm(
                    summed_embeddings, mask=attention_mask, training=training
                )

            else:
                backward_lstm_output = self.forward_lstm(
                    backward_lstm_output, mask=attention_mask, training=training
                )

        bidirectional_lstm_output = self.concat(
            [forward_lstm_output, backward_lstm_output]
        )

        x = self.dense(bidirectional_lstm_output)
        if training:
            x = self.dropout(x, training=training)
        return self.dense2(x)
