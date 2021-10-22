from typing import Callable, List, Optional, Tuple

import numpy as np
import torch.nn.functional as F
from torch import cat, device, nn, no_grad, tensor
from torch.cuda import is_available
from torch.utils.data import DataLoader
from tqdm import tqdm


class BaseClassifier(nn.Module):

    """
    Abstract class to define fit and predict methods.
    """

    def fit(
        self,
        train_iterator: DataLoader,
        loss_function: Callable,
        optimizer: Callable,
        metrics: Optional[List[Callable[[tensor, tensor], tensor]]] = None,
        val_iterator: Optional[DataLoader] = None,
        epochs: int = 1,
        verbose: bool = True,
    ) -> Tuple[np.array, Optional[dict], Optional[dict]]:

        """
        General purpose fit method.

        Arguments:
            train_iterator: DataLoader iterator with training data;
            val_iterator: DataLoader iterator with validation data;
            epochs: number of epochs to train;
            loss_function: loss function to evaluate error;
            optimizer: optimizer to apply backpropagation (must already be initialized with model's parameters);
            verbose: boolean flag to indicate verbose level (show tqdm or not);

        Returns:
            loss: numpy array with epochs' losses;
            train_accuracy: numpy array with model's accuracy on training data for each epoch;
            val_accuracy: numpy array with model's accuracy on validation data for each epoch if val_iterator passed.
        """

        # loss is always returned
        train_loss = np.zeros(epochs)
        val_loss = np.zeros(epochs)

        train_metrics = {metric.__name__: np.zeros(epochs) for metric in metrics}
        val_metrics = {metric.__name__: np.zeros(epochs) for metric in metrics}

        for epoch in tqdm(range(epochs), unit="epoch") if verbose else range(epochs):

            self.train()  # training mode
            epoch_loss = 0

            epoch_train_preds = []  # store predictions of entire training set
            epoch_train_labels = []  # store true labels of entire training set

            for batch_texts, batch_labels in train_iterator:

                optimizer.zero_grad()

                preds = self(batch_texts.to(self.device))

                batch_loss = loss_function(preds, batch_labels.to(self.device))
                batch_loss.backward()
                epoch_loss += batch_loss.item()

                optimizer.step()

                epoch_train_preds.append(preds.to("cpu"))
                epoch_train_labels.append(batch_labels.to("cpu"))

            # list -> tensor
            epoch_train_preds = cat(epoch_train_preds, dim=0)
            epoch_train_labels = cat(epoch_train_labels, dim=0)

            # calculate each metric value at epoch level
            # ignored if no metrics
            for metric in metrics:
                train_metrics[metric.__name__][epoch] = metric(
                    epoch_train_labels, epoch_train_preds
                )

            # loss at epoch level
            train_loss[epoch] = epoch_loss / (
                len(train_iterator) * train_iterator.batch_size
            )

            # if a validation set was passed
            if val_iterator:

                self.eval()  # evaluation mode
                acc_val_loss = 0

                epoch_val_preds = []  # store predictions of entire validation set
                epoch_val_labels = []  # store true labels of entire validation set

                with no_grad():
                    for batch_texts, batch_labels in val_iterator:
                        preds = self(batch_texts.to(self.device))
                        acc_val_loss += loss_function(
                            preds, batch_labels.to(self.device)
                        ).item()
                        epoch_val_preds.append(preds.to("cpu"))
                        epoch_val_labels.append(batch_labels.to("cpu"))

                # list -> tensor
                epoch_val_preds = cat(epoch_val_preds, dim=0)
                epoch_val_labels = cat(epoch_val_labels, dim=0)

                # calculate each metric value for the entire validation set
                # ignored if no metrics
                for metric in metrics:
                    val_metrics[metric.__name__][epoch] = metric(
                        epoch_val_labels, epoch_val_preds
                    )

                # loss at epoch level
                val_loss[epoch] = acc_val_loss / (
                    len(val_iterator) * val_iterator.batch_size
                )

        if val_iterator:
            return train_loss, train_metrics, val_loss, val_metrics
        else:
            return train_loss, train_metrics

    def predict(
        self, eval_iterator: DataLoader, verbose: bool = True
    ) -> Tuple[np.array, np.array]:

        """
        General purpose predict method.
        It returns both pred and true labels to leave to the caller the evaluation metric(s).

        Arguments:
            eval_iterator: DataLoader iterator with test data;
            verbose: boolean flag to indicate verbose level (show tqdm or not);

        Returns:
            true_labels: true labels;
            preds: predicted labels.
        """

        self.eval()  # evaluation mode

        n_instances = len(eval_iterator) * eval_iterator.batch_size
        true_labels, preds = np.zeros(n_instances), np.zeros(n_instances)

        with no_grad():

            for batch, (batch_texts, batch_labels) in (
                tqdm(enumerate(eval_iterator), unit="batch")
                if verbose
                else enumerate(eval_iterator)
            ):

                batch_preds = self(batch_texts.to(self.device))

                begin_idx = batch * eval_iterator.batch_size
                last_idx = begin_idx + len(batch_texts)

                preds[begin_idx:last_idx] = batch_preds.argmax(1)
                true_labels[begin_idx:last_idx] = batch_labels

        return true_labels, preds


class BiLSTMClassifier(BaseClassifier):

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

        self.device = device("cuda" if is_available() else "cpu")

        # layers
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=self.padding_idx,
        )

        # LSTM layer
        self.lstm = nn.LSTM(
            self.embedding_dim,
            self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout,
        )

        # Dense layer
        self.output = nn.Linear(
            self.max_length * 2,  # double neurons because bidirectional=True
            self.num_classes,
        )

    def forward(self, x):

        embeddings = self.embedding(x)

        # pack sequence to ignore padding token
        packed_x = nn.utils.rnn.pack_padded_sequence(
            embeddings,
            lengths=(x != self.padding_idx)
            .sum(1)
            .to("cpu"),  # length of each sentence in the batch must be in cpu
            batch_first=True,
            enforce_sorted=False,
        )

        lstm_output, (hidden_state, cell_state) = self.lstm(packed_x)

        # unpack (pad) sequence
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_output, batch_first=True, total_length=self.max_length
        )

        # concat last hidden states of both directions [batch, seq_len, lstm_hidden_size * 2] (two last positions)
        hidden_concat = cat((lstm_output[:, :, -1], lstm_output[:, :, -2]), dim=1)

        return self.output(hidden_concat)


class TransformerClassifier(BaseClassifier):

    """
    Text Classifier using a Transformer layer followed by a two LSTM layers.

    Arguments:
        vocab_size: vocabulary's size;
        max_length: maximum length of a sentence;
        embedding_dim: size of word embeddings;
        nhead:
        num_encoder_layers:
        padding_idx: index of padding [PAD] token;
        num_classes: number of classes.
    """

    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        embedding_dim: int,
        nhead: int,
        num_encoder_layers: int,
        padding_idx: int,
        num_classes: int,
    ):

        super(TransformerClassifier, self).__init__()

        # configs
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        # Transformer configs
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers

        self.num_classes = num_classes

        self.device = device("cuda" if is_available() else "cpu")

        # layers
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=self.padding_idx,
        )

        # Transformer layer
        self.transformer = nn.Transformer(
            d_model=self.embedding_dim,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            batch_first=True,
        )

        # LSTM layer
        self.lstm = nn.LSTM(
            self.embedding_dim,
            64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )

        # Dense layer
        self.output = nn.Linear(
            self.max_length * 2, self.num_classes
        )  # double neurons because bidirectional=True

    def forward(self, x):

        embeddings = self.embedding(x)

        transformered = self.transformer(
            embeddings,
            embeddings,
            src_key_padding_mask=(x != self.padding_idx),
            tgt_key_padding_mask=(x != self.padding_idx),
        )

        lstm_output, (hidden_state, cell_state) = self.lstm(transformered)

        # concat last hidden states of both directions [batch, seq_len, lstm_hidden_size * 2] (two last positions)
        hidden = cat((lstm_output[:, :, -1], lstm_output[:, :, -2]), dim=1)

        return self.output(hidden)
