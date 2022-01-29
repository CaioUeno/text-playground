from typing import Callable, List, Optional, Tuple

import numpy as np
from torch import cat, device, nn, no_grad, reshape, tensor
from torch.cuda import is_available
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


class BaseTrainer(nn.Module):

    """
    Base class to define fit and predict methods.
    """

    def fit(
        self,
        train_iterator: DataLoader,
        loss_function: _Loss,
        optimizer: Optimizer,
        metrics: Optional[List[Callable[[tensor, tensor], tensor]]] = [],
        val_iterator: Optional[DataLoader] = None,
        epochs: int = 1,
        verbose: bool = True,
    ) -> Tuple[np.array, Optional[dict], np.array, Optional[dict]]:

        # loss is always returned
        train_loss = np.zeros(epochs)
        val_loss = np.zeros(epochs)

        # dict containing metrics values for each epoch
        train_metrics = {metric.__name__: np.zeros(epochs) for metric in metrics}
        val_metrics = {metric.__name__: np.zeros(epochs) for metric in metrics}

        # iterate over epochs - main loop
        for epoch in tqdm(range(epochs), unit="epoch") if verbose else range(epochs):

            # training mode
            self.train()
            self.embedder.train()

            # initialize epoch loss
            epoch_loss = 0

            # store predictions and targets of entire training set
            epoch_train_preds = []
            epoch_train_targets = []

            # iterate over batches
            for batch_pairs, batch_targets in train_iterator:

                optimizer.zero_grad()

                preds = self(batch_pairs)

                batch_loss = loss_function(preds, batch_targets)
                batch_loss.backward()

                # sum batch loss on epoch loss
                epoch_loss += batch_loss.item()

                optimizer.step()

                epoch_train_preds.append(preds.to("cpu"))
                epoch_train_targets.append(batch_targets.to("cpu"))

            # training loss at epoch level - average
            train_loss[epoch] = epoch_loss / (
                len(train_iterator) * train_iterator.batch_size
            )

            # list -> tensor
            epoch_train_preds = cat(epoch_train_preds, dim=0)
            epoch_train_targets = cat(epoch_train_targets, dim=0)

            # calculate each metric value at epoch level
            # ignored if no metrics
            for metric in metrics:
                train_metrics[metric.__name__][epoch] = metric(
                    epoch_train_targets, epoch_train_preds
                )

            # if a validation set is passed
            if val_iterator:

                # evaluation mode
                self.eval()
                self.embedder.eval()

                # initalize validation loss
                cumulative_val_loss = 0

                # store predictions and true targets of entire validation set
                epoch_val_preds = []
                epoch_val_targets = []

                # no grad disables gradient calculation
                with no_grad():

                    # iterate over batches
                    for batch_texts, batch_targets in val_iterator:

                        preds = self(batch_texts)

                        batch_loss = loss_function(preds, batch_targets.to(self.device))

                        cumulative_val_loss += batch_loss.item()

                        epoch_val_preds.append(preds.to("cpu"))
                        epoch_val_targets.append(batch_targets.to("cpu"))

                # validation loss at epoch level
                val_loss[epoch] = cumulative_val_loss / (
                    len(val_iterator) * val_iterator.batch_size
                )

                # list -> tensor
                epoch_val_preds = cat(epoch_val_preds, dim=0)
                epoch_val_labels = cat(epoch_val_labels, dim=0)

                # calculate each metric value for the entire validation set
                # ignored if no metrics
                for metric in metrics:
                    val_metrics[metric.__name__][epoch] = metric(
                        epoch_val_labels, epoch_val_preds
                    )

        if val_iterator:
            return train_loss, train_metrics, val_loss, val_metrics
        else:
            return train_loss, train_metrics


class SimilarityTrainer(BaseTrainer):

    """ """

    def __init__(self, embedder: nn.Module):

        super(SimilarityTrainer, self).__init__()

        self.embedder = embedder

    def forward(self, pairs):

        # same embedder for both of elements (A, B)
        a = self.embedder(pairs[0])
        b = self.embedder(pairs[1])

        # cosine similarity between embeddings
        cos = nn.CosineSimilarity(dim=1)

        return reshape(cos(a, b), (-1, 1))


class BiLSTMEmbedder(nn.Module):

    """ """

    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        word_embedding_dim: int,
        num_layers: int,
        padding_idx: int,
        sentence_embedding_dim: int,
    ):

        super(BiLSTMEmbedder, self).__init__()

        # configs
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.word_embedding_dim = word_embedding_dim
        self.padding_idx = padding_idx
        self.sentence_embedding_dim = sentence_embedding_dim

        # LSTM configs
        self.num_layers = num_layers
        self.hidden_size = 64
        self.dropout = 0.2 if self.num_layers > 1 else 0

        self.device = device("cuda" if is_available() else "cpu")

        # layers
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.word_embedding_dim,
            padding_idx=self.padding_idx,
        )

        # LSTM layer
        self.lstm = nn.LSTM(
            self.word_embedding_dim,
            self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout,
        )

        # Dense layer
        self.output = nn.Linear(
            self.max_length * 2,  # double neurons because bidirectional=True
            self.sentence_embedding_dim,
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
