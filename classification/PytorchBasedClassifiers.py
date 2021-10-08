from typing import Optional
import numpy as np
from tqdm import tqdm
from torch import nn, cat, device, no_grad, flatten
from torch.cuda import is_available
from torch.utils.data import DataLoader
import torch.nn.functional as F


class BiLSTMClassifier(nn.Module):

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

        # useful variables
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

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

        # concat both directions of last hidden states [batch, seq_len, lstm_hidden_size * 2] (two last positions)
        hidden_concat = cat((lstm_output[:, :, -1], lstm_output[:, :, -2]), dim=1)

        return self.output(hidden_concat)

    def fit(
        self,
        train_iterator: DataLoader,
        val_iterator: Optional[DataLoader],
        epochs: int,
        loss_function,
        optimizer,
        verbose: bool = True,
    ):
        """
        Train model.

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

        loss, train_accuracy, val_accuracy = (
            np.zeros(epochs),
            np.zeros(epochs),
            np.zeros(epochs),
        )
        for epoch in tqdm(range(epochs), unit="epoch") if verbose else range(epochs):

            self.train()  # training mode
            epoch_loss = 0  # calculate epoch loss
            epoch_accuracy = 0  # calculate epoch accuracy

            for batch_texts, batch_labels in train_iterator:

                optimizer.zero_grad()

                preds = self(batch_texts.to(self.device))

                batch_loss = loss_function(preds, batch_labels.to(self.device))
                batch_loss.backward()

                optimizer.step()

                epoch_loss += batch_loss.item()

                epoch_accuracy += (
                    (preds.argmax(1) == batch_labels.to(self.device))
                    .float()
                    .sum()
                    .item()
                )

            # store epoch loss (mean of batches losses)
            loss[epoch] = epoch_loss / (len(train_iterator) * train_iterator.batch_size)
            # store model's accuracy on training set for this epoch
            train_accuracy[epoch] = epoch_accuracy / (
                len(train_iterator) * train_iterator.batch_size
            )

            # if a validation set was passed
            if val_iterator:

                self.eval()  # evaluation mode
                accuracy = 0  # calculate model's accuracy on validation set
                with no_grad():
                    for batch_texts, batch_labels in val_iterator:
                        preds = self(batch_texts.to(self.device))
                        accuracy += (
                            (preds.argmax(1) == batch_labels.to(self.device))
                            .float()
                            .sum()
                            .item()
                        )

                # store model's accuracy on validation set for this epoch
                val_accuracy[epoch] = accuracy / (
                    len(val_iterator) * val_iterator.batch_size
                )

        if val_iterator:
            return loss, train_accuracy, val_accuracy
        else:
            return loss, train_accuracy

    def predict(self, eval_iterator: DataLoader, verbose: bool = True):

        """
        Predict data. It returns both pred and true labels to leave to caller the evaluation metric(s).

        Argument:
            eval_iterator: DataLoader iterator with test data;

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


class CNNClassifier(nn.Module):

    """
    Text Classifier using 2D-Convolutional layers.

    Arguments:
        vocab_size: vocabulary's size;
        max_length: maximum length of a sentence;
        embedding_dim: size of word embeddings;
        padding_idx: index of padding [PAD] token;
        num_classes: number of classes.
    """

    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        embedding_dim: int,
        padding_idx: int,
        num_classes: int,
    ):

        super(CNNClassifier, self).__init__()

        # useful variables
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        self.num_classes = num_classes

        self.device = device("cuda" if is_available() else "cpu")

        # layers
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=self.padding_idx,
        )

        # first CNN layer
        self.first_cnn = nn.Conv2d(
            1,
            4,
            kernel_size=5,
            padding="same",
            padding_mode="replicate",
        )
        self.first_drop = nn.Dropout2d(p=0.2)

        # second CNN layer
        self.second_cnn = nn.Conv2d(
            4,
            4,
            kernel_size=5,
            padding="same",
            padding_mode="replicate",
        )
        self.second_drop = nn.Dropout2d(p=0.2)

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Dense layers
        self.hidden = nn.Linear(4 * self.embedding_dim * 2, self.max_length)
        self.output = nn.Linear(self.max_length, self.num_classes)

    def forward(self, x):

        embeddings = self.embedding(x).unsqueeze(1)

        conved = self.pool(F.sigmoid(self.first_cnn(embeddings)))
        dropped = self.first_drop(conved)
        conved = self.pool(F.sigmoid(self.second_cnn(dropped)))
        dropped = self.second_drop(conved)

        hidden = F.sigmoid(self.hidden(flatten(dropped, 1)))
        return F.softmax(self.output(hidden), dim=1)

    def fit(
        self,
        train_iterator: DataLoader,
        val_iterator: Optional[DataLoader],
        epochs: int,
        loss_function,
        optimizer,
        verbose: bool = True,
    ):
        """
        Train model.

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

        loss, train_accuracy, val_accuracy = (
            np.zeros(epochs),
            np.zeros(epochs),
            np.zeros(epochs),
        )
        for epoch in tqdm(range(epochs), unit="epoch") if verbose else range(epochs):

            self.train()  # training mode
            epoch_loss = 0  # calculate epoch loss
            epoch_accuracy = 0  # calculate epoch accuracy

            for batch_texts, batch_labels in train_iterator:

                optimizer.zero_grad()

                preds = self(batch_texts.to(self.device))

                batch_loss = loss_function(preds, batch_labels.to(self.device))
                batch_loss.backward()

                optimizer.step()

                epoch_loss += batch_loss.item()

                epoch_accuracy += (
                    (preds.argmax(1) == batch_labels.to(self.device))
                    .float()
                    .sum()
                    .item()
                )

            # store epoch loss (mean of batches losses)
            loss[epoch] = epoch_loss / (len(train_iterator) * train_iterator.batch_size)
            # store model's accuracy on training set for this epoch
            train_accuracy[epoch] = epoch_accuracy / (
                len(train_iterator) * train_iterator.batch_size
            )

            # if a validation set was passed
            if val_iterator:

                self.eval()  # evaluation mode
                accuracy = 0  # calculate model's accuracy on validation set
                with no_grad():
                    for batch_texts, batch_labels in val_iterator:
                        preds = self(batch_texts.to(self.device))
                        accuracy += (
                            (preds.argmax(1) == batch_labels.to(self.device))
                            .float()
                            .sum()
                            .item()
                        )

                # store model's accuracy on validation set for this epoch
                val_accuracy[epoch] = accuracy / (
                    len(val_iterator) * val_iterator.batch_size
                )

        if val_iterator:
            return loss, train_accuracy, val_accuracy
        else:
            return loss, train_accuracy

    def predict(self, eval_iterator: DataLoader, verbose: bool = True):

        """
        Predict data. It returns both pred and true labels to leave to caller the evaluation metric(s).

        Argument:
            eval_iterator: DataLoader iterator with test data;

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


class TransformerClassifier(nn.Module):

    """
    Text Classifier using a Transformer layer followed by a LSTM layer(s).

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

        # useful variables
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

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

        # concat both directions of last hidden states [batch, seq_len, hidden_size * 2] (two last positions)
        hidden = cat((lstm_output[:, :, -1], lstm_output[:, :, -2]), dim=1)

        return self.output(hidden)

    def fit(
        self,
        train_iterator: DataLoader,
        val_iterator: Optional[DataLoader],
        epochs: int,
        loss_function,
        optimizer,
        verbose: bool = True,
    ):
        """
        Train model.

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

        loss, train_accuracy, val_accuracy = (
            np.zeros(epochs),
            np.zeros(epochs),
            np.zeros(epochs),
        )
        for epoch in tqdm(range(epochs), unit="epoch") if verbose else range(epochs):

            self.train()  # training mode
            epoch_loss = 0  # calculate epoch loss
            epoch_accuracy = 0  # calculate epoch accuracy

            for batch_texts, batch_labels in train_iterator:

                optimizer.zero_grad()

                preds = self(batch_texts.to(self.device))

                batch_loss = loss_function(preds, batch_labels.to(self.device))
                batch_loss.backward()

                optimizer.step()

                epoch_loss += batch_loss.item()

                epoch_accuracy += (
                    (preds.argmax(1) == batch_labels.to(self.device))
                    .float()
                    .sum()
                    .item()
                )

            # store epoch loss (mean of batches losses)
            loss[epoch] = epoch_loss / (len(train_iterator) * train_iterator.batch_size)
            # store model's accuracy on training set for this epoch
            train_accuracy[epoch] = epoch_accuracy / (
                len(train_iterator) * train_iterator.batch_size
            )

            # if a validation set was passed
            if val_iterator:

                self.eval()  # evaluation mode
                accuracy = 0  # calculate model's accuracy on validation set
                with no_grad():
                    for batch_texts, batch_labels in val_iterator:
                        preds = self(batch_texts.to(self.device))
                        accuracy += (
                            (preds.argmax(1) == batch_labels.to(self.device))
                            .float()
                            .sum()
                            .item()
                        )

                # store model's accuracy on validation set for this epoch
                val_accuracy[epoch] = accuracy / (
                    len(val_iterator) * val_iterator.batch_size
                )

        if val_iterator:
            return loss, train_accuracy, val_accuracy
        else:
            return loss, train_accuracy

    def predict(self, eval_iterator: DataLoader, verbose: bool = True):

        """
        Predict data. It returns both pred and true labels to leave to caller the evaluation metric(s).

        Argument:
            eval_iterator: DataLoader iterator with test data;

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
