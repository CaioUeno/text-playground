from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from torch import tensor
from Datasets import SimpleTextDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.special import softmax


class HFBasedTextClassifier(object):

    """
    General Purpose Text Classifier based on Hugging Face library.

    Arguments:
        num_labels: number of classes.
        max_length: instance (text) max length.
    """

    def __init__(self, num_labels: int, max_length: int = 64):

        # pretrained tokenizer and model
        self.model_name = "bert-base-multilingual-cased"
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name, num_labels=num_labels
        )

        self.num_labels = num_labels
        self.max_length = max_length

    def fit(
        self,
        X_train,
        y_train,
        X_valid=None,
        y_valid=None,
        validation_split: float = None,
        batch_size: int = 16,
        epochs: int = 3,
    ):

        """
        Fit classifier on X and y.

        Arguments:
            X: list of texts;
            y: list of corresponding labels;
            X_val: list of validation texts;
            y_val: list of corresponding validation labels;
            validation_split: percentage of training set as validation;
            batch_size: batch size;
            epochs: (max) number of epochs.
        """

        def compute_metrics(p):
            pred, labels = p
            pred = np.argmax(pred, axis=1)
            accuracy = accuracy_score(y_true=labels, y_pred=pred)
            return {"accuracy": accuracy}

        if any(X_valid) and any(y_valid) and validation_split:
            raise ValueError("Use (X_val, y_val) OR validation_split (not both).")

        if validation_split:

            # split train/validation
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_train, y_train, test_size=validation_split, stratify=y_train
            )

        # tokenize texts
        X_train_tokenized = self.tokenizer(
            X_train, padding=True, truncation=True, max_length=self.max_length
        )

        # pre-process labels - it has to be numbers
        self.classes = list(set(y_train))

        # mapping class onto index (number)
        self.cls2idx = {c: i for i, c in enumerate(self.classes)}
        # mapping index onto class
        self.idx2cls = {i: c for i, c in enumerate(self.classes)}

        # string to number (if labels are already number, then it becomes redundant - but it is still usefull)
        y_train_encoded = list(map(lambda x: int(self.cls2idx[x]), y_train))

        train_dataset = SimpleTextDataset(X_train_tokenized, y_train_encoded)

        # initialize validation dataset variable
        valid_dataset = False

        if any(X_valid) and any(y_valid):

            # tokenize valid texts and encode valid labels
            X_valid_tokenized = self.tokenizer(
                X_valid, padding=True, truncation=True, max_length=self.max_length
            )
            y_valid_encoded = list(map(lambda x: int(self.cls2idx[x]), y_valid))

            valid_dataset = SimpleTextDataset(X_valid_tokenized, y_valid_encoded)

        # use Trainer API
        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                "model",
                do_eval=True if valid_dataset else False,
                evaluation_strategy="epoch" if valid_dataset else "no",
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=256,
                num_train_epochs=epochs,
                logging_strategy="epoch",
                save_strategy="epoch" if valid_dataset else "no",
                load_best_model_at_end=True,
            ),
            train_dataset=train_dataset,
            eval_dataset=valid_dataset if valid_dataset else None,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
            if valid_dataset
            else None,
        )

        trainer.train()

    def predict(self, X):

        """
        Predict X.

        Arguments:
            X: list of texts.

        Returns:
            prob_distribution_pred: classes' probability distribution for each instance with shape (n_instances, n_classes).
        """

        X_tokenized = self.tokenizer(
            X, padding=True, truncation=True, max_length=self.max_length
        )
        prediction_dataset = SimpleTextDataset(X_tokenized)

        # Define test trainer
        test_trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                "model", save_strategy="no", per_device_eval_batch_size=256
            ),
        )

        # Make prediction
        raw_pred, _, _ = test_trainer.predict(prediction_dataset)
        prob_distribution_pred = softmax(raw_pred, axis=1)

        return prob_distribution_pred

    def get_class_from_idx(self, y):

        """
        After prediction, if applied an argmax function, it transforms numerical labels into true classes' names.
        """

        return list(map(lambda x: self.idx2cls[x], y))

    def save(self, model_path):
        self.model.save_pretrained(model_path)

    def load(self, model_path):
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        )
