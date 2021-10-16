from typing import List, Optional, Union

import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from Datasets import HFSimpleTextDataset


class HFBasedTextClassifier(object):

    """
    General Purpose Text Classifier based on Hugging Face library.
    Reference: https://huggingface.co/course/chapter3/3?fw=pt

    Arguments:
        num_labels: number of classes.
        max_length: instance (text) max length.
    """

    def __init__(
        self,
        num_labels: int,
        model_name: str = "bert-base-uncased",
        max_length: int = 64,
    ):

        # configs
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length

        # pretrained model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            self.model_name, model_max_length=self.max_length
        )
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        )

    def fit(
        self,
        X_train: List[str],
        y_train: Union[List[str], List[int]],
        X_valid: Optional[List[str]] = None,
        y_valid: Optional[Union[List[str], List[int]]] = None,
        validation_split: float = None,
        batch_size: int = 16,
        epochs: int = 3,
        random_state: int = None,
    ) -> None:

        """
        Fit classifier on X and y.

        Arguments:
            X: list of texts;
            y: list of corresponding labels;
            X_val: list of validation texts;
            y_val: list of corresponding validation labels;
            validation_split: percentage of training set as validation;
            batch_size: batch size;
            epochs: (max) number of epochs;
            random_state: variable forrReproducibility (data split only).
        """

        def compute_classification_metrics(p):
            pred_scores, labels = p
            preds = np.argmax(pred_scores, axis=1)

            return {
                "accuracy": accuracy_score(y_true=labels, y_pred=preds),
                "precision": precision_score(
                    y_true=labels, y_pred=preds, average="weighted", zero_division=1
                ),
                "recall": recall_score(
                    y_true=labels, y_pred=preds, average="weighted", zero_division=1
                ),
                "f1-socre": f1_score(
                    y_true=labels, y_pred=preds, average="weighted", zero_division=1
                ),
            }

        # arguments checks
        if X_valid is not None and y_valid is not None and validation_split is not None:
            raise ValueError("Use (X_valid, y_valid) XOR validation_split (not both).")

        if validation_split is not None and 0 < validation_split < 1:
            # split train/validation
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_train,
                y_train,
                test_size=validation_split,
                stratify=y_train,
                random_state=random_state,
            )

        self._check_X_y(X_train, y_train)

        # pre-process labels - it must be a number
        self.classes = list(set(y_train))

        # mapping class onto an index (number)
        self.cls2idx = {c: i for i, c in enumerate(self.classes)}
        # mapping index onto class
        self.idx2cls = {i: c for i, c in enumerate(self.classes)}

        # string to number (if labels are already number, then it becomes redundant - but it is still usefull)
        y_train_encoded = list(map(lambda x: int(self.cls2idx[x]), y_train))

        # tokenize texts
        X_train_tokenized = self.tokenizer(
            X_train, padding=True, truncation=True, max_length=self.max_length
        )

        train_dataset = HFSimpleTextDataset(X_train_tokenized, y_train_encoded)

        # # initialize valid_dataset variable
        valid_dataset = None

        if X_valid is not None and y_valid is not None:

            self._check_X_y(X_valid, y_valid)

            # tokenize valid texts and encode valid labels
            X_valid_tokenized = self.tokenizer(
                X_valid, padding=True, truncation=True, max_length=self.max_length
            )
            y_valid_encoded = list(map(lambda x: int(self.cls2idx[x]), y_valid))

            valid_dataset = HFSimpleTextDataset(X_valid_tokenized, y_valid_encoded)

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
            compute_metrics=compute_classification_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
            if valid_dataset
            else None,
        )

        trainer.train()

    def predict(self, X: List[str]) -> np.array:

        """
        Predict X.

        Arguments:
            X: list of texts.

        Returns:
            prob_distribution_pred: classes' probability distribution for each instance with shape (n_instances, n_classes).
        """

        # tokenize texts
        X_tokenized = self.tokenizer(
            X, padding=True, truncation=True, max_length=self.max_length
        )

        prediction_dataset = HFSimpleTextDataset(X_tokenized)

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

    def _check_X_y(self, X: list, y: list) -> None:

        """
        Method to check X and y - lengths and elements' type.

        Arguments:
            X: supposed list of texts; 
            y: supposed list of labels.
        """

        if len(X) != len(y):
            raise ValueError("Number of texts is different from number of labels.")

        X_is_valid = all(list(map(lambda label: isinstance(label, str), X)))
        if not X_is_valid:
            raise ValueError("X must be a list of strings (check all elements).")

        y_is_valid = all(list(map(lambda label: isinstance(label, (str, int)), y)))
        if not y_is_valid:
            raise ValueError(
                "y must be a list of either strings or integers (check all elements)."
            )

    def get_class_from_idx(self, y: List[int]) -> List[str]:

        """
        After prediction, if an argmax function is applied, it transforms y (numerical labels) into true classes' names.

        Arguments:
            y: list of numerical labels.

        Returns:
            list: list of classes' names.
        """

        return list(map(lambda x: self.idx2cls[x], y))

    def save(self, model_path: str) -> None:
        self.model.save_pretrained(model_path)

    def load(self, model_path: str) -> None:
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        )
