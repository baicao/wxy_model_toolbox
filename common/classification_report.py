import traceback
import logging
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


class ClassificationReport(object):
    def __init__(self, labels, logger=None):
        self.labels = labels
        self.labels_size = len(self.labels)
        self.column_len = 0
        for i in range(self.labels_size):
            if len(self.labels[i]) > self.column_len:
                self.column_len = len(self.labels[i])
        self.column_len += 10
        if logger is None:
            self.logger = logging
        else:
            self.logger = logger

    def append_blank(self, text):
        return str(text) + " " * (self.column_len - len(str(text)))

    def print_cm(self, cm):
        confusion_matrix_str = (
            "confusion_matrix(left labels: y_true, up labels: y_pred):\n"
        )

        confusion_matrix_str += self.append_blank("labels")
        for i in range(self.labels_size):
            confusion_matrix_str += self.append_blank(self.labels[i])
        confusion_matrix_str += (
            "\n" + "-" * (self.labels_size + 1) * self.column_len + "\n"
        )
        for i, _ in enumerate(cm):
            confusion_matrix_str += self.append_blank(self.labels[i])
            for j in range(len(cm[i])):
                confusion_matrix_str += self.append_blank(cm[i][j])
            confusion_matrix_str += (
                "\n" + "-" * (self.labels_size + 1) * self.column_len + "\n"
            )
        confusion_matrix_str += "\n"
        return confusion_matrix_str

    def show_cm_report(self, y_true, y_pred):
        if type(y_true) is np.ndarray and len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        if type(y_pred) is np.ndarray and len(y_pred.shape) == 2:
            y_pred = np.argmax(y_pred, axis=1)

        cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
        confusion_matrix_str = self.print_cm(cm)
        self.logger.info(self.labels)
        self.logger.info(confusion_matrix_str)
        try:
            report = classification_report(
                y_true=y_true, y_pred=y_pred, target_names=self.labels
            )
            self.logger.info("\r\n" + report)

        except:  # pylint: disable=bare-except
            traceback.print_exc()
