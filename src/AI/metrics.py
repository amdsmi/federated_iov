from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, confusion_matrix, \
    ConfusionMatrixDisplay
import matplotlib.pylab as plt
import tensorflow as tf
from tqdm import tqdm
import numpy as np


class Result:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.labels, self.predictions, self.target = self._calculate_labels

    @property
    def _calculate_labels(self):
        y_true = []
        y_pred = []

        for x, y in tqdm(self.dataset):
            y_pred.append(tf.math.argmax(self.model.predict(x, verbose=0), axis=1).numpy())
            y_true.append(y)
        labels = np.array([x for x in range(43)])

        predictions = np.concatenate(y_pred, axis=0)
        target = np.concatenate(y_true, axis=0)
        return labels, predictions, target

    def calculate_metrics(self):
        f1 = f1_score(self.target, self.predictions, average='macro', zero_division=0),
        acc = accuracy_score(self.target, self.predictions),
        recall = recall_score(self.target, self.predictions, average='macro', zero_division=0),
        precision = precision_score(self.target, self.predictions, average='macro', zero_division=0)
        _, mse, rmse, mae = self.model.evaluate(self.dataset, verbose=0)
        return {'f1': f1[0], 'acc': acc[0], 'recall': recall[0], 'precision': precision, 'mse': mse, 'rmse': rmse, 'mae': mae}

    def plot_confusion(self, name):
        disp = ConfusionMatrixDisplay(
            confusion_matrix(
                self.target, self.predictions
            ),
            display_labels=self.labels,
        )
        fig, ax = plt.subplots(figsize=(30, 30))
        disp.plot(cmap='Blues', ax=ax)
        plt.savefig(name)

