import numpy as np


def gini_impurity(y):
    """
    Gini impurity for binary labels 0/1
    y is a 1D numpy array
    """
    if y.size == 0:
        return 0.0

    classes, counts = np.unique(y, return_counts=True)
    p = counts.astype(float) / counts.sum()
    return 1.0 - np.sum(p ** 2)


def accuracy(y_true, y_pred):
    """
    accuracy metric
    It is the fraction of correct predictions
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())