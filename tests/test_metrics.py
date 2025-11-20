import numpy as np

from src.metrics import gini_impurity, accuracy


def test_gini_pure_binary():
    # all labels are the same
    y = np.array([0, 0, 0, 0])
    g = gini_impurity(y)
    assert g == 0.0


def test_gini_mixed_binary():
    # 50 percent of class 0 and 50 percent of class 1
    y = np.array([0, 1, 0, 1])
    g = gini_impurity(y)
    # for 2 classes with 0.5 / 0.5 gini = 0.5
    assert np.isclose(g, 0.5)


def test_gini_multiclass():
    # three classes with equal counts
    y = np.array([0, 1, 2, 0, 1, 2])
    g = gini_impurity(y)
    # probabilities are 1/3 each, gini = 1 - 3 * (1/3)^2 = 2/3
    assert np.isclose(g, 2.0 / 3.0)


def test_accuracy_simple():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    acc = accuracy(y_true, y_pred)
    # 3 correct out of 4
    assert np.isclose(acc, 0.75)
