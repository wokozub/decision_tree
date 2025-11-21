import numpy as np

from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as SkDecisionTree

from src import DecisionTreeClassifier


def test_tree_vs_sklearn_breast_cancer():
    # binary classification dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # our tree
    my_tree = DecisionTreeClassifier(max_depth=5, min_samples_split=2)
    my_tree.fit(X_train, y_train)
    acc_my = my_tree.score(X_test, y_test)

    # sklearn tree
    sk_tree = SkDecisionTree(max_depth=5, min_samples_split=2, random_state=42)
    sk_tree.fit(X_train, y_train)
    acc_sk = sk_tree.score(X_test, y_test)

    assert acc_my > 0.85

    diff = abs(acc_my - acc_sk)
    assert diff < 0.15


def test_tree_vs_sklearn_wine():
    # multiclass dataset
    data = load_wine()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    my_tree = DecisionTreeClassifier(max_depth=5, min_samples_split=2)
    my_tree.fit(X_train, y_train)
    acc_my = my_tree.score(X_test, y_test)

    sk_tree = SkDecisionTree(max_depth=5, min_samples_split=2, random_state=42)
    sk_tree.fit(X_train, y_train)
    acc_sk = sk_tree.score(X_test, y_test)

    assert acc_my > 0.7

    diff = abs(acc_my - acc_sk)
    assert diff < 0.2
