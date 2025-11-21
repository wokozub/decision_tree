import numpy as np

from src import DecisionTreeClassifier


def test_tree_fits_simple_binary_data():
    X = np.array([
        [0.1, 1.0],
        [0.2, 1.3],
        [1.0, 0.2],
        [1.2, 0.1],
    ])
    y = np.array([0, 0, 1, 1])

    tree = DecisionTreeClassifier(max_depth=3, min_samples_split=1)
    tree.fit(X, y)

    acc = tree.accuracy(X, y)
    assert np.isclose(acc, 1.0)


def test_tree_depth_limit_reduces_accuracy():
    # same data but with very small depth
    X = np.array([
        [0.1, 1.0],
        [0.2, 1.3],
        [1.0, 0.2],
        [1.2, 0.1],
    ])
    y = np.array([0, 0, 1, 1])

    tree = DecisionTreeClassifier(max_depth=0, min_samples_split=1)
    tree.fit(X, y)

    acc = tree.accuracy(X, y)
    # depth 0 means only one leaf node, so accuracy should be less than 1
    assert acc < 1.0


def test_tree_multiclass_toy():
    # small multiclass problem to check that multi class logic works
    X = np.array([
        [0.0],
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
    ])
    y = np.array([0, 0, 1, 1, 2, 2])

    tree = DecisionTreeClassifier(max_depth=3, min_samples_split=1)
    tree.fit(X, y)

    acc = tree.accuracy(X, y)
    # tree should also learn this tiny dataset perfectly
    assert np.isclose(acc, 1.0)
    # check that classes_ contains all class labels
    assert set(tree.classes_) == {0, 1, 2}
