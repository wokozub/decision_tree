from src.node import Node


def test_leaf_node():
    # leaf node has only prediction set
    leaf = Node(prediction=1)
    assert leaf.is_leaf
    assert leaf.prediction == 1
    assert leaf.feature_index is None
    assert leaf.threshold is None


def test_decision_node():
    # decision node has feature, threshold and children
    left = Node(prediction=0)
    right = Node(prediction=1)
    root = Node(feature_index=0, threshold=0.5, left=left, right=right)

    assert not root.is_leaf
    assert root.feature_index == 0
    assert root.threshold == 0.5
    assert root.left is left
    assert root.right is right
