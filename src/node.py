from dataclasses import dataclass


@dataclass
class Node:
    """
    Tree node
    It can be a decision node or a leaf node
    """

    # For decision node
    feature_index: int = None   # column index in X
    threshold: float = None     # split value
    left: "Node" = None         # left child node
    right: "Node" = None        # right child node

    # For leaf node
    prediction: int = None      # class 0 or 1

    @property
    def is_leaf(self):
        return self.left is None and self.right is None