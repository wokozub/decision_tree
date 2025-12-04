import numpy as np

from .node import Node
from .metrics import gini_impurity, accuracy as accuracy_metric


class DecisionTreeClassifier:
    """
    decision tree for binary classification
    Supports multi class labels
    Uses Gini impurity and numeric features
    """

    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        self.root_ = None
        self.n_features_ = None
        self.classes_ = None       # unique class labels
        self.n_classes_ = None     # number of classes

    # ---------- PUBLIC API ----------

    def fit(self, X, y):
        """
        Train the tree

        X: 2D array (n_samples, n_features)
        y: 1D array (n_samples,) with values 0 or 1
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")
        if y.ndim != 1:
            raise ValueError("y must be 1D")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]

        self.root_ = self._build_tree(X, y, depth=0)
        return self

    def predict(self, X):
        """
        Predict class labels for samples in X
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self.root_ is None:
            raise RuntimeError("Call fit() before predict()")

        preds = [self._predict_one(row) for row in X]
        return np.array(preds, dtype=int)

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return accuracy_metric(y, y_pred)
    
    def score(self, X, y):
        """
        Alias for accuracy
        Same name as in scikit learn classifiers
        """
        return self.accuracy(X, y)

    # ---------- HELPERS ----------

    def _build_tree(self, X, y, depth):
        """
        Recursive tree building.
        """
        num_samples = y.shape[0]

        # majority class in this node
        classes, counts = np.unique(y, return_counts=True)
        majority_class = classes[np.argmax(counts)]

        # stopping conditions
        if (
            depth >= self.max_depth
            or num_samples < self.min_samples_split
            or np.all(y == y[0])
        ):
            return Node(prediction=int(majority_class))

        # find best split
        best_feature, best_threshold, best_impurity, best_left_mask = self._best_split(X, y)

        if best_feature is None:
            return Node(prediction=int(majority_class))

        left_mask = best_left_mask
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        # create a decision node
        return Node(
            feature_index=int(best_feature),
            threshold=float(best_threshold),
            left=left_child,
            right=right_child,
            prediction=int(majority_class),
        )

    def _best_split(self, X, y):
        """
        Try all features and thresholds
        Use Gini impurity to choose the best split

        Return:
        - best feature index
        - best threshold value
        - impurity after split
        - boolean mask for left side
        """
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split:
            return None, None, None, None

        parent_impurity = gini_impurity(y)
        best_impurity = 1.0
        best_feature = None
        best_threshold = None
        best_left_mask = None

        for feature_index in range(n_features):
            values = X[:, feature_index]
            unique_values = np.unique(values)

            # test every unique value as a threshold
            for threshold in unique_values:
                left_mask = values <= threshold
                right_mask = ~left_mask

                # skip split that does not separate data
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                impurity_left = gini_impurity(y[left_mask])
                impurity_right = gini_impurity(y[right_mask])

                w_left = left_mask.sum() / n_samples
                w_right = right_mask.sum() / n_samples
                impurity_split = w_left * impurity_left + w_right * impurity_right

                if impurity_split < best_impurity:
                    best_impurity = impurity_split
                    best_feature = feature_index
                    best_threshold = threshold
                    best_left_mask = left_mask

        if best_feature is None or best_impurity >= parent_impurity:
            return None, None, None, None

        return best_feature, best_threshold, best_impurity, best_left_mask

    def _predict_one(self, x):
        """
        Traverse the tree for one sample
        """
        node = self.root_
        while not node.is_leaf:
            # in a decision node, feature_index and threshold are set
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right

        return node.prediction

    def print_tree(self, feature_names=None):
        """
        Print simple if else view of the tree
        """
        if self.root_ is None:
            raise RuntimeError("Call fit() before print_tree()")

        self._print_node(self.root_, depth=0, feature_names=feature_names)

    def _print_node(self, node, depth, feature_names):
        # indentation for nested levels
        indent = "  " * depth

        if node.is_leaf:
            # leaf node, print prediction
            print(f"{indent}predict {node.prediction}")
            return

        # name of feature for display
        if feature_names is not None:
            name = feature_names[node.feature_index]
        else:
            name = f"feature_{node.feature_index}"

        # decision node
        print(f"{indent}if {name} <= {node.threshold:.3f}:")
        self._print_node(node.left, depth + 1, feature_names)
        print(f"{indent}else:")
        self._print_node(node.right, depth + 1, feature_names)
