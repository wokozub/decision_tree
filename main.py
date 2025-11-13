import numpy as np
from src import DecisionTreeClassifier

# X has 2 features: feature 0 and feature 1
X = np.array([
    [0.1, 1.0],
    [0.2, 1.3],
    [1.0, 0.2],
    [1.2, 0.1],
])

# y is the class (label) to predict
y = np.array([0, 0, 1, 1])

tree = DecisionTreeClassifier(max_depth=2, min_samples_split=1)
tree.fit(X, y)

print("Predictions:", tree.predict(X))
print("Accuracy:", tree.accuracy(X, y))
