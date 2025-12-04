# Decision tree

Simple implementation of a decision tree classifier in Python.  
Project for a Python programming course.

The goal is to implement a basic classifier, add unit tests, and compare it with `sklearn.tree.DecisionTreeClassifier` on standard datasets.

---

## Demo - run main.py

Example output:

```bash
=== Breast cancer dataset ===
Our tree accuracy          : 0.9181
Sklearn tree accuracy      : 0.9298
Accuracy difference        : 0.0117

Our tree train time   [s]  : 2.326839
Sklearn train time    [s]  : 0.005548
Train time ratio my/sk     : 419.37

Our tree predict time [s]  : 0.000183
Sklearn predict time  [s]  : 0.000091
Predict time ratio my/sk   : 2.02

=== Wine dataset ===
Our tree accuracy          : 0.9444
Sklearn tree accuracy      : 0.9630
Accuracy difference        : 0.0185

Our tree train time   [s]  : 0.157467
Sklearn train time    [s]  : 0.001076
Train time ratio my/sk     : 146.33

Our tree predict time [s]  : 0.000047
Sklearn predict time  [s]  : 0.000074
Predict time ratio my/sk   : 0.64
```