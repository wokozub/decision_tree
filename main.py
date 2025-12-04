from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as SkDecisionTree

from src import DecisionTreeClassifier


def run_breast_cancer():
    data = load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # our tree
    my_tree = DecisionTreeClassifier(max_depth=5, min_samples_split=2)
    my_tree.fit(X_train, y_train)
    my_tree.print_tree(feature_names=data.feature_names)
    acc_my = my_tree.score(X_test, y_test)

    # sklearn tree
    sk_tree = SkDecisionTree(max_depth=5, min_samples_split=2, random_state=42)
    sk_tree.fit(X_train, y_train)
    acc_sk = sk_tree.score(X_test, y_test)

    print("=== Breast cancer dataset ===")
    print("Our tree accuracy    :", acc_my)
    print("Sklearn tree accuracy:", acc_sk)
    print("Difference           :", abs(acc_my - acc_sk))
    print()


def run_wine():
    data = load_wine()
    X = data.data
    y = data.target

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

    print("=== Wine dataset ===")
    print("Our tree accuracy    :", acc_my)
    print("Sklearn tree accuracy:", acc_sk)
    print("Difference           :", abs(acc_my - acc_sk))
    print()


def main():
    run_breast_cancer()
    run_wine()


if __name__ == "__main__":
    main()
