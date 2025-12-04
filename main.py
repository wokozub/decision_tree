from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as SkDecisionTree

from src import DecisionTreeClassifier
import time


def evaluate_dataset(name, loader, max_depth=5, min_samples_split=2):
    # load dataset
    data = loader()
    X = data.data
    y = data.target

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    my_tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)

    # training
    t0 = time.perf_counter()
    my_tree.fit(X_train, y_train)
    #my_tree.print_tree(feature_names=data.feature_names)
    t1 = time.perf_counter()
    acc_my = my_tree.score(X_test, y_test)

    # prediction
    t2 = time.perf_counter()
    y_pred_my = my_tree.predict(X_test)
    t3 = time.perf_counter()

    train_time_my = t1 - t0
    predict_time_my = t3 - t2

    # sklearn tree
    sk_tree = SkDecisionTree(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
    )

    # training
    t0 = time.perf_counter()
    sk_tree.fit(X_train, y_train)
    t1 = time.perf_counter()
    acc_sk = sk_tree.score(X_test, y_test)

    # prediction
    t2 = time.perf_counter()
    y_pred_sk = sk_tree.predict(X_test)
    t3 = time.perf_counter()

    train_time_sk = t1 - t0
    predict_time_sk = t3 - t2

    print(f"=== {name} dataset ===")
    print("Our tree accuracy          :", f"{acc_my:.4f}")
    print("Sklearn tree accuracy      :", f"{acc_sk:.4f}")
    print("Accuracy difference        :", f"{abs(acc_my - acc_sk):.4f}")
    print()
    print("Our tree train time   [s]  :", f"{train_time_my:.6f}")
    print("Sklearn train time    [s]  :", f"{train_time_sk:.6f}")
    print("Train time ratio my/sk     :", f"{(train_time_my / train_time_sk) if train_time_sk > 0 else float('inf'):.2f}")
    print()
    print("Our tree predict time [s]  :", f"{predict_time_my:.6f}")
    print("Sklearn predict time  [s]  :", f"{predict_time_sk:.6f}")
    print("Predict time ratio my/sk   :", f"{(predict_time_my / predict_time_sk) if predict_time_sk > 0 else float('inf'):.2f}")
    print()


def main():
    evaluate_dataset("Breast cancer", load_breast_cancer)
    evaluate_dataset("Wine", load_wine)


if __name__ == "__main__":
    main()
