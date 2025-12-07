from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def get_param_grid(model_name):
    if model_name == "logistic_regression":
        model = LogisticRegression(max_iter=5000)
        params = {
            "C": [0.01, 0.1, 1, 10],
            "solver": ["liblinear", "lbfgs"],
            "class_weight": [None, "balanced"]
        }

    elif model_name == "decision_tree":
        model = DecisionTreeClassifier()
        params = {
            "max_depth": [5, 10, 20, 30, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "criterion": ["gini", "entropy"]
        }

    elif model_name == "random_forest":
        model = RandomForestClassifier()
        params = {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, 30, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }

    elif model_name == "svm":
        model = SVC()
        params = {
            "C": [0.1, 1, 10],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
            "class_weight": [None, "balanced"]
        }

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return {"model": model, "params": params}


