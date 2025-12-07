from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

def get_model(model_name, **kwargs):
    if model_name == 'logistic_regression':
        return LogisticRegression(**kwargs)
    elif model_name == 'random_forest':
        return RandomForestClassifier(**kwargs)
    elif model_name == 'svm':
        return SVC(**kwargs)
    elif model_name == 'decision_tree':
        return DecisionTreeClassifier(**kwargs)
    elif model_name == 'knn':
        return KNeighborsClassifier(**kwargs)
    elif model_name == 'naive_bayes':
        return GaussianNB(**kwargs)
    else:
        raise ValueError(f"Model {model_name} is not recognized.")