def create_model(model_type, **kwargs):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC

    models = {
        'logistic_regression': LogisticRegression,
        'random_forest': RandomForestClassifier,
        'decision_tree': DecisionTreeClassifier,
        'svm': SVC,
    }

    if model_type in models:
        return models[model_type](**kwargs)
    else:
        raise ValueError(f"Model type '{model_type}' is not supported.")