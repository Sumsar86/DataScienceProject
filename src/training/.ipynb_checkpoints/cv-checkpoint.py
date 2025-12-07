import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

def cross_validate(model, X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        score = accuracy_score(y_val, predictions)
        scores.append(score)

    return scores

def main():
    # Example usage
    from sklearn.ensemble import RandomForestClassifier
    data = pd.read_csv('../data/raw/Australian_Student_PerformanceData (ASPD24).csv')
    X = data.drop('target_column', axis=1)  # Replace 'target_column' with the actual target column name
    y = data['target_column']  # Replace 'target_column' with the actual target column name

    model = RandomForestClassifier()
    scores = cross_validate(model, X, y)
    print(f'Cross-validation scores: {scores}')
    print(f'Mean score: {sum(scores) / len(scores)}')

if __name__ == "__main__":
    main()