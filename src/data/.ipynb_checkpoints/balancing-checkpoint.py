from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import pandas as pd

def balance_dataset(df, target_column, method='smote'):
    """
    Balances the dataset using the specified method.

    Parameters:
    df (pd.DataFrame): The input dataframe to be balanced.
    target_column (str): The name of the target column.
    method (str): The balancing method to use ('smote' or 'undersample').

    Returns:
    pd.DataFrame: The balanced dataframe.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    if method == 'smote':
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
    elif method == 'undersample':
        undersample = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = undersample.fit_resample(X, y)
    else:
        raise ValueError("Method must be 'smote' or 'undersample'.")

    balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
    balanced_df[target_column] = y_resampled

    print(f"Original dataset shape: {Counter(y)}")
    print(f"Balanced dataset shape: {Counter(y_resampled)}")

    return balanced_df