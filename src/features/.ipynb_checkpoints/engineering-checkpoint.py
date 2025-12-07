import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def feature_engineering(df):
    # Example feature engineering steps
    # Assuming 'gender' and 'age' are columns in the dataset
    df['gender'] = df['gender'].map({'male': 0, 'female': 1})
    
    # Create age groups
    df['age_group'] = pd.cut(df['age'], bins=[0, 18, 25, 35, 100], labels=['0-18', '19-25', '26-35', '36+'])
    
    # Drop original 'age' column
    df.drop('age', axis=1, inplace=True)
    
    return df

def preprocess_data(df):
    # Define categorical and numerical features
    categorical_features = ['gender', 'age_group']
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Create a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])
    
    # Apply transformations
    df_processed = preprocessor.fit_transform(df)
    
    return df_processed