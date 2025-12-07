import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    # Split the data into training and temporary sets
    X_train_data, X_temp_data, y_train_data, y_temp_data = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Calculate the new validation size relative to the remaining data
    new_val_size = val_size / (1 - test_size)
    
    # Split the temporary set into validation and test sets
    X_val_data, X_test_data, y_val_data, y_test_data = train_test_split(X_temp_data, y_temp_data, test_size=new_val_size, random_state=random_state)
    
    return X_train_data, X_val_data, X_test_data, y_train_data, y_val_data, y_test_data

# def save_splits(train_data, val_data, test_data, output_dir):
#     train_data.to_csv(f"{output_dir}/train_data.csv", index=False)
#     val_data.to_csv(f"{output_dir}/val_data.csv", index=False)
#     test_data.to_csv(f"{output_dir}/test_data.csv", index=False)