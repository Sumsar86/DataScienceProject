import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(data, test_size=0.2, val_size=0.1, random_state=42):
    # Split the data into training and temporary sets
    train_data, temp_data = train_test_split(data, test_size=test_size, random_state=random_state)
    
    # Calculate the new validation size relative to the remaining data
    new_val_size = val_size / (1 - test_size)
    
    # Split the temporary set into validation and test sets
    val_data, test_data = train_test_split(temp_data, test_size=new_val_size, random_state=random_state)
    
    return train_data, val_data, test_data

# def save_splits(train_data, val_data, test_data, output_dir):
#     train_data.to_csv(f"{output_dir}/train_data.csv", index=False)
#     val_data.to_csv(f"{output_dir}/val_data.csv", index=False)
#     test_data.to_csv(f"{output_dir}/test_data.csv", index=False)