import pandas as pd

def load_data(file_path):
    """Load the dataset from a CSV file."""
    data = pd.read_csv(file_path)
    return data

def one_hot_encode(data):
    """Perform one-hot encoding on specified categorical columns."""
    return pd.get_dummies(data, drop_first=True)

def clean_data(data):
    """Clean the dataset by handling missing values and removing duplicates."""
    # Handle missing values
    data = data.dropna()  # Example: drop rows with missing values
    # Remove duplicates
    data = data.drop_duplicates()
    return data

def preprocess_data(file_path):
    """Load and clean the dataset."""
    data = load_data(file_path)
    cleaned_data = clean_data(data)
    return cleaned_data

if __name__ == "__main__":
    # Example usage
    file_path = '../../data/raw/Australian_Student_PerformanceData (ASPD24).csv'
    cleaned_data = preprocess_data(file_path)
    cleaned_data = one_hot_encode(cleaned_data)
    print(cleaned_data.head())