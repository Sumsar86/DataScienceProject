import unittest
import pandas as pd
from src.data.loader import load_data
from src.data.cleaning import clean_data
from src.data.balancing import balance_data
from src.data.split import split_data

class TestDataProcessing(unittest.TestCase):

    def setUp(self):
        self.file_path = 'data/raw/Australian_Student_PerformanceData (ASPD24).csv'
        self.data = load_data(self.file_path)

    def test_load_data(self):
        self.assertIsInstance(self.data, pd.DataFrame)
        self.assertFalse(self.data.empty)

    def test_clean_data(self):
        cleaned_data = clean_data(self.data)
        self.assertFalse(cleaned_data.isnull().values.any())
        self.assertEqual(cleaned_data.shape[0], self.data.shape[0])  # Assuming no rows are dropped

    def test_balance_data(self):
        balanced_data = balance_data(self.data)
        self.assertEqual(balanced_data['target_column'].value_counts().min(), 
                         balanced_data['target_column'].value_counts().max())  # Replace 'target_column' with actual target column name

    def test_split_data(self):
        train_data, val_data, test_data = split_data(self.data)
        self.assertEqual(len(train_data) + len(val_data) + len(test_data), len(self.data))
        self.assertEqual(len(train_data), int(0.7 * len(self.data)))
        self.assertEqual(len(val_data), int(0.15 * len(self.data)))
        self.assertEqual(len(test_data), int(0.15 * len(self.data)))

if __name__ == '__main__':
    unittest.main()