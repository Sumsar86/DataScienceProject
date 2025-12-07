import unittest
from src.training.trainer import train_model
from src.evaluation.evaluate import evaluate_model
from src.data.loader import load_data
from src.data.cleaning import clean_data
from src.data.balancing import balance_data
from src.data.split import split_data

class TestTraining(unittest.TestCase):

    def setUp(self):
        self.data = load_data('data/raw/Australian_Student_PerformanceData (ASPD24).csv')
        self.cleaned_data = clean_data(self.data)
        self.balanced_data = balance_data(self.cleaned_data)
        self.train_data, self.val_data, self.test_data = split_data(self.balanced_data)

    def test_train_model(self):
        model = train_model(self.train_data)
        self.assertIsNotNone(model)

    def test_evaluate_model(self):
        model = train_model(self.train_data)
        evaluation_results = evaluate_model(model, self.val_data)
        self.assertIn('accuracy', evaluation_results)
        self.assertGreaterEqual(evaluation_results['accuracy'], 0.5)  # Assuming 50% is a baseline

if __name__ == '__main__':
    unittest.main()