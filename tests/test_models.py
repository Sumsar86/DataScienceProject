import unittest
from src.models.model_factory import create_model
from src.models.baseline import BaselineModel
from src.models.sklearn_models import SklearnModel

class TestModels(unittest.TestCase):

    def test_baseline_model(self):
        model = BaselineModel()
        self.assertIsNotNone(model)
        # Add more assertions to test the baseline model's functionality

    def test_sklearn_model_creation(self):
        model_name = 'RandomForest'
        model = create_model(model_name)
        self.assertIsNotNone(model)
        self.assertEqual(model.__class__.__name__, 'RandomForestClassifier')  # Adjust based on the actual class name

    def test_invalid_model_creation(self):
        with self.assertRaises(ValueError):
            create_model('InvalidModelName')

if __name__ == '__main__':
    unittest.main()