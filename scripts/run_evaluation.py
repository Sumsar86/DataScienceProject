import argparse
import pandas as pd
from src.evaluation.evaluate import evaluate_model
from src.data.loader import load_data
from src.utils.logger import setup_logger

def main(model_name, test_data_path):
    # Set up logger
    logger = setup_logger()

    # Load test data
    logger.info("Loading test data...")
    test_data = load_data(test_data_path)

    # Evaluate the model
    logger.info(f"Evaluating model: {model_name}...")
    results = evaluate_model(model_name, test_data)

    # Log results
    logger.info("Evaluation results:")
    logger.info(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model evaluation.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to evaluate.")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the test data.")
    
    args = parser.parse_args()
    main(args.model_name, args.test_data_path)