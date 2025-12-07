import pandas as pd
from src.data.loader import load_data
from src.data.cleaning import clean_data
from src.data.balancing import balance_data
from src.data.split import split_data
from src.training.trainer import train_model
from src.training.hyperparameter_tuning import tune_hyperparameters
from src.utils.logger import setup_logger

def main():
    # Setup logger
    logger = setup_logger()

    # Load data
    logger.info("Loading data...")
    data = load_data("data/raw/Australian_Student_PerformanceData (ASPD24).csv")

    # Clean data
    logger.info("Cleaning data...")
    cleaned_data = clean_data(data)

    # Balance data
    logger.info("Balancing data...")
    balanced_data = balance_data(cleaned_data)

    # Split data
    logger.info("Splitting data...")
    train_data, val_data, test_data = split_data(balanced_data)

    # Train model
    logger.info("Training model...")
    model = train_model(train_data)

    # Hyperparameter tuning
    logger.info("Tuning hyperparameters...")
    best_model = tune_hyperparameters(model, val_data)

    logger.info("Training process completed.")

if __name__ == "__main__":
    main()