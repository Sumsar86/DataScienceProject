# Australian Student Performance Data Machine Learning Project

This project aims to analyze and predict student performance using machine learning techniques. The dataset used for this project is the "Australian Student Performance Data (ASPD24)", which contains various features related to student demographics, academic performance, and other relevant factors.

## Project Structure

The project is organized into the following directories and files:

- **data/**
  - **raw/**: Contains the raw dataset.
    - `Australian_Student_PerformanceData (ASPD24).csv`: The original dataset used for analysis.
  - **processed/**: This directory will store processed data after cleaning and balancing.

- **notebooks/**: Contains Jupyter notebooks for analysis and experimentation.
  - `01-eda.ipynb`: Exploratory Data Analysis notebook.
  - `02-model-experiments.ipynb`: Notebook for experimenting with different machine learning models.

- **src/**: Contains source code for data processing, modeling, training, and evaluation.
  - **data/**: Modules for data handling.
    - `loader.py`: Functions to load the dataset.
    - `cleaning.py`: Functions for cleaning the dataset.
    - `balancing.py`: Functions to balance the dataset.
    - `split.py`: Functions to split the dataset into training, validation, and test sets.
  - **features/**: Module for feature engineering.
    - `engineering.py`: Functions for transforming raw data into features.
  - **models/**: Modules for model creation and implementation.
    - `model_factory.py`: Factory pattern for creating models.
    - `baseline.py`: Baseline model implementation.
    - `sklearn_models.py`: Implementations of scikit-learn models.
  - **training/**: Modules for training models.
    - `trainer.py`: Functions to train models.
    - `cv.py`: Cross-validation techniques.
    - `hyperparameter_tuning.py`: Functions for hyperparameter tuning.
  - **evaluation/**: Modules for model evaluation.
    - `evaluate.py`: Functions to evaluate model performance.
    - `metrics.py`: Performance metrics.
  - **utils/**: Utility functions.
    - `logger.py`: Logging functions.
    - `seed.py`: Functions for setting random seeds.

- **tests/**: Contains unit tests for various components of the project.
  - `test_data.py`: Tests for data handling functions.
  - `test_models.py`: Tests for model implementations.
  - `test_training.py`: Tests for training and evaluation functions.

- **configs/**: Configuration files.
  - `config.yaml`: Configuration settings for the project.

- **scripts/**: Scripts for running the project.
  - `run_training.py`: Script to execute the training process.
  - `run_evaluation.py`: Script to execute the evaluation of models.

- **requirements.txt**: Lists required Python packages for the project.

- **environment.yml**: Specifies environment configuration for package management.

- **.gitignore**: Specifies files and directories to be ignored by version control.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd ASPD24-ml-project
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up the environment (if using conda):
   ```
   conda env create -f environment.yml
   ```

4. Run the Jupyter notebooks for exploratory data analysis and model experimentation.

## Usage

- Use the `notebooks/01-eda.ipynb` for initial data exploration and visualization.
- Experiment with different models in `notebooks/02-model-experiments.ipynb`.
- Use the scripts in the `scripts/` directory to run training and evaluation processes.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for details.