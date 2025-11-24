def evaluate_model(model, X, y, metrics):
    """
    Evaluate the performance of a trained model using specified metrics.

    Parameters:
    - model: The trained machine learning model.
    - X: Features of the dataset to evaluate.
    - y: True labels of the dataset.
    - metrics: A dictionary of metric functions to evaluate the model.

    Returns:
    - results: A dictionary containing the evaluation results for each metric.
    """
    results = {}
    predictions = model.predict(X)

    for metric_name, metric_func in metrics.items():
        results[metric_name] = metric_func(y, predictions)

    return results


def print_evaluation_results(results):
    """
    Print the evaluation results in a readable format.

    Parameters:
    - results: A dictionary containing evaluation results.
    """
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")