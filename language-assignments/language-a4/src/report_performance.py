"""
Generic functions to evaluate a model on a dataset. 
"""
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras import Model
from pathlib import Path


def get_classification_report(
    model,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> str:
    """
    Generates a classification report for the given model.
    """
    predictions = model.predict(x_test)
    return classification_report(y_test, predictions)


def save_classification_report(report: str, filename: str) -> None:
    """Save classification report as txt file"""
    output_dir = Path("output")
    if not output_dir.exists():
        output_dir.mkdir()

    output_path = output_dir / f"{filename}.txt"
    with open(output_path, "w") as f:
        f.write(report)


def evaluate_model(model: Model, x_test: np.ndarray, y_test: np.ndarray) -> str:
    """Evaluate the predictions and output classification report"""
    y_preds = model.predict(x_test)
    y_preds = np.argmax(y_preds, axis=1)
    return classification_report(y_test, y_preds, zero_division=1)
