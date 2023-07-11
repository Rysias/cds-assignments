import numpy as np
from tensorflow.keras import Model  # type: ignore
from sklearn.metrics import classification_report


def evaluate_model(model: Model, x_test: np.ndarray, y_test: np.ndarray) -> str:
    """Evaluate the predictions and output classification report"""
    y_preds = model.predict(x_test)
    y_preds = np.argmax(y_preds, axis=1)
    return classification_report(y_test, y_preds)
