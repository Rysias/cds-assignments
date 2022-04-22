import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.callbacks import History  # type: ignore


def plot_metric(history: History, metric_name: str = "accuracy") -> None:
    assert metric_name in history.history.keys(), ValueError(
        "Metric name not found in history."
    )
    plt.plot(history.history[metric_name], label="train")
    plt.plot(history.history[f"val_{metric_name}"], label="test")
    plt.legend()
    # Save plot to file
    output_path = Path("Output") / f"plot_{metric_name}.png"
    plt.savefig(output_path)

