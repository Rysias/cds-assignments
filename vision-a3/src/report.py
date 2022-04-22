import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.callbacks import History  # type: ignore


def plot_metric(
    history: History, output_dir: Path, metric_name: str = "accuracy"
) -> None:
    assert metric_name in history.history.keys(), ValueError(
        "Metric name not found in history."
    )
    plt.plot(history.history[metric_name], label="train")
    plt.plot(history.history[f"val_{metric_name}"], label="test")
    plt.legend()
    # Save plot to file
    output_path = output_dir / f"plot_{metric_name}.png"
    plt.savefig(output_path)


def plot_metrics(history: History, output_dir: Path) -> None:
    METRICS = ["loss", "accuracy"]
    for metric in METRICS:
        plot_metric(history, output_dir, metric_name=metric)


def save_classification_report(
    report: str, output_dir: Path, filename: str = "report.txt"
) -> None:
    output_path = output_dir / filename
    with open(output_path, "w") as f:
        f.write(report)
