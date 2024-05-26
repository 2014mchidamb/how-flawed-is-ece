import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from typing import list


def plot_multi_dataset_metrics(
    fname: str,
    x_label: str,
    y_label: str,
    xs: np.ndarray,
    metric_means: np.ndarray,
    metric_stds: np.ndarray,
    datasets: list[str],
    custom_colors: list[str] = None,
    custom_lines: list[str] = None,
    x_lim: float = None,
    y_lim: float = None,
):
    """Plots metric for several datasets/model types/etc.

    Args:
        fname (str): File name for saving.
        x_label (str): X-axis label.
        y_label (str): Y-axis label.
        xs (np.ndarray): X values.
        metric_means (np.ndarray): Means of the metric being considered over datasets.
        metric_stds (np.ndarray): 1 Std of the the metric being considered over datasets.
        datasets (list[str]): List of datasets/model types/etc.
        custom_colors (list[str], optional): Custom list of colors. Defaults to None.
        custom_lines (list[str], optional): Custom list of line types. Defaults to None.
        x_lim (float, optional): Limit for X-axis. Defaults to None.
        y_lim (float, optional): Limit for Y-axis. Defaults to None.
    """
    # Plot parameters.
    plt.figure(figsize=(9, 7))
    plt.rc("axes", titlesize=18, labelsize=18)
    plt.rc("xtick", labelsize=15)
    plt.rc("ytick", labelsize=15)
    plt.rc("legend", fontsize=18)
    plt.rc("figure", titlesize=18)

    if x_lim is not None:
        plt.xlim(0, x_lim)
    if y_lim is not None:
        plt.ylim(0, y_lim)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    for i, dataset in enumerate(datasets):
        color = f"C{i}"
        line = "solid"
        if custom_colors is not None:
            color = custom_colors[i]
        if custom_lines is not None:
            line = custom_lines[i]
        plt.plot(xs, metric_means[i], label=dataset, color=color, linestyle=line)
        if metric_stds is not None:
            # One std area around each curve.
            plt.fill_between(
                xs,
                metric_means[i] - metric_stds[i],
                metric_means[i] + metric_stds[i],
                facecolor=color,
                alpha=0.2,
            )

    if len(datasets) > 1:
        plt.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(fname)
