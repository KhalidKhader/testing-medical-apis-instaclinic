"""
Visualization functions for generating plots from evaluation results.

This module provides functions for creating various visualizations of speech-to-text
evaluation results, including model comparisons, language analysis, and more.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union


def set_plotting_style():
    """Set consistent plotting style for all visualizations."""
    sns.set(style="whitegrid")
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["figure.titlesize"] = 16


def create_model_comparison_plot(
    data: pd.DataFrame,
    metric: str = "wer",
    output_dir: str = "results/figures",
    filename: str = "model_performance_comparison.png"
) -> str:
    """
    Create a bar chart comparing model performance across different metrics.
    
    Args:
        data: DataFrame containing evaluation results
        metric: Metric to visualize (default: "wer")
        output_dir: Directory to save the plot
        filename: Filename for the saved plot
        
    Returns:
        Path to the saved plot
    """
    set_plotting_style()
    
    plt.figure(figsize=(14, 8))
    
    # Create the plot
    ax = sns.barplot(
        x="model", 
        y=metric, 
        hue="language",
        data=data,
        palette="viridis",
        errorbar=("ci", 95),
        capsize=0.2
    )
    
    # Customize the plot
    plt.title(f"Model Performance Comparison ({metric.upper()})")
    plt.xlabel("Model")
    plt.ylabel(metric.upper())
    
    # Add value labels on the bars
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.2f}",
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
            color="black",
            xytext=(0, 5),
            textcoords="offset points"
        )
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    return output_path


def create_language_comparison_plot(
    data: pd.DataFrame,
    metrics: List[str] = ["wer", "f1_score", "semantic_similarity"],
    output_dir: str = "results/figures",
    filename: str = "language_performance_comparison.png"
) -> str:
    """
    Create a grouped bar chart comparing language performance across multiple metrics.
    
    Args:
        data: DataFrame containing evaluation results
        metrics: List of metrics to visualize
        output_dir: Directory to save the plot
        filename: Filename for the saved plot
        
    Returns:
        Path to the saved plot
    """
    set_plotting_style()
    
    # Prepare data for plotting
    plot_data = []
    for metric in metrics:
        for language in data["language"].unique():
            language_data = data[data["language"] == language]
            mean_value = language_data[metric].mean()
            ci_low, ci_high = calculate_confidence_interval(language_data[metric])
            
            plot_data.append({
                "language": language,
                "metric": metric,
                "value": mean_value,
                "ci_low": ci_low,
                "ci_high": ci_high
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create plot
    plt.figure(figsize=(14, 8))
    
    ax = sns.barplot(
        x="metric",
        y="value",
        hue="language",
        data=plot_df,
        palette="muted",
        errorbar=None
    )
    
    # Add error bars manually
    for i, row in enumerate(plot_df.itertuples()):
        ax.errorbar(
            x=i % len(metrics) + (i // len(metrics)) * 0.8 - 0.2,
            y=row.value,
            yerr=[[row.value - row.ci_low], [row.ci_high - row.value]],
            fmt="none",
            color="black",
            capsize=5
        )
    
    # Customize the plot
    plt.title("Language Performance Comparison Across Metrics")
    plt.xlabel("Metric")
    plt.ylabel("Score")
    
    # Add value labels
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.2f}",
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
            color="black",
            xytext=(0, 5),
            textcoords="offset points"
        )
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    return output_path


def create_heatmap(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    value_col: str,
    title: str,
    output_dir: str = "results/figures",
    filename: str = "heatmap.png"
) -> str:
    """
    Create a heatmap visualization.
    
    Args:
        data: DataFrame containing evaluation results
        x_col: Column to use for x-axis
        y_col: Column to use for y-axis
        value_col: Column containing values for the heatmap
        title: Plot title
        output_dir: Directory to save the plot
        filename: Filename for the saved plot
        
    Returns:
        Path to the saved plot
    """
    set_plotting_style()
    
    # Pivot data for heatmap
    pivot_data = data.pivot_table(
        index=y_col,
        columns=x_col,
        values=value_col,
        aggfunc="mean"
    )
    
    plt.figure(figsize=(12, 10))
    
    # Create the heatmap
    ax = sns.heatmap(
        pivot_data,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={"label": value_col}
    )
    
    plt.title(title)
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    return output_path


def calculate_confidence_interval(
    data: Union[List[float], np.ndarray, pd.Series],
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate confidence interval for a data series.
    
    Args:
        data: Series of values
        confidence: Confidence level (default: 0.95)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    import scipy.stats as stats
    
    data = np.array(data)
    mean = np.mean(data)
    n = len(data)
    
    # Standard error of the mean
    sem = stats.sem(data)
    
    # Calculate the margin of error
    h = sem * stats.t.ppf((1 + confidence) / 2, n - 1)
    
    return mean - h, mean + h 