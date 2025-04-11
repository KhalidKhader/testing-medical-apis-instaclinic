#!/usr/bin/env python3
"""
Generate custom visualizations for the scientific evaluation report from evaluation data.

Usage:
    python create_visualization.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def main():
    """Generate visualizations for the report."""
    os.makedirs("evaluation_results/custom_viz", exist_ok=True)
    
    # Load model comparison data
    df = pd.read_csv("evaluation_results/statistics/model_comparison_data.csv")
    
    # Filter out suspicious transcripts
    df = df[~df["suspicious"]]
    
    # Create model comparison visualization
    create_model_comparison(df, "evaluation_results/custom_viz")
    
    # Create language comparison visualization
    create_language_comparison(df, "evaluation_results/custom_viz")
    
    # Create noise impact visualization (if noise data is available)
    if "noise_level" in df.columns:
        create_noise_impact(df, "evaluation_results/custom_viz")
    else:
        # Infer noise levels from dataset names
        df["noise_level"] = "Unknown"
        for idx, row in df.iterrows():
            if "Without-noise" in row.get("dataset", ""):
                df.at[idx, "noise_level"] = "No Noise"
            elif "Semi-noise" in row.get("dataset", ""):
                df.at[idx, "noise_level"] = "Semi-Noise"
            elif "Noisy" in row.get("dataset", ""):
                df.at[idx, "noise_level"] = "Full Noise"
        
        # Proceed with noise impact visualization if we have data
        if df["noise_level"].nunique() > 1:
            create_noise_impact(df, "evaluation_results/custom_viz")
    
    # Create specialty comparison
    create_specialty_comparison(df, "evaluation_results/custom_viz")
    
    print("Custom visualizations have been generated in evaluation_results/custom_viz/")

def create_model_comparison(df, output_dir):
    """Create model comparison visualization."""
    plt.figure(figsize=(14, 8))
    
    # Define metrics to show
    metrics = ["wer", "medical_accuracy", "similarity", "speaker_accuracy"]
    pretty_metrics = {
        "wer": "Word Error Rate (↓)",
        "medical_accuracy": "Medical Term F1",
        "similarity": "Text Similarity",
        "speaker_accuracy": "Speaker Accuracy"
    }
    
    # Group by model and language
    model_lang_groups = df.groupby(["model", "language"])
    
    # Create positions for grouped bars
    positions = np.arange(len(metrics))
    bar_width = 0.15
    
    # Color palette
    colors = {
        "Nova-3-medical_en": "#3498db",  # Blue
        "Azure_en": "#2ecc71",           # Green
        "Nova-2_fr": "#e74c3c",          # Red
        "Azure_fr": "#f39c12"            # Orange
    }
    
    # Plot bars for each model-language group
    i = 0
    legend_entries = []
    
    for (model, language), group_df in model_lang_groups:
        group_key = f"{model}_{language}"
        if group_key not in colors:
            continue
            
        means = []
        errors = []
        
        for metric in metrics:
            values = group_df[metric].dropna().values
            if len(values) > 0:
                mean = np.mean(values)
                error = 1.96 * np.std(values) / np.sqrt(len(values))
                means.append(mean)
                errors.append(error)
            else:
                means.append(0)
                errors.append(0)
        
        pos = positions + (i - 1.5) * bar_width
        bars = plt.bar(
            pos, means, width=bar_width, 
            color=colors[group_key], alpha=0.8, 
            edgecolor="black", linewidth=1
        )
        
        # Add error bars
        plt.errorbar(
            pos, means, yerr=errors, 
            fmt="none", ecolor="black", capsize=5
        )
        
        # Add text labels
        for j, (mean, error) in enumerate(zip(means, errors)):
            plt.text(
                pos[j], mean + error + 0.02,
                f"{mean:.2f}",
                ha="center", va="bottom",
                fontsize=8, rotation=0,
                color="black"
            )
        
        legend_entries.append(f"{model} ({language})")
        i += 1
    
    # Add labels and style
    plt.xticks(positions, [pretty_metrics[m] for m in metrics], fontsize=12)
    plt.ylabel("Score (with 95% CI)", fontsize=12)
    plt.title("Scientific Model Performance Comparison by Language", fontsize=14, fontweight="bold")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(legend_entries, title="Model (Language)", loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=len(legend_entries))
    
    # Add annotation about WER
    plt.figtext(0.5, 0.01, "Note: For Word Error Rate (WER), lower values indicate better performance.", 
               ha="center", fontsize=10, fontstyle="italic")
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "model_performance_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()

def create_language_comparison(df, output_dir):
    """Create language comparison visualization."""
    plt.figure(figsize=(12, 6))
    
    # Group by language
    language_groups = df.groupby("language")
    
    # Define metrics to show
    metrics = ["wer", "medical_accuracy", "similarity", "speaker_accuracy"]
    pretty_metrics = {
        "wer": "Word Error Rate (↓)",
        "medical_accuracy": "Medical Term F1",
        "similarity": "Text Similarity",
        "speaker_accuracy": "Speaker Accuracy"
    }
    
    # Set up positions
    positions = np.arange(len(metrics))
    bar_width = 0.4
    
    # Colors
    colors = {
        "en": "#3498db",  # Blue
        "fr": "#e74c3c",  # Red
    }
    
    # Plot bars for each language
    i = 0
    for language, group_df in language_groups:
        lang_code = language.split("-")[0] if "-" in language else language[:2]
        if lang_code not in colors:
            continue
            
        means = []
        errors = []
        
        for metric in metrics:
            values = group_df[metric].dropna().values
            if len(values) > 0:
                mean = np.mean(values)
                error = 1.96 * np.std(values) / np.sqrt(len(values))
                means.append(mean)
                errors.append(error)
            else:
                means.append(0)
                errors.append(0)
        
        pos = positions + (i - 0.5) * bar_width
        bars = plt.bar(
            pos, means, width=bar_width, 
            color=colors[lang_code], alpha=0.8, 
            edgecolor="black", linewidth=1,
            label=f"{language.split('-')[0].upper()} (n={len(group_df)})"
        )
        
        # Add error bars
        plt.errorbar(
            pos, means, yerr=errors, 
            fmt="none", ecolor="black", capsize=5
        )
        
        # Add text labels with confidence intervals
        for j, (mean, error) in enumerate(zip(means, errors)):
            plt.text(
                pos[j], mean + error + 0.02,
                f"{mean:.2f} ± {error:.2f}",
                ha="center", va="bottom",
                fontsize=9, rotation=0
            )
        
        i += 1
    
    # Add labels and style
    plt.xticks(positions, [pretty_metrics[m] for m in metrics], fontsize=12)
    plt.ylabel("Score (with 95% CI)", fontsize=12)
    plt.title("Language Performance Comparison", fontsize=14, fontweight="bold")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(title="Language", loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    
    # Add annotation
    plt.figtext(0.5, 0.01, "Note: For Word Error Rate (WER), lower values indicate better performance.", 
               ha="center", fontsize=10, fontstyle="italic")
    
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "language_performance_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()

def create_noise_impact(df, output_dir):
    """Create noise impact visualization."""
    # Filter out data points without noise level information
    noise_df = df[df["noise_level"] != "Unknown"].copy()
    
    if len(noise_df) < 10:  # Not enough data
        return
    
    # Ensure noise levels are in the correct order
    noise_order = ["No Noise", "Semi-Noise", "Full Noise"]
    noise_df["noise_level"] = pd.Categorical(noise_df["noise_level"], categories=noise_order, ordered=True)
    
    plt.figure(figsize=(12, 8))
    
    # Metrics to analyze
    metrics = ["wer", "medical_accuracy"]
    titles = {
        "wer": "Word Error Rate (WER) by Noise Level",
        "medical_accuracy": "Medical Term F1 Score by Noise Level"
    }
    
    # Create plots
    for i, metric in enumerate(metrics):
        plt.subplot(1, 2, i+1)
        
        # Split by language
        for lang, lang_group in noise_df.groupby("language"):
            lang_code = lang.split("-")[0] if "-" in lang else lang[:2]
            
            # Calculate mean and CI for each noise level
            noise_means = []
            noise_cis = []
            noise_levels = []
            
            for noise_level, noise_group in lang_group.groupby("noise_level"):
                if len(noise_group) >= 3:  # Need at least 3 data points for meaningful stats
                    values = noise_group[metric].dropna().values
                    if len(values) > 0:
                        mean = np.mean(values)
                        ci = 1.96 * np.std(values) / np.sqrt(len(values))
                        noise_means.append(mean)
                        noise_cis.append(ci)
                        noise_levels.append(noise_level)
            
            if len(noise_levels) >= 2:  # Need at least 2 noise levels for a trend
                # Plot the trend
                x_pos = np.arange(len(noise_levels))
                plt.errorbar(
                    x_pos, noise_means, yerr=noise_cis,
                    marker='o', markersize=8, capsize=5,
                    label=f"{lang_code.upper()}",
                    linewidth=2
                )
        
        # Add labels and style
        plt.title(titles[metric], fontsize=12)
        plt.xticks(np.arange(len(noise_order)), noise_order, rotation=0)
        plt.xlabel("Noise Level", fontsize=11)
        
        if metric == "wer":
            plt.ylabel("Word Error Rate (lower is better)", fontsize=11)
        else:
            plt.ylabel("F1 Score", fontsize=11)
            
        plt.grid(linestyle='--', alpha=0.7)
        plt.legend()
    
    plt.suptitle("Impact of Noise on Speech Recognition Performance", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "noise_impact_analysis.png"), dpi=300, bbox_inches="tight")
    plt.close()

def create_specialty_comparison(df, output_dir):
    """Create specialty comparison visualization."""
    plt.figure(figsize=(12, 6))
    
    # Only focus on medical term accuracy for specialties
    metric = "medical_accuracy"
    
    # Set up for grouped bars
    specialties = df["specialty"].unique()
    languages = df["language"].unique()
    
    # Filter to languages we have names for
    languages = [l for l in languages if l.startswith("en") or l.startswith("fr")]
    
    # Language colors
    lang_colors = {
        "en": "#3498db",  # Blue
        "fr": "#e74c3c",  # Red
    }
    
    # Set positions for grouped bars
    positions = np.arange(len(specialties))
    bar_width = 0.8 / len(languages)
    
    # Plot bars for each language
    for i, language in enumerate(languages):
        lang_df = df[df["language"] == language]
        lang_code = language.split("-")[0] if "-" in language else language[:2]
        
        means = []
        errors = []
        
        for specialty in specialties:
            spec_df = lang_df[lang_df["specialty"] == specialty]
            values = spec_df[metric].dropna().values
            
            if len(values) > 0:
                mean = np.mean(values)
                error = 1.96 * np.std(values) / np.sqrt(len(values))
                means.append(mean)
                errors.append(error)
            else:
                means.append(0)
                errors.append(0)
        
        pos = positions + (i - len(languages)/2 + 0.5) * bar_width
        bars = plt.bar(
            pos, means, width=bar_width, 
            color=lang_colors.get(lang_code, "#95a5a6"), alpha=0.8, 
            edgecolor="black", linewidth=1,
            label=f"{lang_code.upper()} (n={len(lang_df)})"
        )
        
        # Add error bars
        plt.errorbar(
            pos, means, yerr=errors, 
            fmt="none", ecolor="black", capsize=5
        )
        
        # Add text labels
        for j, (mean, error) in enumerate(zip(means, errors)):
            if mean > 0:
                plt.text(
                    pos[j], mean + error + 0.02,
                    f"{mean:.2f}",
                    ha="center", va="bottom",
                    fontsize=9, rotation=0
                )
    
    # Add labels and style
    plt.xticks(positions, [s.capitalize() for s in specialties], fontsize=12)
    plt.ylabel("Medical Term F1 Score (with 95% CI)", fontsize=12)
    plt.title("Medical Term Recognition by Specialty", fontsize=14, fontweight="bold")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(title="Language", loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(languages))
    
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "specialty_term_recognition.png"), dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main() 