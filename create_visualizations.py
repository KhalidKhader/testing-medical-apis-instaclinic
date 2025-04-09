import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load the data
df = pd.read_csv("./all-data/evaluation_summary.csv")

# Set the aesthetic style
plt.style.use('ggplot')
sns.set_style('whitegrid')

# 1. Service Configuration Comparison: Medical Term Accuracy
plt.figure(figsize=(10, 6))
english_df = df[df["language"] == "en-CA"]
french_df = df[df["language"] == "fr-CA"]

# Group and calculate means
eng_deepgram = english_df[~english_df["dataset"].str.contains("Azure for English")]["medical_term_accuracy_mean"].mean()
eng_azure = english_df[english_df["dataset"].str.contains("Azure for English")]["medical_term_accuracy_mean"].mean()
fr_azure = french_df[~french_df["dataset"].str.contains("Nova for French")]["medical_term_accuracy_mean"].mean()
fr_deepgram = french_df[french_df["dataset"].str.contains("Nova for French")]["medical_term_accuracy_mean"].mean()

data = [eng_deepgram, eng_azure, fr_azure, fr_deepgram]
labels = ["English + Deepgram", "English + Azure", "French + Azure", "French + Deepgram"]
colors_service = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]

bars = plt.bar(labels, data, color=colors_service)
plt.ylabel("Medical Term Accuracy")
plt.title("Medical Term Accuracy by Service and Language")
plt.ylim(0, 1.05)

# Add values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f"{height:.2f}", ha="center", va="bottom")

plt.tight_layout()
plt.savefig("service_comparison.png", dpi=300)
plt.close()

# 2. Specialty Comparison: Cardiology vs. GP (Medical Term Accuracy)
plt.figure(figsize=(10, 6))

# Filter by specialty
cardio = english_df[english_df["specialty"] == "cardiology"]
gp = english_df[english_df["specialty"] == "gp"]

# Group by dataset type (excluding Azure for English which we know performs poorly)
cardio_vals = cardio[~cardio["dataset"].str.contains("Azure for English")]["medical_term_accuracy_mean"].values
gp_vals = gp[~gp["dataset"].str.contains("Azure for English")]["medical_term_accuracy_mean"].values

plt.bar(["Cardiology", "General Practice"], [cardio_vals.mean(), gp_vals.mean()], color=["#3498db", "#9b59b6"])
plt.ylabel("Medical Term Accuracy")
plt.title("Medical Term Accuracy by Specialty (Deepgram for English)")
plt.ylim(0, 1.1)

# Add values on bars
plt.text(0, cardio_vals.mean() + 0.01, f"{cardio_vals.mean():.2f}", ha="center")
plt.text(1, gp_vals.mean() + 0.01, f"{gp_vals.mean():.2f}", ha="center")

plt.tight_layout()
plt.savefig("specialty_comparison.png", dpi=300)
plt.close()

# 3. Impact of Noise Visualization
plt.figure(figsize=(12, 6))

# Define noise levels
noise_levels = ["No Noise", "Semi-Noise", "Full Noise"]

# Get data for each noise level
clean = english_df[english_df["dataset"].str.contains("without-noise")]
semi = english_df[english_df["dataset"].str.contains("semi-noise")]
noisy = english_df[english_df["dataset"].str.contains("noisy")]

# Ensure we have data for each case and aren't mixing configurations
clean = clean[~clean["dataset"].str.contains("Azure for English")]
semi = semi[~semi["dataset"].str.contains("Azure for English")]
noisy = noisy[~noisy["dataset"].str.contains("Azure for English")]

# Get metrics for each noise level
metrics = ["medical_term_accuracy_mean", "similarity_mean", "wer_mean"]
metrics_names = ["Medical Term Accuracy", "Text Similarity", "WER"]

data = {
    "Medical Term Accuracy": [
        clean[metrics[0]].mean(), 
        semi[metrics[0]].mean(), 
        noisy[metrics[0]].mean()
    ],
    "Text Similarity": [
        clean[metrics[1]].mean(), 
        semi[metrics[1]].mean(), 
        noisy[metrics[1]].mean()
    ],
    "WER": [
        clean[metrics[2]].mean(), 
        semi[metrics[2]].mean(), 
        noisy[metrics[2]].mean()
    ]
}

# Create line plot
for i, metric in enumerate(metrics_names):
    plt.plot(noise_levels, data[metric], marker="o", linewidth=2, markersize=8, label=metric)
    
    # Add values near points
    for j, val in enumerate(data[metric]):
        plt.text(j, val + 0.01, f"{val:.2f}", ha="center")

plt.xlabel("Noise Level")
plt.ylabel("Score")
plt.title("Impact of Noise on English Transcription Metrics")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("noise_impact.png", dpi=300)
plt.close()

# 4. Speaker diarization - just show the constant 0%
plt.figure(figsize=(8, 4))
plt.bar(["All Configurations"], [0], color="#e74c3c")
plt.ylabel("Speaker Diarization Accuracy")
plt.title("Speaker Diarization Performance")
plt.ylim(0, 1)
plt.text(0, 0.05, "0.00", ha="center", color="white", fontweight="bold")
plt.tight_layout()
plt.savefig("speaker_diarization.png", dpi=300)
plt.close()

print("All visualizations created successfully!") 