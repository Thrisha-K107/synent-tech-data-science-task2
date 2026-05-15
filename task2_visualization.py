# ============================================================
# Task 2: Data Visualization — Iris Dataset
# Synent Technologies Data Science Internship
# ============================================================

# Step 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.datasets import load_iris
import warnings

warnings.filterwarnings("ignore")

# Set global plot style
sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.dpi"] = 120

# ============================================================
# Step 2: Load the Iris Dataset
# ============================================================

print("=" * 60)
print("STEP 1: LOADING IRIS DATASET")
print("=" * 60)

# Load from sklearn's built-in datasets
iris_raw = load_iris(as_frame=True)
df = iris_raw.frame

# Rename target column for clarity
df["species"] = df["target"].map({0: "setosa", 1: "versicolor", 2: "virginica"})
df.drop(columns=["target"], inplace=True)

# Rename columns to cleaner names
df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

print(f"Dataset Shape: {df.shape}")
print(f"\nColumn Names: {df.columns.tolist()}")
print(f"\nSpecies Distribution:\n{df['species'].value_counts()}")
print(f"\nFirst 5 Rows:\n{df.head()}")
print(f"\nStatistical Summary:\n{df.describe()}")

# ============================================================
# Step 3: Plot 1 — Bar Chart (Mean Feature Values by Species)
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: CREATING BAR CHART")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 6))

features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
species_list = ["setosa", "versicolor", "virginica"]
colors = ["#3498db", "#2ecc71", "#e74c3c"]
x = np.arange(len(features))    # X positions for groups
bar_width = 0.25                 # Width of each bar

for i, (species, color) in enumerate(zip(species_list, colors)):
    # Filter data for this species and compute mean
    means = df[df["species"] == species][features].mean().values
    # Plot bars, offsetting each species group by bar_width
    bars = ax.bar(x + i * bar_width, means, width=bar_width,
                  label=species.capitalize(), color=color, edgecolor="black", alpha=0.88)

# Axis formatting
ax.set_xticks(x + bar_width)
ax.set_xticklabels([f.replace("_", " ").title() for f in features])
ax.set_xlabel("Feature", fontsize=13)
ax.set_ylabel("Mean Value (cm)", fontsize=13)
ax.set_title("Mean Feature Values by Species — Iris Dataset", fontsize=15, fontweight="bold")
ax.legend(title="Species")
ax.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("iris_bar_chart.png", dpi=150, bbox_inches="tight")
plt.show()
print("✔ Bar chart saved as 'iris_bar_chart.png'")

# ============================================================
# Step 4: Plot 2 — Histograms (Feature Distributions per Species)
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: CREATING HISTOGRAMS")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Feature Distributions by Species — Iris Dataset", fontsize=15, fontweight="bold")

for ax, feature in zip(axes.flatten(), features):
    for species, color in zip(species_list, colors):
        # Plot histogram for each species on same axis (overlapping, transparent)
        subset = df[df["species"] == species][feature]
        ax.hist(subset, bins=15, alpha=0.65, label=species.capitalize(),
                color=color, edgecolor="white")

    ax.set_title(feature.replace("_", " ").title(), fontsize=12, fontweight="bold")
    ax.set_xlabel("Value (cm)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("iris_histograms.png", dpi=150, bbox_inches="tight")
plt.show()
print("✔ Histograms saved as 'iris_histograms.png'")

# ============================================================
# Step 5: Plot 3 — Scatter Plot (Petal Length vs Petal Width)
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: CREATING SCATTER PLOT")
print("=" * 60)

fig, ax = plt.subplots(figsize=(9, 7))

for species, color in zip(species_list, colors):
    subset = df[df["species"] == species]
    ax.scatter(
        subset["petal_length"],
        subset["petal_width"],
        label=species.capitalize(),
        color=color,
        s=80,          # Marker size
        alpha=0.75,
        edgecolors="black",
        linewidths=0.5
    )

ax.set_xlabel("Petal Length (cm)", fontsize=13)
ax.set_ylabel("Petal Width (cm)", fontsize=13)
ax.set_title("Petal Length vs Petal Width by Species", fontsize=15, fontweight="bold")
ax.legend(title="Species", fontsize=11)
ax.grid(linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("iris_scatter_plot.png", dpi=150, bbox_inches="tight")
plt.show()
print("✔ Scatter plot saved as 'iris_scatter_plot.png'")

# ============================================================
# Step 6: Plot 4 — Pairplot (Bonus: all features compared)
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: CREATING PAIRPLOT (BONUS)")
print("=" * 60)

pair_fig = sns.pairplot(
    df,
    hue="species",           # Color by species
    palette={"setosa": "#3498db", "versicolor": "#2ecc71", "virginica": "#e74c3c"},
    diag_kind="kde",         # KDE on diagonal
    plot_kws={"alpha": 0.65, "s": 45, "edgecolor": "white"}
)
pair_fig.fig.suptitle("Pairplot — All Iris Features", y=1.02, fontsize=14, fontweight="bold")

pair_fig.savefig("iris_pairplot.png", dpi=150, bbox_inches="tight")
plt.show()
print("✔ Pairplot saved as 'iris_pairplot.png'")

# ============================================================
# Step 7: Feature Comparison Summary
# ============================================================

print("\n" + "=" * 60)
print("STEP 6: FEATURE COMPARISON BY SPECIES")
print("=" * 60)

comparison = df.groupby("species")[features].mean().round(2)
print(comparison)

print("\n" + "=" * 60)
print("VISUALIZATION COMPLETE — 4 Plots Generated")
print("=" * 60)
