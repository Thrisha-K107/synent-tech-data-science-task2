# synent-tech-data-science-task2
My first Internship with Task 2
# 📊 Task 2: Data Visualization
### Synent Technologies — Data Science Internship

---

## 📌 Problem Statement

Raw numbers in a table tell us little. This project transforms the Iris dataset into meaningful visual stories — using bar charts, histograms, scatter plots, and pairplots — to reveal how the three species of iris flowers differ across four physical measurements.

---

## 📂 Dataset Details

| Property | Details |
|---|---|
| **Name** | Iris Dataset |
| **Source** | `sklearn.datasets.load_iris()` |
| **Rows** | 150 (50 per species) |
| **Columns** | 4 features + 1 target |
| **Species** | Setosa, Versicolor, Virginica |

---

## 🛠️ Tools & Libraries

- **Python 3.x**
- `pandas` — data handling
- `numpy` — numerical computations
- `matplotlib` — core plotting engine
- `seaborn` — statistical plots (pairplot)
- `sklearn` — dataset loading

---

## 🔍 Step-by-Step Code Explanation

### Step 1 — Load and Prepare Dataset
```python
iris_raw = load_iris(as_frame=True)
df = iris_raw.frame
df["species"] = df["target"].map({0: "setosa", 1: "versicolor", 2: "virginica"})
```
> `load_iris(as_frame=True)` returns the dataset as a pandas DataFrame. The `target` column contains numbers (0, 1, 2), which we map to human-readable species names using `.map()`.

---

### Step 2 — Bar Chart (Mean Feature Values by Species)
```python
x = np.arange(len(features))
for i, (species, color) in enumerate(zip(species_list, colors)):
    means = df[df["species"] == species][features].mean().values
    ax.bar(x + i * bar_width, means, ...)
```
> We use **grouped bar charts** — one group per feature, one bar per species. `x + i * bar_width` shifts each species' bar slightly right so they don't overlap. `enumerate()` gives us both the index and value in the loop.

---

### Step 3 — Histograms (Feature Distributions)
```python
for ax, feature in zip(axes.flatten(), features):
    for species, color in zip(species_list, colors):
        subset = df[df["species"] == species][feature]
        ax.hist(subset, bins=15, alpha=0.65, ...)
```
> `axes.flatten()` converts the 2×2 grid of axes into a flat list, so we can loop over all 4 subplots easily. Each species is drawn with `alpha=0.65` (semi-transparent) so overlapping bars remain visible.

---

### Step 4 — Scatter Plot (Petal Length vs Width)
```python
ax.scatter(
    subset["petal_length"],
    subset["petal_width"],
    s=80, alpha=0.75, ...
)
```
> A scatter plot places each flower as a dot. `petal_length` on X-axis and `petal_width` on Y-axis. We can visually see how Setosa forms a separate cluster while Versicolor and Virginica overlap slightly.

---

### Step 5 — Pairplot (Bonus)
```python
sns.pairplot(df, hue="species", diag_kind="kde", ...)
```
> Seaborn's `pairplot` automatically creates scatter plots for every pair of features, and KDE (Kernel Density Estimate) plots on the diagonal. `hue="species"` colors each species differently.

---

### Step 6 — Feature Comparison
```python
comparison = df.groupby("species")[features].mean().round(2)
```
> Groups the dataframe by species and computes the mean of each feature. `.round(2)` keeps 2 decimal places for clean output.

---

## 📊 Key Insights

| Insight | Finding |
|---|---|
| **Setosa** | Smallest petals — clearly separable from others |
| **Virginica** | Largest overall measurements |
| **Petal features** | Best discriminators between species |
| **Sepal width** | Setosa has widest sepals |

---

## 📁 Output Files

| File | Description |
|---|---|
| `iris_bar_chart.png` | Mean feature comparison grouped by species |
| `iris_histograms.png` | Distribution of all 4 features per species |
| `iris_scatter_plot.png` | Petal length vs petal width scatter |
| `iris_pairplot.png` | Full feature pairwise comparison |

---

## ▶️ How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn

python task2_visualization.py
```

---

## 🔗 Repository

`synent-task2-datavisualization-<yourname>`

---

*Synent Technologies Data Science Internship — Task 2*
