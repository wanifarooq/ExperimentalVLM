import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.figsize": (12, 6),
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

files = {
    "worker0": "summary_worker0.txt",
    "worker1": "summary_worker1.txt",
    "worker2": "summary_worker2.txt",
    "worker3": "summary_worker3.txt",
}

def parse_file(path):
    rows = []
    with open(path, 'r') as f:
        for ln in f:
            parts = ln.split()
            if len(parts) >= 3 and parts[0] in [
                "Translation", "Pad/Crop", "Scale", "Scale+Pad", "TextOverlay", "Rotation", "Any"
            ]:
                rows.append([parts[0], float(parts[1]), float(parts[2])])
    return pd.DataFrame(rows, columns=["Type", "AVg", "Ve"])

# Combine summaries
dfs = []
for w, p in files.items():
    df = parse_file(p)
    df["Worker"] = w
    dfs.append(df)

df_all = pd.concat(dfs)
df_mean = df_all.groupby("Type")[["AVg", "Ve"]].mean().reset_index()

# Sort (optional)
order = ["Translation", "Pad/Crop", "Scale", "Scale+Pad", "Rotation", "TextOverlay", "Any"]
df_mean["Type"] = pd.Categorical(df_mean["Type"], categories=order, ordered=True)
df_mean = df_mean.sort_values("Type")

# Color palettes
cmap_avg = plt.cm.Blues(np.linspace(0.4, 0.9, len(df_mean)))
cmap_ve  = plt.cm.Reds(np.linspace(0.4, 0.9, len(df_mean)))

# -----------------------------
# PLOT 1: Mean AVg (bar + labels)
# -----------------------------
plt.figure(figsize=(12, 6))
bars = plt.bar(df_mean["Type"], df_mean["AVg"], color=cmap_avg, edgecolor="black")

# Add value labels
for b in bars:
    h = b.get_height()
    plt.text(b.get_x() + b.get_width()/2, h + 0.002, f"{h:.3f}",
             ha="center", va="bottom", fontsize=10)

plt.title("Mean AVg Across 4 Workers (Prediction Flip Rate)")
plt.ylabel("AVg (Fraction of Perturbations Changing Prediction)")
plt.xlabel("Perturbation Type")
plt.xticks(rotation=30)
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

# -----------------------------
# PLOT 2: Mean Ve (bar + labels)
# -----------------------------
plt.figure(figsize=(12, 6))
bars = plt.bar(df_mean["Type"], df_mean["Ve"], color=cmap_ve, edgecolor="black")

# Add value labels
for b in bars:
    h = b.get_height()
    plt.text(b.get_x() + b.get_width()/2, h + 0.003, f"{h:.3f}",
             ha="center", va="bottom", fontsize=10)

plt.title("Mean Ve Across 4 Workers (Images Affected At Least Once)")
plt.ylabel("Ve (Fraction of Images Impacted)")
plt.xlabel("Perturbation Type")
plt.xticks(rotation=30)
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

# --------------------------------------------------------
# PLOT 3: Side-by-side AVg vs Ve (comparative bar chart)
# --------------------------------------------------------
x = np.arange(len(df_mean))
w = 0.35

plt.figure(figsize=(14, 6))
plt.bar(x - w/2, df_mean["AVg"], width=w, color=cmap_avg, edgecolor="black", label="AVg")
plt.bar(x + w/2, df_mean["Ve"],  width=w, color=cmap_ve,  edgecolor="black", label="Ve")

plt.xticks(x, df_mean["Type"], rotation=30)
plt.ylabel("Metric Value")
plt.title("Comparison of AVg vs Ve Across Perturbation Types")
plt.grid(axis="y", linestyle="--", alpha=0.35)
plt.legend()
plt.tight_layout()
plt.show()
