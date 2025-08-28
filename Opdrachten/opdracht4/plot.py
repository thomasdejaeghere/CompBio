import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Style ---
sns.set_style("whitegrid")
palette = sns.color_palette("deep", 3)  # donkere kleuren voor tijd

# --- Load CSV ---
df = pd.read_csv("results1.csv")

# Extract data_size and amount
df["data_size"] = df["File"].str.extract(r"data_size(\d+)_amount\d+\.py").astype(int)
df["amount"] = df["File"].str.extract(r"amount(\d+)\.py").astype(int)

# Map version numbers to descriptive names
version_map = {
    1: "Versie 1: Init",
    2: "Versie 2: Gebruik van Dict.get() verwijderd",
    3: "Versie 3: Precompute log_prob & trans_mask",
}
df["VersionName"] = df["Version"].map(version_map)

# Sort
df = df.sort_values(["data_size", "amount", "VersionName"])

# --- Prepare plotting positions ---
group_labels = sorted(df[["data_size", "amount"]].drop_duplicates().values.tolist())
n_groups = len(group_labels)
bar_width = 0.2
offsets = np.array([-bar_width, 0, bar_width])  # 3 bars per group
x = np.arange(n_groups)
x_labels = [f"{ds}-{amt}" for ds, amt in group_labels]

# --- Plot Uitvoeringstijd (alle versies) ---
plt.figure(figsize=(12, 6))
for i, version in enumerate(version_map.values()):
    subset = df[df["VersionName"] == version]
    plt.bar(
        x + offsets[i],
        subset["Time (s)"].to_numpy(),
        bar_width,
        label=version,
        color=palette[i],
    )

plt.xticks(x, x_labels, rotation=45)
plt.ylabel("Uitvoeringstijd (s)")
plt.xlabel("Data grootte - Aantal")
plt.title("Gemiddelde uitvoeringstijd per data grootte en aantal")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend(title="Versie")
plt.tight_layout()
plt.show()

# --- Plot Geheugengebruik (alleen versie 1) ---
plt.figure(figsize=(12, 6))
subset_mem = df[df["VersionName"] == "Versie 1: Init"]
plt.bar(
    x,
    subset_mem["Memory (MB)"].to_numpy(),
    bar_width,
    label="Versie 1: Init",
    color="#1b4f72",
)

plt.xticks(x, x_labels, rotation=45)
plt.ylabel("Geheugengebruik (MB)")
plt.xlabel("Data grootte - Aantal")
plt.title("Gemiddeld geheugengebruik per data grootte en aantal (alleen versie 1)")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

# --- Speedup plot voor data_size=400, amount=100 ---
subset_400_100 = df[(df["data_size"] == 400) & (df["amount"] == 100)].sort_values(
    "Version"
)
subset_400_100 = subset_400_100[subset_400_100["Version"].isin([1, 2, 3])]
subset_400_100["VersionName"] = pd.Categorical(
    subset_400_100["VersionName"],
    categories=[
        "Versie 1: Init",
        "Versie 2: Gebruik van Dict.get() verwijderd",
        "Versie 3: Precompute log_prob & trans_mask",
    ],
    ordered=True,
)
subset_400_100 = subset_400_100.sort_values("VersionName")
speedup = subset_400_100["Time (s)"].iloc[0] / subset_400_100["Time (s)"]

plt.figure(figsize=(6, 5))
plt.plot(subset_400_100["Version"], speedup, "o--", color="black")
plt.ylabel("Versnellingsfactor t.o.v. versie 1")
plt.xlabel("Versie")
plt.title("Uitvoeringstijd versnelling voor data_size=400, amount=100")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.xticks([1, 2, 3], ["1", "2", "3"])
plt.tight_layout()
plt.show()
