from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(".")
PROCESSED = ROOT / "results" / "processed"
TABLES = PROCESSED / "tables"
OUT = ROOT / "results" / "figures" / "thesis_plots"
OUT.mkdir(parents=True, exist_ok=True)

def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

def save_plot(filename):
    path = OUT / filename
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print("Saved:", path)

# ------------------------------------------------------------
# 1. Random graphs: p+ vs ILP cost reduction
# ------------------------------------------------------------

df_p = read_csv(TABLES / "complete_vs_new" / "complete_vs_new_random_by_p.csv")
df_p = df_p.sort_values("p_positive")

plt.figure(figsize=(7, 4.5))
plt.plot(df_p["p_positive"], df_p["ilp_cost_reduction_fraction"] * 100, marker="o")
plt.xlabel("Positive-edge probability p+")
plt.ylabel("ILP cost reduction after edge deletion (%)")
plt.title("Random graphs: edge deletion effect by p+")
plt.grid(True, alpha=0.3)
save_plot("01_random_p_vs_cost_reduction.png")

# ------------------------------------------------------------
# 2. Random graphs: n vs complete/new ILP cost
# ------------------------------------------------------------

df_n = read_csv(TABLES / "complete_vs_new" / "complete_vs_new_random_by_n.csv")
df_n = df_n.sort_values("n")

plt.figure(figsize=(7, 4.5))
plt.plot(df_n["n"], df_n["complete_ilp_cost"], marker="o", label="Complete graph")
plt.plot(df_n["n"], df_n["new_ilp_cost"], marker="o", label="New graph")
plt.xlabel("Number of vertices n")
plt.ylabel("Average ILP cost")
plt.title("Random graphs: complete vs new ILP cost")
plt.legend()
plt.grid(True, alpha=0.3)
save_plot("02_random_n_vs_ilp_cost.png")

# ------------------------------------------------------------
# 3. Random graphs: n vs LP/ILP ratio
# ------------------------------------------------------------

plt.figure(figsize=(7, 4.5))
plt.plot(df_n["n"], df_n["complete_lp_ilp_ratio"], marker="o", label="Complete graph")
plt.plot(df_n["n"], df_n["new_lp_ilp_ratio"], marker="o", label="New graph")
plt.xlabel("Number of vertices n")
plt.ylabel("LP/ILP ratio")
plt.title("Random graphs: LP tightness by graph size")
plt.legend()
plt.grid(True, alpha=0.3)
save_plot("03_random_n_vs_lp_ilp_ratio.png")

# ------------------------------------------------------------
# 4. Random graphs: n vs Pivot/ILP ratio
# ------------------------------------------------------------

plt.figure(figsize=(7, 4.5))
plt.plot(df_n["n"], df_n["complete_pivot_ilp_ratio"], marker="o", label="Complete graph")
plt.plot(df_n["n"], df_n["new_pivot_ilp_ratio"], marker="o", label="New graph")
plt.xlabel("Number of vertices n")
plt.ylabel("Pivot/ILP ratio")
plt.title("Random graphs: Pivot quality by graph size")
plt.legend()
plt.grid(True, alpha=0.3)
save_plot("04_random_n_vs_pivot_ilp_ratio.png")

# ------------------------------------------------------------
# 5. Bad triangles vs ILP cost
# ------------------------------------------------------------

flat = read_csv(PROCESSED / "all_runs_flat.csv")

def family_from_file(name):
    name = str(name).lower()
    if "random" in name:
        return "random"
    if "clq" in name:
        return "clique"
    if "fb_" in name or "facebook" in name:
        return "facebook"
    return "other"

flat["family"] = flat["file_name"].apply(family_from_file)

plt.figure(figsize=(7, 4.5))
for fam, group in flat.groupby("family"):
    if fam == "other":
        continue
    plt.scatter(
        group["complete_bad_triangles_total"],
        group["complete_ilp_cost"],
        alpha=0.45,
        label=fam,
        s=18,
    )

plt.xlabel("Number of bad triangles")
plt.ylabel("Complete graph ILP cost")
plt.title("Bad triangles vs ILP cost")
plt.legend()
plt.grid(True, alpha=0.3)
save_plot("05_bad_triangles_vs_ilp_cost.png")

# ------------------------------------------------------------
# 6. Clique graphs: largest clique size vs bad 4-cycles
# ------------------------------------------------------------

clique_path = TABLES / "complete_vs_new" / "complete_vs_new_clique_by_largest_size.csv"

if clique_path.exists():
    df_c = read_csv(clique_path)
    x_col = "largest_clique_size"
    y_col = "new_bad_4_cycles"
else:
    df_c = read_csv(TABLES / "clique" / "clique_by_largest_clique_size.csv")
    x_col = "largest_clique_size"
    y_col = "avg_bad4_cycles"

df_c = df_c.sort_values(x_col)

plt.figure(figsize=(7, 4.5))
plt.scatter(df_c[x_col], df_c[y_col], s=40)
plt.plot(df_c[x_col], df_c[y_col], alpha=0.7)
plt.xlabel("Largest clique size")
plt.ylabel("Average number of bad 4-cycles")
plt.title("Clique graphs: clique size vs bad 4-cycles")
plt.grid(True, alpha=0.3)
save_plot("06_clique_size_vs_bad_4_cycles.png")

print()
print("Done. Open plots with:")
print(f"open {OUT}")
