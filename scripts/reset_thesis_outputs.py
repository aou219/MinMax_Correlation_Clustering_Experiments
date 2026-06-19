from pathlib import Path
import shutil

ROOT = Path(".")

# Alleen gegenereerde output verwijderen. Raw data, JSON results en all_runs_flat.csv blijven staan.
TARGETS = [
    ROOT / "results" / "processed" / "reports",
    ROOT / "results" / "processed" / "plots",
    ROOT / "figures" / "thesis_plots",
]

for path in TARGETS:
    if path.exists():
        shutil.rmtree(path)
        print("Deleted:", path)
    else:
        print("Already clean:", path)

# Nieuwe lege mappen maken
(ROOT / "results" / "processed" / "reports").mkdir(parents=True, exist_ok=True)
(ROOT / "results" / "processed" / "plots").mkdir(parents=True, exist_ok=True)
(ROOT / "figures" / "thesis_plots").mkdir(parents=True, exist_ok=True)

print("Done. Reports and plots are reset.")
