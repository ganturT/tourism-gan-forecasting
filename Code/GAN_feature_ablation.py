import os
import json
import importlib
import numpy as np
import pandas as pd

# ===================== SETTINGS (fill these in) =====================
GAN_MODULE = "GAN_model"              # file name of your GAN script without ".py"
HPARAMS_FILE = "Results/best_hparams_gan.json"   # written by the GAN script after the grid search
FINAL_SEEDS = [0, 1, 2, 3]      # same seeds as the full model

# Names of the feature columns in X_*.npy, in EXACTLY the order used when the arrays were created.
FEATURE_NAMES = ["volume", "search", "sin12", "cos12", "covid"]

# Variants: variant name -> features to remove
VARIANTS = {
    "All features": [],
    "Without search intensity": ["search"],
    "Without COVID dummy": ["covid"],
}
OUT_DIR = "Results/"
METRICS = ["MAE", "RMSE", "MAPE"]
# ====================================================================

os.makedirs(OUT_DIR, exist_ok=True)

g = importlib.import_module(GAN_MODULE)

X_train_full, X_val_full, X_test_full = g.X_train.copy(), g.X_val.copy(), g.X_test.copy()
assert X_train_full.shape[2] == len(FEATURE_NAMES), (
    f"X has {X_train_full.shape[2]} features but FEATURE_NAMES has {len(FEATURE_NAMES)}. "
    "Fix FEATURE_NAMES so that it matches the column order of X.")

with open(HPARAMS_FILE) as f:
    hp = json.load(f)
LR, UNITS, D_UNITS = float(hp["lr"]), int(hp["units"]), int(hp["d_units"])
print(f"Fixed hyperparameters from {HPARAMS_FILE}: lr={LR}, units={UNITS}, d_units={D_UNITS}")

summary_rows = []   # formatted strings (for CSV)
numeric_rows = []   # raw numbers (for LaTeX + bolding)
for variant, drop in VARIANTS.items():
    unknown = set(drop) - set(FEATURE_NAMES)
    assert not unknown, f"Unknown feature(s) {unknown} in variant '{variant}'"
    keep = [i for i, name in enumerate(FEATURE_NAMES) if name not in drop]

    g.X_train = X_train_full[:, :, keep]
    g.X_val = X_val_full[:, :, keep]
    g.X_test = X_test_full[:, :, keep]

    input_dim = g.X_train.shape[1]
    feature_size = g.X_train.shape[2]
    output_dim = g.y_train.shape[1]
    print(f"\n=== {variant} | features: {[FEATURE_NAMES[i] for i in keep]} ===")

    records = []
    for seed in FINAL_SEEDS:
        gan, hist, best_epoch = g.run_gan(UNITS, LR, seed, input_dim, output_dim, feature_size,
                                          d_units=D_UNITS)
        va = g.compute_metrics(g.y_val, gan.predict(g.X_val))
        te = g.compute_metrics(g.y_test, gan.predict(g.X_test))
        records.append({"variant": variant, "seed": seed, "best_epoch": best_epoch,
                        "val_RMSE": va["RMSE"],
                        "test_MAE": te["MAE"], "test_RMSE": te["RMSE"], "test_MAPE": te["MAPE"]})
        print(f"seed={seed} best_epoch={best_epoch:<4} TEST MAE={te['MAE']:.0f} "
              f"RMSE={te['RMSE']:.0f} MAPE={te['MAPE']:.2f}%")

    per_seed = pd.DataFrame(records)
    tag = variant.lower().replace(" ", "_")
    #per_seed.to_csv(f"{OUT_DIR}/per_seed_{tag}.csv", index=False)

    row = {"Variant": variant}
    num = {"Variant": variant}
    for m in METRICS:
        mean, std = per_seed[f"test_{m}"].mean(), per_seed[f"test_{m}"].std(ddof=1)
        row[m] = f"{mean:.2f} ± {std:.2f}"
        num[m] = (mean, std)
    row["val_RMSE"] = f"{per_seed['val_RMSE'].mean():.2f}"
    summary_rows.append(row)
    numeric_rows.append(num)

# ---------------- CSV ----------------
summary = pd.DataFrame(summary_rows)
print("\n=== Ablation results (test set, mean ± std over seeds) ===")
print(summary.to_string(index=False))
#summary.to_csv(f"{OUT_DIR}/ablation_summary.csv", index=False)
print(f"\nSaved: {OUT_DIR}/ablation_summary.csv")

# ---------------- LaTeX ----------------
def latex_escape(s):
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("_", r"\_"), ("#", r"\#"), ("$", r"\$")]:
        s = s.replace(a, b)
    return s

# lowest mean per metric is printed in bold
best = {m: min(r[m][0] for r in numeric_rows) for m in METRICS}

lines = [
    r"\begin{tabular}{lccc}",
    r"\hline",
    r"Input features  & MAE & RMSE & MAPE (\%) \\",
    r"\hline",
]
for r in numeric_rows:
    cells = []
    for m in METRICS:
        mean, std = r[m]
        cell = f"{mean:.2f} $\\pm$ {std:.2f}"
        if np.isclose(mean, best[m]):
            cell = f"\\textbf{{{cell}}}"
        cells.append(cell)
    lines.append(f"{latex_escape(r['Variant'])} & " + " & ".join(cells) + r" \\")
lines += [
    r"\hline",
    r"\end{tabular}",
    f"% GAN ablation: mean $\\pm$ std over {len(FINAL_SEEDS)} seeds, "
    f"fixed hyperparameters (lr={LR}, units={UNITS}, d_units={D_UNITS}).",
]

tex_path = f"{OUT_DIR}/ablation_summary.tex"
with open(tex_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"Saved: {tex_path}")
print("\n".join(lines))