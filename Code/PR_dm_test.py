import numpy as np
import pandas as pd
from pickle import load
from scipy import stats


H = 1                       # forecast horizon (one-step-ahead forecasts)
REFERENCE = "GAN"           # model compared against every other model


FILES = {
    "GAN": ("Results/test_predicted_meanseed_gan.csv", "seeds_mean_prediction"),
    "ANN": ("Results/test_predicted_meanseed_ann.csv", "seeds_mean_prediction"),
    "LSTM": ("Results/test_predicted_meanseed_lstm.csv", "seeds_mean_prediction"),
    "GRU": ("Results/test_predicted_meanseed_gru.csv", "seeds_mean_prediction"),
    "SVR": ("Results/test_predicted_svr.csv", "predicted_arrivals"),
    "ARIMAX": ("Results/test_predicted_arimax.csv", "predicted_arrivals"),
    "SARIMAX": ("Results/test_predicted_sarimax.csv", "predicted_arrivals"),
    "Seasonal naive": ("Results/test_predicted_snaive.csv", "predicted_arrivals"),
}

# Actual test values (same as table2_results.py)
y_test = np.load("Data/y_test.npy", allow_pickle=True).astype("float32")
y_scaler = load(open("Data/y_scaler.pkl", "rb"))
actual = y_scaler.inverse_transform(y_test).ravel().astype(float)


def dm_test(actual, pred1, pred2, h=1, loss="squared"):
    """Returns (DM statistic, two-sided p, one-sided p for 'model 1 is more accurate').
    A negative statistic means model 1 has the smaller loss."""
    e1, e2 = actual - pred1, actual - pred2
    if loss == "squared":
        d = e1 ** 2 - e2 ** 2
    elif loss == "absolute":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("loss must be 'squared' or 'absolute'")

    T = len(d)
    d_bar = d.mean()
    # long-run variance of d: autocovariances up to lag h-1
    gamma = [np.sum((d[k:] - d_bar) * (d[:T - k] - d_bar)) / T for k in range(h)]
    var_d = (gamma[0] + 2 * sum(gamma[1:])) / T
    dm = d_bar / np.sqrt(var_d)

    # Harvey-Leybourne-Newbold small-sample correction, compared with t(T-1)
    hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_star = dm * hln
    p_two = 2 * stats.t.sf(abs(dm_star), df=T - 1)
    p_one = stats.t.cdf(dm_star, df=T - 1)          # H1: model 1 more accurate
    return dm_star, p_two, p_one


# ---------------------------------------------------------------------
# Load forecasts
# ---------------------------------------------------------------------
preds = {}
for name, (path, column) in FILES.items():
    df = pd.read_csv(path)
    assert column in df.columns, f"{name}: column '{column}' not found, available: {list(df.columns)}"
    p = df[column].to_numpy(dtype=float)
    assert len(p) == len(actual), f"{name}: {len(p)} forecasts vs {len(actual)} actual values"
    preds[name] = p

# Sanity check: accuracy of the forecasts used in the test
# (seasonal naive, SVR, ARIMAX and SARIMAX should match Table 2;
#  neural models use the seed-averaged forecast, so they can differ slightly)
print("Accuracy of the forecasts used in the DM test:")
for name, p in preds.items():
    err = actual - p
    print(f"  {name:<15} MAE {np.mean(np.abs(err)):>10,.2f}  "
          f"RMSE {np.sqrt(np.mean(err ** 2)):>10,.2f}  "
          f"MAPE {np.mean(np.abs(err / actual)) * 100:6.2f}%")
print()

# ---------------------------------------------------------------------
# DM tests: GAN vs every other model
# ---------------------------------------------------------------------
rows = []
for other in FILES:
    if other == REFERENCE:
        continue
    for loss in ["squared", "absolute"]:
        dm, p_two, p_one = dm_test(actual, preds[REFERENCE], preds[other], h=H, loss=loss)
        rows.append({"comparison": f"{REFERENCE} vs {other}", "loss": loss,
                     "DM": round(dm, 3), "p_two_sided": round(p_two, 4),
                     "p_one_sided": round(p_one, 4)})

res = pd.DataFrame(rows)
print(f"Test period: {len(actual)} months | horizon h = {H}")
print("DM < 0: the GAN has the smaller loss. p_one_sided tests H1: GAN is more accurate.\n")
print(res.to_string(index=False))
res.to_csv("Results/table4_dm_test.csv", index=False)
print("\nSaved: Results/table4_dm_test.csv")
# ---------------------------------------------------------------------
# Table 4: LaTeX table of the DM test results
# ---------------------------------------------------------------------
TABLE_ORDER = ["Seasonal naive", "LSTM", "GRU", "SVR", "ARIMAX", "SARIMAX", "ANN"]  # as in Table 2


def fmt_dm(x):
    return f"${x:.3f}$"          # in math mode, so the minus sign is typeset correctly


def fmt_p(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


# unrounded results for the table (res above is rounded for printing)
raw = {(other, loss): dm_test(actual, preds[REFERENCE], preds[other], h=H, loss=loss)
       for other in TABLE_ORDER for loss in ["squared", "absolute"]}

lines = [
    r"\begin{table}[H]",
    r"\caption{Diebold--Mariano test of the GAN against each benchmark on the test set ($h = " + str(H) +
    r"$, Harvey--Leybourne--Newbold correction). A negative statistic indicates that the GAN has the "
    r"smaller loss; $p$-values are two-sided. For the neural models, the forecasts averaged over four "
    r"seeds are used.\label{tab:dm}}",
    r"\begin{adjustwidth}{-\extralength}{0cm}",
    r"\newcolumntype{C}{>{\centering\arraybackslash}X}",
    r"\begin{tabularx}{\fulllength}{lCCCC}",
    r"\toprule",
    r" & \multicolumn{2}{c}{\textbf{Squared Loss}} & \multicolumn{2}{c}{\textbf{Absolute Loss}} \\",
    r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
    r"\textbf{Comparison} & \textbf{DM} & \textbf{\boldmath$p$} & \textbf{DM} & \textbf{\boldmath$p$} \\",
    r"\midrule",
]
for other in TABLE_ORDER:
    dm_sq, p_sq, _ = raw[(other, "squared")]
    dm_ab, p_ab, _ = raw[(other, "absolute")]
    label = other[0].lower() + other[1:] if other == "Seasonal naive" else other
    lines.append(f"{REFERENCE} vs. {label} & {fmt_dm(dm_sq)} & {fmt_p(p_sq)} & "
                 f"{fmt_dm(dm_ab)} & {fmt_p(p_ab)} \\\\")
lines += [r"\bottomrule", r"\end{tabularx}", r"\end{adjustwidth}", r"\end{table}"]

with open("Results/table4_dm_test.tex", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("Saved: Results/table4_dm_test.tex")