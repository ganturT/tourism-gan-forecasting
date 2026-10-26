import numpy as np
import pandas as pd
from pickle import load
from sklearn.metrics import mean_absolute_error, mean_squared_error

METRICS = ["MAE", "RMSE", "MAPE"]

# Row order of Table 2: display name -> source
# Neural models: <prefix>_test_summary.csv (written by the multi-seed scripts)
NEURAL = {"LSTM": "lstm", "GRU": "gru", "GAN": "gan", "ANN": "ann"}
# Deterministic models: fitted once, metrics computed from the predicted csv
DETERMINISTIC = {"Seasonal naive": "Results/test_predicted_snaive.csv",
                 "SVR": "Results/test_predicted_svr.csv",
                 "ARIMAX": "Results/test_predicted_arimax.csv",
                 "SARIMAX": "Results/test_predicted_sarimax.csv"}
ROW_ORDER = ["Seasonal naive", "LSTM", "GRU", "SVR", "GAN", "ARIMAX", "SARIMAX", "ANN"]


y_test = np.load("Data/y_test.npy", allow_pickle=True).astype("float32")
y_scaler =load(open("Data/y_scaler.pkl", "rb"))   # fitted on the training period only


rescaled_real = y_scaler.inverse_transform(y_test)

test_dates = pd.date_range(start="2024-04-01", periods=len(rescaled_real), freq="MS")
rescaled_real_df = pd.DataFrame({
    "": test_dates.strftime("%Y-%m-%d"),
    "real_mean": rescaled_real.ravel(),
})

real = rescaled_real_df["real_mean"]
print(real)


def single_run_metrics(path):
    pred = pd.read_csv(path)["predicted_arrivals"]
    return {"MAE": mean_absolute_error(real, pred),
            "RMSE": float(np.sqrt(mean_squared_error(real, pred))),
            "MAPE": float(np.mean(np.abs((real - pred) / real)) * 100)}


means, stds = {}, {}

# --- neural models: mean ± std over seeds ---
for name, prefix in NEURAL.items():
    s = pd.read_csv(f"Results/test_summary_{prefix}.csv", index_col=0)   # rows: MAE/RMSE/MAPE, cols: mean/std
    means[name] = {m: s.loc[m, "mean"] for m in METRICS}
    stds[name] = {m: s.loc[m, "std"] for m in METRICS}

# --- deterministic models (incl. seasonal naive): single run, no std ---
for name, path in DETERMINISTIC.items():
    means[name] = single_run_metrics(path)
    stds[name] = {m: None for m in METRICS}

mean_df = pd.DataFrame(means).T.loc[ROW_ORDER, METRICS]
best = mean_df.idxmin()          # best (lowest mean) model for every metric


def fmt(name, m):
    mu, sd = mean_df.loc[name, m], stds[name][m]
    return f"{mu:.2f} ± {sd:.2f}" if sd is not None else f"{mu:.2f}"


# ---------- console / CSV ----------
table = pd.DataFrame({m: [fmt(n, m) for n in ROW_ORDER] for m in METRICS}, index=ROW_ORDER)
table.index.name = "Models"
table.columns = ["MAE", "RMSE", "MAPE (%)"]
print(table.to_string())
table.to_csv("Results/table2_mean_std.csv")

# ---------- LaTeX (best mean in bold) ----------
lines = [r"\begin{tabular}{lccc}", r"\hline",
         r"Models & MAE & RMSE & MAPE (\%) \\", r"\hline"]
for n in ROW_ORDER:
    cells = []
    for m in METRICS:
        c = fmt(n, m).replace("±", r"$\pm$")
        cells.append(rf"\textbf{{{c}}}" if best[m] == n else c)
    lines.append(f"{n} & " + " & ".join(cells) + r" \\")
lines += [r"\hline", r"\end{tabular}",
          r"% ANN, LSTM, GRU, GAN: mean $\pm$ std over 4 seeds. Seasonal naive, SVR, ARIMAX, SARIMAX: deterministic, single run."]
with open("Results/table2_mean_std.tex", "w") as f:
    f.write("\n".join(lines))
print("\nSaved: table2_mean_std.csv, table2_mean_std.tex")