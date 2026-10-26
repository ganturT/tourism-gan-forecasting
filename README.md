# Tourism Demand Forecasting Using a Generative Adversarial Network

This repository contains the code and data for the study: "Tourism Demand Forecasting Using a Generative Adversarial Network"

The study forecasts monthly inbound tourist arrivals to Mongolia (January 2017 – October 2025) with a GAN that uses a GRU-based generator and a Conv1D-based discriminator. The GAN is compared with a seasonal naive forecast and with ARIMAX, SARIMAX, SVR, ANN, LSTM and GRU benchmarks.

## Repository structure

```
tourism-gan-forecasting/
├── Code/                       Python scripts (see the table below)
├── Data/
│   ├── main_dataset.csv                Monthly dataset used in the study
├── Results/                    Test predictions, metrics and selected hyperparameters
├── requirements.txt
├── LICENSE
└── README.md
```

## Data

`Data/main_dataset.csv` contains 106 monthly observations (January 2017 – October 2025) with the following columns:

| Column   | Description |
|----------|-------------|
| `date`   | First day of the month (YYYY-MM-DD) |
| `volume` | Number of inbound tourist arrivals ([National Statistics Office of Mongolia](https://www.1212.mn)) |
| `search` | log(1 + mean of five Google Trends search indices) |
| `sin12`, `cos12` | Cyclical month terms, sin(2πm/12) and cos(2πm/12) |
| `covid`  | COVID-19 dummy (1 from March 2020 to March 2022, 0 otherwise) |

The five Google Trends search terms are "Mongolia tours", "flight to Ulaanbaatar", "Mongolia visa", "visit Mongolia" and "best time to visit Mongolia" (worldwide, all categories).  Google Trends data are sampled, so indices downloaded on a different day may differ slightly from those used in the paper.

The data are split chronologically into training (63 samples, Jan 2018 – Mar 2023), validation (12 samples, Apr 2023 – Mar 2024) and test (19 samples, Apr 2024 – Oct 2025) sets, using a 12-month input window.

## Installation

Python 3.12 was used. Install the required packages with:

```bash
pip install -r requirements.txt
```

No GPU is required; all models were trained on a CPU.

## How to run

Run all scripts from the **root folder** of the repository (the scripts use the relative paths `Data/` and `Results/`):

```bash
# 1. Prepare the windows, the chronological split and the min-max scalers
python Code/Data_preprocessing.py

# 2. Train and evaluate the proposed GAN and the benchmarks
python Code/GAN_model.py
python Code/BM_ANN.py
python Code/BM_LSTM.py
python Code/BM_GRU.py
python Code/BM_SVR.py
python Code/BM_ARIMAX.py
python Code/BM_SARIMAX.py
python Code/BM_SNAIVE.py

# 3. Feature ablation study (uses the selected GAN hyperparameters)
python Code/GAN_feature_ablation.py

# 4. Build Table 2 and the prediction figures
python Code/PR_testset_table.py
python Code/PR_all_models_plots.py
```

| Script | Output | Paper |
|--------|--------|-------|
| `Data_preprocessing.py` | Training/validation/test arrays, scalers | Section 5, Fig. 5 |
| `GAN_model.py` | Proposed GAN: grid search, 4-seed training, test predictions, D/G loss plot | Tables 1–2, Fig. 9 |
| `BM_ANN.py`, `BM_LSTM.py`, `BM_GRU.py` | Neural benchmarks: grid search, 4-seed training, loss curves | Tables 1–2, Fig. 8 |
| `BM_SVR.py`, `BM_ARIMAX.py`, `BM_SARIMAX.py` | Machine learning and statistical benchmarks | Tables 1–2 |
| `BM_SNAIVE.py` | Seasonal naive forecast | Table 2 |
| `GAN_feature_ablation.py` | GAN retrained without the search index or the COVID-19 dummy | Table 3 |
| `PR_testset_table.py` | Performance comparison table (CSV and LaTeX) | Table 2 |
| `PR_all_models_plots.py` | Test-set prediction plots | Figs. 6–7 |

**Hyperparameter search.** In the neural scripts, the full search grids of Table 1 are listed first and are then overridden by the selected values (e.g. `LR_GRID = [0.001]`), so that the final results can be reproduced quickly. To repeat the full grid search, delete the lines below `# Selected values (Table 1)`. The full GAN search is slow on a CPU (6 × 4 × 4 configurations × 3 seeds).

The neural models are tuned with three seeds (10, 11, 12) and the final models are trained with four seeds (0, 1, 2, 3). Results are reported as mean ± standard deviation over the four final seeds. Because of differences in hardware and library versions, re-running the neural models may give slightly different numbers from those in the paper; the predictions and metrics used in the paper are stored in `Results/`.

## Main results (test set)

| Model | MAE | RMSE | MAPE (%) |
|-------|-----|------|----------|
| Seasonal naive | 13142.84 | 17368.62 | 17.81 |
| LSTM | 14950.56 ± 2852.78 | 17201.96 ± 2871.63 | 22.36 ± 3.98 |
| GRU | 14483.05 ± 1443.23 | 18209.92 ± 3112.23 | 21.10 ± 0.91 |
| SVR | 19436.90 | 22340.18 | 28.07 |
| **GAN** | **11104.60 ± 1837.75** | 14631.07 ± 2805.88 | **14.44 ± 1.71** |
| ARIMAX | 15831.37 | 20694.79 | 22.36 |
| SARIMAX | 12437.86 | 17186.46 | 15.67 |
| ANN | 12255.42 ± 1120.98 | **14237.81 ± 858.59** | 19.99 ± 1.06 |

## GAN Model design

The GRU generator and Conv1D discriminator architecture of the GAN model is adapted from Lin et al. (2021):

> H. Lin, C. Chen, G. Huang and A. Jafari, "Stock price prediction using generative adversarial networks," *Journal of Computer Science*, vol. 17, no. 3, pp. 188–196, 2021, doi: [10.3844/jcssp.2021.188.196](https://doi.org/10.3844/jcssp.2021.188.196).


## License

This project is released under the [MIT License](LICENSE).
