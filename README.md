# Stock Market Forecasting — Prophet vs XGBoost

An end-to-end, reproducible Jupyter Notebook that compares Facebook Prophet and XGBoost for daily stock closing price forecasting (Tesla — TSLA). The notebook includes data loading, exploratory data analysis (EDA), time-series decomposition, feature engineering for XGBoost, model training, side-by-side evaluation, and interactive visualizations.

---

## 🎯 Project Objective

Compare two complementary forecasting approaches and identify which produces more accurate daily close price forecasts for TSLA:

- **Facebook Prophet** — time-series decomposition model that handles trend, seasonality and holidays automatically.
- **XGBoost (XGBRegressor)** — a feature-based gradient boosting regressor using engineered technical indicators and lag features.

Target: a reproducible 90-day backtest and clear evaluation using MAE, RMSE and MAPE.

---

## 📁 Repository Structure

```
📦 stock-price-prediction
│
├── Stock Market Forecast.ipynb   # Main end-to-end notebook (EDA → models → evaluation)
├── plots/                       # Saved images created by the notebook (plots/)
├── requirements.txt             # Python dependencies (created for this project)
└── README.md                    # This file
```

Notes:

- The primary artifact is `Stock Market Forecast.ipynb`. Open it in Jupyter or VS Code and run cells in order.
- Plots produced by the notebook may be saved into `plots/`.

---

## 📊 Dataset Overview

- Expected dataset: daily historical stock prices (Tesla - TSLA).
- Required columns (or equivalent names):
  - `Date` (datetime)
  - `Open`
  - `High`
  - `Low`
  - `Close` (target)
  - `Volume`

Default in-notebook path:

- Edit the `FILE_PATH` variable in `Stock Market Forecast.ipynb` (search for `FILE_PATH =`) and set it to your CSV location.

Behavior when dataset is missing:

- The current notebook expects a CSV at the provided `FILE_PATH`.

---

## 🧭 Data Preprocessing (what the notebook performs)

1. Read CSV with `parse_dates=['Date']` and set `Date` as index.
2. Inspect data shape, dtypes and missing values.
3. Basic cleaning / re-indexing: set datetime index and ensure monotonic dates.
4. For XGBoost: create time-based features (dayofweek, dayofmonth, month, year) and lagged price features (lag_1, lag_5, lag_20). Drop rows created by lag/rolling ops.
5. Train / test split: last 90 trading days used as the test set (split implemented in the notebook to match both Prophet and XGBoost evaluations).

---

## 🧠 Models Implemented

- Prophet

  - Requires a DataFrame containing `ds` (datetime) and `y` (target). The notebook maps the `Date` index to `ds` and the `Close` price to `y`.
  - The notebook fits `Prophet()` (defaults) and generates a `future` DataFrame via `make_future_dataframe`.
  - Note: On Windows, installing `prophet` may require extra steps (see Quick Start). The notebook uses the `prophet` PyPI package.

- XGBoost (XGBRegressor)
  - Uses engineered features (time features, lag features). The notebook trains `xgb.XGBRegressor(objective='reg:squarederror', ...)` on the training set and predicts on the test set.

Modeling highlights:

- The notebook aligns Prophet's daily predictions with the actual trading-day dates (filtering `prophet_forecast['ds']` to match the test index dates) to compute metrics fairly.
- XGBoost predictions are converted into a `pd.Series` with the test dates as index for direct comparison.

---

## 📈 Evaluation Metrics

Metrics calculated on the test set (last 90 trading days):

- MAE — Mean Absolute Error
- RMSE — Root Mean Squared Error
- MAPE — Mean Absolute Percentage Error

The notebook prints these metrics for each model and visualizes actual vs predicted values. The lower MAPE determines the winner in the notebook's comparison.

---

## 🧾 Visual Outputs

Key visualizations created by the notebook (and saved into `plots/` when cells save them):

- Close Price & Volume chart (interactive Plotly)
- Time Series Decomposition (Trend / Seasonal / Residual) using `statsmodels`
- Prophet forecast vs actual (with confidence intervals where available)
- XGBoost forecast vs actual
- Model comparison plots (Prophet vs XGBoost vs Actual)

---

## ⚙️ Quick Start (Windows, Git Bash / WSL - bash.exe)

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
# Activate in Git Bash / WSL (bash.exe)
source .venv/Scripts/activate
# (PowerShell) .\.venv\Scripts\Activate.ps1  OR  (CMD) .\.venv\Scripts\activate.bat
```

2. Install required packages (the project root contains `requirements.txt`):

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If `prophet` fails to install on Windows via pip, try conda:

```bash
conda install -c conda-forge prophet
```

3. Open the notebook and run all cells (or run step-by-step):

```bash
jupyter notebook "Stock Market Forecast.ipynb"
# or open the notebook in VS Code
```

4. Before running: update the `FILE_PATH` variable in the notebook to point to your CSV. If you want, I can add a synthetic fallback so the notebook runs even if the CSV is missing.

---

## ✅ Contract (small and precise)

- Inputs: CSV with daily `Date`, `Open`, `High`, `Low`, `Close`, `Volume`. Notebook parameter: `FILE_PATH`.
- Outputs: Forecast arrays/plots (Prophet and XGBoost), evaluation metrics (MAE/RMSE/MAPE), and optional saved images in `plots/`.
- Error modes: If CSV missing → the notebook currently errors when reading the CSV; recommended fix is to add a synthetic-data fallback or a guarded file-existence check. If Prophet installation fails → guidance to use conda.

---

## ⚠️ Edge Cases & Handling

- Missing CSV: the notebook will fail at the CSV read step; adding a fallback dataset (synthetic) will enable end-to-end runs without data.
- Non-trading days / weekends: Prophet produces daily outputs (including weekends). The notebook contains an alignment function that filters the Prophet forecast to match trading-day dates before metrics calculation.
- NaN rows created by lagging/rolling operations: these rows are dropped prior to training XGBoost to prevent leakage.
- Prophet backend differences: Prophet may use different backends (pystan / cmdstanpy); installation can vary by OS.

---

## 🧪 Quality Gates (smoke checklist)

- Build: N/A — notebook-based analysis.
- Lint / Typecheck: Not included (not essential for exploratory notebook).
- Tests: None included; adding unit tests for preprocessing is recommended.
- Smoke test: The notebook should run end-to-end when `FILE_PATH` points to a valid CSV with the required columns. A recommended quick check is to run only the data load + one model training cell first.

Requirements coverage: This README documents dataset expectations, how to run, preprocessing steps, models used, and evaluation — Done.

---

## 🔬 Key Learnings (from this project)

1. For daily stock-close forecasting on this dataset, feature-engineered models (XGBoost with lags/momentum features) can dramatically outperform off-the-shelf time-series decomposition models if the features capture market dynamics.
2. Aligning model outputs to the exact test trading dates is essential when a forecasting tool returns every calendar day (Prophet) but the observed data excludes non-trading days.
3. Installing `prophet` on Windows may require alternate installation strategies (conda-forge) due to compiled dependencies.

---

## ♻️ Future Improvements

- Add a synthetic-data fallback so the notebook remains runnable when the CSV is missing.
- Expand XGBoost feature engineering with technical indicators (RSI, MACD, moving averages) and re-run ablation tests.
- Add hyperparameter tuning (Optuna / GridSearchCV) and cross-validation (purged/rolling) for more robust evaluation.
- Factor preprocessing and modeling code into reusable Python modules for testing and CI.
- Add unit tests for preprocessing functions and a small sample dataset.
- Add a small CLI wrapper (`run_forecast.py`) accepting `--data-path`, `--plots-path`, and `--save-forecasts`.

---

## 📌 Reproducibility Tips

- Freeze your environment after a successful run:

```bash
pip freeze > requirements-frozen.txt
```

- If `prophet` is troublesome on Windows, prefer the conda-forge package:

```bash
conda install -c conda-forge prophet
```

- Save model outputs (forecasts, metrics, plots) for reproducibility.

---

## 👨‍💻 Author

**Alexander Olomokoro**  
Full-Stack Developer | Machine Learning Engineer  
📍 Nigeria  
🔗 [LinkedIn](https://www.linkedin.com/in/alexander-olumukoro-699255199) • [Twitter/X](https://x.com/xandersavage7) • [GitHub](https://github.com/xandersavage)

---

## 📝 License

MIT
