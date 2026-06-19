# Mičov Stock Prediktion 📈

A machine-learning pipeline that predicts stock price movements for **many tickers** and **multiple time horizons at once** (1h, 4h, 1 day, 1 week). For a given ticker it outputs a **BUY / HOLD / SELL** signal plus a predicted % return for *each* horizon — so the model learns the relationship between short-term (hourly) and longer-term (daily/weekly) moves.

It is trained on a **"home set" of 50 diverse US stocks** (tech, semis, finance, healthcare, consumer, industrials, energy, materials) — each gets its own one-hot feature, so the model **specialises on those 50**. It can still predict **any other ticker**: unknown symbols fall back to the generic technical + market features (one-hot all zero), and `run.py` tells you when a prediction is generic. Context inputs include broad **commodities/resources** — gold, silver, copper, platinum, palladium, oil, natural gas, plus lithium/uranium/base-metal ETFs (a true "silicon" futures doesn't exist; the semiconductor sector ETF + lithium proxy stand in).

> ⚠️ **Disclaimer:** Educational project only. Nothing here is financial advice. On hourly data the realistic directional edge is small; the built-in backtest (`test.py`) is there precisely so you can check honestly whether there is any edge after fees.

---

## What's new in this version

- **Multi-horizon prediction** — one feature set, separate models for `1h / 4h / 1d / 1w`.
- **Leakage fixes** (the important part):
  - Hourly market features merged with `pd.merge_asof(direction="backward", allow_exact_matches=False)` — robust to timestamp gaps, never uses a same/future-hour bar.
  - Daily features lagged by one trading day (`shift(1)`) — no same-day leak from the daily Close.
  - Clean NaN filter (drop rows missing >20 % of features) instead of the old `thresh=60 %`.
  - Stacking uses `KFold(shuffle=False)` (contiguous time blocks), not stratified KFold that shuffles classes across time.
- **Profitability built into training** — long/flat simulation with fees vs. an Always-BUY baseline, printed per horizon.
- **Shared `data.py`** — one download/merge implementation for build, predict and backtest → no train/serve skew.
- **Central `config.py`**, on-disk **parquet caching**, **`.gitignore`**, and **unit tests** (`pytest`).
- **Imputation logging** in `run.py` — warns when too many live features are missing.

---

## Architecture

```
config.py          # all constants: tickers, horizons, thresholds, paths, cache
data.py            # shared download + leak-free merges (asof / daily lag) + cache
features.py        # ~60 technical indicators (RSI, MACD, Bollinger, ATR, ...)
labels.py          # multi-horizon labels (label_<h> + ret_<h>)
build_dataset.py   # downloads everything → dataset.csv
train.py           # trains cls+reg stacking models per horizon → models/
run.py             # live prediction for one ticker, all horizons + charts
tracker.py         # SQLite engine: logs predictions, scores them after 1h/4h
test.py            # strict point-in-time backtest: baselines, look-ahead audit, profit, Sharpe
tests/             # pytest unit tests (synthetic data, no network)
```

Pipeline: `build_dataset.py → train.py → run.py`, with `test.py` for honest evaluation and `tracker.py` for live accuracy tracking.

---

## Installation

```bash
pip install -r requirements.txt
# macOS, if XGBoost fails:
brew install libomp
```
Python 3.10+ and an internet connection (live data via `yfinance`) are required.

---

## Usage

**1. Build the dataset** (downloads the 50 home stocks + 30 macro/commodity/sector context tickers; cached afterwards)
```bash
python build_dataset.py
python build_dataset.py --tickers AAPL MSFT GOOGL      # custom set
python build_dataset.py --no_cache                     # force fresh download
```

**2. Train** (one cls + reg model per horizon)
```bash
python train.py                       # default horizons: 1h 4h 1d
python train.py --horizons 4h 1d
python train.py --no_xgb              # HistGradientBoosting only (no XGBoost)
```

**3. Live prediction** (all trained horizons at once)
```bash
python run.py --ticker AAPL               # in the 50-stock home set → specialised
python run.py --ticker SAP                 # any other ticker → generic prediction
python run.py --ticker AAPL --no_chart
python run.py --ticker AAPL --track       # also log to tracker.py for later scoring
```

**4. Strict backtest / audit**
```bash
python test.py --horizon 4h --snapshots 30
```

**5. Track live accuracy** (run periodically, e.g. via cron)
```bash
python tracker.py evaluate      # score predictions that have matured (1h / 4h)
python tracker.py stats         # accuracy per ticker / signal
python tracker.py clean --days 30
```

**Tests**
```bash
pytest -q
```

---

## Example output (`run.py`)

```
==========================================================
  AAPL   kurz 189.43$   2025-05-24 19:00:00
  ──────────────────────────────────────────────────────
  Horizont Signál   Conf  SELL/HOLD/BUY            Předpověď
  1h       HOLD      48%   18%/ 48%/ 34%      189.6$ (+0.10%)
  4h       BUY       61%   14%/ 25%/ 61%      191.1$ (+0.89%)
  1d       BUY       57%   19%/ 24%/ 57%      193.0$ (+1.88%)
==========================================================
```
Charts are saved to `outputs/`.

---

## Configuration

Everything tunable lives in [`config.py`](config.py): `DEFAULT_TICKERS` (the 50-stock home set), `COMMODITIES` / `MARKET_TICKERS` / `SECTOR_ETFS` (merged into `CONTEXT_TICKERS`), `LABEL_HORIZONS` (in hourly bars), `LABEL_THRESHOLDS`, cache TTL, merge tolerance, fees. Horizons are defined in **hourly bars** (US sessions ≈ 7 bars/day), so `1d ≈ 7`, `1w ≈ 35`.

> ⚠️ **Compute:** 50 stocks + 30 context tickers, each contributing ~60 hourly **and** ~60 daily features, produces **~2000+ features**. Full `XGB+HGB` stacking across 3 horizons is heavy (well over an hour, lots of RAM). To speed up: `--no_xgb`, train one horizon at a time (`--horizons 4h`), or trim `CONTEXT_TICKERS` in `config.py`.

---

## Notes & honest limitations

- sklearn's `StackingClassifier` requires its internal CV to be a *partition*, so a true walk-forward `TimeSeriesSplit` can't be used there. We use `KFold(shuffle=False)`; the leak-free evaluation therefore rests on the strict forward train/test split and the out-of-sample backtest in `test.py`.
- Hourly horizons are bar-based approximations of calendar time (they skip overnight gaps).
- Median imputation of missing live features can distort a signal; `run.py` logs and warns when imputation is high.
