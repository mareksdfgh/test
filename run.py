"""
run.py
======
Živá predikce pro jednu akcii přes VŠECHNY natrénované horizonty
(1h, 4h, 1d, ...) + grafy. Příznaky se staví přes sdílené data.py /
features.py → bitově shodné s tréninkem (žádný train/serve skew).

Spuštění:
  python run.py --ticker AAPL
  python run.py --ticker AAPL --no_chart
  python run.py --ticker MSFT --lookback 200
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

import config
from data import (dl_hourly, dl_daily,
                  merge_daily_to_hourly, merge_market_hourly)
from features import add_features, rsi


# ── Sestavení příznaků (sdílená logika s build_dataset) ──────────────────────

def fetch_features(ticker: str, meta: dict, use_cache: bool):
    """Stáhne živá data a sestaví příznaky identické s build_dataset.py."""
    print(f"  Načítám {ticker}...")
    h = dl_hourly(ticker, use_cache=use_cache)
    if h.empty:
        raise ValueError(f"Žádná data pro {ticker}")

    close_series = h["Close"].copy()
    h_feat = add_features(h.copy(), prefix="")

    d = dl_daily(ticker, use_cache=use_cache)
    if not d.empty:
        d_feat = add_features(d.copy(), prefix="d_")
        h_feat = merge_daily_to_hourly(h_feat, d_feat, "d_main")

    for name, t in config.CONTEXT_TICKERS.items():
        try:
            mh = dl_hourly(t, use_cache=use_cache)
            if not mh.empty:
                h_feat = merge_market_hourly(
                    h_feat, add_features(mh.copy(), prefix=f"{name}_"), name)
            md = dl_daily(t, use_cache=use_cache)
            if not md.empty:
                h_feat = merge_daily_to_hourly(
                    h_feat, add_features(md.copy(), prefix=f"{name}_d_"), name)
        except Exception as e:
            print(f"    VAROVÁNÍ: {name} ({t}) selhal: {e}")

    idx = h_feat.index
    h_feat["hour"]         = idx.hour
    h_feat["day_of_week"]  = idx.dayofweek
    h_feat["month"]        = idx.month
    h_feat["quarter"]      = idx.quarter
    h_feat["is_monday"]    = (idx.dayofweek == 0).astype(int)
    h_feat["is_friday"]    = (idx.dayofweek == 4).astype(int)
    h_feat["week_of_year"] = idx.isocalendar().week.astype(int).values

    for feat in meta.get("features", []):
        if feat.startswith("ticker_oh_"):
            h_feat[feat] = int(ticker == feat.replace("ticker_oh_", ""))

    return h_feat, close_series


def align_matrix(h_feat: pd.DataFrame, meta: dict, lookback: int):
    """
    Sestaví matici příznaků v pořadí z tréninku. Loguje, kolik hodnot
    bylo imputováno mediánem (varování při velkém podílu).
    """
    features = meta["features"]
    medians  = pd.Series(meta["medians"]).astype(float)

    X = h_feat.replace([np.inf, -np.inf], np.nan)
    missing = [f for f in features if f not in X.columns]
    avail   = [f for f in features if f in X.columns]

    sl = X[avail].iloc[-lookback:].copy()
    for m in missing:
        sl[m] = np.nan
    sl = sl[features]

    total = sl.size
    n_nan = int(sl.isna().to_numpy().sum())
    sl = sl.fillna(medians).fillna(0.0)

    pct = 100.0 * n_nan / total if total else 0.0
    msg = f"  Imputováno mediánem: {n_nan}/{total} hodnot ({pct:.1f}%)"
    if missing:
        msg += f"  | chybějící příznaky: {len(missing)}"
    print(msg)
    if pct > 25:
        print("  ⚠ VAROVÁNÍ: vysoký podíl imputace → signál může být zkreslený")

    return sl


# ── Grafy ─────────────────────────────────────────────────────────────────────

def plot_signal_chart(close, signals, ticker, lookback, horizon_name):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib není nainstalován")
        return
    df = pd.DataFrame({"close": close, "signal": signals}).iloc[-lookback:]
    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                             gridspec_kw={"height_ratios": [3, 1, 1]})
    fig.suptitle(f"{ticker} — signály [{horizon_name}]", fontsize=14, fontweight="bold")
    ax = axes[0]
    ax.plot(df.index, df["close"], color="#1f77b4", linewidth=1.2, label="Kurz")
    buy = df[df["signal"] == 2].index
    sell = df[df["signal"] == 0].index
    ax.scatter(buy,  df.loc[buy,  "close"], color="lime", marker="^", s=60, zorder=5, label="BUY")
    ax.scatter(sell, df.loc[sell, "close"], color="red",  marker="v", s=60, zorder=5, label="SELL")
    ax.plot(df.index, df["close"].rolling(20).mean(), color="orange", lw=0.8, alpha=0.7, label="MA20")
    ax.plot(df.index, df["close"].rolling(50).mean(), color="purple", lw=0.8, alpha=0.7, label="MA50")
    ax.set_ylabel("Kurz ($)"); ax.legend(loc="upper left", fontsize=8); ax.grid(True, alpha=0.3)
    ax2 = axes[1]
    colors = {0: "red", 1: "gray", 2: "lime"}
    ax2.bar(df.index, df["signal"], color=[colors.get(s, "gray") for s in df["signal"]], width=0.03, alpha=0.7)
    ax2.set_yticks([0, 1, 2]); ax2.set_yticklabels(["SELL", "HOLD", "BUY"], fontsize=8)
    ax2.set_ylabel("Signál"); ax2.grid(True, alpha=0.3)
    ax3 = axes[2]
    rv = rsi(df["close"], 14)
    ax3.plot(df.index, rv, color="darkorange", lw=0.9)
    ax3.axhline(70, color="red", ls="--", alpha=0.5, lw=0.8)
    ax3.axhline(30, color="lime", ls="--", alpha=0.5, lw=0.8)
    ax3.set_ylim(0, 100); ax3.set_ylabel("RSI(14)"); ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = config.OUTPUT_DIR / f"chart_signals_{ticker}_{horizon_name}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"  ✓ {fname}")
    plt.close(fig)


def plot_prediction_chart(close, preds_pct: dict, ticker):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    recent = close.iloc[-100:]
    ax.plot(recent.index, recent.values, color="#1f77b4", lw=1.5, label="Historický kurz")
    last_price, last_time = recent.iloc[-1], recent.index[-1]
    step = pd.Series(recent.index).diff().median()
    for name, (pct, n_bars) in preds_pct.items():
        pred_price = last_price * (1 + pct)
        pred_time = last_time + step * n_bars
        color = "lime" if pct >= 0 else "red"
        ax.plot([last_time, pred_time], [last_price, pred_price],
                lw=2.0, ls="--", marker="o", markersize=6, color=color,
                label=f"{name}: {pred_price:.2f}$ ({pct*100:+.2f}%)")
    ax.axvline(last_time, color="gray", ls=":", alpha=0.6)
    ax.set_title(f"{ticker} — predikce ceny (více horizontů)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Kurz ($)"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m %H:%M"))
    plt.xticks(rotation=30); plt.tight_layout()
    fname = config.OUTPUT_DIR / f"chart_prediction_{ticker}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"  ✓ {fname}")
    plt.close(fig)


# ── Predikce ──────────────────────────────────────────────────────────────────

def predict(ticker: str, model_dir: str, lookback: int,
            no_chart: bool, use_cache: bool) -> dict:
    model_dir = Path(model_dir)
    with open(model_dir / "model_meta.json") as f:
        meta = json.load(f)
    horizons = meta["horizons"]
    print(f"  Modely: horizonty {list(horizons)} (trénováno do {meta['trained_until']})")

    # Je ticker součástí "domácí" sady (má vlastní one-hot příznak)?
    known = {f.replace("ticker_oh_", "") for f in meta.get("features", [])
             if f.startswith("ticker_oh_")}
    if ticker in known:
        print(f"  Ticker {ticker} je v trénované sadě → specializovaná predikce")
    else:
        print(f"  ⚠ Ticker {ticker} NENÍ v trénované sadě ({len(known)} akcií) "
              f"→ generická predikce jen z technických/tržních příznaků "
              f"(one-hot = 0, interpretuj opatrně)")

    h_feat, close_series = fetch_features(ticker, meta, use_cache)
    X_slice = align_matrix(h_feat, meta, lookback)
    close_window = close_series.iloc[-lookback:].reindex(X_slice.index)
    last_price = float(close_series.iloc[-1])
    label_map = meta["label_map"]

    results, preds_for_chart = {}, {}
    print(f"\n{'='*58}")
    print(f"  {ticker}   kurz {last_price:.2f}$   {close_series.index[-1]}")
    print(f"  {'─'*54}")
    print(f"  {'Horizont':<8} {'Signál':<6} {'Conf':>6}  {'SELL/HOLD/BUY':<22} {'Předpověď':>16}")

    for name, n_bars in horizons.items():
        cls = joblib.load(model_dir / f"model_cls_{name}.joblib")
        reg = joblib.load(model_dir / f"model_reg_{name}.joblib")
        sig = cls.predict(X_slice)
        proba = cls.predict_proba(X_slice)
        pred_ret = reg.predict(X_slice)
        last_sig, last_p, last_r = sig[-1], proba[-1], float(pred_ret[-1])
        pred_price = last_price * (1 + last_r)

        # předikce u BUY/SELL musí pokrýt 3 třídy i když chybí ve výcviku
        p = {c: 0.0 for c in (0, 1, 2)}
        for ci, cls_lbl in enumerate(cls.classes_):
            p[int(cls_lbl)] = last_p[ci]
        print(f"  {name:<8} {label_map[str(last_sig)]:<6} {p[last_sig]:>5.0%}  "
              f"{p[0]:>5.0%}/{p[1]:>4.0%}/{p[2]:>4.0%}        "
              f"{pred_price:>8.2f}$ ({last_r*100:+.2f}%)")

        results[name] = {
            "signal": label_map[str(last_sig)],
            "confidence": float(p[last_sig]),
            "pred_return": last_r,
            "pred_price": float(pred_price),
        }
        preds_for_chart[name] = (last_r, n_bars)
    print(f"{'='*58}\n")

    if not no_chart:
        # signal chart pro nejdelší dostupný horizont (nejčitelnější)
        main = list(horizons)[-1]
        cls_main = joblib.load(model_dir / f"model_cls_{main}.joblib")
        plot_signal_chart(close_window, cls_main.predict(X_slice), ticker, lookback, main)
        plot_prediction_chart(close_series, preds_for_chart, ticker)

    return {
        "ticker": ticker,
        "timestamp": str(close_series.index[-1]),
        "price": last_price,
        "horizons": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker",    default="AAPL")
    parser.add_argument("--model_dir", default=str(config.MODEL_DIR))
    parser.add_argument("--lookback",  default=200, type=int)
    parser.add_argument("--no_chart",  action="store_true")
    parser.add_argument("--no_cache",  action="store_true")
    parser.add_argument("--track",     action="store_true",
                        help="Zaloguj predikci do trackeru (tracker.py) pro pozdější vyhodnocení")
    args = parser.parse_args()
    result = predict(args.ticker, args.model_dir, args.lookback,
                     args.no_chart, use_cache=not args.no_cache)

    if args.track:
        from tracker import PredictionTracker
        # Logujeme hlavní horizont (4h pokud existuje, jinak první dostupný).
        hz = result["horizons"]
        name = "4h" if "4h" in hz else next(iter(hz))
        r = hz[name]
        PredictionTracker().log_prediction(
            ticker=result["ticker"], price=result["price"],
            signal=r["signal"], confidence=r["confidence"],
            pred_return=r["pred_return"])
        print(f"  ✓ Predikce [{name}] zalogována do trackeru")
