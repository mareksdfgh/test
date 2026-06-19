"""
build_dataset.py
================
Sestaví společný multi-ticker, multi-horizont dataset.
Každý řádek = jedna hodina jedné akcie. Příznaky jsou identické pro
trénink i inferenci (vše přes sdílené data.py / features.py).

Labely pro VÍCE horizontů (1h, 4h, 1d, 1w) → model se učí vztah mezi
krátkodobým a dlouhodobým pohybem.

Výstupy:
  dataset.csv          — hlavní dataset
  dataset_info.csv     — describe() statistiky
  dataset_tickers.json — seznam tickerů + one-hot mapa

Spuštění:
  python build_dataset.py
  python build_dataset.py --tickers AAPL MSFT GOOGL --output dataset.csv
  python build_dataset.py --no_cache        # vynutí čerstvé stažení
"""

import argparse
import json

import pandas as pd

import config
from data import (dl_hourly, dl_daily,
                  merge_daily_to_hourly, merge_market_hourly)
from features import add_features
from labels import make_labels, max_horizon


# ── Zpracování jednoho tickeru ────────────────────────────────────────────────

def process_ticker(ticker: str, all_tickers: list,
                   market_h: dict, market_d: dict,
                   use_cache: bool) -> pd.DataFrame:
    print(f"\n  • {ticker}")

    h = dl_hourly(ticker, use_cache=use_cache)
    if h.empty:
        print("    PŘESKOČENO: žádná hodinová data")
        return pd.DataFrame()

    close_raw = h["Close"].copy()
    labels = make_labels(close_raw)            # všechny horizonty najednou

    h_feat = add_features(h.copy(), prefix="")

    # Denní kontext samotné akcie (s 1-denním lagem uvnitř merge).
    d = dl_daily(ticker, use_cache=use_cache)
    if not d.empty:
        d_feat = add_features(d.copy(), prefix="d_")
        h_feat = merge_daily_to_hourly(h_feat, d_feat, prefix="d_main")

    # Hodinové tržní příznaky (asof-merge, bez look-ahead).
    for name, mdf in market_h.items():
        if not mdf.empty:
            h_feat = merge_market_hourly(h_feat, mdf.copy(), name)

    # Denní tržní příznaky (1-denní lag).
    for name, mdf in market_d.items():
        if not mdf.empty:
            h_feat = merge_daily_to_hourly(h_feat, mdf.copy(), prefix=name)

    # Labely všech horizontů.
    h_feat = h_feat.join(labels)
    h_feat["close_raw"] = close_raw            # pro grafy v run.py

    # Časové příznaky.
    idx = h_feat.index
    h_feat["hour"]         = idx.hour
    h_feat["day_of_week"]  = idx.dayofweek
    h_feat["month"]        = idx.month
    h_feat["quarter"]      = idx.quarter
    h_feat["is_monday"]    = (idx.dayofweek == 0).astype(int)
    h_feat["is_friday"]    = (idx.dayofweek == 4).astype(int)
    h_feat["week_of_year"] = idx.isocalendar().week.astype(int).values

    # One-hot ticker.
    for t in all_tickers:
        h_feat[f"ticker_oh_{t}"] = int(ticker == t)
    h_feat["ticker"] = ticker                  # metadata, ne feature

    # Posledních max_horizon řádků zahodíme — future return chybí.
    h_feat = h_feat.iloc[:-max_horizon()]
    return h_feat


# ── Tržní data (stahuje se jednou) ────────────────────────────────────────────

def load_market(use_cache: bool):
    print("\n[1/3] Tržní, komoditní a sektorová data...")
    all_market = config.CONTEXT_TICKERS
    market_h, market_d = {}, {}

    for name, t in all_market.items():
        print(f"  {name} ({t})")
        try:
            h = dl_hourly(t, use_cache=use_cache)
            if not h.empty:
                market_h[name] = add_features(h.copy(), prefix=f"{name}_")
            else:
                print(f"    VAROVÁNÍ: {name} — žádná hodinová data")
        except Exception as e:
            print(f"    CHYBA {name} (hodinové): {e}")
        try:
            d = dl_daily(t, use_cache=use_cache)
            if not d.empty:
                market_d[name] = add_features(d.copy(), prefix=f"{name}_d_")
            else:
                print(f"    VAROVÁNÍ: {name} — žádná denní data")
        except Exception as e:
            print(f"    CHYBA {name} (denní): {e}")

    return market_h, market_d


# ── Hlavní pipeline ───────────────────────────────────────────────────────────

def build_dataset(tickers: list, output_path: str, use_cache: bool = True):
    print(f"\n{'='*55}")
    print(" Multi-Ticker / Multi-Horizont Dataset Builder")
    print(f" Tickerů: {len(tickers)}  |  Horizonty: {list(config.LABEL_HORIZONS)}")
    print(f"{'='*55}")

    market_h, market_d = load_market(use_cache)

    print(f"\n[2/3] Zpracování {len(tickers)} akcií...")
    frames, failed = [], []
    for ticker in tickers:
        try:
            df = process_ticker(ticker, tickers, market_h, market_d, use_cache)
            if not df.empty:
                frames.append(df)
                print(f"    ✓ {ticker}: {len(df)} řádků, {df.shape[1]} sloupců")
        except Exception as e:
            print(f"    ✗ {ticker}: {e}")
            failed.append(ticker)

    if not frames:
        raise ValueError("Žádná data nebyla načtena!")

    print("\n[3/3] Spojení a uložení...")
    combined = pd.concat(frames, axis=0)
    combined.sort_index(inplace=True)

    # Čistý NaN filtr: zahodíme řádek, kde chybí > MAX_ROW_NAN_FRAC příznaků.
    label_cols = [c for c in combined.columns
                  if c.startswith(("label_", "ret_"))]
    feat_cols = [c for c in combined.columns
                 if c not in label_cols + config.META_COLS]

    before = len(combined)
    nan_frac = combined[feat_cols].isna().mean(axis=1)
    combined = combined[nan_frac <= config.MAX_ROW_NAN_FRAC]
    # Řádek musí mít definované VŠECHNY labely (jinak je k ničemu).
    combined = combined.dropna(subset=label_cols)
    print(f"  Řádky: {before} → {len(combined)} (po NaN/label filtru)")

    # Statistiky.
    print(f"\n  Příznaků: {len(feat_cols)}")
    print(f"  Časové rozmezí: {combined.index.min()} → {combined.index.max()}")
    for name in config.LABEL_HORIZONS:
        col = f"label_{name}"
        dist = combined[col].value_counts().sort_index()
        print(f"  [{name:>3}]  SELL={int(dist.get(0,0)):>6}  "
              f"HOLD={int(dist.get(1,0)):>6}  BUY={int(dist.get(2,0)):>6}")

    combined.to_csv(output_path, index=True)
    print(f"\n  ✓ Dataset: {output_path}")

    info_path = output_path.replace(".csv", "_info.csv")
    combined.describe().T.to_csv(info_path)
    print(f"  ✓ Info: {info_path}")

    map_path = output_path.replace(".csv", "_tickers.json")
    with open(map_path, "w") as f:
        json.dump({"tickers": tickers,
                   "ticker_id_map": {t: i for i, t in enumerate(tickers)}},
                  f, indent=2)
    print(f"  ✓ Mapa tickerů: {map_path}")

    if failed:
        print(f"\n  Selhalo: {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=config.DEFAULT_TICKERS)
    parser.add_argument("--output",  default=str(config.DATASET_CSV))
    parser.add_argument("--no_cache", action="store_true",
                        help="Ignoruj cache, stáhni vše čerstvé")
    args = parser.parse_args()
    build_dataset(args.tickers, args.output, use_cache=not args.no_cache)
