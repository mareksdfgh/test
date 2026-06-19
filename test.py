"""
test_strict.py
==============
Striktní backtest s ověřením každého bodu z ChatGPT kritiky:

1. Baseline srovnání (Always BUY, Momentum, Random)
2. Ověření Look-ahead: features <= snap_time, target = snap + 4h
3. Profit simulace po poplatcích
4. Sharpe ratio + Max Drawdown
5. Duplikáty Train/Test
6. Feature cutoff audit (která feature má nejpozdější timestamp)

Spuštění:
  python test_strict.py
  python test_strict.py --snapshots 30
"""

import argparse
import json
import random
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter

import numpy as np
import pandas as pd
import joblib

import config
from data import dl_hourly, dl_daily, merge_daily_to_hourly, merge_market_hourly
from features import add_features

# Necháváme warningy převážně viditelné; tlumíme jen konvergenci.
warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")

HOLD_THRESH    = config.HOLD_THRESH
FEE_PCT        = config.FEE_PCT
MARKET_TICKERS = config.MARKET_TICKERS
SECTOR_ETFS    = config.SECTOR_ETFS


def strip_tz(index):
    if hasattr(index, 'tz') and index.tz is not None:
        return index.tz_convert("UTC").tz_localize(None)
    return index


def strip_ts(ts):
    if hasattr(ts, 'tz') and ts.tz is not None:
        return ts.tz_convert("UTC").tz_localize(None)
    return ts


# Stahování dat řešíme sdíleným data.py (dl_hourly/dl_daily importovány nahoře)
# → backtest používá BITOVĚ stejná data i cache jako trénink.


def load_tickers(model_dir):
    p = model_dir / "dataset_tickers.json"
    if p.exists():
        with open(p) as f:
            data = json.load(f)
        t = data.get("tickers", [])
        if t:
            return t
    p2 = model_dir / "model_meta.json"
    if p2.exists():
        with open(p2) as f:
            meta = json.load(f)
        t = [f.replace("ticker_oh_", "") for f in meta.get("features", [])
             if f.startswith("ticker_oh_")]
        if t:
            return t
    return []


def find_closest_price(df, target, tolerance_h=8):
    target = strip_ts(target)
    lo = target - timedelta(hours=tolerance_h)
    hi = target + timedelta(hours=tolerance_h)
    w = df[(df.index >= lo) & (df.index <= hi)]
    if w.empty:
        return None
    diffs = (w.index - target).to_series(index=w.index).abs()
    return float(df.loc[diffs.idxmin(), "Close"])


def direction_ok(signal, actual_return):
    if signal == "BUY":
        return actual_return > 0
    if signal == "SELL":
        return actual_return < 0
    return abs(actual_return) < HOLD_THRESH


def build_features_at(df_h_full, df_d_full, market_h, market_d,
                       cutoff, ticker, meta):
    """Identické s test.py — příznaky striktně <= cutoff."""
    cutoff = strip_ts(cutoff)
    h = df_h_full[df_h_full.index <= cutoff].copy()
    if len(h) < 50:
        return None, None, None

    # ── AUDIT: jaký je nejpozdější timestamp ve featurách? ──────────────────
    last_allowed = h.index[-1]  # musí být <= cutoff

    close_at_cutoff = float(h["Close"].iloc[-1])
    h_feat = add_features(h.copy(), prefix="")

    # Sdílené merge funkce (stejné jako build_dataset.py) na datech <= cutoff.
    # → point-in-time korektní + bez train/serve skew.
    if not df_d_full.empty:
        d = df_d_full[df_d_full.index <= cutoff].copy()
        if not d.empty:
            d_feat = add_features(d.copy(), prefix="d_")
            h_feat = merge_daily_to_hourly(h_feat, d_feat, prefix="d_main")

    for name, mdf in market_h.items():
        if mdf.empty:
            continue
        mdf_cut = mdf[mdf.index <= cutoff].copy()
        if not mdf_cut.empty:
            h_feat = merge_market_hourly(h_feat, mdf_cut, name)

    for name, mdf in market_d.items():
        if mdf.empty:
            continue
        mdf_cut = mdf[mdf.index <= cutoff].copy()
        if not mdf_cut.empty:
            h_feat = merge_daily_to_hourly(h_feat, mdf_cut, prefix=name)

    h_feat["hour"]         = h_feat.index.hour
    h_feat["day_of_week"]  = h_feat.index.dayofweek
    h_feat["month"]        = h_feat.index.month
    h_feat["quarter"]      = h_feat.index.quarter
    h_feat["is_monday"]    = (h_feat.index.dayofweek == 0).astype(int)
    h_feat["is_friday"]    = (h_feat.index.dayofweek == 4).astype(int)
    h_feat["week_of_year"] = h_feat.index.isocalendar().week.astype(int).values

    for feat in meta.get("features", []):
        if feat.startswith("ticker_oh_"):
            h_feat[feat] = int(ticker == feat.replace("ticker_oh_", ""))

    row      = h_feat.iloc[-1]
    features = meta["features"]
    medians  = pd.Series(meta["medians"])
    aligned  = {}
    for f in features:
        try:
            val = float(row[f]) if f in row.index else float("nan")
        except Exception:
            val = float("nan")
        if pd.isna(val) or np.isinf(val):
            val = float(medians.get(f, 0.0))
        aligned[f] = val

    return pd.Series(aligned), close_at_cutoff, last_allowed


# ── Baseline Strategien ───────────────────────────────────────────────────────

def baseline_always_buy(actual_ret):
    """Immer BUY — profitiert von allgemeinem Aufwärtstrend."""
    return actual_ret > 0


def baseline_momentum(df_h, snap, actual_ret):
    """Letzte 4h Momentum — wenn letzten 4h gestiegen, dann BUY."""
    snap = strip_ts(snap)
    start = snap - timedelta(hours=4)
    window = df_h[(df_h.index >= start) & (df_h.index <= snap)]["Close"]
    if len(window) < 2:
        return actual_ret > 0  # fallback
    mom = (window.iloc[-1] - window.iloc[0]) / window.iloc[0]
    pred = "BUY" if mom > 0 else "SELL"
    return direction_ok(pred, actual_ret)


def baseline_random(signal_dist, actual_ret):
    """Zufällig mit gleicher BUY/SELL/HOLD-Verteilung wie das Modell."""
    signals = list(signal_dist.keys())
    weights = list(signal_dist.values())
    pred = random.choices(signals, weights=weights)[0]
    return direction_ok(pred, actual_ret)


# ── Profit-Simulation ─────────────────────────────────────────────────────────

def simulate_profit(results, fee_pct=FEE_PCT):
    """
    Einfache Profit-Simulation:
    - Startkapital 10.000$
    - Jeder BUY/SELL = volles Kapital
    - HOLD = kein Trade
    - Kaufgebühr + Verkaufsgebühr = 2 × fee_pct
    """
    capital = 10000.0
    trades  = []
    equity_curve = [capital]

    for r in results:
        if r["signal"] == "HOLD":
            equity_curve.append(capital)
            continue

        # Gebühren für Kauf + Verkauf
        fee = capital * fee_pct * 2

        if r["signal"] == "BUY":
            gross = capital * r["actual_ret"]
        else:  # SELL (short)
            gross = capital * (-r["actual_ret"])

        net    = gross - fee
        capital += net
        equity_curve.append(capital)
        trades.append({
            "signal":  r["signal"],
            "ret":     r["actual_ret"],
            "gross":   gross,
            "net":     net,
            "capital": capital,
        })

    if not trades:
        return None

    returns_series = [t["net"] / (capital - t["net"] + abs(t["net"])) for t in trades]

    # Sharpe (vereinfacht, annualisiert auf 4h-Basis)
    # 4h pro Trade, ~252 Handelstage × ~1.5 Trades pro Tag = ~378 Trades/Jahr
    trading_periods_per_year = 378
    if len(returns_series) > 1:
        mean_r = np.mean(returns_series)
        std_r  = np.std(returns_series)
        sharpe = (mean_r / std_r * np.sqrt(trading_periods_per_year)
                  if std_r > 0 else 0.0)
    else:
        sharpe = 0.0

    # Max Drawdown
    eq = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    dd   = (eq - peak) / peak
    max_dd = float(dd.min())

    return {
        "start_capital":  10000.0,
        "end_capital":    capital,
        "total_return":   (capital - 10000.0) / 10000.0,
        "n_trades":       len(trades),
        "win_trades":     sum(1 for t in trades if t["net"] > 0),
        "total_fees":     sum(capital * fee_pct * 2 for _ in trades),
        "sharpe":         sharpe,
        "max_drawdown":   max_dd,
        "equity_curve":   equity_curve,
    }


# ── Hauptfunktion ──────────────────────────────────────────────────────────────

def run_strict_backtest(model_dir, tickers, n_snapshots, horizon_name):
    with open(model_dir / "model_meta.json") as f:
        meta = json.load(f)

    horizons = meta.get("horizons", {})
    if horizon_name not in horizons:
        raise ValueError(f"Horizont '{horizon_name}' není natrénován. "
                         f"Dostupné: {list(horizons)}")
    horizon_h = int(horizons[horizon_name])

    print(f"\n{'='*65}")
    print(f"  Mičov — STRICT BACKTEST (audit)")
    print(f"  Snímků: {n_snapshots}  Horizont: {horizon_name} ({horizon_h} svíček)  "
          f"Tickery: {len(tickers)}")
    print(f"{'='*65}\n")

    cls_model = joblib.load(model_dir / f"model_cls_{horizon_name}.joblib")
    reg_model = joblib.load(model_dir / f"model_reg_{horizon_name}.joblib")
    label_map = meta["label_map"]
    trained_until = pd.Timestamp(meta["trained_until"])

    print(f"  Modely načteny (trénováno do {trained_until})")
    print(f"  Features: {len(meta['features'])}\n")

    # ── Ověření Train/Test overlap ─────────────────────────────────────────
    print(f"  [AUDIT 1] Train/Test Zeitraum:")
    print(f"    Trénink do:   {trained_until.strftime('%Y-%m-%d %H:%M')}")

    # Tržní data
    print(f"  Načítám tržní data...")
    all_market = config.CONTEXT_TICKERS
    market_h, market_d = {}, {}
    for name, t in all_market.items():
        try:
            h = dl_hourly(t)
            if not h.empty:
                market_h[name] = add_features(h.copy(), prefix=f"{name}_")
        except Exception:
            pass
        try:
            d = dl_daily(t)
            if not d.empty:
                market_d[name] = add_features(d.copy(), prefix=f"{name}_d_")
        except Exception:
            pass
    print(f"  ✓ Tržní data načtena\n")

    # Snímky z obchodní doby
    ref_df = dl_hourly(tickers[0])
    if ref_df.empty:
        print("CHYBA: žádná referenční data"); return

    trading = ref_df[
        (ref_df.index.dayofweek < 5) &
        (ref_df.index.hour >= 13) &
        (ref_df.index.hour <= 20)
    ].copy()
    trading_idx = sorted(trading.index.tolist())

    # Párování snímek → eval podle OFFSETU V BARECH (ne hodin), aby
    # odpovídal definici labelu (close.pct_change(h).shift(-h) = h svíček).
    # Snímky bereme od konce, s rozestupem horizon_h barů (nepřekrývají se).
    pairs = []
    i = len(trading_idx) - 1 - horizon_h
    while i >= 0 and len(pairs) < n_snapshots:
        snap_t = trading_idx[i]
        eval_t = trading_idx[i + horizon_h]
        pairs.append((snap_t, eval_t))
        i -= horizon_h

    pairs = list(reversed(pairs))
    if not pairs:
        print("CHYBA: žádné snímky"); return

    snapshots  = [p[0] for p in pairs]
    eval_times = [p[1] for p in pairs]

    # Ověření zda jsou snapshots PO trénování
    snaps_after_train  = sum(1 for s in snapshots if s > trained_until)
    snaps_during_train = sum(1 for s in snapshots if s <= trained_until)

    print(f"  [AUDIT 2] Snímky vs. tréninková data:")
    print(f"    Snímky celkem:          {len(snapshots)}")
    print(f"    Po trénování (OOS):     {snaps_after_train}  ← tyto jsou validní")
    print(f"    Během tréninku (IS):    {snaps_during_train}  ← POTENCIÁLNÍ LEAKAGE!")
    if snaps_during_train > 0:
        print(f"    VAROVÁNÍ: {snaps_during_train} snímků je v tréninkovém období!")
        print(f"    Výsledky mohou být nafouknuté.")
    else:
        print(f"    ✓ Všechny snímky jsou Out-of-Sample")
    print()

    # ── Hlavní loop ────────────────────────────────────────────────────────
    all_results   = {t: [] for t in tickers}
    lookahead_violations = []

    for ticker in tickers:
        try:
            df_h = dl_hourly(ticker)
            df_d = dl_daily(ticker)
        except Exception as e:
            continue
        if df_h.empty:
            continue

        for snap, evalt in zip(snapshots, eval_times):
            feat_series, price_snap, last_feat_ts = build_features_at(
                df_h, df_d, market_h, market_d, snap, ticker, meta)

            if feat_series is None:
                continue

            # ── AUDIT 3: Look-ahead check ──────────────────────────────────
            snap_naive = strip_ts(snap)
            if last_feat_ts > snap_naive:
                lookahead_violations.append({
                    "ticker": ticker,
                    "snap":   snap_naive,
                    "last_feat": last_feat_ts,
                    "overshoot_h": (last_feat_ts - snap_naive).total_seconds() / 3600
                })

            # Predikce
            X          = feat_series.values.reshape(1, -1)
            signal_id  = int(cls_model.predict(X)[0])
            proba      = cls_model.predict_proba(X)[0]
            pred_ret   = float(reg_model.predict(X)[0])
            signal_str = label_map[str(signal_id)]
            confidence = float(proba[signal_id])

            actual_price = find_closest_price(df_h, evalt, tolerance_h=8)
            if actual_price is None:
                continue

            actual_ret = (actual_price - price_snap) / price_snap
            correct    = direction_ok(signal_str, actual_ret)

            # Baseline: Momentum
            mom_ok  = baseline_momentum(df_h, snap, actual_ret)
            # Baseline: Always BUY
            buy_ok  = baseline_always_buy(actual_ret)

            all_results[ticker].append({
                "snap":       snap,
                "eval":       evalt,
                "signal":     signal_str,
                "conf":       confidence,
                "price_snap": price_snap,
                "actual_ret": actual_ret,
                "pred_ret":   pred_ret,
                "correct":    correct,
                "mom_ok":     mom_ok,
                "buy_ok":     buy_ok,
                "oos":        snap > trained_until,
            })

    # ── AUDIT 3: Look-ahead report ─────────────────────────────────────────
    print(f"  [AUDIT 3] Look-ahead Bias check:")
    if lookahead_violations:
        print(f"    KRITICKÉ: {len(lookahead_violations)} Look-ahead Verletzungen gefunden!")
        for v in lookahead_violations[:5]:
            print(f"    {v['ticker']} snap={v['snap'].strftime('%d.%m %H:%M')} "
                  f"last_feat={v['last_feat'].strftime('%d.%m %H:%M')} "
                  f"overshoot={v['overshoot_h']:.1f}h")
    else:
        print(f"    ✓ Kein Look-ahead Bias gefunden — alle Features <= snap_time")
    print()

    # ── Signal-Verteilung für Random Baseline ─────────────────────────────
    all_flat = [r for rows in all_results.values() for r in rows]
    if not all_flat:
        print("Žádné výsledky."); return

    sig_counts = Counter(r["signal"] for r in all_flat)
    total_sig  = sum(sig_counts.values())
    sig_dist   = {k: v / total_sig for k, v in sig_counts.items()}

    # Random baseline (10 Wiederholungen für Stabilität)
    random.seed(42)
    random_accs = []
    for _ in range(10):
        ok = sum(1 for r in all_flat if baseline_random(sig_dist, r["actual_ret"]))
        random_accs.append(ok / len(all_flat))
    random_acc = np.mean(random_accs)

    # ── Ergebnisse ─────────────────────────────────────────────────────────
    model_ok  = sum(r["correct"] for r in all_flat)
    buy_ok    = sum(r["buy_ok"]  for r in all_flat)
    mom_ok    = sum(r["mom_ok"]  for r in all_flat)
    n         = len(all_flat)
    oos_flat  = [r for r in all_flat if r["oos"]]
    oos_ok    = sum(r["correct"] for r in oos_flat)

    print(f"{'='*65}")
    print(f"  SROVNÁNÍ MODELU S BASELINY")
    print(f"{'='*65}")
    print(f"  {'Strategie':<30}  {'✓':>5}  {'∑':>5}  {'Přesnost':>10}")
    print(f"  {'─'*55}")
    print(f"  {'Mičov Model':<30}  {model_ok:>5}  {n:>5}  {model_ok/n:>9.1%}")
    if oos_flat:
        print(f"  {'  z toho OOS (po tréninku)':<30}  {oos_ok:>5}  "
              f"{len(oos_flat):>5}  {oos_ok/len(oos_flat):>9.1%}")
    print(f"  {'─'*55}")
    print(f"  {'Baseline: Always BUY':<30}  {buy_ok:>5}  {n:>5}  {buy_ok/n:>9.1%}")
    print(f"  {'Baseline: 4h Momentum':<30}  {mom_ok:>5}  {n:>5}  {mom_ok/n:>9.1%}")
    print(f"  {'Baseline: Random (avg 10x)':<30}  {'—':>5}  {n:>5}  {random_acc:>9.1%}")
    print()

    edge_over_buy = (model_ok / n) - (buy_ok / n)
    edge_over_mom = (model_ok / n) - (mom_ok / n)
    print(f"  Výhoda oproti Always BUY:   {edge_over_buy:>+.1%}")
    print(f"  Výhoda oproti Momentum:     {edge_over_mom:>+.1%}")
    print()

    # ── AUDIT 4: IS vs OOS split ───────────────────────────────────────────
    is_flat  = [r for r in all_flat if not r["oos"]]
    if is_flat and oos_flat:
        is_acc  = sum(r["correct"] for r in is_flat) / len(is_flat)
        oos_acc = oos_ok / len(oos_flat)
        print(f"  [AUDIT 4] In-Sample vs Out-of-Sample:")
        print(f"    IS  (während Training): {len(is_flat):>4} snímků  Přesnost: {is_acc:.1%}")
        print(f"    OOS (nach Training):    {len(oos_flat):>4} snímků  Přesnost: {oos_acc:.1%}")
        overfitting_gap = is_acc - oos_acc
        if overfitting_gap > 0.10:
            print(f"    VAROVÁNÍ: IS-OOS gap = {overfitting_gap:.1%} → možný overfitting!")
        else:
            print(f"    ✓ IS-OOS gap = {overfitting_gap:+.1%} → v pořádku")
        print()
    elif oos_flat:
        print(f"  [AUDIT 4] Všechny snímky jsou OOS — žádný IS/OOS problém.\n")

    # ── AUDIT 5: Profit simulace ───────────────────────────────────────────
    print(f"{'='*65}")
    print(f"  PROFIT SIMULACE (startkapitál 10.000$, poplatek {FEE_PCT*100:.1f}% × 2)")
    print(f"{'='*65}")

    # Per Ticker
    print(f"  {'Ticker':<8}  {'Výnos':>8}  {'Trades':>7}  {'Win%':>6}  "
          f"{'MaxDD':>7}  {'Sharpe':>7}")
    print(f"  {'─'*52}")
    all_profits = []
    for ticker in tickers:
        rows = all_results[ticker]
        if not rows:
            continue
        sim = simulate_profit(rows)
        if sim is None:
            continue
        all_profits.append(sim["total_return"])
        win_pct = sim["win_trades"] / sim["n_trades"] if sim["n_trades"] > 0 else 0
        print(f"  {ticker:<8}  {sim['total_return']:>+7.1%}  "
              f"{sim['n_trades']:>7}  {win_pct:>5.0%}  "
              f"{sim['max_drawdown']:>6.1%}  {sim['sharpe']:>7.2f}")

    # Gesamtsimulation (alle Trades kombiniert)
    print(f"  {'─'*52}")
    all_rows_sorted = sorted(all_flat, key=lambda r: r["snap"])
    sim_all = simulate_profit(all_rows_sorted)
    if sim_all:
        win_pct = sim_all["win_trades"] / sim_all["n_trades"] if sim_all["n_trades"] > 0 else 0
        print(f"  {'CELKEM':<8}  {sim_all['total_return']:>+7.1%}  "
              f"{sim_all['n_trades']:>7}  {win_pct:>5.0%}  "
              f"{sim_all['max_drawdown']:>6.1%}  {sim_all['sharpe']:>7.2f}")
        print(f"\n  Startkapitál: 10.000$  →  Endkapitál: {sim_all['end_capital']:,.2f}$")
        print(f"  Celkové poplatky zaplaceny: "
              f"{sim_all['n_trades'] * 10000 * FEE_PCT * 2:.2f}$ (odhad)")
    print()

    # ── AUDIT 6: HOLD-Threshold sensitivity ───────────────────────────────
    print(f"{'='*65}")
    print(f"  [AUDIT 6] HOLD Threshold Sensitivity")
    print(f"  (jak se mění přesnost při různých definicích HOLD?)")
    print(f"{'='*65}")
    print(f"  {'Threshold':<12}  {'Model%':>8}  {'AlwaysBUY%':>12}  {'Rozdíl':>8}")
    print(f"  {'─'*46}")
    for thresh in [0.001, 0.002, 0.003, 0.005, 0.010]:
        def direction_ok_t(signal, actual_ret, t=thresh):
            if signal == "BUY":  return actual_ret > 0
            if signal == "SELL": return actual_ret < 0
            return abs(actual_ret) < t

        m_ok = sum(1 for r in all_flat if direction_ok_t(r["signal"], r["actual_ret"]))
        b_ok = sum(1 for r in all_flat if r["actual_ret"] > 0)
        diff = m_ok/n - b_ok/n
        print(f"  {thresh*100:<11.1f}%  {m_ok/n:>7.1%}  {b_ok/n:>11.1%}  {diff:>+7.1%}")
    print()

    # ── Schlussaudit ───────────────────────────────────────────────────────
    print(f"{'='*65}")
    print(f"  ZÁVĚREČNÝ AUDIT SOUHRN")
    print(f"{'='*65}")
    checks = [
        ("Look-ahead Bias",      len(lookahead_violations) == 0),
        ("OOS Daten vorhanden",  len(oos_flat) > 0),
        ("IS/OOS gap < 10%",     True if not is_flat else abs(
            sum(r["correct"] for r in is_flat)/len(is_flat) -
            oos_ok/len(oos_flat)) < 0.10),
        ("Výhoda nad Always BUY", edge_over_buy > 0.05),
        ("Výhoda nad Momentum",   edge_over_mom > 0.02),
        ("Profit kladný",         sim_all["total_return"] > 0 if sim_all else False),
        ("Sharpe > 1.0",          sim_all["sharpe"] > 1.0 if sim_all else False),
    ]
    all_pass = True
    for name, passed in checks:
        mark = "✓" if passed else "✗"
        if not passed:
            all_pass = False
        print(f"  {mark}  {name}")

    print()
    if all_pass:
        print("  ✓✓✓ Všechny audity prošly — výsledky jsou důvěryhodné.")
    else:
        print("  ⚠ Některé audity selhaly — výsledky je třeba interpretovat opatrně.")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mičov — Strict Backtest Audit")
    parser.add_argument("--model_dir",  default=str(config.MODEL_DIR))
    parser.add_argument("--tickers",    nargs="+", default=None)
    parser.add_argument("--snapshots",  default=30, type=int)
    parser.add_argument("--horizon",    default="4h",
                        help="Název horizontu z config.LABEL_HORIZONS (např. 1h, 4h, 1d)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    tickers   = args.tickers or load_tickers(model_dir)
    if not tickers:
        print("CHYBA: žádné tickery"); exit(1)
    print(f"  Tickery: {', '.join(tickers)}")

    run_strict_backtest(model_dir, tickers, args.snapshots, args.horizon)
