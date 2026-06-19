"""
labels.py
=========
Tvorba labelů pro VÍCE horizontů najednou.

Pro každý horizont h (v hodinových svíčkách) vytvoříme:
  label_<h>  ∈ {0=SELL, 1=HOLD, 2=BUY}   — klasifikace
  ret_<h>    ∈ ℝ                          — budoucí výnos (regrese)

Bez look-ahead: cílem je BUDOUCÍ výnos, tj. close.pct_change(h).shift(-h).
Poslední max(h) řádků se v build_dataset.py zahodí (future return chybí).
"""

import pandas as pd

import config


def future_return(close: pd.Series, horizon: int) -> pd.Series:
    """Výnos za příštích `horizon` svíček, zarovnaný na aktuální řádek."""
    return close.pct_change(horizon).shift(-horizon)


def make_labels(close: pd.Series, horizons: dict = None,
                thresholds: dict = None) -> pd.DataFrame:
    """
    Vrátí DataFrame se sloupci label_<name> a ret_<name> pro každý horizont.
    """
    horizons   = horizons   or config.LABEL_HORIZONS
    thresholds = thresholds or config.LABEL_THRESHOLDS

    out = pd.DataFrame(index=close.index)
    for name, h in horizons.items():
        thr = thresholds.get(name, 0.003)
        ret = future_return(close, h)
        lab = pd.Series(1, index=close.index)   # HOLD
        lab[ret >  thr] = 2                      # BUY
        lab[ret < -thr] = 0                      # SELL
        out[f"ret_{name}"]   = ret
        out[f"label_{name}"] = lab
    return out


def max_horizon(horizons: dict = None) -> int:
    horizons = horizons or config.LABEL_HORIZONS
    return max(horizons.values())
