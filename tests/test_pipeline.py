"""
tests/test_pipeline.py
======================
Jednotkové testy nad SYNTETICKÝMI daty (žádná síť, žádné yfinance).
Ověřují hlavně to, co snadno tiše regresuje: absenci look-ahead leaku
v mergích a labelech.

Spuštění:
  pytest -q
"""

import numpy as np
import pandas as pd
import pytest

import config
from features import add_features, rsi
from labels import make_labels, future_return, max_horizon
from data import merge_daily_to_hourly, merge_market_hourly


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _hourly_ohlcv(n=400, start="2024-01-01 09:30", freq="1h", seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq=freq)
    price = 100 + np.cumsum(rng.normal(0, 0.5, n))
    df = pd.DataFrame({
        "Open":   price + rng.normal(0, 0.1, n),
        "High":   price + np.abs(rng.normal(0, 0.3, n)),
        "Low":    price - np.abs(rng.normal(0, 0.3, n)),
        "Close":  price,
        "Volume": rng.integers(1e5, 1e6, n).astype(float),
    }, index=idx)
    return df


def _daily_ohlcv(n=60, start="2023-12-01", seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="1D")
    price = 100 + np.cumsum(rng.normal(0, 1.0, n))
    return pd.DataFrame({
        "Open": price, "High": price + 1, "Low": price - 1,
        "Close": price, "Volume": rng.integers(1e6, 9e6, n).astype(float),
    }, index=idx)


# ── Příznaky ──────────────────────────────────────────────────────────────────

def test_add_features_drops_raw_ohlcv():
    df = add_features(_hourly_ohlcv(), prefix="")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        assert col not in df.columns
    assert df.shape[1] > 30          # ~60 příznaků


def test_rsi_in_range():
    s = _hourly_ohlcv()["Close"]
    r = rsi(s, 14).dropna()
    assert (r >= 0).all() and (r <= 100).all()


# ── Labely (žádný look-ahead u feature, cíl je v budoucnu) ─────────────────────

def test_future_return_is_forward_looking():
    close = pd.Series(np.arange(1, 21, dtype=float))
    fr = future_return(close, 4)
    # ret na indexu i = (close[i+4]/close[i] - 1); poslední 4 jsou NaN
    assert np.isnan(fr.iloc[-4:]).all()
    expected = close.iloc[4] / close.iloc[0] - 1
    assert fr.iloc[0] == pytest.approx(expected)


def test_make_labels_has_all_horizons():
    close = _hourly_ohlcv()["Close"]
    lab = make_labels(close)
    for name in config.LABEL_HORIZONS:
        assert f"label_{name}" in lab.columns
        assert f"ret_{name}" in lab.columns
        assert set(lab[f"label_{name}"].dropna().unique()) <= {0, 1, 2}


def test_max_horizon():
    assert max_horizon() == max(config.LABEL_HORIZONS.values())


# ── Merge bez look-ahead ───────────────────────────────────────────────────────

def test_daily_merge_no_same_day_leak():
    """
    Denní příznaky musí na den D pocházet z D-1. Vložíme do denního Close
    monotónně rostoucí hodnotu = pořadové číslo dne; po mergi musí hodinový
    řádek dne D nést hodnotu dne D-1 (tj. o 1 nižší), ne dne D.
    """
    h = _hourly_ohlcv(n=200, start="2024-01-02 09:30")
    h_feat = add_features(h.copy(), prefix="")

    # Denní rámec, kde "d_close_raw" = index dne (0,1,2,...).
    days = pd.date_range("2024-01-01", periods=20, freq="1D")
    d = pd.DataFrame({"d_marker": np.arange(len(days), dtype=float)}, index=days)

    merged = merge_daily_to_hourly(h_feat, d, prefix="dtest")
    assert "d_marker" in merged.columns

    # Pro každý den D je hodnota markeru = (index D) - 1.
    day_index = {ts.normalize(): i for i, ts in enumerate(days)}
    sample = merged.dropna(subset=["d_marker"]).iloc[len(merged)//2]
    d_norm = sample.name.normalize()
    if d_norm in day_index:
        assert sample["d_marker"] == pytest.approx(day_index[d_norm] - 1)


def test_market_merge_uses_only_past_bar():
    """
    merge_market_hourly (asof backward, allow_exact_matches=False) nesmí
    nikdy použít tržní svíčku ze STEJNÉ nebo budoucí hodiny.
    """
    h = _hourly_ohlcv(n=100, start="2024-01-02 09:30")
    h_feat = add_features(h.copy(), prefix="")

    # Tržní rámec se stejným indexem, marker = pořadí svíčky.
    mkt_idx = h.index
    mkt = pd.DataFrame({"mkt_marker": np.arange(len(mkt_idx), dtype=float)},
                       index=mkt_idx)

    merged = merge_market_hourly(h_feat, mkt, "m",
                                 tolerance="48h")
    # Na pozici i musí být marker <= i-1 (striktně minulost).
    pos = {ts: i for i, ts in enumerate(h_feat.index)}
    vals = merged["mkt_marker"].dropna()
    for ts, v in vals.items():
        assert v <= pos[ts] - 1


def test_market_merge_index_preserved():
    h = _hourly_ohlcv(n=50)
    h_feat = add_features(h.copy(), prefix="")
    mkt = pd.DataFrame({"x": np.arange(len(h), dtype=float)}, index=h.index)
    merged = merge_market_hourly(h_feat, mkt, "m")
    assert merged.index.equals(h_feat.index)
