"""
features.py
===========
Společný modul pro výpočet technických indikátorů.
Importován v build_dataset.py, train.py i run.py — jedna definice, žádná duplikace.
"""

import numpy as np
import pandas as pd


# ── Technické indikátory ───────────────────────────────────────────────────────

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relativní index síly (RSI)."""
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26,
         signal: int = 9):
    """MACD: rychlá EMA - pomalá EMA, signální linie, histogram."""
    ef = series.ewm(span=fast, adjust=False).mean()
    es = series.ewm(span=slow, adjust=False).mean()
    ml = ef - es
    sl = ml.ewm(span=signal, adjust=False).mean()
    return ml, sl, ml - sl


def bollinger(series: pd.Series, period: int = 20) -> pd.Series:
    """Bollinger Bands — vrátí Z-score pozici ceny v pásu."""
    ma  = series.rolling(period).mean()
    std = series.rolling(period).std()
    return (series - ma) / std.replace(0, np.nan)


def stoch_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Stochastický RSI — normalizovaný RSI na rozsah 0–1."""
    r  = rsi(series, period)
    mn = r.rolling(period).min()
    mx = r.rolling(period).max()
    return (r - mn) / (mx - mn).replace(0, np.nan)


# ── Hlavní funkce pro výpočet příznaků ────────────────────────────────────────

def add_features(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """
    Přidá ~60 technických příznaků ke OHLCV DataFrame.

    Opravy oproti starší verzi:
      - mom_5 / mom_20 jsou nyní pct_change (ne absolutní rozdíl)
        → srovnatelné napříč různými cenovými hladinami
      - Všechny přídavné sloupce mají prefix pro kolize při mergi
    """
    c = df["Close"]
    p = prefix

    # Výnosy (procentuální změna za N period)
    for n in [1, 2, 4, 8, 16, 32]:
        df[f"{p}ret_{n}"] = c.pct_change(n)

    # Vzdálenost od klouzavých průměrů (normalizováno cenou)
    for n in [5, 10, 20, 50, 100, 200]:
        ma = c.rolling(n).mean()
        df[f"{p}ma{n}_dist"] = (c - ma) / c

    # Vzdálenost od exponenciálních klouzavých průměrů
    for n in [9, 21, 55]:
        ema = c.ewm(span=n, adjust=False).mean()
        df[f"{p}ema{n}_dist"] = (c - ema) / c

    # Oscilátory
    df[f"{p}rsi_7"]     = rsi(c, 7)
    df[f"{p}rsi_14"]    = rsi(c, 14)
    df[f"{p}rsi_21"]    = rsi(c, 21)
    df[f"{p}stoch_rsi"] = stoch_rsi(c, 14)
    df[f"{p}bb_z"]      = bollinger(c, 20)

    ml, sl, mh = macd(c)
    df[f"{p}macd"]          = ml
    df[f"{p}macd_signal"]   = sl
    df[f"{p}macd_hist"]     = mh
    df[f"{p}macd_hist_chg"] = mh.diff()

    # Volatilita (klouzavá směrodatná odchylka výnosů)
    ret = c.pct_change()
    for n in [8, 20, 50]:
        df[f"{p}vol_{n}"] = ret.rolling(n).std()

    # High/Low poměry (pokud jsou k dispozici)
    if "High" in df.columns and "Low" in df.columns:
        hl_range = (df["High"] - df["Low"]).replace(0, np.nan)
        df[f"{p}hl_ratio"]  = (df["High"] - df["Low"]) / c
        df[f"{p}close_pos"] = (c - df["Low"]) / hl_range

    # Objem (normalizovaný)
    if "Volume" in df.columns:
        vol = df["Volume"]
        df[f"{p}vol_ma5_ratio"]  = vol / vol.rolling(5).mean().replace(0, np.nan)
        df[f"{p}vol_ma20_ratio"] = vol / vol.rolling(20).mean().replace(0, np.nan)
        # Spike detekce: objem > průměr + 2× std
        vol_mean = vol.rolling(20).mean()
        vol_std  = vol.rolling(20).std()
        df[f"{p}vol_spike"] = (vol > vol_mean + 2 * vol_std).astype(int)

    # Momentum — pct_change místo absolutního rozdílu (oprava!)
    df[f"{p}mom_5"]  = c.pct_change(5)
    df[f"{p}mom_20"] = c.pct_change(20)

    # Průměrný skutečný rozsah (zjednodušené ATR)
    hl = df["High"] - df["Low"] if "High" in df.columns else c.diff().abs()
    df[f"{p}atr_14"] = hl.rolling(14).mean() / c

    # Odstranění surových OHLCV sloupců
    drop_cols = [col for col in
                 ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]
                 if col in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    return df
