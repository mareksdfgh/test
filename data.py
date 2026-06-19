"""
data.py
=======
Sdílená vrstva pro stahování a slučování dat. JEDINÝ zdroj pravdy pro
získávání OHLCV — importují odsud build_dataset.py, run.py i test.py,
takže trénink a inference používají BITOVĚ stejnou logiku (žádný
train/serve skew).

Klíčové opravy oproti původní verzi:
  • merge_market_hourly() používá pd.merge_asof(direction="backward",
    allow_exact_matches=False) → robustní vůči nesouladu timestampů,
    nikdy nepoužije tržní svíčku ze stejné nebo budoucí hodiny.
  • merge_daily_to_hourly() lupne denní příznaky o 1 obchodní den dozadu
    (shift(1)) → na den D vidíme jen denní hodnoty z D-1, žádný
    same-day leak z denního Close.
  • Stažená data se cachují na disk (parquet) s TTL → opakované běhy
    netahají ~50 tickerů znovu.
"""

import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import config

# yfinance importujeme líně uvnitř dl_* funkcí → merge utility jdou
# používat (a testovat) i bez nainstalovaného yfinance.
# Warningy NEumlčujeme globálně — jen konkrétní známý šum.
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")


# ── Cache ───────────────────────────────────────────────────────────────────

def _cache_path(ticker: str, interval: str):
    safe = ticker.replace("^", "_").replace("=", "_").replace(".", "_").replace("-", "_")
    return config.DATA_DIR / f"{safe}_{interval}.parquet"


def _fresh(path) -> bool:
    """Je cache soubor mladší než CACHE_TTL_HOURS?"""
    if not path.exists():
        return False
    age_h = (time.time() - path.stat().st_mtime) / 3600.0
    return age_h < config.CACHE_TTL_HOURS


def _normalize(df: pd.DataFrame, intraday: bool) -> pd.DataFrame:
    """Sjednotí sloupce (MultiIndex → flat) a index na naive datetime."""
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if intraday:
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    else:
        df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


# ── Stahování ───────────────────────────────────────────────────────────────

def dl_hourly(ticker: str, use_cache: bool = True) -> pd.DataFrame:
    """Stáhne hodinová OHLCV data (s cache)."""
    import yfinance as yf
    path = _cache_path(ticker, "1h")
    if use_cache and _fresh(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            pass

    df = yf.download(ticker, period=config.HOURLY_PERIOD, interval="1h",
                     auto_adjust=True, progress=False)
    df = _normalize(df, intraday=True)
    if not df.empty and use_cache:
        try:
            df.to_parquet(path)
        except Exception:
            pass
    return df


def dl_daily(ticker: str, use_cache: bool = True) -> pd.DataFrame:
    """Stáhne denní OHLCV data za posledních DAILY_YEARS let (s cache)."""
    import yfinance as yf
    path = _cache_path(ticker, "1d")
    if use_cache and _fresh(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            pass

    start = (datetime.now() - timedelta(days=config.DAILY_YEARS * 365)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, interval="1d",
                     auto_adjust=True, progress=False)
    df = _normalize(df, intraday=False)
    if not df.empty and use_cache:
        try:
            df.to_parquet(path)
        except Exception:
            pass
    return df


# ── Slučování (bez look-ahead) ──────────────────────────────────────────────

def merge_daily_to_hourly(df_hourly: pd.DataFrame,
                          df_daily: pd.DataFrame,
                          prefix: str) -> pd.DataFrame:
    """
    Přidá denní příznaky k hodinovým datům BEZ same-day leaku.

    Oprava: denní příznaky se posunou o 1 obchodní den (shift(1)), takže
    na obchodní den D model vidí pouze denní hodnoty uzavřené v den D-1.
    Dříve se připojoval celý dnešní denní řádek (včetně dnešního Close) na
    každou hodinu téhož dne → leak.
    """
    if df_daily.empty:
        return df_hourly

    df_daily = df_daily.copy()
    df_daily.index = pd.to_datetime(df_daily.index).normalize()
    df_daily = df_daily[~df_daily.index.duplicated(keep="last")].sort_index()

    # KLÍČOVÉ: posun o jeden obchodní den dozadu.
    df_daily = df_daily.shift(1)

    # Přejmenování kolizních sloupců.
    overlap = [c for c in df_daily.columns if c in df_hourly.columns]
    df_daily = df_daily.rename(columns={c: f"{c}__{prefix}" for c in overlap})

    orig_idx = df_hourly.index.copy()
    date_idx = pd.to_datetime(df_hourly.index).normalize()

    out = df_hourly.copy()
    out.index = date_idx
    out = out.join(df_daily, how="left")
    out.index = orig_idx
    return out


def merge_market_hourly(df_stock: pd.DataFrame,
                        df_market: pd.DataFrame,
                        name: str,
                        tolerance: str = None) -> pd.DataFrame:
    """
    Přidá hodinové tržní příznaky pomocí merge_asof (backward).

    Oprava: místo slepého shift(1) + join (který selhává při nesouladu
    timestampů) použijeme asof-merge. Pro každý timestamp T akcie vezme
    POSLEDNÍ dostupnou tržní svíčku STRIKTNĚ PŘED T (allow_exact_matches=
    False) v rámci tolerance. → robustní vůči mezerám a nikdy neleakuje
    svíčku ze stejné/budoucí hodiny.
    """
    if df_market.empty:
        return df_stock

    tolerance = tolerance or config.MARKET_MERGE_TOLERANCE

    overlap = [c for c in df_market.columns if c in df_stock.columns]
    df_market = df_market.rename(columns={c: f"{c}__{name}" for c in overlap})

    left  = df_stock.sort_index()
    right = df_market.sort_index()

    merged = pd.merge_asof(
        left, right,
        left_index=True, right_index=True,
        direction="backward",
        allow_exact_matches=False,
        tolerance=pd.Timedelta(tolerance),
    )
    merged.index = left.index
    return merged
