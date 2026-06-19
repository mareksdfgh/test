"""
config.py
=========
Centrální konfigurace celého projektu — jedna pravda, žádné duplikované
konstanty roztroušené po souborech.

Sem patří: seznamy tickerů, horizonty predikce, prahy labelů, cesty,
parametry stahování a cache. Importují odsud build_dataset.py, train.py,
run.py, tracker.py i test.py.
"""

from pathlib import Path

# ── Cesty ───────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent
DATA_DIR   = ROOT_DIR / "data_cache"      # cache stažených OHLCV dat
MODEL_DIR  = ROOT_DIR / "models"          # uložené modely + metadata
OUTPUT_DIR = ROOT_DIR / "outputs"         # grafy, exporty

DATASET_CSV = ROOT_DIR / "dataset.csv"

for _d in (DATA_DIR, MODEL_DIR, OUTPUT_DIR):
    _d.mkdir(exist_ok=True)


# ── Tickery ─────────────────────────────────────────────────────────────────
# 50 likvidních US akcií napříč nejrůznějšími odvětvími. Toto je "domácí"
# sada, na kterou se model SPECIALIZUJE (každý ticker má vlastní one-hot
# příznak). Model umí predikovat i ticker MIMO tento seznam — pak se opírá
# jen o obecné technické/tržní příznaky (viz run.py, generalizace).
DEFAULT_TICKERS = [
    # Tech / polovodiče / software
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO",
    "ORCL", "ADBE", "CRM", "AMD", "INTC", "QCOM", "CSCO",
    # Komunikace / média
    "NFLX", "DIS", "CMCSA", "VZ", "T",
    # Finance
    "JPM", "BAC", "GS", "MS", "WFC", "V", "MA", "AXP",
    # Healthcare
    "JNJ", "UNH", "PFE", "LLY", "ABBV", "MRK", "TMO",
    # Spotřební
    "WMT", "HD", "MCD", "KO", "PEP", "NKE", "COST", "PG",
    # Průmysl
    "BA", "CAT", "GE", "HON",
    # Energie
    "XOM", "CVX",
    # Materiály / těžba (měď)
    "FCX",
]

# Makro / tržní kontext — stahuje se jednou, sdílí se napříč tickery.
MARKET_TICKERS = {
    "sp500":  "^GSPC",
    "nasdaq": "^IXIC",
    "dow":    "^DJI",
    "russ":   "^RUT",
    "dax":    "^GDAXI",
    "vix":    "^VIX",
    "dxy":    "DX-Y.NYB",
    "bonds":  "^TNX",      # 10Y výnos
    "bonds2": "^FVX",      # 5Y výnos
    "btc":    "BTC-USD",   # riziková nálada / 24-7 likvidita
}

# Suroviny / komodity jako vstupní kontext (zlato, stříbro, měď, ropa, ...).
# Pozn.: čistý "silicium" futures na yfinance neexistuje → pokrýváme ho
# proxy přes lithium/baterie (LIT) a polovodičový sektor (XLK v SECTOR_ETFS).
COMMODITIES = {
    "gold":      "GC=F",
    "silver":    "SI=F",
    "copper":    "HG=F",
    "platinum":  "PL=F",
    "palladium": "PA=F",
    "oil":       "CL=F",
    "natgas":    "NG=F",
    "lithium_etf": "LIT",   # baterie / "tech suroviny" (proxy za silicium)
    "uranium_etf": "URA",   # jaderné palivo
    "metals_etf":  "DBB",   # koš základních kovů
}

SECTOR_ETFS = {
    "tech_etf":     "XLK",
    "finance_etf":  "XLF",
    "health_etf":   "XLV",
    "energy_etf":   "XLE",
    "consumer_etf": "XLY",
    "staples_etf":  "XLP",
    "industrial_etf": "XLI",
    "utilities_etf": "XLU",
    "realestate_etf": "XLRE",
    "materials_etf":  "XLB",
}

# Veškerý sdílený kontext (makro + komodity + sektory) — stahuje se jednou.
CONTEXT_TICKERS = {**MARKET_TICKERS, **COMMODITIES, **SECTOR_ETFS}


# ── Horizonty predikce ──────────────────────────────────────────────────────
# Hodnota = počet HODINOVÝCH svíček dopředu. US akcie mají ~7 obchodních
# hodin/den, takže:
#   1h  ≈ příští hodina        (krátký intradenní šum)
#   4h  ≈ konec obchodního dne
#   1d  ≈ příští obchodní den  (~7 svíček)
#   1w  ≈ příští týden         (~35 svíček)
# Model trénuje VŠECHNY horizonty nad stejnými příznaky → učí se vztah mezi
# krátkodobým (hodinovým) a dlouhodobým (denním/týdenním) pohybem.
LABEL_HORIZONS = {
    "1h": 1,
    "4h": 4,
    "1d": 7,
    "1w": 35,
}

# Práh pro klasifikaci BUY/SELL na daném horizontu (absolutní výnos).
# Delší horizont → větší očekávaný pohyb → vyšší práh.
LABEL_THRESHOLDS = {
    "1h": 0.0015,
    "4h": 0.003,
    "1d": 0.006,
    "1w": 0.015,
}

# Výchozí horizonty pro trénink (lze přepsat přes --horizons).
# Záměrně pokrývá hodinový i denní rozsah, ale ne všechny čtyři naráz
# (4 × 2 modely × stacking je výpočetně drahé).
DEFAULT_TRAIN_HORIZONS = ["1h", "4h", "1d"]


# ── Parametry stahování dat ─────────────────────────────────────────────────
HOURLY_PERIOD = "730d"   # yfinance limit pro 1h interval
DAILY_YEARS   = 20

# Tolerance pro merge_asof tržních příznaků (musí překlenout noční mezeru
# ~17,5 h mezi 16:00 a 9:30 následujícího dne).
MARKET_MERGE_TOLERANCE = "26h"

# Cache: jak dlouho je stažený soubor považován za čerstvý (hodiny).
CACHE_TTL_HOURS = 6


# ── Dataset / kvalita ───────────────────────────────────────────────────────
# Sloupce, které NEJSOU příznaky (metadata, labely).
META_COLS = ["ticker", "close_raw"]

# Maximální podíl chybějících hodnot v ŘÁDKU, jinak se řádek zahodí.
# (Dříve dropna(thresh=60%) propouštěl polovičatě vyplněné řádky.)
MAX_ROW_NAN_FRAC = 0.20


# ── Backtest / profit simulace ──────────────────────────────────────────────
FEE_PCT     = 0.001   # 0.1 % poplatek na obchod
HOLD_THRESH = 0.002   # ±0.2 % → směr "flat" je správný při HOLD
