"""
tracker.py
==========
Rozšířený engine pro sledování přesnosti předpovědí.

  1h vyhodnocení:  Je směr po 1 hodině správný?
  4h vyhodnocení:  Je směr po 4 hodinách správný? (hlavní label horizont)
  Podobnost grafu: Jak podobná je předpovídaná cenová trajektorie skutečné?
                   (kosinová podobnost + MAE)

Schéma tabulky predictions:
  ticker, ts_predicted, price_at_pred, signal, confidence, pred_return
  -- 1h vyhodnocení --
  price_1h, actual_return_1h, direction_ok_1h, evaluated_1h
  -- 4h vyhodnocení --
  price_4h, actual_return_4h, direction_ok_4h, evaluated_4h
  -- Podobnost grafu (po 4h datech kurzu) --
  chart_cosine, chart_mae, chart_evaluated
"""

import sqlite3
import threading
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

import config

HOLD_THRESH = config.HOLD_THRESH        # ±0.2% → HOLD je správný
DB_PATH     = str(config.ROOT_DIR / "micov_tracker.db")


class PredictionTracker:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._lock   = threading.Lock()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        """Vytvoří nové spojení s SQLite databází."""
        return sqlite3.connect(self.db_path, timeout=10,
                                check_same_thread=False)

    def _init_db(self):
        """Inicializuje schéma databáze + migrace starých DB bez nových sloupců."""
        with self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker           TEXT NOT NULL,
                    ts_predicted     TEXT NOT NULL,
                    price_at_pred    REAL NOT NULL,
                    signal           TEXT NOT NULL,
                    confidence       REAL NOT NULL,
                    pred_return      REAL NOT NULL,

                    price_1h         REAL,
                    actual_ret_1h    REAL,
                    direction_ok_1h  INTEGER DEFAULT 0,
                    evaluated_1h     INTEGER DEFAULT 0,

                    price_4h         REAL,
                    actual_ret_4h    REAL,
                    direction_ok_4h  INTEGER DEFAULT 0,
                    evaluated_4h     INTEGER DEFAULT 0,

                    chart_cosine     REAL,
                    chart_mae        REAL,
                    chart_evaluated  INTEGER DEFAULT 0
                )
            """)
            # Migrace — přidání sloupců pokud starší verze DB chybí
            existing = [r[1] for r in c.execute(
                "PRAGMA table_info(predictions)").fetchall()]
            for col, typedef in [
                ("price_1h",        "REAL"),
                ("actual_ret_1h",   "REAL"),
                ("direction_ok_1h", "INTEGER DEFAULT 0"),
                ("evaluated_1h",    "INTEGER DEFAULT 0"),
                ("price_4h",        "REAL"),
                ("actual_ret_4h",   "REAL"),
                ("direction_ok_4h", "INTEGER DEFAULT 0"),
                ("evaluated_4h",    "INTEGER DEFAULT 0"),
                ("chart_cosine",    "REAL"),
                ("chart_mae",       "REAL"),
                ("chart_evaluated", "INTEGER DEFAULT 0"),
            ]:
                if col not in existing:
                    c.execute(f"ALTER TABLE predictions ADD COLUMN {col} {typedef}")

            # Indexy pro rychlejší dotazy
            c.execute("""
                CREATE INDEX IF NOT EXISTS idx_eval1h
                ON predictions(evaluated_1h, ticker)
            """)
            c.execute("""
                CREATE INDEX IF NOT EXISTS idx_eval4h
                ON predictions(evaluated_4h, ticker)
            """)
            c.commit()

    # ── Uložení předpovědi ─────────────────────────────────────────────────────

    def log_prediction(self, ticker: str, price: float, signal: str,
                       confidence: float, pred_return: float, ts=None):
        """Uloží novou předpověď do databáze."""
        ts = ts or datetime.now()
        with self._lock:
            with self._conn() as c:
                c.execute("""
                    INSERT INTO predictions
                    (ticker, ts_predicted, price_at_pred, signal,
                     confidence, pred_return)
                    VALUES (?,?,?,?,?,?)
                """, (ticker, ts.isoformat(), price,
                      signal, confidence, pred_return))
                c.commit()

    # ── Vyhodnocení ────────────────────────────────────────────────────────────

    def evaluate_pending(self) -> list:
        """
        Vyhodnotí čekající předpovědi:
          - 1h: předpovědi starší než 1 hodina
          - 4h: předpovědi starší než 4 hodiny + podobnost grafu
        Vrátí seznam nově vyhodnocených záznamů.
        """
        now   = datetime.now()
        cut1h = (now - timedelta(hours=1)).isoformat()
        cut4h = (now - timedelta(hours=4)).isoformat()
        results = []

        # Zjistíme všechny tickery, které potřebují vyhodnocení
        with self._conn() as c:
            rows_1h = c.execute("""
                SELECT id, ticker, ts_predicted, price_at_pred, signal
                FROM predictions
                WHERE evaluated_1h=0 AND ts_predicted <= ?
                LIMIT 100
            """, (cut1h,)).fetchall()

            rows_4h = c.execute("""
                SELECT id, ticker, ts_predicted, price_at_pred, signal, pred_return
                FROM predictions
                WHERE evaluated_4h=0 AND ts_predicted <= ?
                LIMIT 100
            """, (cut4h,)).fetchall()

        # Stažení aktuálních cen — každý ticker stáhneme jen jednou
        all_tickers = set(r[1] for r in rows_1h) | set(r[1] for r in rows_4h)
        prices_now  = {}   # ticker → aktuální cena
        hist_data   = {}   # ticker → DataFrame posledních hodinových dat

        for ticker in all_tickers:
            try:
                df = yf.download(ticker, period="2d", interval="1h",
                                  auto_adjust=True, progress=False)
                if df.empty:
                    print(f"    VAROVÁNÍ: {ticker} — žádná data pro vyhodnocení")
                    continue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
                prices_now[ticker] = float(df["Close"].iloc[-1])
                hist_data[ticker]  = df
            except Exception as e:
                # Explicitní logování — tiché except: pass je zakázáno
                print(f"    CHYBA při stahování {ticker}: {e}")

        with self._lock:
            with self._conn() as c:
                # ── 1h vyhodnocení ─────────────────────────────────────────────
                for row_id, ticker, ts_pred_str, price_pred, signal in rows_1h:
                    p1h = prices_now.get(ticker)
                    if p1h is None:
                        continue
                    ret1h = (p1h - price_pred) / price_pred
                    ok1h  = _direction_ok(signal, ret1h)
                    c.execute("""
                        UPDATE predictions
                        SET price_1h=?, actual_ret_1h=?,
                            direction_ok_1h=?, evaluated_1h=1
                        WHERE id=?
                    """, (p1h, ret1h, ok1h, row_id))
                    results.append({
                        "horizon": "1h", "id": row_id,
                        "ticker": ticker, "signal": signal,
                        "price_pred": price_pred, "price_eval": p1h,
                        "actual_ret": ret1h, "direction_ok": ok1h,
                    })

                # ── 4h vyhodnocení + podobnost grafu ──────────────────────────
                for row_id, ticker, ts_pred_str, price_pred, signal, pred_ret \
                        in rows_4h:
                    p4h = prices_now.get(ticker)
                    if p4h is None:
                        continue
                    ret4h = (p4h - price_pred) / price_pred
                    ok4h  = _direction_ok(signal, ret4h)

                    # Podobnost grafu předpovědi a skutečnosti
                    cosine, mae = _chart_similarity(
                        hist_data.get(ticker), ts_pred_str,
                        price_pred, pred_ret, n_steps=4)

                    c.execute("""
                        UPDATE predictions
                        SET price_4h=?, actual_ret_4h=?,
                            direction_ok_4h=?, evaluated_4h=1,
                            chart_cosine=?, chart_mae=?, chart_evaluated=1
                        WHERE id=?
                    """, (p4h, ret4h, ok4h, cosine, mae, row_id))
                    results.append({
                        "horizon": "4h", "id": row_id,
                        "ticker": ticker, "signal": signal,
                        "price_pred": price_pred, "price_eval": p4h,
                        "actual_ret": ret4h, "direction_ok": ok4h,
                        "chart_cosine": cosine, "chart_mae": mae,
                    })

                c.commit()

        return results

    # ── Statistiky ─────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Vrátí celkové statistiky + statistiky per ticker a per signál."""
        with self._conn() as c:
            # Celkové statistiky
            row = c.execute("""
                SELECT
                    COUNT(*),
                    SUM(CASE WHEN evaluated_1h=1 THEN 1 END),
                    SUM(direction_ok_1h),
                    SUM(CASE WHEN evaluated_4h=1 THEN 1 END),
                    SUM(direction_ok_4h),
                    AVG(CASE WHEN chart_evaluated=1 THEN chart_cosine END),
                    AVG(CASE WHEN chart_evaluated=1 THEN chart_mae END),
                    COUNT(CASE WHEN evaluated_4h=0 THEN 1 END)
                FROM predictions
            """).fetchone()

            total      = row[0] or 0
            eval_1h    = row[1] or 0
            ok_1h      = row[2] or 0
            eval_4h    = row[3] or 0
            ok_4h      = row[4] or 0
            avg_cosine = row[5]
            avg_mae    = row[6]
            pending    = row[7] or 0
            acc_1h     = ok_1h / eval_1h if eval_1h > 0 else None
            acc_4h     = ok_4h / eval_4h if eval_4h > 0 else None

            # Statistiky per signál
            sig_rows = c.execute("""
                SELECT signal,
                       COUNT(CASE WHEN evaluated_1h=1 THEN 1 END),
                       SUM(direction_ok_1h),
                       COUNT(CASE WHEN evaluated_4h=1 THEN 1 END),
                       SUM(direction_ok_4h)
                FROM predictions
                GROUP BY signal
            """).fetchall()
            by_signal = {}
            for sig, ev1, ok1, ev4, ok4 in sig_rows:
                by_signal[sig] = {
                    "eval_1h": ev1 or 0, "ok_1h": ok1 or 0,
                    "acc_1h":  (ok1 or 0) / (ev1 or 1) if ev1 else None,
                    "eval_4h": ev4 or 0, "ok_4h": ok4 or 0,
                    "acc_4h":  (ok4 or 0) / (ev4 or 1) if ev4 else None,
                }

            # Statistiky per ticker
            tick_rows = c.execute("""
                SELECT ticker,
                       COUNT(*),
                       SUM(CASE WHEN evaluated_1h=1 THEN 1 END),
                       SUM(direction_ok_1h),
                       SUM(CASE WHEN evaluated_4h=1 THEN 1 END),
                       SUM(direction_ok_4h),
                       AVG(CASE WHEN chart_evaluated=1 THEN chart_cosine END),
                       AVG(CASE WHEN chart_evaluated=1 THEN chart_mae END),
                       AVG(CASE WHEN evaluated_4h=1 THEN actual_ret_4h END),
                       COUNT(CASE WHEN evaluated_4h=0 THEN 1 END)
                FROM predictions
                GROUP BY ticker ORDER BY ticker
            """).fetchall()
            by_ticker = {}
            for (tk, tot, ev1, ok1, ev4, ok4,
                 cos, mae, avg_r, pend) in tick_rows:
                ev1 = ev1 or 0; ok1 = ok1 or 0
                ev4 = ev4 or 0; ok4 = ok4 or 0
                by_ticker[tk] = {
                    "total":   tot,
                    "eval_1h": ev1,  "ok_1h": ok1,
                    "acc_1h":  ok1 / ev1 if ev1 > 0 else None,
                    "eval_4h": ev4,  "ok_4h": ok4,
                    "acc_4h":  ok4 / ev4 if ev4 > 0 else None,
                    "cosine":  cos,  "mae": mae,
                    "avg_ret": avg_r or 0,
                    "pending": pend or 0,
                }

            # Posledních 30 vyhodnocených předpovědí (4h)
            recent = c.execute("""
                SELECT ticker, ts_predicted, signal, confidence,
                       price_at_pred, price_4h,
                       actual_ret_1h, direction_ok_1h,
                       actual_ret_4h, direction_ok_4h,
                       chart_cosine,  chart_mae
                FROM predictions
                WHERE evaluated_4h=1
                ORDER BY rowid DESC LIMIT 30
            """).fetchall()

        return {
            "total":      total,
            "eval_1h":    eval_1h,  "ok_1h":    ok_1h,    "acc_1h":    acc_1h,
            "eval_4h":    eval_4h,  "ok_4h":    ok_4h,    "acc_4h":    acc_4h,
            "avg_cosine": avg_cosine, "avg_mae": avg_mae,
            "pending":    pending,
            "by_signal":  by_signal,
            "by_ticker":  by_ticker,
            "recent":     recent,
        }

    def get_accuracy_over_time(self, ticker: str = None) -> pd.DataFrame:
        """Rolling přesnost 1h a 4h odděleně — pro grafy."""
        where = f"AND ticker='{ticker}'" if ticker else ""
        with self._conn() as c:
            rows = c.execute(f"""
                SELECT ts_predicted,
                       direction_ok_1h, evaluated_1h,
                       direction_ok_4h, evaluated_4h,
                       chart_cosine, ticker
                FROM predictions
                WHERE (evaluated_1h=1 OR evaluated_4h=1) {where}
                ORDER BY ts_predicted ASC
            """).fetchall()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=[
            "ts", "ok_1h", "ev_1h", "ok_4h", "ev_4h", "cosine", "ticker"])
        df["ts"]       = pd.to_datetime(df["ts"])
        df["ok_1h"]    = df["ok_1h"].where(df["ev_1h"] == 1).astype(float)
        df["ok_4h"]    = df["ok_4h"].where(df["ev_4h"] == 1).astype(float)
        df["roll_1h"]  = df["ok_1h"].rolling(10, min_periods=1).mean()
        df["roll_4h"]  = df["ok_4h"].rolling(10, min_periods=1).mean()
        df["roll_cos"] = df["cosine"].rolling(10, min_periods=1).mean()
        return df

    def delete_old(self, days: int = 30):
        """Smaže záznamy starší než zadaný počet dní."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        with self._lock:
            with self._conn() as c:
                c.execute("DELETE FROM predictions WHERE ts_predicted < ?",
                           (cutoff,))
                c.commit()


# ── Pomocné funkce ─────────────────────────────────────────────────────────────

def _direction_ok(signal: str, actual_ret: float) -> int:
    """Vrátí 1 pokud je směr předpovědi správný, jinak 0."""
    if signal == "BUY":  return 1 if actual_ret > 0 else 0
    if signal == "SELL": return 1 if actual_ret < 0 else 0
    # HOLD: správný pokud pohyb nepřekročil práh
    return 1 if abs(actual_ret) < HOLD_THRESH else 0


def _chart_similarity(df, ts_pred_str: str, price_pred: float,
                       pred_return: float, n_steps: int = 4):
    """
    Porovná předpovídaný lineární cenový průběh s reálným.

    Předpověď: lineární cesta od price_pred na price_pred*(1+pred_return)
               v n_steps krocích.
    Skutečnost: příštích n_steps hodinových svíček po ts_pred.

    Metriky:
      cosine   0–1  (1 = stejný směr/tvar)
      mae_pct  MAE v procentech (menší = lepší)
    """
    if df is None or df.empty:
        return None, None
    try:
        ts_pred = pd.to_datetime(ts_pred_str)
        # Data bezprostředně po okamžiku předpovědi
        future = df[df.index > ts_pred]["Close"].iloc[:n_steps]
        if len(future) < 2:
            return None, None

        # Skutečný průběh (normalizovaný na price_pred)
        actual = future.values / price_pred - 1.0

        # Předpovídaný lineární průběh
        steps = len(actual)
        pred  = np.linspace(0, pred_return, steps)

        # Kosinová podobnost
        dot    = np.dot(actual, pred)
        norm   = np.linalg.norm(actual) * np.linalg.norm(pred)
        cosine = float(dot / norm) if norm > 1e-10 else 0.0
        cosine = (cosine + 1) / 2   # normalizace na 0–1

        # MAE v procentech
        mae = float(np.mean(np.abs(actual - pred))) * 100

        return round(cosine, 4), round(mae, 4)
    except Exception as e:
        print(f"    VAROVÁNÍ _chart_similarity: {e}")
        return None, None


# ── CLI ─────────────────────────────────────────────────────────────────────

def _print_stats(stats: dict):
    print(f"\n{'='*55}\n  TRACKER STATISTIKY\n{'='*55}")
    print(f"  Předpovědí celkem: {stats['total']}  |  čeká: {stats['pending']}")
    if stats["acc_1h"] is not None:
        print(f"  Přesnost 1h: {stats['acc_1h']:.1%}  "
              f"({stats['ok_1h']}/{stats['eval_1h']})")
    if stats["acc_4h"] is not None:
        print(f"  Přesnost 4h: {stats['acc_4h']:.1%}  "
              f"({stats['ok_4h']}/{stats['eval_4h']})")
    if stats["avg_cosine"] is not None:
        print(f"  Podobnost grafu (cosine): {stats['avg_cosine']:.3f}")
    if stats["by_ticker"]:
        print(f"\n  {'Ticker':<8} {'4h přesn.':>9} {'avg ret':>9} {'čeká':>6}")
        for tk, s in sorted(stats["by_ticker"].items()):
            acc = f"{s['acc_4h']:.0%}" if s["acc_4h"] is not None else "—"
            print(f"  {tk:<8} {acc:>9} {s['avg_ret']:>+8.2%} {s['pending']:>6}")
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Mičov — Prediction Tracker")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("evaluate", help="Vyhodnotí čekající předpovědi (1h/4h)")
    sub.add_parser("stats",    help="Vypíše statistiky přesnosti")
    p_clean = sub.add_parser("clean", help="Smaže staré záznamy")
    p_clean.add_argument("--days", type=int, default=30)
    args = parser.parse_args()

    tracker = PredictionTracker()
    if args.cmd == "evaluate":
        res = tracker.evaluate_pending()
        print(f"  Nově vyhodnoceno: {len(res)} záznamů")
        _print_stats(tracker.get_stats())
    elif args.cmd == "stats":
        _print_stats(tracker.get_stats())
    elif args.cmd == "clean":
        tracker.delete_old(args.days)
        print(f"  Smazány záznamy starší než {args.days} dní")
