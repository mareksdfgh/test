"""
train.py
========
Trénink stacking ensemblů pro VÍCE horizontů najednou (1h, 4h, 1d, ...).
Pro každý horizont vzniká klasifikační (BUY/HOLD/SELL) i regresní
(budoucí výnos) model. Společné příznaky → model chápe vztah mezi
hodinovým a denním pohybem.

Klíčové opravy:
  • Vnitřní CV stackingu = KFold(shuffle=False) místo stratifikovaného
    KFoldu → žádné promíchání tříd napříč časem (viz pozn. u CV níže).
  • Časový train/test split (test je vždy novější).
  • Vestavěná PROFIT simulace na testu vs. baseline (Always-BUY) →
    klasifikační metriky ≠ peníze.

Spuštění:
  python train.py
  python train.py --horizons 4h 1d
  python train.py --no_xgb            # bez XGBoost (jen HGB)
"""

import argparse
import json
import warnings

import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor,
                              StackingClassifier, StackingRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (classification_report, confusion_matrix,
                             mean_absolute_error, r2_score)
from sklearn.model_selection import KFold

import config

sklearn.set_config(enable_metadata_routing=True)
# Necháváme warningy viditelné; tlumíme jen konkrétní convergence šum.
warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")

# Vnitřní CV stackingu. POZOR: StackingClassifier interně volá
# cross_val_predict, který vyžaduje, aby CV byla PARTICE (každý vzorek
# přesně v jednom test foldu) → TimeSeriesSplit zde NELZE použít
# ("only works for partitions"). Volíme proto KFold(shuffle=False):
#   • zachovává časové pořadí (souvislé bloky, žádné stratifikované
#     promíchání tříd napříč časem — to byl hlavní leak původní verze),
#   • je validní partice.
# Zbytkové omezení: meta-příznaky pro prostřední fold mohou využít model
# trénovaný i na pozdějších blocích. Skutečně leak-free vyhodnocení proto
# stojí na striktním forward train/test splitu níže + OOS backtestu (test.py).
CV = KFold(n_splits=5, shuffle=False)


def load_data(path: str) -> pd.DataFrame:
    print(f"Načítám dataset: {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    print(f"  Tvar: {df.shape}")
    if "ticker" in df.columns:
        print(f"  Tickerů: {df['ticker'].nunique()}")
    print(f"  Časové rozmezí: {df.index.min()} → {df.index.max()}")
    return df


def feature_columns(df: pd.DataFrame) -> list:
    """Sloupce, které jsou skutečné příznaky (ne labely/metadata)."""
    drop = config.META_COLS + [c for c in df.columns
                               if c.startswith(("label_", "ret_"))]
    num = df.drop(columns=[c for c in drop if c in df.columns]) \
            .select_dtypes(include=[np.number])
    return num.columns.tolist()


def time_split(n: int, ratio: float):
    cut = int(n * (1 - ratio))
    return cut


def sample_weights(y: pd.Series) -> np.ndarray:
    counts = y.value_counts()
    total, nc = len(y), len(counts)
    w = {cls: total / (nc * cnt) for cls, cnt in counts.items()}
    return y.map(w).values


# ── Stavba modelů ─────────────────────────────────────────────────────────────

def build_cls_model(use_xgb: bool) -> StackingClassifier:
    hgb = HistGradientBoostingClassifier(
        max_iter=600, max_depth=7, learning_rate=0.04,
        min_samples_leaf=15, l2_regularization=0.1,
        max_features=0.8, random_state=42)
    hgb.set_fit_request(sample_weight=True)
    estimators = [("hgb", hgb)]

    if use_xgb:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=600, max_depth=6, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.75,
            reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="mlogloss", random_state=42,
            n_jobs=-1, verbosity=0)
        xgb.set_fit_request(sample_weight=True)
        estimators.append(("xgb", xgb))

    meta = LogisticRegression(max_iter=500, C=1.0)
    meta.set_fit_request(sample_weight=True)

    return StackingClassifier(
        estimators=estimators, final_estimator=meta,
        cv=CV,     # ← oprava: ne StratifiedKFold (viz pozn. nahoře)
        stack_method="predict_proba", n_jobs=-1, passthrough=False)


def build_reg_model(use_xgb: bool) -> StackingRegressor:
    hgb = HistGradientBoostingRegressor(
        max_iter=600, max_depth=6, learning_rate=0.04,
        min_samples_leaf=15, l2_regularization=0.1,
        max_features=0.8, random_state=42)
    estimators = [("hgb", hgb)]

    if use_xgb:
        from xgboost import XGBRegressor
        xgb = XGBRegressor(
            n_estimators=600, max_depth=6, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.75,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbosity=0)
        estimators.append(("xgb", xgb))

    return StackingRegressor(
        estimators=estimators, final_estimator=Ridge(alpha=1.0),
        cv=CV, n_jobs=-1, passthrough=False)


# ── Vyhodnocení ───────────────────────────────────────────────────────────────

def eval_cls(model, X_test, y_test, tag: str):
    y_pred = model.predict(X_test)
    print(f"\n── Klasifikace [{tag}] ──────────────────────────")
    print(classification_report(y_test, y_pred,
                                target_names=["SELL", "HOLD", "BUY"],
                                zero_division=0))
    cm = pd.DataFrame(confusion_matrix(y_test, y_pred, labels=[0, 1, 2]),
                      index=["Sk.SELL", "Sk.HOLD", "Sk.BUY"],
                      columns=["Př.SELL", "Př.HOLD", "Př.BUY"])
    print(cm)


def eval_reg(model, X_test, y_test, tag: str) -> np.ndarray:
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    dir_acc = np.mean(np.sign(y_pred) == np.sign(y_test))
    print(f"\n── Regrese [{tag}] ──────────────────────────────")
    print(f"  MAE:            {mae:.5f}  ({mae*100:.3f}%)")
    print(f"  R²:             {r2:.4f}")
    print(f"  Přesnost směru: {dir_acc:.1%}")
    return y_pred


def profit_sim(signals: np.ndarray, actual_ret: np.ndarray, tag: str):
    """
    Jednoduchá long/flat simulace na testu vs. baseline Always-BUY.
    Vstup do pozice na BUY (signal==2), jinak flat. Poplatek při změně pozice.
    Klasifikační metriky ≠ peníze — tohle ukazuje skutečný edge.
    """
    pos = (signals == 2).astype(float)
    trades = np.abs(np.diff(np.concatenate([[0], pos])))
    fees = trades * config.FEE_PCT
    strat_ret = pos * actual_ret - fees

    def stats(r):
        r = np.asarray(r, dtype=float)
        total = np.prod(1 + r) - 1
        sharpe = (r.mean() / r.std() * np.sqrt(len(r))) if r.std() > 0 else 0.0
        eq = np.cumprod(1 + r)
        dd = (eq / np.maximum.accumulate(eq) - 1).min()
        return total, sharpe, dd

    s_tot, s_sh, s_dd = stats(strat_ret)
    b_tot, b_sh, b_dd = stats(actual_ret)        # always-in baseline
    n_tr = int(trades.sum())

    print(f"\n── Profit simulace [{tag}] (po {config.FEE_PCT:.1%} popl.) ──")
    print(f"  Model:     výnos={s_tot:+.2%}  Sharpe={s_sh:5.2f}  "
          f"MaxDD={s_dd:.2%}  obchodů={n_tr}")
    print(f"  Always-BUY:výnos={b_tot:+.2%}  Sharpe={b_sh:5.2f}  MaxDD={b_dd:.2%}")
    edge = s_tot - b_tot
    print(f"  → Edge nad baseline: {edge:+.2%}  "
          f"{'✓ překonává' if edge > 0 else '✗ nepřekonává'}")


# ── Hlavní ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(config.DATASET_CSV))
    parser.add_argument("--test_ratio", default=0.15, type=float)
    parser.add_argument("--output_dir", default=str(config.MODEL_DIR))
    parser.add_argument("--horizons", nargs="+",
                        default=config.DEFAULT_TRAIN_HORIZONS,
                        help="Které horizonty trénovat (z config.LABEL_HORIZONS)")
    parser.add_argument("--no_xgb", action="store_true")
    args = parser.parse_args()

    use_xgb = not args.no_xgb
    if use_xgb:
        try:
            import xgboost  # noqa
            print("  XGBoost dostupný ✓")
        except Exception as e:
            print(f"  XGBoost NENÍ dostupný ({e}) → přepínám na --no_xgb")
            use_xgb = False

    horizons = [h for h in args.horizons if h in config.LABEL_HORIZONS]
    if not horizons:
        raise ValueError(f"Neznámé horizonty. Dostupné: {list(config.LABEL_HORIZONS)}")
    print(f"  Horizonty k tréninku: {horizons}")

    from pathlib import Path
    out = Path(args.output_dir)
    out.mkdir(exist_ok=True)

    df = load_data(args.data)
    feats = feature_columns(df)
    X = df[feats].replace([np.inf, -np.inf], np.nan)
    print(f"  Počet příznaků: {len(feats)}")

    cut = time_split(len(df), args.test_ratio)
    X_train, X_test = X.iloc[:cut], X.iloc[cut:]

    # Vyřazení degenerovaných příznaků POSOUZENO NA TRÉNINKU:
    #   • konstantní (< 2 distinct hodnoty) → HistGradientBoosting padá na
    #     "window shape cannot be larger than input array shape" v binningu,
    #   • vysoce-NaN (> 30 %) → typicky tržní ticker bez překryvu, k ničemu
    #     a riskuje all-NaN podfold uvnitř stacking CV.
    nunique  = X_train.nunique(dropna=True)
    nan_frac = X_train.isna().mean()
    feats = [c for c in feats if nunique[c] >= 2 and nan_frac[c] <= 0.30]
    n_dropped = len(X_train.columns) - len(feats)
    if n_dropped:
        print(f"  Vyřazeno {n_dropped} konstantních / vysoce-NaN příznaků "
              f"→ zbývá {len(feats)}")
    X_train, X_test = X_train[feats], X_test[feats]

    print(f"\n  Trénink: {len(X_train)}  |  Test: {len(X_test)}")
    print(f"  Trénink do: {X_train.index[-1]}  |  Test od: {X_test.index[0]}")

    trained = []
    for name in horizons:
        print(f"\n{'='*55}\n  HORIZONT {name}\n{'='*55}")
        y_cls = df[f"label_{name}"].astype(int)
        y_reg = df[f"ret_{name}"].astype(float)
        y_cls_tr, y_cls_te = y_cls.iloc[:cut], y_cls.iloc[cut:]
        y_reg_tr, y_reg_te = y_reg.iloc[:cut], y_reg.iloc[cut:]

        sw = sample_weights(y_cls_tr)

        print(f"  [1/2] Klasifikace ({'XGB+HGB' if use_xgb else 'HGB'})...")
        cls = build_cls_model(use_xgb)
        cls.fit(X_train, y_cls_tr, sample_weight=sw)
        eval_cls(cls, X_test, y_cls_te, name)
        sig_test = cls.predict(X_test)
        profit_sim(sig_test, y_reg_te.values, name)

        print(f"\n  [2/2] Regrese...")
        reg = build_reg_model(use_xgb)
        reg.fit(X_train, y_reg_tr)
        eval_reg(reg, X_test, y_reg_te, name)

        joblib.dump(cls, out / f"model_cls_{name}.joblib")
        joblib.dump(reg, out / f"model_reg_{name}.joblib")
        trained.append(name)
        print(f"  ✓ Modely [{name}] uloženy")

    meta = {
        "features":      feats,
        "medians":       X_train.median().to_dict(),
        "use_xgb":       use_xgb,
        "trained_until": str(X_train.index[-1]),
        "horizons":      {h: config.LABEL_HORIZONS[h] for h in trained},
        "thresholds":    {h: config.LABEL_THRESHOLDS[h] for h in trained},
        "label_map":     {"0": "SELL", "1": "HOLD", "2": "BUY"},
    }
    with open(out / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"\n  ✓ model_meta.json uložen ({len(trained)} horizontů)")


if __name__ == "__main__":
    main()
