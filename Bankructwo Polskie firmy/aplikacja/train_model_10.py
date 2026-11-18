"""
Trening modelu na 10 wskaźnikach – ta sama metodologia co w notebooku
(model_bankructwo_pipeline.ipynb):

  * deduplikacja, podział 80/20 stratyfikowany (random_state=42) PRZED jakimkolwiek dopasowaniem,
  * winsoryzacja 1%/99% + imputacja medianą wewnątrz Pipeline,
  * wybór modelu 5-fold CV na zbiorze treningowym (metryka: Average Precision),
  * strojenie RandomizedSearchCV (AP),
  * próg decyzyjny maksymalizujący F2 na predykcjach out-of-fold (tylko trening),
  * jednorazowa ocena na zbiorze testowym,
  * zapis w formacie {'pipeline', 'threshold', 'feature_names'}.

Różnica względem notebooka (i powód): w tym zbiorze BRAK wartości w Attr21/Attr27 niesie
informację o klasie (96% firm bez Attr21 to bankruci) – to artefakt danych. W aplikacji
użytkownik zawsze podaje wartości, więc taki skrót nigdy nie zadziała. Dlatego:
  - oceniamy modele na firmach z KOMPLETNYMI danymi (jak w aplikacji),
  - porównujemy warianty: z flagami braków / bez flag / uczenie tylko na kompletnych wierszach.

Uruchomienie:
    python train_model_10.py [ścieżka/do/5year.arff] [wyjście.joblib]
"""
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             fbeta_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

from custom_transformers import QuantileClipper

warnings.filterwarnings("ignore")
RS = 42
ARFF = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("5year.arff")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("model_10.joblib")

# 10 wskaźników z aplikacji (kolejność jak w app.py)
FEATURES = ["Attr27", "Attr21", "Attr34", "Attr58", "Attr46",
            "Attr24", "Attr35", "Attr56", "Attr39", "Attr5"]


# ----------------------------------------------------------------------------
# Dane
# ----------------------------------------------------------------------------
def load_data(path: Path):
    rek, _ = arff.loadarff(str(path))
    cols = [n for n in rek.dtype.names if n != "class"]
    X = pd.DataFrame({n: rek[n].astype(float) for n in cols})
    y = pd.Series([int(k.decode("ascii")) for k in rek["class"]], name="class")
    d = X.copy()
    d["class"] = y
    d = d.drop_duplicates().reset_index(drop=True)
    y = d.pop("class").astype(int)
    return d, y


def make_pipe(model, flags: bool, scale: bool = False) -> Pipeline:
    """Ta sama struktura co w notebooku: preprocess(numeric: clip -> impute) -> model."""
    steps = [("clip", QuantileClipper(0.01, 0.99)),
             ("impute", SimpleImputer(strategy="median", add_indicator=flags))]
    if scale:
        steps.append(("scale", RobustScaler()))
    pre = ColumnTransformer([("numeric", Pipeline(steps), FEATURES)],
                            remainder="drop", sparse_threshold=0.0)
    return Pipeline([("preprocess", pre), ("model", model)])


def oof_predict(factory, X, y, cv, fit_on_complete: bool) -> np.ndarray:
    """Predykcje out-of-fold; opcjonalnie uczenie tylko na kompletnych wierszach foldu."""
    p = np.zeros(len(X))
    for tr, va in cv.split(X, y):
        Xtr, ytr = X.iloc[tr], y.iloc[tr]
        if fit_on_complete:
            m = Xtr.notna().all(axis=1)
            Xtr, ytr = Xtr[m], ytr[m]
        pipe = factory()
        pipe.fit(Xtr, ytr)
        p[va] = pipe.predict_proba(X.iloc[va])[:, 1]
    return p


def base_models():
    return {
        "HGB": (lambda: HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=30,
            l2_regularization=1.0, random_state=RS), False),
        "XGBoost": (lambda: XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", n_estimators=300, max_depth=3,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            random_state=RS, n_jobs=1), False),
        # bez class_weight – żeby prawdopodobieństwa były interpretowalne w aplikacji
        "LogReg": (lambda: LogisticRegression(C=0.2, max_iter=3000, random_state=RS), True),
    }


def main():
    X_all, y_all = load_data(ARFF)
    print(f"Po deduplikacji: {X_all.shape}, bankructwa: {int(y_all.sum())} ({y_all.mean():.2%})")

    # Ten sam podział co w notebooku
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.20, stratify=y_all, random_state=RS)
    X_train, X_test = X_train[FEATURES], X_test[FEATURES]
    cm_train = X_train.notna().all(axis=1)
    cm_test = X_test.notna().all(axis=1)
    print(f"Train: {len(X_train)} (kompletne: {int(cm_train.sum())}, bankruci w kompletnych: "
          f"{int(y_train[cm_train].sum())})")
    print(f"Test : {len(X_test)} (kompletne: {int(cm_test.sum())}, bankruci w kompletnych: "
          f"{int(y_test[cm_test].sum())})")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)

    # ------------------------------------------------------------------
    # 1) Porównanie modeli i wariantów (CV na treningu)
    # ------------------------------------------------------------------
    variants = [
        ("A: wszystkie wiersze + flagi braków (jak w notebooku)", True, False),
        ("B: wszystkie wiersze, bez flag", False, False),
        ("C: uczenie tylko na kompletnych wierszach, bez flag", False, True),
    ]
    rows = []
    for vname, flags, only_complete in variants:
        for mname, (mk, scale) in base_models().items():
            p = oof_predict(lambda: make_pipe(mk(), flags, scale), X_train, y_train, cv, only_complete)
            rows.append({
                "wariant": vname[:2], "model": mname,
                "AP (wszystkie)": average_precision_score(y_train, p),
                "AP (kompletne)": average_precision_score(y_train[cm_train], p[cm_train]),
                "ROC (kompletne)": roc_auc_score(y_train[cm_train], p[cm_train]),
            })
    cmp_df = pd.DataFrame(rows).sort_values("AP (kompletne)", ascending=False)
    print("\n=== Porównanie 5-fold CV (tylko zbiór treningowy) ===")
    print("Warianty:", *[v[0] for v in variants], sep="\n  ")
    print(cmp_df.round(4).to_string(index=False))

    # ------------------------------------------------------------------
    # 2) Wybór modelu: wariant C (zgodny z użyciem w aplikacji), najlepszy wg AP na kompletnych
    # ------------------------------------------------------------------
    best_row = cmp_df[cmp_df["wariant"] == "C:"].iloc[0]
    best_name = best_row["model"]
    print(f"\nWybrany model: {best_name} (wariant C), AP(kompletne) = {best_row['AP (kompletne)']:.4f}")

    # ------------------------------------------------------------------
    # 3) Strojenie (AP, 3-fold), tylko na kompletnych wierszach treningu
    # ------------------------------------------------------------------
    Xc, yc = X_train[cm_train], y_train[cm_train]
    mk, scale = base_models()[best_name]
    pipe0 = make_pipe(mk(), flags=False, scale=scale)
    if best_name == "HGB":
        grid = {"model__max_iter": [120, 200, 280], "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_leaf_nodes": [7, 15, 31], "model__min_samples_leaf": [20, 40, 60],
                "model__l2_regularization": [0.0, 1.0, 10.0]}
    elif best_name == "XGBoost":
        grid = {"model__n_estimators": [150, 250, 400], "model__learning_rate": [0.02, 0.05, 0.1],
                "model__max_depth": [2, 3, 4, 5], "model__min_child_weight": [1, 3, 5, 10],
                "model__subsample": [0.7, 0.85, 1.0], "model__colsample_bytree": [0.7, 0.85, 1.0],
                "model__reg_lambda": [1, 5, 10]}
    else:
        grid = {"model__C": [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]}
    search = RandomizedSearchCV(
        pipe0, grid, n_iter=min(25, int(np.prod([len(v) for v in grid.values()]))),
        scoring="average_precision", cv=StratifiedKFold(3, shuffle=True, random_state=RS),
        refit=True, random_state=RS, n_jobs=1)
    search.fit(Xc, yc)
    best = search.best_estimator_
    print(f"Najlepsze AP w CV (strojenie): {search.best_score_:.4f}")
    print("Parametry:", search.best_params_)

    # ------------------------------------------------------------------
    # 4) Próg F2 – out-of-fold, tylko trening, tylko kompletne wiersze
    # ------------------------------------------------------------------
    p_oof = oof_predict(lambda: clone(best), X_train, y_train, cv, True)
    prec, rec, thr = precision_recall_curve(y_train[cm_train], p_oof[cm_train])
    f2 = 5 * prec[:-1] * rec[:-1] / (4 * prec[:-1] + rec[:-1] + 1e-12)
    threshold = float(thr[int(np.argmax(f2))])
    print(f"Próg (F2, OOF): {threshold:.5f}  (F2={f2.max():.4f})")

    # ------------------------------------------------------------------
    # 5) Jednorazowy test końcowy (firmy z kompletnymi danymi = scenariusz aplikacji)
    # ------------------------------------------------------------------
    Xt, yt = X_test[cm_test], y_test[cm_test]
    p_test = best.predict_proba(Xt)[:, 1]
    pred = (p_test >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
    rng = np.random.default_rng(RS)
    boots = []
    for _ in range(1000):
        i = rng.integers(0, len(yt), len(yt))
        if yt.iloc[i].sum() > 0:
            boots.append(average_precision_score(yt.iloc[i], p_test[i]))
    metrics = {
        "AP": average_precision_score(yt, p_test),
        "AP_95CI": (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))),
        "ROC_AUC": roc_auc_score(yt, p_test),
        "Precision": precision_score(yt, pred, zero_division=0),
        "Recall": recall_score(yt, pred, zero_division=0),
        "F2": fbeta_score(yt, pred, beta=2, zero_division=0),
        "Brier": brier_score_loss(yt, p_test),
        "TN_FP_FN_TP": (int(tn), int(fp), int(fn), int(tp)),
        "n_test": int(len(yt)), "n_bankrupt_test": int(yt.sum()),
    }
    print("\n=== TEST (firmy z kompletnymi danymi) ===")
    for k, v in metrics.items():
        print(f"{k:16s}", round(v, 4) if isinstance(v, float) else v)

    # ------------------------------------------------------------------
    # 6) Zapis w formacie wymaganym przez aplikację
    # ------------------------------------------------------------------
    joblib.dump({"pipeline": best, "threshold": threshold, "feature_names": FEATURES,
                 "model_type": best_name, "test_metrics": metrics,
                 "notes": "10 wskaźników; uczone na kompletnych wierszach, bez flag braków."}, OUT)
    print(f"\nZapisano: {OUT}")


if __name__ == "__main__":
    main()
