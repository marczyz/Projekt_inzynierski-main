"""
Logika aplikacji bez Streamlita – dzięki temu można ją testować osobno (test_model.py).

Zawiera: opisy wskaźników, przeliczanie danych finansowych na wskaźniki, ładowanie modelu,
liczenie prawdopodobieństwa, prognozę trendu, wpływ pojedynczych wskaźników i ewaluację.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             roc_auc_score)

# WAŻNE: kolejność jak przy trenowaniu modelu (train_model_10.py)
FEATURES = ["Attr27", "Attr21", "Attr34", "Attr58", "Attr46",
            "Attr24", "Attr35", "Attr56", "Attr39", "Attr5"]

MAX_MISSING = 2   # ile wskaźników może być niedostępnych (model użyje wtedy mediany)

INDICATORS = {
    "Attr27": dict(
        name="Pokrycie kosztów finansowych zyskiem operacyjnym", short="Pokrycie kosztów fin.",
        formula="Zysk z działalności operacyjnej / Koszty finansowe",
        info="Ile razy zysk operacyjny „pokrywa” odsetki i inne koszty finansowe. "
             "Wartość poniżej 1 oznacza, że firma nie zarabia na obsługę długu."),
    "Attr21": dict(
        name="Dynamika sprzedaży", short="Dynamika sprzedaży",
        formula="Przychody ze sprzedaży (ten rok) / Przychody (rok wcześniej)",
        info="Powyżej 1 – sprzedaż rośnie, poniżej 1 – spada. Np. 1,10 = wzrost o 10%."),
    "Attr34": dict(
        name="Koszty operacyjne do zobowiązań", short="Koszty oper./zobowiązania",
        formula="Koszty operacyjne / Zobowiązania ogółem",
        info="Relacja kosztów prowadzenia działalności do całkowitego zadłużenia firmy."),
    "Attr58": dict(
        name="Udział kosztów w przychodach", short="Koszty/przychody",
        formula="Koszty całkowite / Przychody ze sprzedaży ogółem",
        info="Poniżej 1 – firma wychodzi na plus, powyżej 1 – koszty przewyższają przychody (strata)."),
    "Attr46": dict(
        name="Płynność szybka", short="Płynność szybka",
        formula="(Aktywa obrotowe − Zapasy) / Zobowiązania krótkoterminowe",
        info="Czy firma ma czym szybko spłacić bieżące długi bez wyprzedawania zapasów. "
             "Wartość ok. 1 lub więcej jest zwykle bezpieczna."),
    "Attr24": dict(
        name="Rentowność aktywów (3 lata)", short="Rentowność akt. (3 lata)",
        formula="Zysk brutto z ostatnich 3 lat (suma) / Aktywa ogółem",
        info="Ile zysku brutto (suma z 3 lat) przypada na złotówkę majątku firmy."),
    "Attr35": dict(
        name="Rentowność aktywów ze sprzedaży", short="Rentowność aktywów",
        formula="Zysk ze sprzedaży / Aktywa ogółem",
        info="Jak efektywnie majątek firmy generuje zysk ze sprzedaży."),
    "Attr56": dict(
        name="Marża brutto na sprzedaży", short="Marża brutto",
        formula="(Przychody − Koszt sprzedanych produktów) / Przychody",
        info="Jaka część przychodów zostaje po pokryciu bezpośrednich kosztów sprzedanych produktów."),
    "Attr39": dict(
        name="Marża zysku ze sprzedaży", short="Marża sprzedaży",
        formula="Zysk ze sprzedaży / Przychody ze sprzedaży",
        info="Ile groszy zysku firma zarabia na każdej złotówce sprzedaży."),
    "Attr5": dict(
        name="Zapas płynności w dniach", short="Płynność (dni)",
        formula="[(Środki pieniężne + Papiery krótkoterminowe + Należności − Zobowiązania krótkoterminowe) "
                "/ (Koszty operacyjne − Amortyzacja)] × 365",
        info="Przez ile dni firma mogłaby pokrywać codzienne koszty operacyjne z płynnych środków, "
             "po spłacie zobowiązań krótkoterminowych. Wartość ujemna = niedobór płynności."),
}

# Surowe dane z bilansu / rachunku wyników: (klucz, etykieta, skąd wziąć)
RAW = [
    ("przychody", "Przychody ze sprzedaży (ten rok)", "Rachunek zysków i strat"),
    ("przychody_prev", "Przychody ze sprzedaży (rok wcześniej)", "RZiS z poprzedniego roku"),
    ("koszt_sprzedanych", "Koszt sprzedanych produktów/towarów", "RZiS"),
    ("zysk_sprzedaz", "Zysk ze sprzedaży", "RZiS"),
    ("zysk_oper", "Zysk z działalności operacyjnej", "RZiS (EBIT)"),
    ("koszty_fin", "Koszty finansowe", "RZiS (odsetki itp.)"),
    ("koszty_oper", "Koszty operacyjne", "RZiS"),
    ("koszty_calk", "Koszty całkowite", "Wszystkie koszty działalności"),
    ("amortyzacja", "Amortyzacja", "RZiS / rachunek przepływów pieniężnych"),
    ("aktywa", "Aktywa ogółem", "Bilans"),
    ("aktywa_obr", "Aktywa obrotowe", "Bilans"),
    ("zapasy", "Zapasy", "Bilans"),
    ("srodki", "Środki pieniężne + papiery wartościowe krótkoterminowe", "Bilans"),
    ("naleznosci", "Należności krótkoterminowe", "Bilans"),
    ("zob_kr", "Zobowiązania krótkoterminowe", "Bilans"),
    ("zob_ogol", "Zobowiązania ogółem", "Bilans"),
    ("zysk_brutto_3l", "Zysk brutto z ostatnich 3 lat (suma)", "Suma zysków brutto z 3 ostatnich lat"),
]
RAW_KEYS = [k for k, _, _ in RAW]
RAW_LABELS = {k: lbl for k, lbl, _ in RAW}


# ----------------------------------------------------------------------------
# Przykładowe dane (tys. zł) – 5 lat, żeby wykresy miały co pokazać
# ----------------------------------------------------------------------------
_DET = {  # firma, która z roku na rok traci rentowność i wpada w stratę (t-4 … t), tys. zł
    "przychody":         [10000, 10800, 11000, 9800, 8200],
    "koszt_sprzedanych": [7000, 7500, 7900, 7600, 6800],
    "zysk_sprzedaz":     [1400, 1500, 1000, 0, -900],
    "zysk_oper":         [1200, 1250, 800, -200, -1100],
    "koszty_fin":        [200, 220, 300, 420, 550],
    "koszty_oper":       [8800, 9550, 10200, 10000, 9300],
    "koszty_calk":       [9000, 9770, 10500, 10420, 9850],
    "amortyzacja":       [400, 420, 450, 450, 450],
    "aktywa":            [8000, 8600, 9400, 9500, 9000],
    "aktywa_obr":        [3500, 3800, 4200, 4000, 3500],
    "zapasy":            [900, 1000, 1300, 1500, 1500],
    "srodki":            [700, 650, 500, 200, 80],
    "naleznosci":        [1600, 1750, 1900, 1700, 1300],
    "zob_kr":            [2300, 2700, 3300, 3900, 4300],
    "zob_ogol":          [3800, 4500, 5400, 6200, 6900],
    "zysk_brutto_3l":    [2850, 2980, 2530, 910, -1770],
}
EXAMPLE_KINDS = ["Firma pogarszająca się", "Firma zdrowa", "Firma poprawiająca się", "Puste (własne dane)"]


def example_data(kind: str, n_years: int, last_year: int) -> pd.DataFrame:
    """Tabela (wiersze = pozycje RAW, kolumny = lata) z przykładowymi danymi."""
    n_years = max(1, min(5, n_years))
    if kind == EXAMPLE_KINDS[0]:
        cols = {k: list(v) for k, v in _DET.items()}
        prev = [9500] + cols["przychody"][:-1]
    elif kind == EXAMPLE_KINDS[1]:   # struktura z „dobrego” roku, wzrost 5% rocznie
        cols = {k: [v[0] * 1.05 ** i for i in range(5)] for k, v in _DET.items()}
        prev = [cols["przychody"][0] / 1.05] + cols["przychody"][:-1]
    elif kind == EXAMPLE_KINDS[2]:   # odwrócona historia firmy pogarszającej się
        cols = {k: list(reversed(v)) for k, v in _DET.items()}
        prev = [cols["przychody"][0] * 0.9] + cols["przychody"][:-1]
    else:
        cols = {k: [np.nan] * 5 for k in RAW_KEYS}
        prev = [np.nan] * 5
    cols["przychody_prev"] = prev
    years = [int(last_year) - (n_years - 1 - j) for j in range(n_years)]
    df = pd.DataFrame({str(y): [cols[k][5 - n_years + j] for k in RAW_KEYS]
                       for j, y in enumerate(years)}, index=RAW_KEYS).astype(float).round(0)
    return df


# ----------------------------------------------------------------------------
# Wskaźniki
# ----------------------------------------------------------------------------
def div(a, b):
    """Bezpieczne dzielenie – NaN przy braku danych lub dzieleniu przez 0."""
    if a is None or b is None or pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return a / b


def compute_indicators(d: dict) -> dict:
    """Wylicza 10 wskaźników modelu z danych finansowych (wzory wg opisu zbioru UCI)."""
    attr5 = div(d["srodki"] + d["naleznosci"] - d["zob_kr"], d["koszty_oper"] - d["amortyzacja"])
    return {
        "Attr27": div(d["zysk_oper"], d["koszty_fin"]),
        "Attr21": div(d["przychody"], d["przychody_prev"]),
        "Attr34": div(d["koszty_oper"], d["zob_ogol"]),
        "Attr58": div(d["koszty_calk"], d["przychody"]),
        "Attr46": div(d["aktywa_obr"] - d["zapasy"], d["zob_kr"]),
        "Attr24": div(d["zysk_brutto_3l"], d["aktywa"]),
        "Attr35": div(d["zysk_sprzedaz"], d["aktywa"]),
        "Attr56": div(d["przychody"] - d["koszt_sprzedanych"], d["przychody"]),
        "Attr39": div(d["zysk_sprzedaz"], d["przychody"]),
        "Attr5": attr5 * 365 if not pd.isna(attr5) else np.nan,
    }


def indicators_table(raw: pd.DataFrame) -> pd.DataFrame:
    """raw: wiersze = klucze RAW, kolumny = lata -> DataFrame (index = rok int, kolumny = FEATURES)."""
    rows = {}
    for col in raw.columns:
        d = {k: (float(raw.loc[k, col]) if pd.notna(raw.loc[k, col]) else np.nan) for k in RAW_KEYS}
        rows[int(col)] = compute_indicators(d)
    return pd.DataFrame(rows).T[FEATURES].astype(float)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def load_bundle(path: str) -> dict:
    """
    Wczytuje model zapisany jako {'pipeline', 'threshold', 'feature_names'} (lub zwykły model).
    Zwraca też `fill` = mediany z treningu („typowa firma”).
    """
    import __main__
    from custom_transformers import QuantileClipper
    __main__.QuantileClipper = QuantileClipper     # zgodność z modelami zapisanymi z notebooka

    obj = joblib.load(path)
    if isinstance(obj, dict):
        pipe, thr = obj["pipeline"], obj.get("threshold")
        names = list(obj.get("feature_names") or [])
        extra = {k: v for k, v in obj.items() if k not in ("pipeline", "threshold", "feature_names")}
    else:
        pipe, thr, names, extra = obj, None, list(getattr(obj, "feature_names_in_", [])), {}

    fill = {n: np.nan for n in names}
    try:
        imp = pipe.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"]
        fill = dict(zip(names, imp.statistics_))
    except Exception:
        pass
    return {"pipeline": pipe, "threshold": thr, "feature_names": names, "fill": fill, "extra": extra}


def predict_proba(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    """
    Prawdopodobieństwo bankructwa (klasa 1) dla wierszy X (kolumny = FEATURES).
    NaN w X są uzupełniane medianą przez imputer w pipeline.
    Jeśli model ma więcej cech niż 10, pozostałe dostają mediany z treningu.
    """
    pipe, names, fill = bundle["pipeline"], bundle["feature_names"], bundle["fill"]
    X = X[FEATURES].astype(float)
    if names and set(names) != set(FEATURES):
        full = pd.DataFrame([fill] * len(X), columns=names, dtype=float)
        for f in FEATURES:
            full[f] = X[f].to_numpy()
        proba = pipe.predict_proba(full)
    else:
        proba = pipe.predict_proba(X[names] if names else X)
    classes = list(getattr(pipe, "classes_", [0, 1]))
    return proba[:, classes.index(1) if 1 in classes else -1]


def predict_rows(bundle: dict, ind: pd.DataFrame, max_missing: int = MAX_MISSING) -> pd.Series:
    """Prawdopodobieństwa dla wierszy ind; NaN, gdy brakuje więcej niż max_missing wskaźników."""
    out = pd.Series(np.nan, index=ind.index, dtype=float)
    ok = ind[FEATURES].isna().sum(axis=1) <= max_missing
    if ok.any():
        out[ok] = predict_proba(bundle, ind.loc[ok, FEATURES])
    return out


def typical_firm(bundle: dict) -> pd.Series:
    return pd.Series({f: bundle["fill"].get(f, np.nan) for f in FEATURES})


def swap_impacts(bundle: dict, base: pd.Series, other: pd.Series):
    """
    Dla każdego wskaźnika: o ile zmieni się prawdopodobieństwo, gdy w firmie `base`
    zamienimy TYLKO ten wskaźnik na wartość z `other` (reszta bez zmian).
    Zwraca (Series zmian, p_base). To analiza „po jednym”, więc zmiany nie muszą się sumować.
    """
    base = base[FEATURES].astype(float)
    rows = []
    for f in FEATURES:
        r = base.copy()
        r[f] = other[f]
        rows.append(r)
    P = predict_proba(bundle, pd.DataFrame(rows, columns=FEATURES))
    p0 = float(predict_proba(bundle, base.to_frame().T)[0])
    return pd.Series(P - p0, index=FEATURES), p0


def risk_label(p, low, high):
    if p < low:
        return "NISKIE", "#2e9e5b"
    if p < high:
        return "PODWYŻSZONE", "#e0a526"
    return "WYSOKIE", "#d64545"


# ----------------------------------------------------------------------------
# Prognoza trendu
# ----------------------------------------------------------------------------
def forecast_indicators(ind: pd.DataFrame, n_ahead: int, damping: float) -> pd.DataFrame:
    """
    Przedłuża trend liniowy każdego wskaźnika na kolejne lata.
      damping = 0   -> „nic się nie zmienia” (wskaźniki zostają na poziomie ostatniego roku)
      damping = 1   -> trend utrzymany bez zmian
      0 < d < 1     -> trend stopniowo wygasa
    Wartości są ograniczone do rozsądnego zakresu wokół historii.
    """
    years = ind.index.to_numpy(dtype=float)
    last = years.max()
    future = np.arange(last + 1, last + 1 + n_ahead)
    out = pd.DataFrame(index=future.astype(int), columns=ind.columns, dtype=float)
    steps = np.arange(1, n_ahead + 1)
    eff = np.cumsum(np.power(float(damping), steps))
    for col in ind.columns:
        y = ind[col].to_numpy(dtype=float)
        ok = ~np.isnan(y)
        if ok.sum() == 0:
            continue
        base = y[ok][-1]
        if ok.sum() == 1:
            out[col] = base
            continue
        slope, _ = np.polyfit(years[ok], y[ok], 1)
        pred = base + slope * eff
        lo, hi = y[ok].min(), y[ok].max()
        span = max(hi - lo, abs(y[ok]).mean() * 0.1, 1e-6)
        out[col] = np.clip(pred, lo - span, hi + span)
    return out


# ----------------------------------------------------------------------------
# Ewaluacja na danych z etykietami
# ----------------------------------------------------------------------------
def evaluate(y, p, thr: float) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    res = dict(n=int(len(y)), n_pos=int(y.sum()), TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp),
               recall=tp / (tp + fn) if tp + fn else np.nan,
               precision=tp / (tp + fp) if tp + fp else np.nan,
               brier=float(brier_score_loss(y, p)))
    if 0 < y.sum() < len(y):
        res["AP"] = float(average_precision_score(y, p))
        res["ROC_AUC"] = float(roc_auc_score(y, p))
    return res


CAL_EDGES = [0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 1.0001]


def calibration_table(y, p, edges=CAL_EDGES) -> pd.DataFrame:
    """Średnie przewidywane prawdopodobieństwo vs rzeczywisty odsetek bankructw w przedziałach."""
    d = pd.DataFrame({"y": np.asarray(y).astype(int), "p": np.asarray(p, dtype=float)})
    d["bin"] = pd.cut(d["p"], bins=edges, right=False)
    g = d.groupby("bin", observed=True).agg(n=("y", "size"), n_pos=("y", "sum"),
                                            pred=("p", "mean"), obs=("y", "mean")).reset_index()
    g["label"] = [f"{int(round(b.left * 100))}–{min(100, int(round(b.right * 100)))}%" for b in g["bin"]]
    return g


def threshold_curve(y, p, thresholds) -> pd.DataFrame:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    rows = []
    for t in thresholds:
        pred = p >= t
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        tn = int((~pred & (y == 0)).sum())
        rows.append(dict(threshold=t,
                         recall=tp / (tp + fn) if tp + fn else np.nan,
                         precision=tp / (tp + fp) if tp + fp else np.nan,
                         false_alarm=fp / (fp + tn) if fp + tn else np.nan))
    return pd.DataFrame(rows)
