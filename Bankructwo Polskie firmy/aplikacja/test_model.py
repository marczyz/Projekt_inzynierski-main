"""
Testy poprawności: czy model i aplikacja liczą to, co powinny.

Uruchomienie (w folderze z aplikacją):
    python test_model.py
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import core_logic as cl

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'OK ' if ok else 'BŁĄD'}] {name}" + (f"  –  {detail}" if detail else ""))


bundle = cl.load_bundle(str(HERE / "model.joblib"))
data = pd.read_csv(HERE / "dane_testowe.csv")
test = data[data.zbior == "test"]
thr = bundle["threshold"]

# 1. model wczytany poprawnie
check("Model ma 10 oczekiwanych cech w dobrej kolejności", bundle["feature_names"] == cl.FEATURES)
check("Próg decyzyjny jest sensowny (0–1)", 0 < thr < 1, f"{thr:.4f}")

# 2. predykcje przez aplikację = bezpośrednio przez pipeline
p_app = cl.predict_proba(bundle, test[cl.FEATURES])
p_direct = bundle["pipeline"].predict_proba(test[cl.FEATURES])[:, 1]
check("Predykcje aplikacji = predykcje pipeline'u", np.allclose(p_app, p_direct, atol=1e-12),
      f"max różnica {np.abs(p_app - p_direct).max():.2e}")
check("Prawdopodobieństwa mieszczą się w [0, 1]", ((p_app >= 0) & (p_app <= 1)).all())

# 3. metryki odtwarzają te zapisane przy treningu (dowód, że plik modelu jest nienaruszony)
ev = cl.evaluate(test["class"], p_app, thr)
saved = bundle["extra"].get("test_metrics")
if saved:
    check("AP na teście = zapisane przy treningu", abs(ev["AP"] - saved["AP"]) < 1e-6,
          f"{ev['AP']:.4f}")
    check("ROC-AUC na teście = zapisane przy treningu", abs(ev["ROC_AUC"] - saved["ROC_AUC"]) < 1e-6,
          f"{ev['ROC_AUC']:.4f}")
    check("Macierz pomyłek = zapisana przy treningu",
          (ev["TN"], ev["FP"], ev["FN"], ev["TP"]) == tuple(saved["TN_FP_FN_TP"]),
          f"TN={ev['TN']} FP={ev['FP']} FN={ev['FN']} TP={ev['TP']}")

# 4. model rozróżnia bankrutów od reszty
check("ROC-AUC na teście > 0,85", ev["ROC_AUC"] > 0.85)
check("Średnie ryzyko bankrutów > 5× średnie ryzyko pozostałych",
      p_app[test["class"] == 1].mean() > 5 * p_app[test["class"] == 0].mean(),
      f"{p_app[test['class'] == 1].mean():.1%} vs {p_app[test['class'] == 0].mean():.1%}")
top = test.assign(p=p_app).sort_values("p", ascending=False).head(int(len(test) * 0.1))
check("Górne 10% firm wg ryzyka zawiera ≥ 60% bankrutów", top["class"].sum() / test["class"].sum() >= 0.6,
      f"{top['class'].sum() / test['class'].sum():.0%}")

# 5. kalibracja: średnia prognoza ≈ rzeczywisty odsetek bankructw
check("Średnia prognoza ≈ rzeczywisty odsetek bankructw (±1,5 p.p.)",
      abs(p_app.mean() - test["class"].mean()) < 0.015,
      f"prognoza {p_app.mean():.2%}, rzeczywistość {test['class'].mean():.2%}")

# 6. przeliczanie danych finansowych na wskaźniki (test „w kółko”: wskaźniki -> dane -> wskaźniki)
target = dict(Attr27=2.5, Attr21=1.1, Attr34=1.8, Attr58=0.96, Attr46=0.9,
              Attr24=0.2, Attr35=0.08, Attr56=0.3, Attr39=0.05, Attr5=25.0)
rev = 10000.0
raw = dict(przychody=rev, przychody_prev=rev / target["Attr21"], koszty_fin=1000.0,
           koszty_oper=8000.0, amortyzacja=500.0, zob_kr=3000.0, zapasy=1000.0, srodki=400.0)
raw["zysk_oper"] = target["Attr27"] * raw["koszty_fin"]
raw["zob_ogol"] = raw["koszty_oper"] / target["Attr34"]
raw["koszty_calk"] = target["Attr58"] * rev
raw["aktywa_obr"] = target["Attr46"] * raw["zob_kr"] + raw["zapasy"]
raw["zysk_sprzedaz"] = target["Attr39"] * rev
raw["aktywa"] = raw["zysk_sprzedaz"] / target["Attr35"]
raw["zysk_brutto_3l"] = target["Attr24"] * raw["aktywa"]
raw["koszt_sprzedanych"] = rev * (1 - target["Attr56"])
raw["naleznosci"] = (target["Attr5"] * (raw["koszty_oper"] - raw["amortyzacja"]) / 365
                     + raw["zob_kr"] - raw["srodki"])
got = cl.compute_indicators(raw)
check("Wzory wskaźników odtwarzają zadane wartości", all(np.isclose(got[k], target[k]) for k in target),
      ", ".join(f"{k}={got[k]:.3g}" for k in target))
check("Dzielenie przez 0 daje brak wartości (NaN), a nie błąd",
      np.isnan(cl.compute_indicators({**raw, "koszty_fin": 0.0})["Attr27"]))

# 7. brakujące wskaźniki nie psują predykcji
row = test[cl.FEATURES].iloc[[0]].copy()
row.iloc[0, 0] = np.nan
check("Brak 1 wskaźnika -> model nadal zwraca prawdopodobieństwo", np.isfinite(cl.predict_proba(bundle, row)[0]))

# 8. kierunki: gorsze wskaźniki -> wyższe ryzyko
typ = cl.typical_firm(bundle)
bad = typ.copy()
bad[["Attr27", "Attr39", "Attr35", "Attr24", "Attr46", "Attr5"]] = [-1.5, -0.1, -0.1, -0.2, 0.3, -80]
good = typ.copy()
good[["Attr27", "Attr39", "Attr35", "Attr24", "Attr46", "Attr5"]] = [6, 0.12, 0.15, 0.4, 2.0, 60]
pb, pt, pg = (float(cl.predict_proba(bundle, s.to_frame().T)[0]) for s in (bad, typ, good))
check("Firma zagrożona > typowa > zdrowa", pb > pt > pg, f"{pb:.1%} > {pt:.1%} > {pg:.1%}")

# 9. prognoza
hist = pd.DataFrame({"Attr27": [3.0, 2.0, 1.0]}, index=[2023, 2024, 2025])
check("Prognoza „bez zmian” utrzymuje ostatnią wartość",
      np.allclose(cl.forecast_indicators(hist, 3, 0.0)["Attr27"], 1.0))
check("Prognoza „trend utrzymany” kontynuuje spadek",
      cl.forecast_indicators(hist, 2, 1.0)["Attr27"].iloc[0] < 1.0)

# 10. cała aplikacja uruchamia się bez błędów
try:
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(HERE / "app.py"), default_timeout=120).run()
    check("Aplikacja uruchamia się bez błędów", not at.exception and not at.error,
          str([e.value for e in at.exception][:1]))
except ImportError:
    print("[ – ] Pominięto test aplikacji (brak streamlit)")

print(f"\nWynik: {sum(results)}/{len(results)} testów zaliczonych")
raise SystemExit(0 if all(results) else 1)
