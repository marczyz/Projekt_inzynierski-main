"""
Aplikacja: ocena ryzyka bankructwa firmy na podstawie własnego modelu ML.

Uruchomienie:
    pip install -r requirements.txt
    streamlit run app.py
"""
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Ryzyko bankructwa firmy", page_icon="📉", layout="wide")

# ----------------------------------------------------------------------------
# KONFIGURACJA
# ----------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent   # folder, w którym leży app.py
MODEL_PATH = Path(os.environ.get("MODEL_PATH", APP_DIR / "model_10.joblib"))

# WAŻNE: kolejność musi być taka sama, jak kolejność kolumn przy trenowaniu modelu!
FEATURES = ["Attr27", "Attr21", "Attr34", "Attr58", "Attr46",
            "Attr24", "Attr35", "Attr56", "Attr39", "Attr5"]

INDICATORS = {
    "Attr27": dict(
        name="Pokrycie kosztów finansowych zyskiem operacyjnym",
        formula="Zysk z działalności operacyjnej / Koszty finansowe",
        info="Ile razy zysk operacyjny „pokrywa” odsetki i inne koszty finansowe. "
             "Wartość poniżej 1 oznacza, że firma nie zarabia na obsługę długu.",
        default=2.8),
    "Attr21": dict(
        name="Dynamika sprzedaży",
        formula="Przychody ze sprzedaży (bieżący rok) / Przychody (rok poprzedni)",
        info="Powyżej 1 – sprzedaż rośnie, poniżej 1 – spada. Np. 1,10 = wzrost o 10%.",
        default=1.09),
    "Attr34": dict(
        name="Koszty operacyjne do zobowiązań",
        formula="Koszty operacyjne / Zobowiązania ogółem",
        info="Relacja kosztów prowadzenia działalności do całkowitego zadłużenia firmy.",
        default=2.17),
    "Attr58": dict(
        name="Udział kosztów w przychodach",
        formula="Koszty całkowite / Przychody ze sprzedaży ogółem",
        info="Poniżej 1 – firma wychodzi na plus, powyżej 1 – koszty przewyższają przychody (strata).",
        default=0.975),
    "Attr46": dict(
        name="Płynność szybka",
        formula="(Aktywa obrotowe − Zapasy) / Zobowiązania krótkoterminowe",
        info="Czy firma ma czym szybko spłacić bieżące długi bez wyprzedawania zapasów. "
             "Przyjmuje się, że wartość ok. 1 lub więcej jest bezpieczna.",
        default=0.94),
    "Attr24": dict(
        name="Rentowność aktywów (3 lata)",
        formula="Zysk brutto z ostatnich 3 lat / Aktywa ogółem",
        info="Ile zysku brutto (suma z 3 lat) przypada na złotówkę majątku firmy.",
        default=0.20),
    "Attr35": dict(
        name="Rentowność aktywów ze sprzedaży",
        formula="Zysk ze sprzedaży / Aktywa ogółem",
        info="Jak efektywnie majątek firmy generuje zysk ze sprzedaży.",
        default=0.12),
    "Attr56": dict(
        name="Marża brutto na sprzedaży",
        formula="(Przychody − Koszt sprzedanych produktów) / Przychody",
        info="Jaka część przychodów zostaje po pokryciu bezpośrednich kosztów sprzedanych produktów.",
        default=0.27),
    "Attr39": dict(
        name="Marża zysku ze sprzedaży",
        formula="Zysk ze sprzedaży / Przychody ze sprzedaży",
        info="Ile groszy zysku firma zarabia na każdej złotówce sprzedaży.",
        default=0.09),
    "Attr5": dict(
        name="Zapas płynności w dniach",
        formula="[(Środki pieniężne + Papiery krótkoterminowe + Należności − Zobowiązania krótkoterminowe) "
                "/ (Koszty operacyjne − Amortyzacja)] × 365",
        info="Przez ile dni firma mogłaby pokrywać codzienne koszty operacyjne z płynnych środków, "
             "po spłacie zobowiązań krótkoterminowych. Wartość ujemna = niedobór płynności.",
        default=20.0),
}

# Surowe dane z bilansu / rachunku wyników (klucz, etykieta, podpowiedź)
RAW = [
    ("przychody", "Przychody ze sprzedaży (ten rok)", "Rachunek zysków i strat"),
    ("przychody_prev", "Przychody ze sprzedaży (rok wcześniej)", "Do liczenia dynamiki sprzedaży"),
    ("koszt_sprzedanych", "Koszt sprzedanych produktów/towarów", "RZiS"),
    ("zysk_sprzedaz", "Zysk ze sprzedaży", "RZiS"),
    ("zysk_oper", "Zysk z działalności operacyjnej", "RZiS (EBIT)"),
    ("koszty_fin", "Koszty finansowe", "RZiS (odsetki itp.)"),
    ("koszty_oper", "Koszty operacyjne", "RZiS"),
    ("koszty_calk", "Koszty całkowite", "Wszystkie koszty działalności"),
    ("amortyzacja", "Amortyzacja", "RZiS / rachunek przepływów"),
    ("aktywa", "Aktywa ogółem", "Bilans"),
    ("aktywa_obr", "Aktywa obrotowe", "Bilans"),
    ("zapasy", "Zapasy", "Bilans"),
    ("srodki", "Środki pieniężne + papiery wartościowe krótkoterminowe", "Bilans"),
    ("naleznosci", "Należności krótkoterminowe", "Bilans"),
    ("zob_kr", "Zobowiązania krótkoterminowe", "Bilans"),
    ("zob_ogol", "Zobowiązania ogółem", "Bilans"),
    ("zysk_brutto_3l", "Zysk brutto z ostatnich 3 lat (suma)", "Suma zysków brutto z 3 lat"),
]
RAW_LABELS = {k: lbl for k, lbl, _ in RAW}

# Przykładowe dane (tys. zł) dla ostatniego roku – tylko jako wartości startowe
EXAMPLE = dict(
    przychody=12000, przychody_prev=11000, koszt_sprzedanych=8800, zysk_sprzedaz=1100,
    zysk_oper=700, koszty_fin=250, koszty_oper=11300, koszty_calk=11700, amortyzacja=400,
    aktywa=9000, aktywa_obr=4200, zapasy=1300, srodki=600, naleznosci=1900,
    zob_kr=3100, zob_ogol=5200, zysk_brutto_3l=1800,
)


# ----------------------------------------------------------------------------
# FUNKCJE POMOCNICZE
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Ładowanie modelu…")
def load_model(path: str) -> dict:
    """
    Wczytuje plik modelu. Obsługuje:
      * słownik {'pipeline', 'threshold', 'feature_names'} (Twój przypadek),
      * zwykły model / Pipeline z metodą predict_proba.
    Zwraca słownik z pipeline'em, progiem i wartościami domyślnymi (medianami z treningu)
    dla cech, których użytkownik nie podaje.
    """
    import __main__
    from custom_transformers import QuantileClipper
    # model był zapisany z klasą z modułu __main__ – udostępniamy ją pod tą nazwą
    __main__.QuantileClipper = QuantileClipper

    obj = joblib.load(path)
    if isinstance(obj, dict):
        pipe = obj["pipeline"]
        threshold = obj.get("threshold")
        names = list(obj.get("feature_names") or [])
    else:
        pipe, threshold = obj, None
        names = list(getattr(obj, "feature_names_in_", []))

    fill = None
    if names:
        fill = {n: np.nan for n in names}
        try:  # mediany z imputera -> „typowa firma” dla cech, których nie podajemy
            imp = pipe.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"]
            fill = dict(zip(names, imp.statistics_))
        except Exception:
            pass
    return {"pipeline": pipe, "threshold": threshold, "feature_names": names, "fill": fill}


def div(a, b):
    """Bezpieczne dzielenie – zwraca NaN przy braku danych lub dzieleniu przez 0."""
    if a is None or b is None or pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return a / b


def compute_indicators(d: dict) -> dict:
    """Wylicza 10 wskaźników modelu z danych finansowych."""
    attr5 = div(d["srodki"] + d["naleznosci"] - d["zob_kr"],
                d["koszty_oper"] - d["amortyzacja"])
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


def predict_bankruptcy(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    """
    Zwraca prawdopodobieństwo klasy 1 (bankructwo).
    Model wymaga wszystkich cech (np. 64), a aplikacja zbiera 10 najważniejszych –
    pozostałe są uzupełniane medianami z treningu (czyli „typową firmą”).
    """
    pipe, names, fill = bundle["pipeline"], bundle["feature_names"], bundle["fill"]
    X = X[FEATURES]
    if names:
        full = pd.DataFrame([fill] * len(X), columns=names, dtype=float)
        for f in FEATURES:
            full[f] = X[f].to_numpy(dtype=float)
        proba = pipe.predict_proba(full)
    else:
        try:
            proba = pipe.predict_proba(X)
        except Exception:
            proba = pipe.predict_proba(X.values)
    classes = list(getattr(pipe, "classes_", [0, 1]))
    idx = classes.index(1) if 1 in classes else -1
    return proba[:, idx]


def risk_label(p, low, high):
    if p < low:
        return "NISKIE", "#2e9e5b"
    if p < high:
        return "PODWYŻSZONE", "#e0a526"
    return "WYSOKIE", "#d64545"


def gauge(p, low, high):
    _, color = risk_label(p, low, high)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100,
        number={"suffix": " %", "valueformat": ".1f"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
               "steps": [{"range": [0, low * 100], "color": "#e3f4ea"},
                         {"range": [low * 100, high * 100], "color": "#fbf0d6"},
                         {"range": [high * 100, 100], "color": "#f9dcdc"}]}))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=10))
    return fig


def forecast_indicators(ind: pd.DataFrame, n_ahead: int, damping: float) -> pd.DataFrame:
    """
    Prognoza „jeśli nic się nie zmieni”: dopasowuje trend liniowy do każdego
    wskaźnika i przedłuża go na kolejne lata. Wartości są ograniczone do
    rozsądnego zakresu wokół historii, a `damping` (0–1) wygasza trend.
    """
    years = ind.index.to_numpy(dtype=float)
    last = years.max()
    future = np.arange(last + 1, last + 1 + n_ahead)
    out = pd.DataFrame(index=future.astype(int), columns=ind.columns, dtype=float)
    for col in ind.columns:
        y = ind[col].to_numpy(dtype=float)
        ok = ~np.isnan(y)
        if ok.sum() == 0:
            continue
        if ok.sum() == 1:
            out[col] = y[ok][0]
            continue
        slope, intercept = np.polyfit(years[ok], y[ok], 1)
        base = y[ok][-1]
        steps = np.arange(1, n_ahead + 1)
        # tłumienie trendu: kolejne kroki rosną coraz wolniej
        effective_steps = np.cumsum(damping ** steps)
        pred = base + slope * effective_steps
        lo, hi = y[ok].min(), y[ok].max()
        span = max(hi - lo, abs(y[ok]).mean() * 0.1, 1e-6)
        out[col] = np.clip(pred, lo - span, hi + span)
    return out


# ----------------------------------------------------------------------------
# PASEK BOCZNY
# ----------------------------------------------------------------------------
st.sidebar.header("⚙️ Ustawienia")
_thr = None
if MODEL_PATH.exists():
    _thr = load_model(str(MODEL_PATH)).get("threshold")
low_default = round(float(_thr), 2) if _thr else 0.20
high_default = max(0.25, round(low_default * 3, 2))
low_thr = st.sidebar.slider("Próg: ryzyko podwyższone od", 0.01, 0.9, low_default, 0.01, format="%.2f")
high_thr = st.sidebar.slider("Próg: ryzyko wysokie od", 0.05, 0.99, high_default, 0.01, format="%.2f")
if high_thr <= low_thr:
    st.sidebar.warning("Próg „wysokie” powinien być większy niż „podwyższone”.")
st.sidebar.caption(
    "Próg „podwyższone” ustawiono na próg decyzyjny zapisany w modelu "
    "(powyżej niego model klasyfikuje firmę jako zagrożoną). Próg „wysokie” możesz dowolnie zmienić."
)
st.sidebar.markdown("---")
st.sidebar.caption(f"Plik modelu: `{MODEL_PATH}`")

# ----------------------------------------------------------------------------
# MODEL
# ----------------------------------------------------------------------------
st.title("📉 Ocena ryzyka bankructwa firmy")
st.caption("Model ML + 10 wskaźników finansowych. Narzędzie pomocnicze – nie zastępuje analizy eksperta.")

if not MODEL_PATH.exists():
    st.error(f"Nie znaleziono pliku modelu: **{MODEL_PATH}**. "
             "Zapisz model przez `joblib.dump(model, 'model.pkl')` i umieść go obok app.py.")
    st.stop()
model = load_model(str(MODEL_PATH))

tab_data, tab_forecast, tab_manual, tab_help = st.tabs(
    ["1️⃣ Dane finansowe i wynik", "2️⃣ Trend i prognoza", "3️⃣ Wskaźniki ręcznie", "📖 Opis wskaźników"])

# ----------------------------------------------------------------------------
# TAB 1: dane finansowe -> wskaźniki -> prawdopodobieństwo
# ----------------------------------------------------------------------------
with tab_data:
    st.subheader("Wpisz dane z bilansu i rachunku zysków i strat")
    st.write("Wypełnij tabelę (każda kolumna to jeden rok). **Ostatnia kolumna = rok oceniany.** "
             "Pozostałe lata służą do pokazania trendu i prognozy. "
             "Wszystkie kwoty w tej samej jednostce (np. tys. zł).")

    c1, c2 = st.columns(2)
    last_year = c1.number_input("Ostatni rok sprawozdawczy", 2000, 2100, 2025, step=1)
    n_years = c2.slider("Liczba lat danych (min. 2 do prognozy)", 1, 5, 3)

    years = [int(last_year) - (n_years - 1 - j) for j in range(n_years)]
    default = {}
    for j, y in enumerate(years):
        k = n_years - 1 - j
        scale = 0.92 ** k
        col = {key: EXAMPLE[key] * scale for key in EXAMPLE}
        col["przychody_prev"] = EXAMPLE["przychody"] * 0.92 ** (k + 1)
        default[str(y)] = col
    df_default = pd.DataFrame(default, index=list(EXAMPLE.keys())).round(0)
    df_default.index = [RAW_LABELS[k] for k in df_default.index]

    edited = st.data_editor(df_default, key=f"editor_{n_years}_{last_year}",
                            width="stretch", height=640)
    label_to_key = {v: k for k, v in RAW_LABELS.items()}

    ind_rows = {}
    for y in years:
        col = {label_to_key[idx]: (float(v) if pd.notna(v) else np.nan)
               for idx, v in edited[str(y)].items()}
        ind_rows[y] = compute_indicators(col)
    ind_df = pd.DataFrame(ind_rows).T[FEATURES]

    probs = pd.Series(np.nan, index=ind_df.index, dtype=float)
    for y in ind_df.index:
        row = ind_df.loc[[y]]
        if not row.isna().any(axis=1).iloc[0]:
            probs[y] = predict_bankruptcy(model, row)[0]

    st.markdown("---")
    st.subheader(f"Wynik dla roku {years[-1]}")
    latest = ind_df.iloc[[-1]]
    if latest.isna().any(axis=1).iloc[0]:
        missing = [f"{c} ({INDICATORS[c]['name']})" for c in FEATURES if pd.isna(latest[c].iloc[0])]
        st.error("Nie można obliczyć wszystkich wskaźników (brak danych lub dzielenie przez 0): "
                 + ", ".join(missing))
    else:
        p = float(probs.iloc[-1])
        label, color = risk_label(p, low_thr, high_thr)
        g1, g2 = st.columns([1, 1])
        g1.plotly_chart(gauge(p, low_thr, high_thr), width="stretch")
        g2.markdown(f"### Ryzyko: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
        g2.metric("Prawdopodobieństwo bankructwa", f"{p:.1%}")
        if len(probs.dropna()) >= 2:
            delta = probs.dropna().iloc[-1] - probs.dropna().iloc[-2]
            g2.metric("Zmiana względem poprzedniego roku", f"{delta*100:+.1f} p.p.",
                      delta=f"{delta*100:+.1f}", delta_color="inverse")

    st.markdown("#### Obliczone wskaźniki")
    show = ind_df.T.copy()
    show.columns = [str(c) for c in show.columns]
    show.insert(0, "Wskaźnik", [INDICATORS[c]["name"] for c in show.index])
    st.dataframe(show.round(3), width="stretch")

# ----------------------------------------------------------------------------
# TAB 2: trend i prognoza
# ----------------------------------------------------------------------------
with tab_forecast:
    st.subheader("Prognoza „jeśli nic się nie zmieni”")
    st.write("Aplikacja przedłuża dotychczasowy trend każdego wskaźnika na kolejne lata "
             "i sprawdza, jak zmieniałoby się ryzyko bankructwa. To **ekstrapolacja trendu**, "
             "a nie pewna przepowiednia – im mniej lat danych, tym mniej wiarygodna.")

    valid_hist = ind_df.dropna()
    if len(valid_hist) < 2:
        st.info("Podaj kompletne dane z co najmniej 2 lat (zakładka 1), aby zobaczyć prognozę.")
    else:
        f1, f2 = st.columns(2)
        n_ahead = f1.slider("Ile lat do przodu", 1, 5, 3)
        damping = f2.slider("Wygaszanie trendu (1 = trend liniowy bez zmian)", 0.5, 1.0, 0.85, 0.05)

        fut = forecast_indicators(valid_hist, n_ahead, damping)
        fut_p = pd.Series(predict_bankruptcy(model, fut), index=fut.index)
        hist_p = probs.dropna()

        fig = go.Figure()
        fig.add_hrect(y0=0, y1=low_thr * 100, fillcolor="#2e9e5b", opacity=0.08, line_width=0)
        fig.add_hrect(y0=low_thr * 100, y1=high_thr * 100, fillcolor="#e0a526", opacity=0.08, line_width=0)
        fig.add_hrect(y0=high_thr * 100, y1=100, fillcolor="#d64545", opacity=0.08, line_width=0)
        fig.add_trace(go.Scatter(x=hist_p.index, y=hist_p * 100, mode="lines+markers",
                                 name="Historia", line=dict(width=3)))
        bridge_x = [hist_p.index[-1]] + list(fut_p.index)
        bridge_y = [hist_p.iloc[-1] * 100] + list(fut_p * 100)
        fig.add_trace(go.Scatter(x=bridge_x, y=bridge_y, mode="lines+markers",
                                 name="Prognoza", line=dict(width=3, dash="dash")))
        fig.update_layout(title="Prawdopodobieństwo bankructwa w czasie",
                          xaxis=dict(title="Rok", dtick=1), yaxis=dict(title="%", range=[0, 100]),
                          height=420, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width="stretch")

        st.markdown("#### Trend wybranego wskaźnika")
        chosen = st.selectbox("Wskaźnik", FEATURES,
                              format_func=lambda c: f"{c} – {INDICATORS[c]['name']}")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=valid_hist.index, y=valid_hist[chosen], mode="lines+markers",
                                  name="Historia"))
        fig2.add_trace(go.Scatter(x=[valid_hist.index[-1]] + list(fut.index),
                                  y=[valid_hist[chosen].iloc[-1]] + list(fut[chosen]),
                                  mode="lines+markers", name="Prognoza", line=dict(dash="dash")))
        fig2.update_layout(height=320, xaxis=dict(title="Rok", dtick=1),
                           margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig2, width="stretch")
        st.caption(INDICATORS[chosen]["formula"])

        out = pd.concat([hist_p.rename("Prawdop. (historia)"),
                         fut_p.rename("Prawdop. (prognoza)")], axis=1)
        st.dataframe(out.style.format("{:.1%}", na_rep="–"), width="stretch")

# ----------------------------------------------------------------------------
# TAB 3: ręczne wpisanie wskaźników
# ----------------------------------------------------------------------------
with tab_manual:
    st.subheader("Mam już gotowe wskaźniki")
    st.write("Wpisz wartości wskaźników bezpośrednio (np. gdy liczysz je w innym systemie).")
    vals = {}
    cols = st.columns(2)
    for i, f in enumerate(FEATURES):
        meta = INDICATORS[f]
        with cols[i % 2]:
            vals[f] = st.number_input(f"{f} – {meta['name']}", value=float(meta["default"]),
                                      format="%.4f", step=0.01, help=f"{meta['formula']}\n\n{meta['info']}",
                                      key=f"manual_{f}")
    if st.button("Oblicz ryzyko", type="primary"):
        p = float(predict_bankruptcy(model, pd.DataFrame([vals]))[0])
        label, color = risk_label(p, low_thr, high_thr)
        st.plotly_chart(gauge(p, low_thr, high_thr), width="stretch")
        st.markdown(f"### Ryzyko: <span style='color:{color}'>{label}</span> ({p:.1%})",
                    unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# TAB 4: opis wskaźników
# ----------------------------------------------------------------------------
with tab_help:
    st.subheader("Co oznaczają wskaźniki?")
    for f in FEATURES:
        m = INDICATORS[f]
        with st.expander(f"{f} – {m['name']}"):
            st.markdown(f"**Wzór:** {m['formula']}")
            st.write(m["info"])
