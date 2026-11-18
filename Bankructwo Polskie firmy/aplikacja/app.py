"""
Aplikacja: ocena ryzyka bankructwa firmy na podstawie własnego modelu ML.

Uruchomienie (w folderze z plikami):
    python -m pip install -r requirements.txt
    python -m streamlit run app.py
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import core_logic as cl
from core_logic import EXAMPLE_KINDS, FEATURES, INDICATORS, MAX_MISSING, RAW, RAW_KEYS, RAW_LABELS

st.set_page_config(page_title="Ryzyko bankructwa firmy", page_icon="📉", layout="wide")

APP_DIR = Path(__file__).resolve().parent      # folder, w którym leży app.py
MODEL_PATH = Path(os.environ.get("MODEL_PATH", APP_DIR / "model_10.joblib"))
TEST_CSV = APP_DIR / "dane_testowe.csv"

C_HIST, C_NOCHANGE, C_DAMPED, C_TREND = "#1f3a5f", "#8a8f98", "#2f80ed", "#e67e22"
C_RED, C_GREEN = "#d64545", "#2e9e5b"


# ----------------------------------------------------------------------------
# Pomocnicze
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Ładowanie modelu…")
def load_model(path: str) -> dict:
    return cl.load_bundle(path)


@st.cache_data(show_spinner="Liczenie predykcji na danych testowych…")
def load_scored(csv_path: str, model_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["p"] = cl.predict_proba(load_model(model_path), df[FEATURES])
    return df


def pct(x, d=1):
    return "–" if x is None or pd.isna(x) else f"{x * 100:.{d}f}%"


def gauge(p, low, high):
    _, color = cl.risk_label(p, low, high)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100,
        number={"suffix": " %", "valueformat": ".1f"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
               "steps": [{"range": [0, low * 100], "color": "#e3f4ea"},
                         {"range": [low * 100, high * 100], "color": "#fbf0d6"},
                         {"range": [high * 100, 100], "color": "#f9dcdc"}]}))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=10))
    return fig


def impact_chart(delta: pd.Series, title: str, xtitle: str, positive_is_bad=True):
    """Słupki: wpływ każdego wskaźnika (p.p. prawdopodobieństwa)."""
    s = (delta * 100).sort_values()
    bad = s > 0 if positive_is_bad else s < 0
    colors = [C_RED if b else C_GREEN for b in bad]
    labels = [f"{INDICATORS[f]['short']} ({f})" for f in s.index]
    fig = go.Figure(go.Bar(x=s.values, y=labels, orientation="h", marker_color=colors,
                           text=[f"{v:+.1f}" for v in s.values], textposition="outside",
                           hovertemplate="%{y}: %{x:+.2f} p.p.<extra></extra>"))
    pad = max(1.0, float(np.abs(s.values).max()) * 0.25)
    fig.update_layout(title=title, height=380, margin=dict(l=10, r=30, t=50, b=40),
                      xaxis=dict(title=xtitle, zeroline=True, zerolinecolor="#999",
                                 range=[min(0, s.min()) - pad, max(0, s.max()) + pad]),
                      yaxis=dict(automargin=True))
    return fig


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
st.title("📉 Ocena ryzyka bankructwa firmy")
st.caption("Model ML + 10 wskaźników finansowych. Narzędzie pomocnicze – nie zastępuje analizy eksperta.")

if not MODEL_PATH.exists():
    st.error(f"Nie znaleziono pliku modelu: **{MODEL_PATH}**. Umieść `model.joblib` obok `app.py`.")
    st.stop()
try:
    bundle = load_model(str(MODEL_PATH))
except Exception as e:  # najczęściej: inna wersja xgboost / scikit-learn niż przy zapisie modelu
    st.error("Nie udało się wczytać modelu. Najczęstsza przyczyna: inna wersja `xgboost` lub "
             "`scikit-learn` niż ta, w której model zapisano.")
    st.code(f"{type(e).__name__}: {e}")
    st.markdown("**Rozwiązanie:** wytrenuj model u siebie (kilka minut), dzięki czemu będzie zgodny z Twoimi "
                "wersjami bibliotek:\n\n"
                "`python train_model_10.py sciezka\\do\\5year.arff model.joblib`")
    st.stop()

typ = cl.typical_firm(bundle)
thr_model = float(bundle["threshold"]) if bundle["threshold"] else 0.20
p_typ = float(cl.predict_proba(bundle, typ.to_frame().T)[0])

# ----------------------------------------------------------------------------
# Pasek boczny
# ----------------------------------------------------------------------------
st.sidebar.header("⚙️ Ustawienia")
low_default = round(thr_model, 2)
high_default = max(0.25, round(low_default * 3, 2))
low_thr = st.sidebar.slider("Ryzyko „podwyższone” od", 0.01, 0.9, low_default, 0.01, format="%.2f")
high_thr = st.sidebar.slider("Ryzyko „wysokie” od", 0.05, 0.99, high_default, 0.01, format="%.2f")
if high_thr <= low_thr:
    st.sidebar.warning("Próg „wysokie” powinien być większy niż „podwyższone”.")
st.sidebar.caption(f"Domyślny próg „podwyższone” = próg decyzyjny modelu ({thr_model:.1%}): powyżej niego "
                   "model klasyfikuje firmę jako zagrożoną.")
with st.sidebar.expander("ℹ️ O modelu"):
    ex = bundle["extra"]
    tm = ex.get("test_metrics", {})
    st.write(f"**Typ:** {ex.get('model_type', type(bundle['pipeline']).__name__)}")
    if tm:
        st.write(f"**Test (firmy z kompletnymi danymi):** AP {tm['AP']:.2f}, ROC-AUC {tm['ROC_AUC']:.2f}, "
                 f"czułość {tm['Recall']:.0%}, precyzja {tm['Precision']:.0%}")
    st.caption("Dane: polskie firmy z lat 2000–2013 (UCI, `5year`), prognoza bankructwa w ciągu roku. "
               "W zbiorze ok. 7% firm to bankruci, więc prawdopodobieństwa są zawyżone względem "
               "całej populacji firm.")

tab_data, tab_forecast, tab_manual, tab_test, tab_help = st.tabs(
    ["1️⃣ Dane i wynik", "2️⃣ Ścieżka i prognoza", "3️⃣ Wskaźniki ręcznie", "🧪 Test na danych", "📖 Opis wskaźników"])

# ----------------------------------------------------------------------------
# TAB 1: dane -> wskaźniki -> prawdopodobieństwo
# ----------------------------------------------------------------------------
with tab_data:
    st.subheader("Dane finansowe firmy")
    st.write("Wpisz dane z bilansu i rachunku zysków i strat – jedna kolumna na rok, **ostatnia kolumna = rok oceniany**. "
             "Wszystkie kwoty w tej samej jednostce (np. tys. zł). Starsze lata służą do pokazania trendu.")

    c1, c2, c3 = st.columns([1, 1, 1.4])
    last_year = c1.number_input("Ostatni rok sprawozdawczy", 2000, 2100, 2025, step=1)
    n_years = c2.slider("Liczba lat danych", 1, 5, 4)
    example = c3.selectbox("Dane startowe", EXAMPLE_KINDS, index=0,
                           help="Wybierz przykład albo „Puste” i wpisz własne dane.")

    df_default = cl.example_data(example, n_years, int(last_year))
    df_show = df_default.copy()
    df_show.index = [RAW_LABELS[k] for k in df_show.index]
    edited = st.data_editor(df_show, key=f"editor_{example}_{n_years}_{last_year}", width="stretch", height=640)
    raw = edited.copy()
    raw.index = RAW_KEYS

    with st.expander("Skąd wziąć te dane?"):
        st.dataframe(pd.DataFrame(RAW, columns=["klucz", "Pozycja", "Gdzie szukać"]).drop(columns="klucz"),
                     width="stretch", hide_index=True)

    ind_df = cl.indicators_table(raw)
    probs = cl.predict_rows(bundle, ind_df)

    st.markdown("---")
    last = ind_df.index[-1]
    row = ind_df.loc[last]
    missing = [f"{f} ({INDICATORS[f]['name']})" for f in FEATURES if pd.isna(row[f])]
    p_last = None
    st.subheader(f"Wynik dla roku {last}")
    if len(missing) > MAX_MISSING:
        st.error("Za mało danych: nie da się obliczyć wskaźników " + ", ".join(missing) +
                 f". Model akceptuje brak co najwyżej {MAX_MISSING} wskaźników.")
    else:
        if missing:
            st.warning("Nie da się obliczyć: " + ", ".join(missing) +
                       " (brak danych lub dzielenie przez 0). Model użył dla nich wartości typowej, "
                       "więc wynik jest mniej pewny.")
        p_last = float(probs[last])
        label, color = cl.risk_label(p_last, low_thr, high_thr)
        g1, g2 = st.columns([1, 1])
        g1.plotly_chart(gauge(p_last, low_thr, high_thr), width="stretch")
        g2.markdown(f"### Ryzyko: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
        m1, m2 = g2.columns(2)
        m1.metric("Prawdopodobieństwo bankructwa", pct(p_last))
        m2.metric("Typowa firma ze zbioru", pct(p_typ), delta=f"×{p_last / p_typ:.1f}" if p_typ > 0 else None,
                  delta_color="off")
        prev_ok = probs.iloc[:-1].dropna()
        if len(prev_ok):
            d = p_last - prev_ok.iloc[-1]
            g2.metric(f"Zmiana względem {prev_ok.index[-1]}", f"{d * 100:+.1f} p.p.", delta=f"{d * 100:+.1f}",
                      delta_color="inverse")
        g2.write("**Decyzja modelu:** " + ("⚠️ firma zagrożona" if p_last >= thr_model else "✅ firma niezagrożona")
                 + f" (próg decyzyjny {thr_model:.1%}).")

        imp, _ = cl.swap_impacts(bundle, row, typ)
        st.plotly_chart(
            impact_chart(-imp, "Co podnosi, a co obniża ryzyko tej firmy (względem firmy typowej)",
                         "wpływ na prawdopodobieństwo (p.p.)  →  czerwone: podnosi, zielone: obniża"),
            width="stretch")
        st.caption("Dla każdego wskaźnika sprawdzam, o ile zmieniłoby się ryzyko, gdyby miał wartość typową "
                   "(mediana ze zbioru treningowego), a reszta zostałaby bez zmian. To analiza „po jednym” – "
                   "wpływy mogą się nie sumować, bo model uwzględnia interakcje.")

    st.markdown("#### Obliczone wskaźniki")
    show = ind_df.T.copy()
    show.columns = [str(c) for c in show.columns]
    show.insert(0, "Wskaźnik", [INDICATORS[f]["name"] for f in show.index])
    show["Typowa firma"] = [typ[f] for f in show.index]
    st.dataframe(show.round(3), width="stretch")

# ----------------------------------------------------------------------------
# TAB 2: ścieżka i prognoza
# ----------------------------------------------------------------------------
with tab_forecast:
    st.subheader("Ścieżka ryzyka w czasie")
    st.write("Pokazuję, jak ryzyko zmieniało się w latach, które wpisałeś, i jak będzie się zmieniać, "
             "jeśli firma pójdzie jedną z trzech ścieżek: **nic się nie zmienia**, **trend stopniowo wygasa** "
             "albo **trend trwa bez zmian**.")

    hist_ind = ind_df[ind_df.isna().sum(axis=1) <= MAX_MISSING]
    if len(hist_ind) < 2:
        st.info("Do prognozy potrzebne są dane z co najmniej 2 lat z możliwie kompletnymi wskaźnikami "
                "(zakładka 1).")
    else:
        f1, f2 = st.columns(2)
        n_ahead = f1.slider("Horyzont prognozy (lata)", 1, 5, 3)
        damping = f2.slider("Wygaszanie trendu w scenariuszu środkowym", 0.3, 1.0, 0.8, 0.05,
                            help="1 = trend bez zmian, im mniej, tym szybciej trend wygasa.")
        scen = {"Bez zmian": (0.0, C_NOCHANGE), "Trend wygaszony": (damping, C_DAMPED),
                "Trend utrzymany": (1.0, C_TREND)}
        fut_ind = {n: cl.forecast_indicators(hist_ind, n_ahead, d) for n, (d, _) in scen.items()}
        fut_p = {n: cl.predict_rows(bundle, f) for n, f in fut_ind.items()}
        hist_p = probs.loc[hist_ind.index]
        last_y, last_p = int(hist_p.index[-1]), float(hist_p.iloc[-1])
        fut_years = list(fut_p["Bez zmian"].index)

        # --- podsumowanie scenariuszy
        cols = st.columns(len(scen))
        for col, (n, _) in zip(cols, scen.items()):
            end = fut_p[n].iloc[-1]
            col.metric(f"{n} → {fut_years[-1]}", pct(end),
                       delta=None if pd.isna(end) else f"{(end - last_p) * 100:+.1f} p.p.", delta_color="inverse")

        # --- wykres 1: ścieżka prawdopodobieństwa
        all_vals = [hist_p.max()] + [s.max() for s in fut_p.values()]
        ymax = min(100.0, max(10.0, 125 * max(np.nanmax(all_vals), low_thr * 0.9)))
        fig = go.Figure()
        for a, b, color in [(0, low_thr, C_GREEN), (low_thr, high_thr, "#e0a526"), (high_thr, 1, C_RED)]:
            if a * 100 < ymax:
                fig.add_hrect(y0=a * 100, y1=min(b * 100, ymax), fillcolor=color, opacity=0.07,
                              line_width=0, layer="below")
        for a, b, txt, color in [(0, low_thr, "niskie ryzyko", C_GREEN),
                                 (low_thr, high_thr, "podwyższone", "#b9851a"), (high_thr, 1, "wysokie", C_RED)]:
            if a * 100 < ymax:
                fig.add_annotation(xref="paper", x=0.995, xanchor="right", y=(a * 100 + min(b * 100, ymax)) / 2,
                                   text=txt, showarrow=False, font=dict(size=11, color=color))
        xs_f = [last_y] + fut_years
        up = [last_p * 100] + list(np.nanmax([fut_p[n].values for n in scen], axis=0) * 100)
        lo = [last_p * 100] + list(np.nanmin([fut_p[n].values for n in scen], axis=0) * 100)
        fig.add_trace(go.Scatter(x=xs_f, y=up, mode="lines", line=dict(width=0), showlegend=False,
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xs_f, y=lo, mode="lines", line=dict(width=0), fill="tonexty",
                                 fillcolor="rgba(47,128,237,0.12)", name="Zakres scenariuszy", hoverinfo="skip"))
        for n, (_, color) in scen.items():
            ys = [last_p * 100] + list(fut_p[n].values * 100)
            fig.add_trace(go.Scatter(x=xs_f, y=ys, mode="lines+markers", name=n,
                                     line=dict(color=color, width=2.5, dash="dash"), marker=dict(size=7),
                                     hovertemplate=f"{n}: " + "%{y:.1f}%<extra>%{x}</extra>"))
        fig.add_trace(go.Scatter(x=list(hist_p.index), y=list(hist_p.values * 100), mode="lines+markers+text",
                                 name="Historia", line=dict(color=C_HIST, width=4), marker=dict(size=9),
                                 text=[f"{v * 100:.1f}%" for v in hist_p.values], textposition="top center",
                                 hovertemplate="Historia: %{y:.1f}%<extra>%{x}</extra>"))
        fig.add_vline(x=last_y, line_dash="dot", line_color="#999")
        for n in ("Trend wygaszony", "Trend utrzymany"):      # kiedy ryzyko przekroczy próg?
            s = fut_p[n]
            hit = s[s >= low_thr]
            if len(hit) and last_p < low_thr:
                yr = int(hit.index[0])
                fig.add_annotation(x=yr, y=float(hit.iloc[0]) * 100, text=f"{n}:<br>próg ryzyka w {yr}",
                                   showarrow=True, arrowhead=2, ay=-45, font=dict(size=10, color="#333"),
                                   bgcolor="rgba(255,255,255,0.85)", bordercolor="#bbb")
        fig.update_layout(height=470, margin=dict(l=10, r=10, t=30, b=10),
                          xaxis=dict(title="Rok", dtick=1, range=[hist_p.index[0] - 0.4, fut_years[-1] + 0.4]),
                          yaxis=dict(title="Prawdopodobieństwo bankructwa (%)", range=[0, ymax]),
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width="stretch")

        # --- wykres 2: ścieżki wskaźników
        st.markdown("#### Jak zmieniają się poszczególne wskaźniki")
        fig2 = make_subplots(rows=2, cols=5, subplot_titles=[INDICATORS[f]["short"] for f in FEATURES],
                             vertical_spacing=0.2, horizontal_spacing=0.05)
        mid = fut_ind["Trend wygaszony"]
        all_years = list(hist_ind.index) + list(mid.index)
        for i, f in enumerate(FEATURES):
            r, c = i // 5 + 1, i % 5 + 1
            fig2.add_trace(go.Scatter(x=all_years, y=[typ[f]] * len(all_years), mode="lines",
                                      line=dict(color="#aaa", width=1.2, dash="dot"), hoverinfo="skip"), r, c)
            fig2.add_trace(go.Scatter(x=list(hist_ind.index), y=list(hist_ind[f]), mode="lines+markers",
                                      line=dict(color=C_HIST, width=2.5), marker=dict(size=6),
                                      hovertemplate="%{y:.3f}<extra>%{x}</extra>"), r, c)
            fig2.add_trace(go.Scatter(x=[last_y] + list(mid.index), y=[hist_ind[f].iloc[-1]] + list(mid[f]),
                                      mode="lines+markers", line=dict(color=C_DAMPED, width=2, dash="dash"),
                                      marker=dict(size=5), hovertemplate="%{y:.3f}<extra>%{x}</extra>"), r, c)
        fig2.update_xaxes(dtick=1, tickformat="d", tickfont=dict(size=9))
        fig2.update_yaxes(tickfont=dict(size=9))
        fig2.update_annotations(font_size=11)
        fig2.update_layout(height=520, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig2, width="stretch")
        st.caption("Linia ciągła – historia, przerywana – prognoza (scenariusz „Trend wygaszony”), "
                   "kropkowana szara – wartość typowa dla firm w zbiorze treningowym.")

        # --- wykres 3: co napędza zmianę
        base_row = hist_ind.iloc[-1]
        end_row = mid.iloc[-1]
        imp2, p0 = cl.swap_impacts(bundle, base_row, end_row)
        p_end = fut_p["Trend wygaszony"].iloc[-1]
        st.plotly_chart(
            impact_chart(imp2, f"Co zmienia ryzyko między {last_y} a {fut_years[-1]} "
                               f"(scenariusz „Trend wygaszony”: {pct(p0)} → {pct(p_end)})",
                         "zmiana prawdopodobieństwa (p.p.), gdy zmienia się tylko ten wskaźnik"),
            width="stretch")

        # --- tabela
        tab = pd.DataFrame({"Historia": hist_p}).reindex(list(hist_p.index) + fut_years)
        for n in scen:
            tab[n] = fut_p[n]
        st.dataframe(tab.style.format("{:.1%}", na_rep="–"), width="stretch")
        st.caption("⚠️ To ekstrapolacja trendu z kilku lat, a nie prognoza ekonomiczna – im mniej lat danych, "
                   "tym mniej wiarygodna. Służy do pokazania kierunku, w którym zmierza firma.")

# ----------------------------------------------------------------------------
# TAB 3: ręczne wpisanie wskaźników
# ----------------------------------------------------------------------------
with tab_manual:
    st.subheader("Mam już gotowe wskaźniki")
    st.write("Wpisz wartości wskaźników bezpośrednio. Domyślnie ustawiona jest firma typowa "
             "(mediany ze zbioru treningowego).")
    vals = {}
    cols = st.columns(2)
    for i, f in enumerate(FEATURES):
        meta = INDICATORS[f]
        with cols[i % 2]:
            vals[f] = st.number_input(f"{f} – {meta['name']}", value=float(round(typ[f], 4)), format="%.4f",
                                      step=0.01, help=f"{meta['formula']}\n\n{meta['info']}", key=f"manual_{f}")
    pm = float(cl.predict_proba(bundle, pd.DataFrame([vals]))[0])
    label, color = cl.risk_label(pm, low_thr, high_thr)
    mm1, mm2 = st.columns([1, 1])
    mm1.plotly_chart(gauge(pm, low_thr, high_thr), width="stretch")
    mm2.markdown(f"### Ryzyko: <span style='color:{color}'>{label}</span> ({pct(pm)})", unsafe_allow_html=True)
    mm2.write("**Decyzja modelu:** " + ("⚠️ firma zagrożona" if pm >= thr_model else "✅ firma niezagrożona"))
    imp_m, _ = cl.swap_impacts(bundle, pd.Series(vals), typ)
    st.plotly_chart(impact_chart(-imp_m, "Co podnosi, a co obniża ryzyko (względem firmy typowej)",
                                 "wpływ na prawdopodobieństwo (p.p.)"), width="stretch")

# ----------------------------------------------------------------------------
# TAB 4: test na danych z etykietami
# ----------------------------------------------------------------------------
with tab_test:
    st.subheader("Test modelu na prawdziwych firmach")
    st.write("Tu sprawdzisz, czy model zwraca to, co powinien: bierzemy firmy ze zbioru, dla których **znamy "
             "prawdziwy wynik** (czy zbankrutowały), i porównujemy z prognozą modelu.")
    if not TEST_CSV.exists():
        st.info("Brak pliku `dane_testowe.csv` obok `app.py` – ta zakładka go potrzebuje.")
    else:
        df_all = load_scored(str(TEST_CSV), str(MODEL_PATH))
        choice = st.radio("Zbiór", ["Testowy (model go nie widział)",
                                    "Treningowy (model się na nim uczył – wyniki zawyżone)"], horizontal=True)
        is_test = choice.startswith("Testowy")
        df = df_all[df_all["zbior"] == ("test" if is_test else "trening")].reset_index(drop=True)
        if not is_test:
            st.warning("Na zbiorze treningowym model „zna odpowiedzi”, więc wyniki są nierealnie dobre. "
                       "Rzetelną ocenę daje tylko zbiór testowy.")
        ev = cl.evaluate(df["class"], df["p"], thr_model)

        k = st.columns(6)
        k[0].metric("Firm", f"{ev['n']}")
        k[1].metric("Bankrutów", f"{ev['n_pos']}")
        k[2].metric("ROC-AUC", f"{ev['ROC_AUC']:.3f}")
        k[3].metric("Average Precision", f"{ev['AP']:.3f}")
        k[4].metric("Czułość", pct(ev["recall"], 0))
        k[5].metric("Precyzja", pct(ev["precision"], 0))
        st.write(f"Przy progu decyzyjnym {thr_model:.1%} model wykrył **{ev['TP']} z {ev['n_pos']}** bankructw, "
                 f"przeoczył {ev['FN']} i wywołał {ev['FP']} fałszywych alarmów "
                 f"(na {ev['n'] - ev['n_pos']} firm niezagrożonych).")

        t1, t2 = st.columns(2)
        cal = cl.calibration_table(df["class"], df["p"])
        figc = go.Figure()
        figc.add_trace(go.Bar(x=cal["label"], y=cal["pred"] * 100, name="Prognoza modelu (średnia)",
                              marker_color=C_DAMPED))
        figc.add_trace(go.Bar(x=cal["label"], y=cal["obs"] * 100, name="Rzeczywisty odsetek bankructw",
                              marker_color=C_RED, text=[f"{a}/{b}" for a, b in zip(cal["n_pos"], cal["n"])],
                              textposition="outside"))
        figc.update_layout(barmode="group", height=380, title="Czy prawdopodobieństwa się zgadzają?",
                           xaxis_title="Przedział prognozy (liczba bankrutów/firm w przedziale)",
                           yaxis_title="%", legend=dict(orientation="h", y=-0.35), margin=dict(l=10, r=10, t=50, b=10))
        t1.plotly_chart(figc, width="stretch")

        grid = np.unique(np.concatenate([np.linspace(0.005, 0.1, 40), np.linspace(0.1, 0.9, 30)]))
        tc = cl.threshold_curve(df["class"], df["p"], grid)
        figt = go.Figure()
        for col, name, color in [("recall", "Czułość (wykryte bankructwa)", C_GREEN),
                                 ("precision", "Precyzja (trafność alarmów)", C_DAMPED),
                                 ("false_alarm", "Fałszywe alarmy (% firm zdrowych)", C_RED)]:
            figt.add_trace(go.Scatter(x=tc["threshold"] * 100, y=tc[col] * 100, mode="lines", name=name,
                                      line=dict(color=color, width=2.5)))
        figt.add_vline(x=thr_model * 100, line_dash="dot", line_color="#555",
                       annotation_text="próg modelu", annotation_position="top")
        figt.update_layout(height=380, title="Jak próg wpływa na wykrywalność", xaxis_title="Próg decyzyjny (%)",
                           xaxis_type="log", yaxis_title="%", legend=dict(orientation="h", y=-0.35),
                           margin=dict(l=10, r=10, t=50, b=10))
        t2.plotly_chart(figt, width="stretch")
        st.caption("Przedziały z niewielką liczbą bankrutów (liczby nad słupkami) są obarczone dużym szumem "
                   "– zbiór testowy ma tylko kilkadziesiąt bankructw.")

        # --- pojedyncza firma
        st.markdown("---")
        st.markdown("#### Sprawdź pojedynczą firmę")
        fc1, fc2 = st.columns([1.3, 1])
        filt = fc1.radio("Pokaż firmy", ["Wszystkie", "Tylko bankruci", "Tylko niezagrożone"], horizontal=True)
        sort_risk = fc2.checkbox("Sortuj wg prognozy (od najwyższego ryzyka)", value=True)
        sub = df if filt == "Wszystkie" else df[df["class"] == (1 if filt == "Tylko bankruci" else 0)]
        sub = (sub.sort_values("p", ascending=False) if sort_risk else sub).reset_index(drop=True)

        if "firm_idx" not in st.session_state:
            st.session_state["firm_idx"] = 1
        st.session_state["firm_idx"] = int(min(max(1, st.session_state["firm_idx"]), len(sub)))

        def _random_firm():
            st.session_state["firm_idx"] = int(np.random.randint(1, len(sub) + 1))

        n1, n2 = st.columns([1, 1])
        idx = n1.number_input(f"Numer firmy (1–{len(sub)})", 1, len(sub), key="firm_idx")
        n2.write("")
        n2.button("🎲 Losowa firma", on_click=_random_firm)

        firm = sub.iloc[int(idx) - 1]
        p_f = float(firm["p"])
        truth = int(firm["class"])
        label, color = cl.risk_label(p_f, low_thr, high_thr)
        flagged = p_f >= thr_model
        if truth == 1 and flagged:
            verdict = "✅ Zgodne – bankrut został wykryty."
        elif truth == 0 and not flagged:
            verdict = "✅ Zgodne – firma niezagrożona i tak oceniona."
        elif truth == 0:
            verdict = "❌ Fałszywy alarm – firma nie zbankrutowała, a model ją oznaczył."
        else:
            verdict = "❌ Przeoczone bankructwo – model nie oznaczył firmy, a ta zbankrutowała."

        r1, r2 = st.columns([1, 1])
        r1.plotly_chart(gauge(p_f, low_thr, high_thr), width="stretch")
        r2.markdown(f"### Prognoza: <span style='color:{color}'>{label}</span> ({pct(p_f)})", unsafe_allow_html=True)
        r2.write(f"**Rzeczywiście:** {'🔴 zbankrutowała' if truth else '🟢 nie zbankrutowała'}")
        r2.write(f"**Decyzja modelu:** {'⚠️ zagrożona' if flagged else '✅ niezagrożona'}")
        r2.write(verdict)
        tbl = pd.DataFrame({"Wskaźnik": [INDICATORS[f]["name"] for f in FEATURES],
                            "Wartość": [firm[f] for f in FEATURES], "Typowa firma": [typ[f] for f in FEATURES]},
                           index=FEATURES)
        st.dataframe(tbl.round(3), width="stretch")

        with st.expander("🔧 Zmień wskaźniki tej firmy i zobacz, jak zmieni się ryzyko"):
            wv = {}
            wcols = st.columns(2)
            for i, f in enumerate(FEATURES):
                wv[f] = wcols[i % 2].number_input(f"{f} – {INDICATORS[f]['short']}", value=float(round(firm[f], 4)),
                                                  format="%.4f", step=0.01,
                                                  key=f"wi_{'t' if is_test else 'tr'}_{int(firm['id'])}_{f}")
            p_new = float(cl.predict_proba(bundle, pd.DataFrame([wv]))[0])
            st.metric("Nowe prawdopodobieństwo", pct(p_new), delta=f"{(p_new - p_f) * 100:+.1f} p.p.",
                      delta_color="inverse")

# ----------------------------------------------------------------------------
# TAB 5: opis wskaźników
# ----------------------------------------------------------------------------
with tab_help:
    st.subheader("Co oznaczają wskaźniki?")
    for f in FEATURES:
        m = INDICATORS[f]
        with st.expander(f"{f} – {m['name']}   (typowo: {typ[f]:.3g})"):
            st.markdown(f"**Wzór:** {m['formula']}")
            st.write(m["info"])
