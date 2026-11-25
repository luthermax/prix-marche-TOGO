from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

FR = {
    "Maize (white)": "Maïs blanc",
    "Cassava meal (gari)": "Gari (manioc)",
    "Beans (white)": "Haricots blancs",
    "Beans (red)": "Haricots rouges",
    "Soybeans": "Soja",
    "Sorghum (red)": "Sorgho rouge",
    "Rice (local)": "Riz local",
    "Millet": "Mil",
}
MOIS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
        "septembre", "octobre", "novembre", "décembre"]
MOIS_C = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]
NIVEAU_FR = {"forte": "Alerte forte", "vigilance": "Vigilance", "": "—"}
GOLD, GREY = "#F2A900", "#8A93A3"

# Résultats du backtest 2022-2026 (calculés dans les notebooks 02, 03 et 04)
MAPE = pd.DataFrame({
    "Horizon": ["1 mois", "2 mois", "3 mois"] * 3,
    "Méthode": ["Naïve"] * 3 + ["Saisonnière"] * 3 + ["Modèle LightGBM"] * 3,
    "MAPE (%)": [10.0, 14.5, 17.8, 23.8, 23.8, 23.8, 10.0, 13.9, 16.4],
})
PRECISION_H3 = pd.DataFrame({
    "Règle": ["Hasard", "Momentum 3 mois", "Rebond", "Modèle"],
    "Précision (%)": [21.6, 14.1, 49.0, 58.1],
})
GAIN_ANNEE = pd.DataFrame({"Année": ["2022", "2023", "2024", "2025", "2026"],
                           "Gain vs naïve (%)": [10.2, 6.5, 2.5, 11.2, 6.9]})
GAIN_PRODUIT = pd.DataFrame({
    "Produit": ["Haricots blancs", "Maïs blanc", "Mil", "Haricots rouges", "Sorgho rouge",
                "Soja", "Riz local", "Gari (manioc)"],
    "Gain vs naïve (%)": [11.0, 7.9, 7.8, 7.7, 7.0, 6.9, 6.4, 6.1]})

st.set_page_config(page_title="Prix des marchés au Togo", layout="wide")


def mois_fr(ts):
    return f"{MOIS[ts.month - 1]} {ts.year}"


@st.cache_data(show_spinner="Chargement des données…")
def load():
    f = pd.read_csv(PROC / "previsions.csv", parse_dates=["date_base", "date_cible"])
    p = pd.read_csv(PROC / "panel_long_brut.csv", parse_dates=["date"])
    f["alerte"] = f["alerte"].fillna("")
    for d in (f, p):
        d["produit"] = d["commodity"].map(FR).fillna(d["commodity"])
    c_last, c_n = None, None
    path = PROC / "marches_courts.csv"
    if path.exists():
        c = pd.read_csv(path, parse_dates=["date"]).dropna(subset=["price"])
        c["produit"] = c["commodity"].map(FR).fillna(c["commodity"])
        c_n = c.groupby("date")["market"].nunique()
        c_last = c.sort_values("date").groupby(["market", "commodity"]).tail(1)
    return f, p, c_last, c_n


def style(fig, height=380):
    fig.update_layout(template="plotly_white", height=height, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(size=13),
                      xaxis=dict(gridcolor="#E8ECF1"), yaxis=dict(gridcolor="#E8ECF1"))
    return fig


def make_map(df, **kw):
    if hasattr(px, "scatter_map"):
        return px.scatter_map(df, lat="latitude", lon="longitude",
                              map_style="carto-darkmatter", **kw)
    return px.scatter_mapbox(df, lat="latitude", lon="longitude",
                             mapbox_style="carto-darkmatter", **kw)


f, p, c_last, c_n = load()
base = f["date_base"].iloc[0]

# ------------------------------------------------------------------ filtres
with st.sidebar:
    st.header("Filtres")
    produit = st.selectbox("Produit", sorted(f["produit"].unique()))
    h = st.segmented_control("Horizon de la carte", [1, 2, 3], default=3,
                             format_func=lambda x: f"{x} mois")
    regions = sorted(f["admin1"].dropna().unique())
    sel_reg = st.multiselect("Région", regions, default=regions) or regions
    show_courts = st.checkbox("Afficher les marchés récents (sans prévision)", value=True)
    st.divider()
    st.caption(f"Dernier mois connu : **{mois_fr(base)}**")
    st.caption("Source : PAM (WFP) via HDX. Projet de prévision des prix des denrées au Togo.")

d = f[(f["produit"] == produit) & (f["h"] == h) & (f["admin1"].isin(sel_reg))].copy()
if d.empty:
    st.warning("Aucune donnée pour cette sélection.")
    st.stop()
d["pct"] = (100 * d["hausse_prevue"]).round(1)
d["niveau"] = d["alerte"].map(NIVEAU_FR)
target = d["date_cible"].iloc[0]
lim = max(10.0, float(d["pct"].abs().quantile(0.98)))

st.title("Prix des marchés au Togo")
st.caption(
    f"Prix mensuels jusqu’à {mois_fr(base)} · "
    f"prévision à {h} mois, jusqu’à {mois_fr(target)}"
)

k1, k2, k3 = st.columns(3)
k1.metric("Marchés couverts", d["market"].nunique())
k2.metric("Prix médian actuel", f"{d['prix_actuel'].median():.0f} FCFA/kg")
k3.metric("Variation médiane prévue", f"{d['pct'].median():+.1f} %")

t_map, t_evo, t_al, t_analyse = st.tabs(
    ["Marchés", "Évolution", "Alertes", "Analyses"]
)
with t_analyse:
    st.caption("Saisonnalité, comparaison des produits, performance du modèle et qualité des données.")
    t_sai, t_prod, t_mod, t_dat = st.tabs(
        ["Saisonnalité", "Produits", "Modèle", "Qualité des données"]
    )

# Marchés
with t_map:
    left, right = st.columns([3, 2])
    with left:
        d["taille"] = np.where(d["alerte"] != "", 2.0, 1.0)
        fig = make_map(
            d, color="pct", size="taille", size_max=16, hover_name="market",
            hover_data={"pct": ":.1f", "prix_actuel": ":.0f", "prix_prevu": ":.0f",
                        "niveau": True, "taille": False, "latitude": False, "longitude": False},
            color_continuous_scale="RdBu_r", range_color=[-lim, lim],
            labels={"pct": "Variation (%)", "prix_actuel": "Prix actuel", "prix_prevu": "Prix prévu",
                    "niveau": "Alerte"},
            center=dict(lat=8.6, lon=1.2), zoom=5.6, height=620)
        if show_courts and c_last is not None:
            cc = c_last[(c_last["produit"] == produit) & (c_last["admin1"].isin(sel_reg))].copy()
            if len(cc):
                cc["groupe"] = "Marchés récents (prix actuels, sans prévision)"
                cc["prix"] = cc["price"].round(0)
                f2 = make_map(cc, color="groupe", color_discrete_sequence=[GREY], hover_name="market",
                              hover_data={"prix": True, "groupe": False, "latitude": False,
                                          "longitude": False},
                              labels={"prix": "Prix actuel (FCFA/kg)"})
                f2.update_traces(marker=dict(size=7, opacity=0.7))
                fig.add_traces(list(f2.data))
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)",
                          legend=dict(x=0.01, y=0.99, bgcolor="rgba(22,27,38,0.8)", title_text=""),
                          coloraxis_colorbar=dict(title="Variation %", len=0.5))
        st.plotly_chart(
            fig,
            alt="Carte des hausses et baisses prévues par marché, avec les alertes mises en évidence",
        )
        st.caption("Rouge : hausse prévue · bleu : baisse prévue · points plus grands : alerte.")
    with right:
        top = d.nlargest(10, "pct").sort_values("pct")
        fig = px.bar(top, x="pct", y="market", orientation="h", color="pct", text="pct",
                     color_continuous_scale="RdBu_r", range_color=[-lim, lim],
                     title="Plus fortes hausses prévues (%)", labels={"pct": "%", "market": ""})
        fig.update_coloraxes(showscale=False)
        fig.update_traces(texttemplate="%{text:+.1f}")
        st.plotly_chart(style(fig, 360), alt="Dix marchés avec les plus fortes hausses prévues")

# Évolution
with t_evo:
    c1, c2 = st.columns([2, 1])
    m = c1.selectbox("Marché", sorted(d["market"].unique()))
    nmois = c2.slider("Fenêtre d'historique (mois)", 24, 120, 60, step=12)

    hs = p[(p["produit"] == produit) & (p["admin1"].isin(sel_reg))]
    hs = hs[hs["date"] >= base - pd.DateOffset(months=nmois)]
    band = hs.groupby("date")["price"].quantile([0.1, 0.5, 0.9]).unstack()
    mk = hs[hs["market"] == m].set_index("date")["price"].sort_index()
    fc = f[(f["market"] == m) & (f["produit"] == produit)].sort_values("h")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=band.index, y=band[0.9], line=dict(width=0),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=band.index, y=band[0.1], fill="tonexty",
                             fillcolor="rgba(138,147,163,0.20)", line=dict(width=0),
                             name="Fourchette 10-90 % des marchés", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=band.index, y=band[0.5], name="Médiane des marchés",
                             line=dict(color=GREY, width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=mk.index, y=mk.values, name=m, line=dict(color=GOLD, width=2.5)))
    if len(fc) and len(mk):
        fx = [mk.index[-1]] + list(fc["date_cible"])
        fy = [mk.iloc[-1]] + list(fc["prix_prevu"])
        fig.add_trace(go.Scatter(x=fx, y=fy, name="Prévision", mode="lines+markers",
                                 line=dict(color="#ff5c5c", width=2.5, dash="dash"),
                                 marker=dict(size=9)))
    fig.update_layout(title=f"{produit} · {m}", yaxis_title="FCFA/kg", hovermode="x unified",
                      legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(style(fig, 460), alt="Historique des prix du marché sélectionné et prévisions à trois mois")

    if len(fc):
        cols = st.columns(len(fc))
        for col, (_, r) in zip(cols, fc.iterrows()):
            col.metric(f"Prévu en {mois_fr(r['date_cible'])}", f"{r['prix_prevu']:.0f} FCFA/kg",
                       f"{100 * r['hausse_prevue']:+.1f} %", delta_color="inverse")
            if r["alerte"]:
                col.warning(f"{NIVEAU_FR[r['alerte']]} à {int(r['h'])} mois")

# Analyses · saisonnalité
with t_sai:
    s = (p[(p["produit"] == produit) & (p["admin1"].isin(sel_reg))]
         .groupby("date")["price"].median())
    s = s[s.index >= "2015-01-01"]
    heat = pd.DataFrame({"p": s.values, "année": s.index.year, "mois": s.index.month})
    pv = heat.pivot(index="année", columns="mois", values="p")
    pv.columns = [MOIS_C[i - 1] for i in pv.columns]
    fig = px.imshow(pv, aspect="auto", color_continuous_scale="YlOrRd",
                    labels=dict(color="FCFA/kg", x="Mois", y="Année"),
                    title=f"{produit} · prix médian par mois et par année")
    st.plotly_chart(style(fig, 440), alt="Prix médians par mois et par année pour le produit sélectionné")

    full = s.groupby(s.index.year).filter(lambda x: len(x) == 12)
    if len(full) >= 24:
        lg = np.log(full)
        dev = lg - lg.groupby(lg.index.year).transform("mean")
        prof = (np.exp(dev.groupby(dev.index.month).mean()) - 1) * 100
        fig = px.bar(x=[MOIS_C[i - 1] for i in prof.index], y=prof.values,
                     color=prof.values > 0, color_discrete_map={True: "#ff5c5c", False: "#4aa3ff"},
                     title="Écart moyen au prix annuel, par mois (%)",
                     labels={"x": "", "y": "% vs moyenne de l'année"})
        fig.update_layout(showlegend=False)
        st.plotly_chart(style(fig, 320), alt="Écart saisonnier moyen du prix par rapport à la moyenne annuelle")
        st.caption("Moyenne sur les années complètes depuis 2015. À lire avec prudence : elle inclut "
                   "des mois où les prix ont été recopiés d'un mois à l'autre (voir l'onglet Données).")
    else:
        st.info("Pas assez d'années complètes pour tracer le profil saisonnier.")

# Analyses · produits
with t_prod:
    a = f[(f["h"] == h) & (f["admin1"].isin(sel_reg))]
    prod = a.groupby("produit").agg(
        variation=("hausse_prevue", "median"), alertes=("alerte", lambda x: int((x != "").sum())),
        marchés=("market", "nunique")).reset_index()
    prod["pct"] = (100 * prod["variation"]).round(1)
    fig = px.bar(prod.sort_values("pct"), x="pct", y="produit", orientation="h", color="pct",
                 text="pct", color_continuous_scale="RdBu_r", range_color=[-lim, lim],
                 hover_data={"alertes": True, "marchés": True, "pct": False},
                 title=f"Variation médiane prévue à {h} mois, par produit (%)",
                 labels={"pct": "%", "produit": ""})
    fig.update_coloraxes(showscale=False)
    fig.update_traces(texttemplate="%{text:+.1f}")
    st.plotly_chart(style(fig, 360), alt="Variation médiane prévue par produit")

    hm = a.pivot_table(index="market", columns="produit", values="hausse_prevue") * 100
    if len(hm):
        idx = hm.abs().max(axis=1).nlargest(30).index
        fig = px.imshow(hm.loc[idx], aspect="auto", color_continuous_scale="RdBu_r",
                        zmin=-lim, zmax=lim, labels=dict(color="%", x="", y=""),
                        title=f"Variation prévue à {h} mois : marchés × produits (30 marchés les plus mouvementés)")
        st.plotly_chart(style(fig, 560), alt="Carte thermique des variations prévues par marché et par produit")

# Alertes
with t_al:
    st.subheader(f"Alertes · {produit}")
    st.caption("Prévisions à 2 et 3 mois. Une alerte signale un risque de hausse, sans garantir qu’elle se produira.")
    with st.expander("Seuils et précision du backtest"):
        st.dataframe(pd.DataFrame({
            "Horizon": ["2 mois", "3 mois"],
            "Vigilance": ["≥ 5,6 %", "≥ 8,0 %"],
            "Alerte forte": ["≥ 13,9 %", "≥ 18,6 %"],
            "Précision observée (vigilance / forte)": ["32 % / 48 %", "43 % / 58 %"],
        }), hide_index=True, alt="Seuils d’alerte et précision mesurée lors du backtest")
    al = f[(f["alerte"] != "") & (f["produit"] == produit) & (f["admin1"].isin(sel_reg))].copy()
    if al.empty:
        st.info("Aucune alerte avec les filtres actuels.")
    else:
        al["Niveau"] = al["alerte"].map(NIVEAU_FR)
        al["Hausse prévue (%)"] = (100 * al["hausse_prevue"]).round(1)
        al["Cible"] = al["date_cible"].map(mois_fr)
        out = (al.sort_values("Hausse prévue (%)", ascending=False)
                 .rename(columns={"market": "Marché", "admin1": "Région", "produit": "Produit",
                                  "h": "Horizon (mois)", "prix_actuel": "Prix actuel",
                                  "prix_prevu": "Prix prévu"})
                 [["Marché", "Région", "Produit", "Horizon (mois)", "Cible", "Prix actuel",
                   "Prix prévu", "Hausse prévue (%)", "Niveau"]])
        out[["Prix actuel", "Prix prévu"]] = out[["Prix actuel", "Prix prévu"]].round(0)
        st.dataframe(out, hide_index=True, alt="Alertes de hausse prévues pour le produit sélectionné")
        st.download_button("Télécharger les alertes (CSV)", out.to_csv(index=False).encode("utf-8"),
                           file_name="alertes.csv", mime="text/csv")


# Analyses · modèle
with t_mod:
    st.markdown("Résultats du backtest 2022-2026 : chaque année est prévue avec un modèle entraîné "
                "uniquement sur le passé. Le modèle est comparé à des règles simples.")
    a1, a2 = st.columns(2)
    fig = px.bar(MAPE, x="Horizon", y="MAPE (%)", color="Méthode", barmode="group", text_auto=".1f",
                 title="Erreur moyenne (MAPE) : plus bas = mieux",
                 color_discrete_map={"Naïve": GREY, "Saisonnière": "#5b6475", "Modèle LightGBM": GOLD})
    a1.plotly_chart(style(fig, 380), alt="Erreur de prévision comparée entre les méthodes et les horizons")
    fig = px.bar(PRECISION_H3, x="Règle", y="Précision (%)", text_auto=".1f",
                 title="Alertes à 3 mois (top 5 %) : part d'alertes justifiées",
                 color="Règle", color_discrete_map={"Hasard": "#5b6475", "Momentum 3 mois": GREY,
                                                    "Rebond": "#4aa3ff", "Modèle": GOLD})
    fig.update_layout(showlegend=False)
    a2.plotly_chart(style(fig, 380), alt="Précision des alertes à trois mois selon la règle utilisée")
    b1, b2 = st.columns(2)
    fig = px.bar(GAIN_ANNEE, x="Année", y="Gain vs naïve (%)", text_auto=".1f",
                 title="Gain relatif vs naïve à 3 mois, par année", color_discrete_sequence=[GOLD])
    b1.plotly_chart(style(fig, 320), alt="Gain du modèle par rapport à la prévision naïve, par année")
    fig = px.bar(GAIN_PRODUIT, x="Gain vs naïve (%)", y="Produit", orientation="h", text_auto=".1f",
                 title="Gain relatif vs naïve à 3 mois, par produit", color_discrete_sequence=[GOLD])
    b2.plotly_chart(style(fig, 320), alt="Gain du modèle par rapport à la prévision naïve, par produit")
    with st.expander("Limites à connaître"):
        st.markdown(
            "- **À 1 mois, le modèle ne bat pas la naïve** (« prix du mois dernier ») : gain d'environ 4 % à "
            "2 mois et 8 % à 3 mois.\n"
            "- **Il bat une règle simple de rebond, mais de peu** (58 % de précision contre 49 %).\n"
            "- **Il a presque raté la forte montée de 2024** : peu d'alertes cette année-là.\n"
            "- **Les alertes suivent surtout la saison** (octobre à février), pas des chocs imprévus.\n"
            "- **Des prix sont recopiés d'un mois à l'autre** sur certaines périodes, ce qui fausse "
            "légèrement les évaluations.\n"
            "- **81 séries sans prix en septembre 2026 sont exclues** de la prévision, et les marchés "
            "récents (moins de 6 mois d'historique) ne reçoivent pas de prévision.")

# Analyses · qualité des données
with t_dat:
    d1, d2 = st.columns(2)
    stale = (p.groupby("date")["stale"].mean() * 100)
    stale = stale[stale.index >= "2019-01-01"].reset_index()
    stale.columns = ["date", "pct"]
    fig = px.line(stale, x="date", y="pct", title="Part des prix identiques au mois précédent (%)",
                  labels={"pct": "%", "date": ""}, color_discrete_sequence=[GOLD])
    d1.plotly_chart(style(fig, 360), alt="Part mensuelle des prix inchangés par rapport au mois précédent")

    cnt = pd.DataFrame({"Marchés avec historique long": p.groupby("date")["market"].nunique()})
    if c_n is not None:
        cnt["Marchés récents (2026)"] = c_n
    cnt = cnt.fillna(0)
    cnt.index.name = "date"
    cnt = cnt.reset_index()
    fig = px.area(cnt, x="date", y=[col for col in cnt.columns if col != "date"],
                  title="Nombre de marchés suivis par mois", labels={"value": "Marchés", "date": "",
                                                                       "variable": ""})
    d2.plotly_chart(style(fig, 360), alt="Nombre de marchés suivis au fil du temps")

    x1, x2, x3 = st.columns(3)
    x1.metric("Observations (marchés longs)", f"{len(p):,}".replace(",", " "))
    x2.metric("Période", f"{p['date'].min():%Y} → {p['date'].max():%m/%Y}")
    x3.metric("Séries prévues", f["market"].astype(str).add(f["commodity"]).nunique())
    st.caption("Source : prix de détail relevés par le PAM, en FCFA par kg. Les valeurs très éloignées de "
               "leurs voisines ont été repérées (111 points) ; le nettoyage ne change presque pas les résultats.")