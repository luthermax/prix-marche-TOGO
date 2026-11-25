# src/data/clean.py
from pathlib import Path
import pandas as pd
import numpy as np
import sys 

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "wfp_food_prices_tgo.csv"
OUT = ROOT / "data" / "processed"

PRODUITS = ["Maize (white)", "Cassava meal (gari)", "Beans (white)", "Beans (red)",
            "Soybeans", "Sorghum (red)", "Rice (local)", "Millet"]
MIN_MOIS = 60      # au-dessous : marché "court"
MAX_TROU = 2       # on ne comble que les trous de 2 mois maximum


def load_raw():
    df = pd.read_csv(RAW, skiprows=[1])                      # skiprows : ligne HXL
    df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()  # 15 → 1er du mois
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df[df["commodity"].isin(PRODUITS)].dropna(subset=["price"])


def split_markets(df):
    nb_mois = df.groupby("market")["date"].nunique()
    longs = nb_mois[nb_mois > MIN_MOIS].index
    return df[df["market"].isin(longs)], df[~df["market"].isin(longs)]


def build_panel(df_long):
    parts = []
    for (m, c), g in df_long.groupby(["market", "commodity"]):
        idx = pd.date_range(g["date"].min(), g["date"].max(), freq="MS")   # calendrier mensuel complet
        s = g.set_index("date")["price"].reindex(idx)                      # les mois absents deviennent NaN
        filled = s.interpolate(limit=MAX_TROU, limit_area="inside")        # comble les petits trous
        parts.append(pd.DataFrame({"date": idx, "market": m, "commodity": c,
                                   "price": filled.values,
                                   "imputed": (s.isna() & filled.notna()).values}))
    return pd.concat(parts, ignore_index=True).dropna(subset=["price"])

def flag_outliers(df, lo=0.55, hi=1.8):
    df = df.sort_values(["market", "commodity", "date"]).copy()

    # 1) rapport au prix habituel de la propre série (médiane glissante sur 5 mois)
    roll = (df.groupby(["market", "commodity"])["price"]
              .transform(lambda s: s.rolling(5, center=True, min_periods=3).median()))
    df["r_own"] = df["price"] / roll

    # 2) rapport à la médiane de tous les marchés, même mois, même produit
    med = df.groupby(["date", "commodity"])["price"].transform("median")
    df["r_cross"] = df["price"] / med

    trop_haut = (df["r_own"] > hi) & (df["r_cross"] > hi)
    trop_bas = (df["r_own"] < lo) & (df["r_cross"] < lo)
    df["outlier"] = trop_haut | trop_bas
    return df

if __name__ == "__main__":
    CLEAN = "--no-clean" not in sys.argv
    suffix = "" if CLEAN else "_brut"

    OUT.mkdir(parents=True, exist_ok=True)
    df = flag_outliers(load_raw())
    print("Points marqués :", int(df["outlier"].sum()), "| nettoyage :", CLEAN)
    if CLEAN:
        df.loc[df["outlier"], "price"] = np.nan

    longs, courts = split_markets(df)
    meta = (df[["market", "admin1", "admin2", "latitude", "longitude"]]
              .drop_duplicates("market"))

    panel = build_panel(longs).merge(meta, on="market", how="left")
    panel = panel.sort_values(["market", "commodity", "date"])
    panel["stale"] = (panel.groupby(["market", "commodity"])["price"].diff() == 0) & ~panel["imputed"]
    panel.to_csv(OUT / f"panel_long{suffix}.csv", index=False)
    courts.to_csv(OUT / f"marches_courts{suffix}.csv", index=False)

    print("Panel :", panel.shape, "|", panel["market"].nunique(), "marchés")
    print("Points imputés :", int(panel["imputed"].sum()))