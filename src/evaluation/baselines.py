from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
TEST_START = "2022-01-01"


def load_panel(brut=True):
    name = "panel_long_brut.csv" if brut else "panel_long.csv"
    return pd.read_csv(PROC / name, parse_dates=["date"])


def bad_months(panel, seuil=0.5):
    """Mois où plus de 50 % des prix sont figés, + le mois suivant (rattrapage)."""
    share = panel.groupby("date")["stale"].mean()
    stale = share[share > seuil].index
    return set(stale) | set(stale + pd.offsets.MonthBegin(1))


def to_wide(panel, col="price"):
    """Tableau : 1 ligne par mois, 1 colonne par série (marché, produit)."""
    return panel.pivot_table(index="date", columns=["market", "commodity"],
                             values=col).asfreq("MS")


def evaluate(panel, h, method, exclude_bad=False):
    y = to_wide(panel)
    imp = (to_wide(panel.assign(imputed=panel["imputed"].astype(int)), "imputed")
           .reindex_like(y).fillna(0) > 0)

    lag = h if method == "naive" else 12      # de combien de mois on remonte pour prévoir
    pred = y.shift(lag)                        # shift(k) : décale les valeurs de k mois vers l'avenir

    ok = y.notna() & pred.notna() & ~imp       # on n'évalue que sur de vrais prix observés
    ok = ok.loc[y.index >= TEST_START]

    if exclude_bad:
        b = pd.Series(y.index.isin(list(bad_months(panel))), index=y.index)
        bad_rows = b | b.shift(lag, fill_value=False)   # mois cible OU mois d'origine non fiable
        ok = ok.loc[~bad_rows.loc[ok.index].values]

    y_t, p_t = y.loc[ok.index], pred.loc[ok.index]
    err = (y_t - p_t).abs().where(ok)
    ape = (err / y_t).where(ok)
    return pd.DataFrame({"sum_err": err.sum(axis=1), "sum_ape": ape.sum(axis=1),
                         "n": ok.sum(axis=1)})


def summarize(t, by_year=False):
    g = t.groupby(t.index.year).sum() if by_year else t.sum().to_frame().T
    return pd.DataFrame({"MAE": g["sum_err"] / g["n"],
                         "MAPE_%": 100 * g["sum_ape"] / g["n"],
                         "n": g["n"].astype(int)})