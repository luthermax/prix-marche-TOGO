import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from src.evaluation.baselines import bad_months

HORIZONS = (1, 2, 3)
NUM = ["d1", "d3", "d12", "dev6", "vol6", "rel", "nat_d3", "lat", "lon", "month_t"]
CAT = ["commodity", "admin1"]


def build_table(panel):
    key = ["market", "commodity"]
    W = panel.pivot_table(index="date", columns=key, values="price").asfreq("MS")
    I = (panel.assign(imp=panel["imputed"].astype(float))
              .pivot_table(index="date", columns=key, values="imp")
              .reindex_like(W).fillna(0))
    L = np.log(W)                                                  # log des prix : grille mois x séries
    Lc = L.T.groupby(level="commodity").transform("median").T      # prix national médian par produit

    F = {
        "lp": L,
        "d1": L - L.shift(1),
        "d3": L - L.shift(3),
        "d12": L - L.shift(12),
        "dev6": L - L.rolling(6, min_periods=3).mean(),
        "vol6": L.diff().rolling(6, min_periods=3).std(),
        "rel": L - Lc,
        "nat_d3": Lc - Lc.shift(3),
    }
    for h in HORIZONS:
        F[f"y{h}"] = L.shift(-h) - L            # cible : variation du log-prix dans h mois
        F[f"imp{h}"] = I.shift(-h)              # la cible est-elle une valeur imputée ?

    n_d, n_s = L.shape                           # mise à plat de la grille en tableau long
    T = pd.DataFrame({
        "date": np.repeat(L.index.values, n_s),
        "market": np.tile(L.columns.get_level_values("market"), n_d),
        "commodity": np.tile(L.columns.get_level_values("commodity"), n_d),
    })
    for name, X in F.items():
        T[name] = X.values.ravel()
    meta = panel.drop_duplicates("market").set_index("market")
    T["admin1"] = T["market"].map(meta["admin1"])
    T["lat"] = T["market"].map(meta["latitude"])
    T["lon"] = T["market"].map(meta["longitude"])
    return T


def run_backtest(train_panel, test_panel, years=range(2022, 2027), params=None):
    A, B = build_table(train_panel), build_table(test_panel)
    for c in CAT:
        cats = sorted(set(A[c].dropna()) | set(B[c].dropna()))
        A[c] = pd.Categorical(A[c], categories=cats)
        B[c] = pd.Categorical(B[c], categories=cats)
    bad = bad_months(test_panel)
    params = params or dict(objective="l1", n_estimators=400, learning_rate=0.03,
                            num_leaves=31, min_child_samples=50, subsample=0.8,
                            subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    feats = NUM + CAT
    out = []
    for h in HORIZONS:
        for T_ in (A, B):
            T_["tdate"] = T_["date"] + pd.offsets.MonthBegin(h)    # date du prix à prévoir
            T_["month_t"] = T_["tdate"].dt.month
        ok = lambda T_: T_["lp"].notna() & T_[f"y{h}"].notna() & (T_[f"imp{h}"] == 0)
        for Y in years:
            start, end = pd.Timestamp(f"{Y}-01-01"), pd.Timestamp(f"{Y + 1}-01-01")
            tr = A[ok(A) & (A["tdate"] < start)]                    # on n'apprend que sur le passé
            te = B[ok(B) & (B["tdate"] >= start) & (B["tdate"] < end)]
            if len(te) == 0:
                continue
            model = lgb.LGBMRegressor(**params).fit(tr[feats], tr[f"y{h}"])
            yhat = model.predict(te[feats])
            out.append(pd.DataFrame({
                "h": h, "year": Y,"date": te["date"].values,
                "d3": te["d3"].values,
                "market": te["market"].values, "commodity": te["commodity"].values,
                "true": np.exp(te["lp"] + te[f"y{h}"]).values,
                "pred_model": np.exp(te["lp"] + yhat).values,
                "pred_naive": np.exp(te["lp"]).values,
                "bad": (te["date"].isin(bad) | te["tdate"].isin(bad)).values}))
    res = pd.concat(out, ignore_index=True)
    for k in ("model", "naive"):
        res[f"ae_{k}"] = (res[f"pred_{k}"] - res["true"]).abs()
        res[f"ape_{k}"] = res[f"ae_{k}"] / res["true"]
    return res