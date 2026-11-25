import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from src.models.lgbm import build_table, HORIZONS, NUM, CAT

PROC = ROOT / "data" / "processed"
# (seuil vigilance = top 20 %, seuil alerte forte = top 5 %) issus du backtest
SEUILS = {2: (0.056, 0.139), 3: (0.080, 0.186)}
PARAMS = dict(objective="l1", n_estimators=400, learning_rate=0.03, num_leaves=31,
              min_child_samples=50, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, verbose=-1)


def niveau(h, x):
    if h not in SEUILS:
        return ""
    vig, forte = SEUILS[h]
    return "forte" if x >= forte else ("vigilance" if x >= vig else "")


def forecast(panel):
    T = build_table(panel)
    for c in CAT:
        T[c] = pd.Categorical(T[c], categories=sorted(T[c].dropna().unique()))
    last = T["date"].max()
    feats = NUM + CAT
    out = []
    for h in HORIZONS:
        T["month_t"] = (T["date"] + pd.offsets.MonthBegin(h)).dt.month
        tr = T[T["lp"].notna() & T[f"y{h}"].notna() & (T[f"imp{h}"] == 0)]
        now = T[(T["date"] == last) & T["lp"].notna()]
        model = lgb.LGBMRegressor(**PARAMS).fit(tr[feats], tr[f"y{h}"])
        yhat = model.predict(now[feats])
        out.append(pd.DataFrame({
            "market": now["market"].values, "commodity": now["commodity"].values,
            "date_base": last, "date_cible": last + pd.offsets.MonthBegin(h), "h": h,
            "prix_actuel": np.exp(now["lp"]).values,
            "prix_prevu": np.exp(now["lp"].values + yhat),
            "hausse_prevue": np.exp(yhat) - 1}))
    f = pd.concat(out, ignore_index=True)
    f["alerte"] = [niveau(h, x) for h, x in zip(f["h"], f["hausse_prevue"])]
    return f


if __name__ == "__main__":
    panel = pd.read_csv(PROC / "panel_long_brut.csv", parse_dates=["date"])
    f = forecast(panel)
    meta = panel.drop_duplicates("market")[["market", "admin1", "latitude", "longitude"]]
    f = f.merge(meta, on="market", how="left")
    f.to_csv(PROC / "previsions.csv", index=False)

    print("Base :", f["date_base"].iloc[0].date(), "| prévisions :", len(f))
    print(f.groupby(["h", "alerte"]).size().to_string())