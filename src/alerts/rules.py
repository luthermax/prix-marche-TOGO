import numpy as np
import pandas as pd


def add_changes(res):
    res = res.copy()
    res["pred_chg"] = res["pred_model"] / res["pred_naive"] - 1   # hausse prévue par le modèle
    res["true_chg"] = res["true"] / res["pred_naive"] - 1         # hausse réellement observée
    res["mom_chg"] = np.exp(res["d3"]) - 1                        # hausse des 3 derniers mois
    return res


def evaluate_alerts(res, h, event=0.15, top_shares=(0.05, 0.10, 0.20), exclude_bad=True):
    r = res[res["h"] == h].dropna(subset=["d3"])
    if exclude_bad:
        r = r[~r["bad"]]
    real = r["true_chg"] >= event
    base = real.mean()

    rows = []
    for name, signal in [("modèle", r["pred_chg"]), ("momentum 3 mois", r["mom_chg"])]:
        for q in top_shares:
            alert = signal >= signal.quantile(1 - q)       # on alerte sur les q % de signaux les plus forts
            n_alert = int(alert.sum())
            tp = int((alert & real).sum())
            rows.append({"règle": name, "alertes_%": int(q * 100), "alertes": n_alert,
                         "précision_%": round(100 * tp / n_alert, 1),
                         "rappel_%": round(100 * tp / real.sum(), 1),
                         "lift": round((tp / n_alert) / base, 2)})
    print(f"h={h} | hausse réelle >= {int(event*100)}% | fréquence de base : {100*base:.1f}% | n={len(r)}")
    return pd.DataFrame(rows)