# src/data/download.py
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[2]      # racine du projet
RAW_DIR = ROOT / "data" / "raw"
DATASET_ID = "wfp-food-prices-for-togo"
API = f"https://data.humdata.org/api/3/action/package_show?id={DATASET_ID}"


def download_togo_prices() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # 1. L'API HDX renvoie la liste des fichiers du dataset
    meta = requests.get(API, timeout=30).json()["result"]

    # 2. On garde le CSV principal (pas la version QuickCharts)
    resources = [
        r for r in meta["resources"]
        if r["format"].upper() == "CSV" and "quickcharts" not in r["name"].lower()
    ]
    res = resources[0]
    print("Fichier :", res["name"], "| modifié :", res.get("last_modified"))

    # 3. Téléchargement vers data/raw
    save_path = RAW_DIR / "wfp_food_prices_tgo.csv"
    content = requests.get(res["download_url"], timeout=60).content
    save_path.write_bytes(content)
    print("Enregistré dans :", save_path)
    return save_path


if __name__ == "__main__":
    download_togo_prices()