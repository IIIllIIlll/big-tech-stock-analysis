"""Hem train_model.py hem main.py tarafından kullanılan ortak yardımcılar."""
from pathlib import Path
import numpy as np
import pandas as pd

KLASOR = Path(__file__).resolve().parent

FEATURES = ["unemployment_rate", "cpi", "inflation_rate",
            "mortgage_interest_rate", "corporate_bond_yield"]

KESIM = "2020-01-01"   # notebook'taki B1 hücresiyle aynı: 2020 öncesi eğitim, sonrası test


def csv_bul(ad: str) -> Path:
    """CSV'yi api klasöründe, data/ klasöründe ya da bir üst klasörde arar."""
    for yer in (KLASOR, KLASOR / "data", KLASOR.parent):
        yol = yer / ad
        if yol.exists():
            return yol
    raise FileNotFoundError(
        f"{ad} bulunamadı. Dosyayı {KLASOR} klasörüne (ya da data/ alt klasörüne) koyun.")


def fiyatlari_yukle() -> pd.DataFrame:
    df = pd.read_csv(csv_bul("big_tech_stock_prices.csv"))
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["stock_symbol", "date"]).reset_index(drop=True)


def birlesik_veri_yukle() -> pd.DataFrame:
    """Notebook'taki birleştirme: makro veri aylık, her işlem gününe o ayın değeri bağlanır."""
    fiyat = fiyatlari_yukle()
    makro = pd.read_csv(csv_bul("DATA.csv")).rename(columns={
        "UNRATE(%)": "unemployment_rate",
        "CPIALLITEMS": "cpi",
        "INFLATION(%)": "inflation_rate",
        "MORTGAGE INT. MONTHLY AVG(%)": "mortgage_interest_rate",
        "CORP. BOND YIELD(%)": "corporate_bond_yield",
    })
    # DÜZELTME (notebook'ta da var): tarihler gün-ay-yıl biçiminde
    makro["date"] = pd.to_datetime(makro["DATE"], dayfirst=True)
    makro["ay"] = makro["date"].dt.to_period("M")
    fiyat["ay"] = fiyat["date"].dt.to_period("M")
    birlesik = pd.merge(fiyat, makro[["ay"] + FEATURES], on="ay", how="inner")
    return birlesik.sort_values("date").reset_index(drop=True)


VOL_OZELLIKLERI = ["v5", "v20", "v60"]
VOL_EGITIM_BITIS = "2019-11-30"   # hedef 20 gün ileri baktığı için eğitimi testten ~1 ay önce keseriz (sızıntı önlemi)


def vol_tablosu(fiyat: pd.DataFrame) -> pd.DataFrame:
    """Günlük getiri, geçmiş 5/20/60 gün log-volatilite (v5, v20, v60) ve hedef y
    (sonraki 20 işlem gününün log-volatilitesi; son 20 günde NaN)."""
    f = fiyat.sort_values(["stock_symbol", "date"]).copy()
    f["lr"] = np.log(f["close"]).groupby(f["stock_symbol"]).diff()
    g = f.groupby("stock_symbol")["lr"]
    for w in (5, 20, 60):
        f[f"v{w}"] = np.log(g.transform(lambda s: s.rolling(w).std()) + 1e-9)
    f["y"] = np.log(g.transform(lambda s: s.rolling(20).std().shift(-20)) + 1e-9)
    return f
