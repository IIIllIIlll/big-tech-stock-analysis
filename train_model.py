"""Makro model eğitimi (notebook'taki Hücre B1 ile aynı mantık).

Kullanım:  python train_model.py
Çıktı   :  model.joblib

Eğitim: 2020-01-01 öncesi. Test: 2020 ve sonrası (tarihe göre bölme, rastgele değil).
Bu yüzden API'nin sunduğu model, raporda ve slaytta anlattığımız test sonuçlarına aittir.
"""
import joblib
import numpy as np
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ortak import (FEATURES, KESIM, KLASOR, VOL_EGITIM_BITIS, VOL_OZELLIKLERI,
                   birlesik_veri_yukle, fiyatlari_yukle, vol_tablosu)


def mape(y, p):
    y, p = np.array(y), np.array(p)
    return float(np.mean(np.abs((y - p) / y)) * 100)


def metrikler(y, p):
    return {
        "RMSE": round(float(mean_squared_error(y, p) ** 0.5), 3),
        "MAE": round(float(mean_absolute_error(y, p)), 3),
        "MAPE": round(mape(y, p), 3),
        "R2": round(float(r2_score(y, p)), 3),
    }


def volatilite_modeli():
    """Sonraki 20 işlem gününün volatilitesi: HAR tarzı doğrusal model (geçmiş 5/20/60 gün volatilitesi).
    Referans (naive): 'gelecek volatilite = geçmiş 20 günlük volatilite'."""
    v = vol_tablosu(fiyatlari_yukle()).dropna(subset=VOL_OZELLIKLERI + ["y"])
    tr, te = v[v["date"] < VOL_EGITIM_BITIS], v[v["date"] >= KESIM]
    model = LinearRegression().fit(tr[VOL_OZELLIKLERI], tr["y"])
    ref = tr["y"].mean()
    def r2(p):
        return round(float(1 - ((te["y"] - p) ** 2).sum() / ((te["y"] - ref) ** 2).sum()), 3)
    sonuc = {"model": model, "ozellikler": VOL_OZELLIKLERI,
             "test_r2_har": r2(model.predict(te[VOL_OZELLIKLERI])),
             "test_r2_naive": r2(te["v20"]),
             "egitim_satir": int(len(tr)), "test_satir": int(len(te))}
    print(f"volatilite: HAR R2={sonuc['test_r2_har']} | naive R2={sonuc['test_r2_naive']}")
    return sonuc


def main():
    df = birlesik_veri_yukle()
    train, test = df[df["date"] < KESIM], df[df["date"] >= KESIM]
    print(f"Toplam: {len(df)} | Eğitim: {len(train)} | Test: {len(test)}")
    print(f"Eğitim: {train['date'].min().date()} → {train['date'].max().date()}")
    print(f"Test  : {test['date'].min().date()} → {test['date'].max().date()}")

    X_tr, y_tr = train[FEATURES], train["close"]
    X_te, y_te = test[FEATURES], test["close"]

    modeller = {
        "baseline": DummyRegressor(strategy="mean").fit(X_tr, y_tr),
        "linear_regression": LinearRegression().fit(X_tr, y_tr),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42).fit(X_tr, y_tr),
    }
    test_metrikleri = {ad: metrikler(y_te, m.predict(X_te)) for ad, m in modeller.items()}
    for ad, m in test_metrikleri.items():
        print(f"{ad:20s} {m}")

    paket = {
        "modeller": modeller,
        "ozellikler": FEATURES,
        "egitim_araligi": {
            "baslangic": str(train["date"].min().date()),
            "bitis": str(train["date"].max().date()),
            "satir": int(len(train)),
        },
        "test_araligi": {
            "baslangic": str(test["date"].min().date()),
            "bitis": str(test["date"].max().date()),
            "satir": int(len(test)),
        },
        # Eğitimde görülen değer aralığı: dışına çıkan girdiler için uyarı verilir
        "ozellik_araliklari": {f: [float(X_tr[f].min()), float(X_tr[f].max())] for f in FEATURES},
        "test_metrikleri": test_metrikleri,
        "volatilite": volatilite_modeli(),
        "sklearn_surumu": sklearn.__version__,
    }
    joblib.dump(paket, KLASOR / "model.joblib")
    print("Kaydedildi: model.joblib")


if __name__ == "__main__":
    main()
