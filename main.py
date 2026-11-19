"""Big Tech hisse analizi projesi: model servisi (FastAPI).

Çalıştırma:  uvicorn main:app --reload
Dokümantasyon (Swagger): http://127.0.0.1:8000/docs

Uç noktalar
  GET  /health          servis durumu
  GET  /symbols         14 şirketin listesi
  GET  /model-info      makro modelin eğitim/test bilgileri ve test metrikleri
  POST /predict/macro   5 makro değerden kapanış fiyatı tahmini (model.joblib)
  GET  /forecast/{sembol}?days=30   ARIMA(5,1,0) ile ileri tahmin
  GET  /volatility/{sembol}         sonraki 20 işlem günü volatilite tahmini ve risk seviyesi
  GET  /history/{sembol}?days=120   son kapanış fiyatları (arayüz grafiği için)
  GET  /                            web arayüzü (index.html)

Hataların hepsi {"hata": "..."} biçimindedir.
"""
import warnings
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from ortak import FEATURES, KLASOR, csv_bul, fiyatlari_yukle, vol_tablosu

ARIMA_ORDER = (5, 1, 0)   # notebook'tan geldi (orijinal çalışmadaki seçim), biz seçmedik
MAX_GUN = 60

YATIRIM_NOTU = ("Bu bir yatırım tavsiyesi değildir. Makro model test döneminde (2020-2022) "
                "zayıf kaldı (test R2 negatif); ARIMA tahmini de naive (dünün fiyatı) "
                "referansını belirgin şekilde geçemedi.")

app = FastAPI(
    docs_url=None,   # /docs aşağıda arayüzle aynı görünümde özel olarak sunulur
    title="Big Tech Hisse Analizi API",
    version="1.0.0",
    description=(
        "Makroekonomik göstergelerden hisse kapanış fiyatı tahmini ve ARIMA ile ileri tahmin.\n\n"
        "**Uyarı:** Eğitim amaçlıdır. " + YATIRIM_NOTU
    ),
)


# ----------------------------------------------------------------------------
# Hata yönetimi: tüm hatalar {"hata": ...} biçimindedir
# ----------------------------------------------------------------------------
@app.exception_handler(StarletteHTTPException)
async def http_hatasi(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"hata": exc.detail})


@app.exception_handler(RequestValidationError)
async def dogrulama_hatasi(request, exc):
    detay = []
    for e in exc.errors():
        alan = ".".join(str(x) for x in e["loc"][1:]) or "govde"
        detay.append({"alan": alan, "mesaj": e["msg"]})
    return JSONResponse(
        status_code=422,
        content={"hata": "Geçersiz istek: eksik, fazla, yanlış türde ya da aralık dışı değer var.",
                 "detay": detay},
    )


@app.exception_handler(Exception)
async def beklenmeyen_hata(request, exc):
    return JSONResponse(status_code=500, content={"hata": "Beklenmeyen sunucu hatası."})


# ----------------------------------------------------------------------------
# Yükleyiciler (ilk istekte yüklenir, sonra bellekte tutulur)
# ----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def paket_yukle():
    yol = KLASOR / "model.joblib"
    if not yol.exists():
        raise HTTPException(503, "model.joblib bulunamadı. Önce `python train_model.py` çalıştırın.")
    return joblib.load(yol)


@lru_cache(maxsize=1)
def fiyatlar_yukle():
    try:
        return fiyatlari_yukle()
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))


@lru_cache(maxsize=1)
def sirketler_yukle():
    """{'AAPL': 'Apple Inc.', ...}  (big_tech_companies.csv yoksa şirket adı boş kalır)"""
    semboller = sorted(fiyatlar_yukle()["stock_symbol"].unique())
    try:
        c = pd.read_csv(csv_bul("big_tech_companies.csv"))
        adlar = dict(zip(c["stock_symbol"], c["company"]))
    except FileNotFoundError:
        adlar = {}
    return {s: adlar.get(s) for s in semboller}


@lru_cache(maxsize=16)
def arima_fit(sembol: str):
    """Sembolün tüm kapanış serisiyle ARIMA(5,1,0) eğitir. Sonuç önbellekte tutulur."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        raise HTTPException(503, "statsmodels kurulu değil. `pip install statsmodels` çalıştırın.")
    s = fiyatlar_yukle()
    s = s[s["stock_symbol"] == sembol].sort_values("date")
    seri = s["close"].reset_index(drop=True)   # notebook'taki gibi sıralı indeks
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sonuc = ARIMA(seri, order=ARIMA_ORDER).fit()
    return sonuc, s["date"].iloc[-1], float(seri.iloc[-1])


# ----------------------------------------------------------------------------
# Girdi modeli
# ----------------------------------------------------------------------------
class MakroGirdi(BaseModel):
    """5 makroekonomik gösterge. Alan adları notebook'taki sütun adlarıyla aynıdır."""
    model_config = ConfigDict(
        extra="forbid",   # tanımsız alan gönderilirse 422
        json_schema_extra={"example": {
            "unemployment_rate": 3.6, "cpi": 108.4, "inflation_rate": 2.3,
            "mortgage_interest_rate": 3.7, "corporate_bond_yield": 3.0}},
    )
    unemployment_rate: float = Field(..., ge=0, le=30, allow_inf_nan=False,
                                     description="İşsizlik oranı (%)")
    cpi: float = Field(..., ge=50, le=500, allow_inf_nan=False,
                       description="Tüketici fiyat endeksi (CPI)")
    inflation_rate: float = Field(..., ge=-10, le=30, allow_inf_nan=False,
                                  description="Enflasyon oranı (%)")
    mortgage_interest_rate: float = Field(..., ge=0, le=25, allow_inf_nan=False,
                                          description="İpotek faizi (%)")
    corporate_bond_yield: float = Field(..., ge=0, le=25, allow_inf_nan=False,
                                        description="Kurumsal tahvil getirisi (%)")


# ----------------------------------------------------------------------------
# Uç noktalar
# ----------------------------------------------------------------------------
@app.get("/health", tags=["Genel"])
def health():
    """Servis çalışıyor mu? Model ve veri dosyaları mevcut mu?"""
    try:
        csv_bul("big_tech_stock_prices.csv")
        veri = True
    except FileNotFoundError:
        veri = False
    return {"durum": "çalışıyor", "model_dosyasi_var": (KLASOR / "model.joblib").exists(),
            "fiyat_verisi_var": veri}


@app.get("/symbols", tags=["Genel"])
def symbols():
    """Desteklenen 14 şirketin sembolleri ve adları."""
    sirketler = sirketler_yukle()
    return {"adet": len(sirketler),
            "semboller": [{"sembol": s, "sirket": a} for s, a in sirketler.items()]}


@app.get("/model-info", tags=["Genel"])
def model_info():
    """Makro modelin eğitim/test aralığı ve test metrikleri (raporla aynı sayılar)."""
    p = paket_yukle()
    return {
        "bolme": "tarihe göre (eğitim: 2020 öncesi, test: 2020 ve sonrası)",
        "egitim_araligi": p["egitim_araligi"],
        "test_araligi": p["test_araligi"],
        "test_metrikleri": p["test_metrikleri"],
        "volatilite_test": ({k: p["volatilite"][k] for k in ("test_r2_har", "test_r2_naive")}
                            if "volatilite" in p else None),
        "egitimde_gorulen_aralik": p["ozellik_araliklari"],
        "not": YATIRIM_NOTU,
    }


@app.post("/predict/macro", tags=["Tahmin"])
def predict_macro(girdi: MakroGirdi):
    """5 makro değerden hisse kapanış fiyatı tahmini.

    Model şirketi tanımaz: aynı makro değerler için 14 şirketin hepsine aynı sayı döner.
    Ana tahmin doğrusal regresyondur (test R2 -0.17); Random Forest değeri de yanda verilir.
    """
    p = paket_yukle()
    satir = pd.DataFrame([girdi.model_dump()])[p["ozellikler"]]
    lr = float(p["modeller"]["linear_regression"].predict(satir)[0])
    rf = float(p["modeller"]["random_forest"].predict(satir)[0])

    uyarilar = []
    for alan, (alt, ust) in p["ozellik_araliklari"].items():
        v = getattr(girdi, alan)
        if v < alt or v > ust:
            uyarilar.append(f"{alan} = {v}, eğitim aralığının ({alt:.2f} - {ust:.2f}) dışında. "
                            "Model bu koşulları eğitimde görmedi, tahmin güvenilmez.")
    if lr <= 0:
        uyarilar.append("Doğrusal model sıfır ya da negatif fiyat üretti; bu değerin anlamı yok.")

    return {
        "tahmin_kapanis_fiyati": round(lr, 2),
        "model": "linear_regression",
        "diger_modeller": {"random_forest": round(rf, 2)},
        "girdi": girdi.model_dump(),
        "uyarilar": uyarilar,
        "not": YATIRIM_NOTU,
    }


@app.get("/forecast/{symbol}", tags=["Tahmin"])
def forecast(symbol: str, days: int = Query(30, ge=1, le=MAX_GUN,
                                            description=f"Kaç iş günü ileri tahmin (1-{MAX_GUN})")):
    """Seçilen şirket için ARIMA(5,1,0) ile ileri tahmin (%95 güven aralığıyla)."""
    sembol = symbol.strip().upper()
    sirketler = sirketler_yukle()
    if sembol not in sirketler:
        raise HTTPException(404, f"'{symbol}' bulunamadı. Geçerli semboller: {', '.join(sirketler)}")

    sonuc, son_tarih, son_kapanis = arima_fit(sembol)
    tahmin = sonuc.get_forecast(steps=days)
    orta = tahmin.predicted_mean.to_numpy()
    aralik = tahmin.conf_int(alpha=0.05)
    alt, ust = aralik.iloc[:, 0].to_numpy(), aralik.iloc[:, 1].to_numpy()
    tarihler = pd.bdate_range(start=son_tarih + pd.Timedelta(days=1), periods=days)

    return {
        "sembol": sembol,
        "sirket": sirketler[sembol],
        "model": f"ARIMA{ARIMA_ORDER}",
        "son_gercek_tarih": str(son_tarih.date()),
        "son_gercek_kapanis": round(son_kapanis, 2),
        "gun_sayisi": days,
        "tahminler": [{"tarih": str(t.date()), "tahmin": round(float(o), 2),
                       "alt_95": round(float(a), 2), "ust_95": round(float(u), 2)}
                      for t, o, a, u in zip(tarihler, orta, alt, ust)],
        "naive_referans": round(son_kapanis, 2),
        "uyari": YATIRIM_NOTU + " Tarihler hafta içi günlerdir, resmi tatiller dikkate alınmaz.",
    }


@lru_cache(maxsize=1)
def vol_verisi():
    return vol_tablosu(fiyatlar_yukle())


@app.get("/volatility/{symbol}", tags=["Tahmin"])
def volatility(symbol: str):
    """Sonraki 20 işlem günü için günlük volatilite tahmini (HAR tarzı doğrusal model).

    Fiyat yönünün aksine volatilite tahmin edilebilir (kümelenir); test R2'si naive referansından yüksektir.
    Risk seviyesi, tahminin şirketin kendi geçmiş 20 günlük volatilite dağılımındaki yerine göredir.
    """
    sembol = symbol.strip().upper()
    sirketler = sirketler_yukle()
    if sembol not in sirketler:
        raise HTTPException(404, f"'{symbol}' bulunamadı. Geçerli semboller: {', '.join(sirketler)}")
    p = paket_yukle()
    if "volatilite" not in p:
        raise HTTPException(503, "model.joblib eski sürüm. `python train_model.py` ile yeniden üretin.")
    v = vol_verisi()
    v = v[v["stock_symbol"] == sembol]
    son = v.dropna(subset=p["volatilite"]["ozellikler"]).iloc[-1]
    log_tahmin = float(p["volatilite"]["model"].predict(
        pd.DataFrame([son[p["volatilite"]["ozellikler"]].astype(float)]))[0])
    gunluk = float(np.exp(log_tahmin))
    gecmis = np.exp(v["v20"].dropna().to_numpy())
    yuzdelik = float((gecmis <= gunluk).mean() * 100)
    seviye = "düşük" if yuzdelik < 33 else "orta" if yuzdelik < 67 else "yüksek"
    return {
        "sembol": sembol, "sirket": sirketler[sembol],
        "son_veri_tarihi": str(son["date"].date()),
        "mevcut_20g_gunluk_volatilite_yuzde": round(float(np.exp(son["v20"])) * 100, 2),
        "tahmin_gunluk_volatilite_yuzde": round(gunluk * 100, 2),
        "tahmin_yillik_volatilite_yuzde": round(gunluk * np.sqrt(252) * 100, 1),
        "risk_seviyesi": seviye,
        "gecmise_gore_yuzdelik": round(yuzdelik, 1),
        "model_test_r2": p["volatilite"]["test_r2_har"],
        "naive_test_r2": p["volatilite"]["test_r2_naive"],
        "not": "Volatilite = fiyat değişiminin büyüklüğü, yönü değil. Yatırım tavsiyesi değildir.",
    }


@app.get("/history/{symbol}", tags=["Genel"])
def history(symbol: str, days: int = Query(120, ge=5, le=1000, description="Son kaç işlem günü (5-1000)")):
    """Arayüz grafiği için son kapanış fiyatları."""
    sembol = symbol.strip().upper()
    sirketler = sirketler_yukle()
    if sembol not in sirketler:
        raise HTTPException(404, f"'{symbol}' bulunamadı. Geçerli semboller: {', '.join(sirketler)}")
    s = fiyatlar_yukle()
    s = s[s["stock_symbol"] == sembol].sort_values("date").tail(days)
    return {"sembol": sembol, "tarihler": [str(d.date()) for d in s["date"]],
            "kapanis": [round(float(c), 2) for c in s["close"]]}

DOCS_CSS = """<style>
:root{--bg:#f6f7f9;--tx:#1c2230;--ac:#2457d6}
html,body{margin:0;background:var(--bg);color:var(--tx)}
body{font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.geri{max-width:1200px;margin:0 auto;padding:16px 24px 0;font-size:14px}
.geri a{color:var(--ac);text-decoration:none;font-weight:600}
.swagger-ui .topbar{display:none}
.swagger-ui{max-width:1200px;margin:0 auto}
.swagger-ui,.swagger-ui .info .title,.swagger-ui .tab li,.swagger-ui .btn,.swagger-ui select,.swagger-ui input,
.swagger-ui textarea,.swagger-ui table thead tr td,.swagger-ui table thead tr th,.swagger-ui .opblock-tag,
.swagger-ui .opblock .opblock-summary-path,.swagger-ui .opblock .opblock-summary-description,
.swagger-ui .parameter__name,.swagger-ui .response-col_status,.swagger-ui .model-title{font-family:inherit}
.swagger-ui .opblock{border-radius:10px;overflow:hidden}
.swagger-ui .scheme-container{background:transparent;box-shadow:none}
@media (prefers-color-scheme: dark){
  :root{--bg:#12151c;--tx:#e7eaf0;--ac:#6c93f5}
  .swagger-ui{filter:invert(88%) hue-rotate(180deg)}
  .swagger-ui .microlight{filter:invert(100%) hue-rotate(180deg)}
  .swagger-ui .opblock .opblock-summary-method{color:#000 !important;text-shadow:none}
}
</style>"""


@app.get("/docs", include_in_schema=False)
def docs():
    html = get_swagger_ui_html(openapi_url="/openapi.json",
                               title="Big Tech Hisse Analizi API: Dokümantasyon").body.decode()
    html = html.replace("</head>", DOCS_CSS + "</head>", 1)
    html = html.replace("<body>", '<body><div class="geri"><a href="/">← Arayüze dön</a></div>', 1)
    return HTMLResponse(html)


@app.get("/", include_in_schema=False)
def arayuz():
    yol = KLASOR / "index.html"
    if not yol.exists():
        raise HTTPException(404, "index.html bulunamadı.")
    return FileResponse(yol)
