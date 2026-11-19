"""API test betiği. Sunucu çalışırken (uvicorn main:app --reload) başka bir terminalde:

    python test_api.py

İlk /forecast istekleri ARIMA modeli eğittiği için birkaç saniye sürebilir.
"""
import sys

import requests

URL = "http://127.0.0.1:8000"

GECERLI = {"unemployment_rate": 3.6, "cpi": 108.4, "inflation_rate": 2.3,
           "mortgage_interest_rate": 3.7, "corporate_bond_yield": 3.0}
EGITIM_DISI = dict(GECERLI, cpi=123.3, inflation_rate=8.6,
                   mortgage_interest_rate=5.2, corporate_bond_yield=4.1)

sonuclar = []


def baslik(metin):
    print(f"\n--- {metin} ---")


def kontrol(ad, kosul, ayrinti=""):
    sonuclar.append(bool(kosul))
    print(f"{'GEÇTİ ' if kosul else 'KALDI '} {ad}  {ayrinti}")


def dene(ad, yontem, yol, beklenen_kod, **kw):
    try:
        r = requests.request(yontem, URL + yol, timeout=180, **kw)
    except requests.exceptions.ConnectionError:
        print("Sunucuya bağlanılamadı. Önce `uvicorn main:app --reload` çalıştırın.")
        sys.exit(1)
    kontrol(ad, r.status_code == beklenen_kod, f"(kod {r.status_code}, beklenen {beklenen_kod})")
    try:
        return r.json()
    except ValueError:
        return {}


# ---------------------------------------------------------------- Genel
baslik("Genel uç noktalar")
j = dene("health", "GET", "/health", 200)
kontrol("health: durum alanı var", "durum" in j)
kontrol("health: model ve veri dosyası var", j.get("model_dosyasi_var") and j.get("fiyat_verisi_var"))

j = dene("symbols", "GET", "/symbols", 200)
kontrol("symbols: adet = 14", j.get("adet") == 14)
semboller = {s["sembol"]: s["sirket"] for s in j.get("semboller", [])}
kontrol("symbols: AAPL = Apple Inc.", semboller.get("AAPL") == "Apple Inc.")
kontrol("symbols: 14 sembolün hepsi var",
        set(semboller) == {"AAPL", "ADBE", "AMZN", "CRM", "CSCO", "GOOGL", "IBM",
                           "INTC", "META", "MSFT", "NFLX", "NVDA", "ORCL", "TSLA"})

j = dene("model-info", "GET", "/model-info", 200)
tm = j.get("test_metrikleri", {})
kontrol("model-info: test metrikleri var", bool(tm))
kontrol("model-info: doğrusal R2 = -0.174 (notebook B1 ile aynı)",
        tm.get("linear_regression", {}).get("R2") == -0.174)
kontrol("model-info: Random Forest R2 = -0.346 (notebook B1 ile aynı)",
        tm.get("random_forest", {}).get("R2") == -0.346)
kontrol("model-info: eğitim 34502, test 8512 satır",
        j.get("egitim_araligi", {}).get("satir") == 34502 and j.get("test_araligi", {}).get("satir") == 8512)

# ---------------------------------------------------------------- Makro tahmin
baslik("POST /predict/macro: normal kullanım")
j = dene("geçerli girdi", "POST", "/predict/macro", 200, json=GECERLI)
kontrol("tahmin = 112.57 (doğrusal regresyon)", abs(j.get("tahmin_kapanis_fiyati", 0) - 112.57) < 0.01,
        f"({j.get('tahmin_kapanis_fiyati')})")
kontrol("random forest = 122.46", abs(j.get("diger_modeller", {}).get("random_forest", 0) - 122.46) < 0.01)
kontrol("aralık içi girdide uyarı yok", j.get("uyarilar") == [])

j = dene("eğitim dışı girdi", "POST", "/predict/macro", 200, json=EGITIM_DISI)
kontrol("eğitim dışı girdide 3 uyarı (cpi, enflasyon, ipotek)", len(j.get("uyarilar", [])) == 3,
        f"({len(j.get('uyarilar', []))} uyarı)")

baslik("POST /predict/macro: sınır değerler")
dene("işsizlik = 0 (alt sınır, kabul)", "POST", "/predict/macro", 200,
     json=dict(GECERLI, unemployment_rate=0))
dene("işsizlik = 30 (üst sınır, kabul)", "POST", "/predict/macro", 200,
     json=dict(GECERLI, unemployment_rate=30))
dene("işsizlik = 30.1 (üst sınırı aşıyor)", "POST", "/predict/macro", 422,
     json=dict(GECERLI, unemployment_rate=30.1))
dene("işsizlik = -1 (negatif)", "POST", "/predict/macro", 422,
     json=dict(GECERLI, unemployment_rate=-1))
dene("işsizlik = 99", "POST", "/predict/macro", 422, json=dict(GECERLI, unemployment_rate=99))
dene("cpi = 49 (alt sınırın altında)", "POST", "/predict/macro", 422, json=dict(GECERLI, cpi=49))
dene("cpi negatif", "POST", "/predict/macro", 422, json=dict(GECERLI, cpi=-5))
dene("enflasyon = 50", "POST", "/predict/macro", 422, json=dict(GECERLI, inflation_rate=50))
dene("ipotek faizi = 26", "POST", "/predict/macro", 422, json=dict(GECERLI, mortgage_interest_rate=26))
dene("tahvil getirisi negatif", "POST", "/predict/macro", 422,
     json=dict(GECERLI, corporate_bond_yield=-0.5))

baslik("POST /predict/macro: bozuk girdi")
eksik = {k: v for k, v in GECERLI.items() if k != "cpi"}
j = dene("eksik alan (cpi yok)", "POST", "/predict/macro", 422, json=eksik)
kontrol("hata yanıtı {'hata': ...} biçiminde", "hata" in j)
kontrol("hata detayı eksik alanı (cpi) gösteriyor",
        any(d.get("alan") == "cpi" for d in j.get("detay", [])))

j = dene("boş gövde {}", "POST", "/predict/macro", 422, json={})
kontrol("boş gövdede 5 alan da eksik görünüyor", len(j.get("detay", [])) == 5,
        f"({len(j.get('detay', []))} alan)")

dene("fazla alan", "POST", "/predict/macro", 422, json=dict(GECERLI, ekstra=1))
dene("yanlış tür (cpi = 'abc')", "POST", "/predict/macro", 422, json=dict(GECERLI, cpi="abc"))
dene("null değer (cpi = null)", "POST", "/predict/macro", 422, json=dict(GECERLI, cpi=None))
dene("bozuk JSON", "POST", "/predict/macro", 422, data="{bozuk json",
     headers={"Content-Type": "application/json"})
dene("yanlış yöntem (GET /predict/macro)", "GET", "/predict/macro", 405)

# ---------------------------------------------------------------- ARIMA tahmini
baslik("GET /forecast/{sembol}: normal kullanım")
j = dene("AAPL, days=5", "GET", "/forecast/AAPL?days=5", 200)
tah = j.get("tahminler", [])
kontrol("5 tahmin döndü", len(tah) == 5)
kontrol("AAPL son gerçek tarih = 2022-12-29", j.get("son_gercek_tarih") == "2022-12-29")
kontrol("alt_95 < tahmin < ust_95", bool(tah) and all(t["alt_95"] < t["tahmin"] < t["ust_95"] for t in tah))
kontrol("tarihler artan sırada ve son gerçek tarihten sonra",
        bool(tah) and [t["tarih"] for t in tah] == sorted(t["tarih"] for t in tah)
        and tah[0]["tarih"] > j.get("son_gercek_tarih", ""))
kontrol("güven aralığı günler geçtikçe genişliyor",
        bool(tah) and (tah[-1]["ust_95"] - tah[-1]["alt_95"]) > (tah[0]["ust_95"] - tah[0]["alt_95"]))
kontrol("naive referans = son gerçek kapanış", j.get("naive_referans") == j.get("son_gercek_kapanis"))

j = dene("küçük harf sembol (aapl)", "GET", "/forecast/aapl?days=3", 200)
kontrol("sembol büyük harfe çevrildi", j.get("sembol") == "AAPL")

j = dene("META (veri 2023-01-24'e kadar)", "GET", "/forecast/META?days=2", 200)
kontrol("META son gerçek tarih = 2023-01-24", j.get("son_gercek_tarih") == "2023-01-24")

dene("MSFT", "GET", "/forecast/MSFT?days=3", 200)
dene("TSLA", "GET", "/forecast/TSLA?days=3", 200)

j = dene("varsayılan days (30)", "GET", "/forecast/AAPL", 200)
kontrol("varsayılan 30 tahmin döndü", len(j.get("tahminler", [])) == 30)

baslik("GET /forecast/{sembol}: days sınırları")
dene("days = 1 (alt sınır)", "GET", "/forecast/AAPL?days=1", 200)
dene("days = 60 (üst sınır)", "GET", "/forecast/AAPL?days=60", 200)
dene("days = 61 (üst sınırı aşıyor)", "GET", "/forecast/AAPL?days=61", 422)
dene("days = 0", "GET", "/forecast/AAPL?days=0", 422)
dene("days = -5", "GET", "/forecast/AAPL?days=-5", 422)
dene("days = abc", "GET", "/forecast/AAPL?days=abc", 422)
dene("days = 1.5 (ondalık)", "GET", "/forecast/AAPL?days=1.5", 422)

baslik("GET /forecast/{sembol}: hatalı sembol ve yol")
j = dene("bilinmeyen sembol (XXXX)", "GET", "/forecast/XXXX", 404)
kontrol("404 yanıtı {'hata': ...} biçiminde", "hata" in j)
dene("olmayan yol (/yok)", "GET", "/yok", 404)

# ---------------------------------------------------------------- Volatilite
baslik("GET /volatility/{sembol}")
j = dene("AAPL volatilite", "GET", "/volatility/AAPL", 200)
kontrol("risk seviyesi düşük/orta/yüksek", j.get("risk_seviyesi") in ("düşük", "orta", "yüksek"))
kontrol("günlük volatilite makul aralıkta (%0.1 - %20)", 0.1 < j.get("tahmin_gunluk_volatilite_yuzde", 0) < 20,
        f"({j.get('tahmin_gunluk_volatilite_yuzde')})")
kontrol("yıllık = günlük * sqrt(252) (±%1)",
        abs(j.get("tahmin_yillik_volatilite_yuzde", 0) - j.get("tahmin_gunluk_volatilite_yuzde", 0) * 252 ** 0.5)
        < 0.01 * j.get("tahmin_yillik_volatilite_yuzde", 1) + 0.1)
kontrol("HAR test R2 > naive test R2", j.get("model_test_r2", -1) > j.get("naive_test_r2", 9))
kontrol("yüzdelik 0-100 arasında", 0 <= j.get("gecmise_gore_yuzdelik", -1) <= 100)
j = dene("küçük harf (tsla)", "GET", "/volatility/tsla", 200)
kontrol("sembol büyük harfe çevrildi", j.get("sembol") == "TSLA")
j = dene("bilinmeyen sembol", "GET", "/volatility/XXXX", 404)
kontrol("404 yanıtı {'hata': ...} biçiminde", "hata" in j)

baslik("GET /history/{sembol}")
j = dene("AAPL, days=10", "GET", "/history/AAPL?days=10", 200)
kontrol("10 kayıt, tarih ve fiyat sayısı eşit", len(j.get("kapanis", [])) == 10 == len(j.get("tarihler", [])))
kontrol("son tarih = 2022-12-29", j.get("tarihler", [""])[-1] == "2022-12-29")
dene("days = 4 (alt sınırın altında)", "GET", "/history/AAPL?days=4", 422)
dene("days = 1001 (üst sınırı aşıyor)", "GET", "/history/AAPL?days=1001", 422)
dene("bilinmeyen sembol", "GET", "/history/XXXX", 404)

baslik("GET /docs: özel Swagger sayfası")
r = requests.get(URL + "/docs", timeout=30)
kontrol("/docs 200 ve Swagger arayüzü", r.status_code == 200 and "swagger-ui" in r.text)
kontrol("/docs arayüze dönüş bağlantısı var", "Arayüze dön" in r.text)

baslik("GET /: web arayüzü")
r = requests.get(URL + "/", timeout=30)
kontrol("ana sayfa 200 ve HTML", r.status_code == 200 and "text/html" in r.headers.get("content-type", ""))
kontrol("sayfada başlık var", "Big Tech Hisse Analizi" in r.text)

print(f"\n{sum(sonuclar)}/{len(sonuclar)} test geçti")
sys.exit(0 if all(sonuclar) else 1)
