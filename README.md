# Big Tech Hisse Analizi: Model Servisi (API)

Notebook'taki analizin iki parçasını HTTP üzerinden sunan küçük bir FastAPI servisi:

1. **Makro model**: 5 makroekonomik değerden hisse kapanış fiyatı tahmini (Hücre B1'deki modeller).
2. **ARIMA(5,1,0) tahmini**: seçilen şirket için ileri tarihli fiyat tahmini (Hücre B3/B4'teki model).

> **Uyarı:** Eğitim amaçlıdır, yatırım tavsiyesi değildir. Makro model test döneminde (2020-2022) zayıf kaldı
> (test R² negatif) ve ARIMA tahmini naive (dünün fiyatı) referansını belirgin şekilde geçemedi.

## Dosyalar

| Dosya | Görevi |
|---|---|
| `ortak.py` | CSV'leri bulur, okur, notebook'taki gibi birleştirir |
| `train_model.py` | Makro modelleri eğitir, `model.joblib` üretir |
| `main.py` | FastAPI uygulaması |
| `index.html` | Web arayüzü (API ile birlikte `/` adresinde açılır) |
| `test_api.py` | Çalışan API'ye istek gönderen test betiği |
| `requirements.txt` | Gerekli kütüphaneler |

Veri dosyaları `big_tech_stock_prices.csv`, `DATA.csv` ve `big_tech_companies.csv` bu klasörde (ya da `data/` alt klasöründe) bulunmalıdır.

## Kurulum ve çalıştırma (Windows PowerShell)

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
uvicorn main:app --reload
```

Sonra tarayıcıda: http://127.0.0.1:8000/docs (otomatik Swagger arayüzü).
Testleri çalıştırmak için sunucu açıkken **başka bir terminalde** (aynı klasörde, `venv` aktif): `python test_api.py`

`model.joblib` repoya konmaz; scikit-learn sürümü farklıysa yüklenmeyebilir. Her bilgisayarda `python train_model.py` ile üretilir.

## Model nasıl eğitildi?

- Veri: hisse fiyatları (14 şirket) ile aylık makro veri, ay bazında birleştirilmiş (43.014 satır).
- Bölme **tarihe göre**: eğitim 2010-01-04 → 2019-12-31 (34.502 satır), test 2020-01-02 → 2022-05-31 (8.512 satır).
- Test metrikleri (notebook'taki B1 tablosuyla aynı):

| Model | RMSE | MAE | MAPE (%) | R² |
|---|---|---|---|---|
| Baseline (eğitim ortalaması) | 198,791 | 138,886 | 55,025 | -0,880 |
| Doğrusal regresyon | 157,076 | 107,840 | 65,005 | -0,174 |
| Random Forest | 168,204 | 114,121 | 57,609 | -0,346 |

Modelin test R²'si negatif çünkü 2022'nin makro değerleri (CPI ≈ 123, enflasyon ≈ %8,6) eğitim aralığının çok dışındadır
(eğitimde en yüksek CPI ≈ 108,6, en yüksek enflasyon ≈ %3,9). Bu yüzden `/predict/macro` böyle girdilerde uyarı döndürür.

## Uç noktalar

| Yöntem | Yol | Açıklama |
|---|---|---|
| GET | `/health` | Servis çalışıyor mu, model ve veri dosyası var mı |
| GET | `/symbols` | 14 şirketin sembolü ve adı |
| GET | `/model-info` | Eğitim/test aralığı, test metrikleri, eğitimde görülen değer aralıkları |
| POST | `/predict/macro` | 5 makro değerden kapanış fiyatı tahmini |
| GET | `/forecast/{sembol}?days=30` | ARIMA(5,1,0) ile 1-60 iş günü ileri tahmin |
| GET | `/volatility/{sembol}` | Sonraki 20 işlem günü volatilite tahmini ve risk seviyesi (düşük/orta/yüksek) |
| GET | `/history/{sembol}?days=120` | Son kapanış fiyatları (arayüz grafiği için) |
| GET | `/` | Web arayüzü |

### POST /predict/macro

İstek:

```json
{
  "unemployment_rate": 3.6,
  "cpi": 108.4,
  "inflation_rate": 2.3,
  "mortgage_interest_rate": 3.7,
  "corporate_bond_yield": 3.0
}
```

Yanıt (sayılar bu girdi için gerçek çıktıdır; `uyarilar` ve `not` alanları kısaltıldı):

```json
{
  "tahmin_kapanis_fiyati": 112.57,
  "model": "linear_regression",
  "diger_modeller": {"random_forest": 122.46},
  "girdi": {"unemployment_rate": 3.6, "cpi": 108.4, "inflation_rate": 2.3,
            "mortgage_interest_rate": 3.7, "corporate_bond_yield": 3.0},
  "uyarilar": [],
  "not": "Bu bir yatırım tavsiyesi değildir. ..."
}
```

Model şirketi tanımaz: aynı makro değerler için 14 şirketin hepsine aynı sayı döner.

### GET /forecast/AAPL?days=5

Yanıt (gerçek çıktı; `uyari` alanı kısaltıldı):

```json
{
  "sembol": "AAPL",
  "sirket": "Apple Inc.",
  "model": "ARIMA(5, 1, 0)",
  "son_gercek_tarih": "2022-12-29",
  "son_gercek_kapanis": 129.61,
  "gun_sayisi": 5,
  "tahminler": [
    {"tarih": "2022-12-30", "tahmin": 129.41, "alt_95": 126.66, "ust_95": 132.16},
    {"tarih": "2023-01-02", "tahmin": 129.42, "alt_95": 125.65, "ust_95": 133.2},
    {"tarih": "2023-01-03", "tahmin": 129.23, "alt_95": 124.68, "ust_95": 133.77},
    {"tarih": "2023-01-04", "tahmin": 129.03, "alt_95": 123.87, "ust_95": 134.19},
    {"tarih": "2023-01-05", "tahmin": 129.24, "alt_95": 123.52, "ust_95": 134.95}
  ],
  "naive_referans": 129.61,
  "uyari": "Bu bir yatırım tavsiyesi değildir. ..."
}
```

Tahmin son kapanışa (129,61) çok yakındır ve güven aralığı günler geçtikçe genişler; ARIMA naive referansından belirgin biçimde ayrışmaz.

Not: ARIMA, sembolün tüm verisiyle eğitilir (notebook'taki B3/B4 ise son 60 günü test için ayırır). İlk istekte model eğitildiği için birkaç saniye sürebilir, sonraki istekler hızlıdır.

## Web arayüzü

Sunucu çalışırken tarayıcıda **http://127.0.0.1:8000/** adresini aç (Swagger için `/docs` yine çalışır). Arayüzde:
şirket seçip ARIMA tahmini ve %95 güven aralığı grafiği, volatilite/risk kartı, makro değerlerden fiyat tahmini ve model bilgisi bulunur.
Ek kütüphane gerekmez; sayfa tek bir `index.html` dosyasıdır.

## Volatilite modeli (`/volatility/{sembol}`)

Fiyat yönünü tahmin etmek bu veriyle başarısız oldu (yukarıdaki makro model ve ARIMA sonuçları). Volatilite ise kümelendiği için tahmin edilebilir:
sonraki 20 işlem gününün günlük volatilitesi, geçmiş 5/20/60 günlük volatiliteyle doğrusal modelle (HAR tarzı) tahmin edilir.

| Yöntem | Test R² (2020 ve sonrası) |
|---|---|
| Naive (geçmiş 20 günlük volatilite) | 0,331 |
| HAR tarzı doğrusal model | 0,459 |

Risk seviyesi, tahminin şirketin kendi geçmiş 20 günlük volatilite dağılımındaki yerine göre belirlenir (alt üçte bir: düşük, orta üçte bir: orta, üst üçte bir: yüksek).
Bu, fiyatın yönü hakkında bilgi vermez, yalnızca değişimin ne kadar büyük olabileceğini gösterir. Eğitim 2019-11-30'da biter (hedef 20 gün ileri baktığı için testten ayrılır).

## Hata yönetimi

Tüm hatalar `{"hata": "..."}` biçimindedir.

| Durum | Kod |
|---|---|
| Eksik, fazla alan, yanlış tür ya da aralık dışı değer | 422 (`detay` listesiyle hangi alan olduğu söylenir) |
| Bilinmeyen sembol | 404 |
| `model.joblib` ya da CSV yok, `statsmodels` kurulu değil | 503 |
| Beklenmeyen hata | 500 |

Kabul edilen aralıklar: işsizlik 0-30, CPI 50-500, enflasyon -10 ile 30, ipotek faizi 0-25, tahvil getirisi 0-25.

## Sınırlılıklar

- Aylık makro değer ayın tüm işlem günlerine uygulanmıştır. Bu veriler ay bittikten sonra açıklandığı için hafif bir ileri bakma (look-ahead) yanlılığı vardır.
- Volatilite modeli tek bir test penceresiyle değerlendirilmiştir; yıl yıl doğrulama yapılmamıştır.
- Tek test penceresi vardır, anlamlılık testi yapılmamıştır.
- Model hangi şirket olduğunu bilmez.
- Doğrusal modelin katsayıları (ipotek faizi ile tahvil getirisi korelasyonu 0,87) tek tek yorumlanamaz.
- ARIMA derecesi (5,1,0) orijinal çalışmadan alınmıştır, biz seçmedik.
- Tahmin tarihleri hafta içi günlerdir, resmi tatiller dikkate alınmaz.

## Kaynaklar ve yapay zekâ kullanımı

- Notebook'un temeli: Kaggle'daki Pinar Topuz'un çalışması. Veri setleri: Kaggle (Big Tech Stock Prices, Macroeconomic Factors Affecting US Housing Prices). Raporun kaynakçasında tam bağlantılarını verin.
- Bu API'nin kodu yapay zekâ (Claude) yardımıyla yazılmıştır; kullanım raporda ve sunumda ayrıca belirtilmiştir.
