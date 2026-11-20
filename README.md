# Görsel İllüzyonlar Bağlamında İnsan Görsel Algısı ile Yapay Sinir Ağlarının Karşılaştırmalı Analizi

## Proje Hakkında

Bu proje, Ebbinghaus görsel illüzyonu üzerinden çevresel görsel uyaranların insanın boyut algısını nasıl etkilediğini ve bu etkinin yapay sinir ağlarının boyut tahminleriyle nasıl karşılaştırılabileceğini incelemeyi amaçlamaktadır.

## Projenin Amaçları

- Ebbinghaus illüzyonuna ait deney görselleri üretmek.
- İnsan katılımcıların merkez dairelerin boyutlarına ilişkin algılarını incelemek.
- Bir yapay sinir ağı geliştirerek merkez dairenin çapını tahmin etmek.
- İnsan algısı ile model tahminlerini karşılaştırmak.
- Algısal yanılgıların çevresel uyaranlarla ve eğitim verilerindeki ilişkilerle bağlantısını değerlendirmek.

## Yöntem

Proje kapsamında sentetik görseller oluşturulacak, uygun etik ve izin koşulları sağlandığında insan deneyi yürütülecek ve yapay sinir ağı farklı veri koşullarında eğitilerek değerlendirilecektir.

## Klasör Yapısı

- `stimuli/`: Deney görsellerinin üretilmesi ve saklanması
- `human_experiment/`: İnsan deneyi ve katılımcı yanıtları
- `model/`: Yapay sinir ağı ve eğitim kodları
- `analysis/`: İstatistiksel analizler ve PSE hesaplamaları
- `results/`: Sonuçlar, grafikler ve değerlendirme çıktıları
- `docs/`: Proje dokümantasyonu

## Uyaranların Üretilmesi

Kök klasördeki betikler:

- `stimuli.py`: Ortak uyaran üreteci (insan deneyi ve model aynı fonksiyonu kullanır). Doğrudan çalıştırılınca `check_all()` tüm koşullarda taşma kontrolü yapar.
- `make_stimuli.py`: İnsan deneyi için 80 deneme görseli (`stimuli/png/`) ve koşul tablosu (`stimuli/stimuli.csv`) üretir.
- `kontrol.py`: Dosya sayısı, koşul/taraf dengesi ve daire çaplarını doğrular.

    pip install -r requirements.txt
    python stimuli.py
    python make_stimuli.py
    python kontrol.py

Seed 2026 sabittir; her bilgisayarda aynı dosyalar üretilir.

**Tasarım:** 8 koşul (merkez tek, 5 oran: 0.4/0.7/1.0/1.5/2.0, 2 uzak kontrol) × 5 karşılaştırma boyutu (0.84–1.16 × D) × 2 taraf = 80 görsel. Görsel 1120 × 560 px; merkez D = 60 px.

**PsychoPy notları:**
- Döngü: conditionsFile = `stimuli/stimuli.csv`, loopType = random, nReps = 3 → 240 deneme.
- Görsel birimi pix, ekran ölçeklendirmesi %100.
- `corrAns`: fiziksel olarak büyük dairenin tuşu (left/right); karşılaştırma 1.00 satırlarında boş.
- Görsel yolları şu an repo köküne göredir (`stimuli/png/...`). Deney dosyası `human_experiment/` içine konursa `make_stimuli.py`'de yol `../stimuli/png/...` yapılmalı.

## Kullanılan Teknolojiler

Python, NumPy, pandas, Matplotlib ve PyTorch veya TensorFlow (uygulama tercihine göre).

## Proje Durumu

Proje geliştirme aşamasındadır.
