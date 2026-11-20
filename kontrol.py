import csv
import math
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

KLASOR = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(KLASOR / "stimuli" / "stimuli.csv", encoding="utf-8")))

# 1) Satır sayısı ve dosyaların varlığı
print("Satır sayısı:", len(rows))
eksik = [r["image"] for r in rows if not (KLASOR / r["image"]).exists()]
print("Eksik dosya:", len(eksik))

# 2) Dağılımlar
print("Koşul başına:", Counter(r["condition"] for r in rows))
print("Doğru cevaplar:", Counter(r["corrAns"] or "eşit" for r in rows))

# 3) Karşılaştırma dairesinin çapını görselden ölç
hatalar = []
for r in rows:
    im = np.asarray(Image.open(KLASOR / r["image"]), dtype=float)
    yari = im[:, 560:] if r["context_side"] == "left" else im[:, :560]
    alan = (255 - yari).sum() / 255          # siyah piksel alanı
    olculen = 2 * math.sqrt(alan / math.pi)  # alan = pi * r^2  ->  çap
    hatalar.append(olculen - float(r["comparison_d"]))

print(f"Çap hatası (px): en küçük {min(hatalar):.2f}, en büyük {max(hatalar):.2f}")