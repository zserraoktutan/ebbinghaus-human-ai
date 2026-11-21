"""Model eğitim verisi: rejim A ve rejim B-fiz (anında üretim).

Rejim A     : yalnız merkez daire, çevre yok.
Rejim B-fiz : %80 rastgele çevreli + %20 tek daire. Çevrenin boyutu, sayısı,
              uzaklığı ve dönüşü merkezden BAĞIMSIZ örneklenir; etiket fizikseldir.

Her iki rejimde etiket y = merkez çapı / D (D = referans çap, 1.0 = referans).
Görüntü: 1 x 224 x 224, daireler 1.0, zemin 0.0.

Tekrarlanabilirlik: her görsel (seed, bölüm, epoch, indeks) dörtlüsünden
üretilir. Aynı dörtlü her makinede, her num_workers değerinde aynı görseli verir.
"""
import math

import numpy as np
from PIL import Image, ImageDraw

import stimuli as st

try:                                   # Colab'da torch vardır; yerel testte olmayabilir
    import torch
    from torch.utils.data import Dataset
except ImportError:                    # pragma: no cover
    torch = None
    Dataset = object

# ---------------------------------------------------------------- sabitler
CANVAS = st.MODEL["canvas"]            # 224
D = st.MODEL["center_d"]               # 24 px, referans merkez çapı
MAX_SHIFT = st.MODEL["max_shift"]      # 8 px
SS = 4                                 # süper örnekleme (stimuli.draw_config ile aynı)
MARGIN = 2.0                           # stimuli.fits ile aynı

CENTER_RANGE = (0.6, 1.4)              # merkez çapı, D cinsinden (eşit dağılımlı)

# B-fiz çevre parametreleri (hepsi D cinsinden, merkezden bağımsız).
# Test koşulları bu aralıkların içinde kalır (check_test_coverage ile kontrol edilir).
P_SINGLE = 0.2                         # B-fiz'de tek daire oranı
SURR_D_RANGE = (0.3, 2.5)              # çevre daire çapı, log-eşit (test: 0.4–2.0)
INNER_RANGE = (0.7, 2.2)               # merkezden çevre iç kenarına uzaklık (test: 0.7 ve 2.0)
N_RANGE = (3, 8)                       # çevre daire sayısı (test: 3–8)
MIN_SPACING = 1.1                      # komşu çevre daireleri arası merkez uzaklığı ≥ 1.1·çap

# INNER_RANGE[0] = 0.7 olduğundan en büyük merkez (yarıçap 0.7·D) bile çevreye
# değmez; böylece çakışmayı önlemek için merkezle ilgili bir ret gerekmez ve
# çevre parametreleri etiketten tamamen bağımsız kalır.
assert INNER_RANGE[0] >= CENTER_RANGE[1] / 2

REGIMES = ("A", "B-fiz")


# ---------------------------------------------------------------- çizim
def render(circles, shift=(0.0, 0.0), canvas=CANVAS, ss=SS):
    """(x, y, çap) listesini çizer. stimuli.draw_config ile piksel piksel aynı
    yöntem (4x çizim + LANCZOS küçültme); check_render_matches_stimuli test eder."""
    big = canvas * ss
    img = Image.new("L", (big, big), 255)
    dr = ImageDraw.Draw(img)
    c0 = big / 2
    for x, y, d in circles:
        cx, cy, r = c0 + (x + shift[0]) * ss, c0 + (y + shift[1]) * ss, d * ss / 2
        # stimuli.draw_config'teki düzeltmenin aynısı: PIL kutuyu kapsayıcı
        # çizdiği için (çap +1 px) kutu 0.5 px içeri alınır.
        dr.ellipse([cx - r + 0.5, cy - r + 0.5, cx + r - 0.5, cy + r - 0.5], fill=0)
    return img.resize((canvas, canvas), Image.Resampling.LANCZOS) if ss > 1 else img


def to_array(img):
    """PIL (siyah daire / beyaz zemin) -> float32 dizi (daire 1, zemin 0).
    test.py de AYNI fonksiyonu kullanmalı."""
    return 1.0 - np.asarray(img, dtype=np.float32) / 255.0


# ---------------------------------------------------------------- örnekleme
def _max_n(ds, R):
    """Yarıçapı R olan çembere, aralarında MIN_SPACING·ds bırakarak sığan en fazla daire."""
    s = MIN_SPACING * ds / (2 * R)
    if s >= 1:
        return 0
    return int(math.floor(math.pi / math.asin(s)))


def sample_surround(rng):
    """B-fiz için merkezden bağımsız rastgele çevre. Döndürür: (x, y, çap) listesi, bilgi."""
    limit = CANVAS / 2 - MARGIN - MAX_SHIFT          # kayma ne olursa olsun taşmasın
    while True:                                       # ret yalnızca çevre parametrelerine bağlı
        ds = D * math.exp(rng.uniform(*np.log(SURR_D_RANGE)))
        inner = D * rng.uniform(*INNER_RANGE)
        R = inner + ds / 2
        if R + ds / 2 > limit:
            continue
        n_hi = min(N_RANGE[1], _max_n(ds, R))
        if n_hi < N_RANGE[0]:
            continue
        n = int(rng.integers(N_RANGE[0], n_hi + 1))
        rot = rng.uniform(0, 2 * math.pi)
        circ = [(R * math.cos(rot + 2 * math.pi * k / n),
                 R * math.sin(rot + 2 * math.pi * k / n), ds) for k in range(n)]
        return circ, dict(surr_d=ds / D, inner=inner / D, n=n)


def sample(regime, rng):
    """Tek bir eğitim örneği. Döndürür: PIL görüntü, etiket, bilgi sözlüğü."""
    if regime not in REGIMES:
        raise ValueError(f"Bilinmeyen rejim: {regime!r}; seçenekler: {REGIMES}")
    y = rng.uniform(*CENTER_RANGE)                    # etiket = merkez çapı / D
    circles = [(0.0, 0.0, y * D)]
    info = dict(regime=regime, y=y, surr_d=np.nan, inner=np.nan, n=0)
    if regime == "B-fiz" and rng.random() >= P_SINGLE:
        surr, sinfo = sample_surround(rng)
        circles += surr
        info.update(sinfo)
    shift = tuple(rng.uniform(-MAX_SHIFT, MAX_SHIFT, size=2))
    assert st.fits(circles, CANVAS, shift, MARGIN)
    info["shift"] = shift
    return render(circles, shift), y, info


# ---------------------------------------------------------------- Dataset
_SPLIT_ID = {"train": 0, "val": 1}


class EbbinghausDataset(Dataset):
    """Anında üretilen eğitim/doğrulama verisi.

    regime : "A" veya "B-fiz"
    split  : "train" (her epoch yeni görseller) veya "val" (sabit set)
    n      : görsel sayısı (plan: train 20.000, val 2.000)
    seed   : deney seed'i (model "katılımcısı")

    Her epoch başında train setinde set_epoch(e) çağırın. DataLoader'ı
    persistent_workers=False (varsayılan) ile kullanın ki yeni epoch işçilere geçsin.
    """

    def __init__(self, regime, split="train", n=20_000, seed=0):
        if split not in _SPLIT_ID:
            raise ValueError(f"split 'train' ya da 'val' olmalı, gelen: {split!r}")
        if regime not in REGIMES:
            raise ValueError(f"Bilinmeyen rejim: {regime!r}; seçenekler: {REGIMES}")
        self.regime, self.split, self.n, self.seed = regime, split, n, seed
        self.epoch = 0

    def set_epoch(self, epoch):
        if self.split == "train":
            self.epoch = epoch

    def __len__(self):
        return self.n

    def rng_for(self, idx):
        regime_id = REGIMES.index(self.regime)
        return np.random.default_rng([self.seed, regime_id, _SPLIT_ID[self.split], self.epoch, idx])

    def __getitem__(self, idx):
        img, y, _ = sample(self.regime, self.rng_for(idx))
        x = to_array(img)[None]                         # 1 x H x W
        if torch is None:
            return x, np.float32(y)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


# ---------------------------------------------------------------- kontroller
def check_render_matches_stimuli():
    """render() ile stimuli.draw_config aynı pikselleri üretiyor mu? (eğitim = test çizimi)"""
    worst = 0
    for _, ratio, gap in st.condition_list():
        for rot, shift in [(0.0, (0.0, 0.0)), (0.7, (MAX_SHIFT, -MAX_SHIFT))]:
            a = np.asarray(st.draw_config(canvas=CANVAS, center_d=D, ratio=ratio, gap=gap,
                                          ref_d=D, rotation=rot, shift=shift), dtype=int)
            b = np.asarray(render(st.layout(D, ratio, gap, D, rot), shift), dtype=int)
            worst = max(worst, int(np.abs(a - b).max()))
    assert worst == 0, f"render ile draw_config farklı (en büyük fark {worst})"
    return worst


def check_test_coverage():
    """8 test koşulunun çevre parametreleri B-fiz eğitim aralığında mı?"""
    rows = []
    for name, ratio, gap in st.condition_list():
        if ratio is None:
            rows.append((name, None, None, 0, True))
            continue
        circ = st.layout(D, ratio, gap, D)
        surr_d = ratio
        inner = 0.5 + gap
        n = len(circ) - 1
        R = (inner + surr_d / 2) * D
        ok = (SURR_D_RANGE[0] <= surr_d <= SURR_D_RANGE[1]
              and INNER_RANGE[0] <= inner <= INNER_RANGE[1]
              and N_RANGE[0] <= n <= min(N_RANGE[1], _max_n(surr_d * D, R))
              and R + surr_d * D / 2 <= CANVAS / 2 - MARGIN - MAX_SHIFT)
        rows.append((name, surr_d, inner, n, ok))
    bad = [r[0] for r in rows if not r[4]]
    assert not bad, f"Eğitim dağılımı dışında kalan test koşulları: {bad}"
    return rows


def check_independence(n=4000, seed=123):
    """B-fiz'de etiket ile çevre parametreleri arasındaki korelasyon ≈ 0 olmalı."""
    rng = np.random.default_rng(seed)
    infos = []
    for _ in range(n):
        y = rng.uniform(*CENTER_RANGE)
        if rng.random() >= P_SINGLE:
            _, inf = sample_surround(rng)
            infos.append((y, inf["surr_d"], inf["inner"], inf["n"]))
    a = np.array(infos)
    r = {k: float(np.corrcoef(a[:, 0], a[:, i])[0, 1]) for i, k in enumerate(["surr_d", "inner", "n"], 1)}
    assert all(abs(v) < 0.06 for v in r.values()), f"Etiket–çevre korelasyonu yüksek: {r}"
    return r


def check_all(verbose=True):
    w = check_render_matches_stimuli()
    cov = check_test_coverage()
    r = check_independence()
    if verbose:
        print(f"[1] render == stimuli.draw_config   : en büyük piksel farkı {w}  ✓")
        print("[2] Test koşulları B-fiz aralığında  :")
        for name, sd, inner, n, ok in cov:
            if sd is None:
                print(f"      {name:<10} çevre yok                           ✓")
            else:
                print(f"      {name:<10} çap {sd:.1f}·D  iç kenar {inner:.1f}·D  n={n}   ✓")
        print("[3] Etiket–çevre korelasyonu (B-fiz) : " +
              ", ".join(f"{k}: {v:+.3f}" for k, v in r.items()) + "  ✓")
    return True


if __name__ == "__main__":
    check_all()
