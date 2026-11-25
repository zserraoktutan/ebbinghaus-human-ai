"""Ebbinghaus uyaran üreteci (insan deneyi ve modeller için ortak)."""

import math
from PIL import Image, ImageDraw

RATIOS = [0.4, 0.7, 1.0, 1.5, 2.0]      # çevre çapı / referans merkez çapı
NEAR_GAP = 0.2                          # kenar-kenar boşluk (referans çap cinsinden)
FAR_GAP = 1.5                           # uzak kontrol boşluğu
COMPARISON = [0.80, 0.90, 1.00, 1.10, 1.20]  # karşılaştırma boyutları (x D)  

HUMAN = dict(canvas=560, center_d=60, max_shift=10)  # insan deneyi
MODEL = dict(canvas=224, center_d=24, max_shift=8)   # model girdisi


def layout(center_d, ratio=None, gap=NEAR_GAP, ref_d=None, rotation=0.0):
    """Dairelerin (x, y, çap) listesi; merkez (0, 0) etrafında.

    ref_d: çevreyi belirleyen referans çap. Merkez çapı değişse de çevre
    sabit kalsın diye ayrı tutulur (varsayılan: center_d).
    """
    ref_d = center_d if ref_d is None else ref_d
    circles = [(0.0, 0.0, center_d)]

    if ratio is not None:
        ds = ratio * ref_d
        R = ref_d / 2.0 + gap * ref_d + ds / 2.0

        # Daire sayısı yakın düzene göre belirlenir; uzak kontrolde
        # yalnızca mesafe değişsin diye aynı n kullanılır.
        R_near = ref_d / 2.0 + NEAR_GAP * ref_d + ds / 2.0
        n = max(3, min(8, int(2.0 * math.pi * R_near / (1.25 * ds))))

        for k in range(n):
            a = rotation + 2.0 * math.pi * k / n
            circles.append((R * math.cos(a), R * math.sin(a), ds))

    return circles


def extent(circles):
    """Merkezden en dış kenara uzaklık (px)."""
    return max(math.hypot(x, y) + d / 2.0 for x, y, d in circles)


def fits(circles, canvas, shift=(0.0, 0.0), margin=2.0):
    """Kaydırılmış düzenin hiçbir dairesi tuvalden taşmıyor mu?"""
    half = canvas / 2.0
    for x, y, d in circles:
        cx, cy = x + shift[0], y + shift[1]
        if (abs(cx) + d / 2.0 > half - margin) or (abs(cy) + d / 2.0 > half - margin):
            return False
    return True


def draw_config(
    canvas=560,
    center_d=60,
    ratio=None,
    gap=NEAR_GAP,
    ref_d=None,
    rotation=0.0,
    shift=(0.0, 0.0),
    fg=0,
    bg=255,
    ss=4,
):
    """ratio=None -> merkez tek başına (Kontrol 1).

    shift: tüm düzenin (dx, dy) konum kayması, px.
    ss: süper örnekleme katsayısı (kenar yumuşatma); küçük tuvallerde
    yüzde birkaçlık çap farkları piksel yuvarlamasında kaybolmasın diye.
    """
    circles = layout(center_d, ratio, gap, ref_d, rotation)
    if not fits(circles, canvas, shift):
        raise ValueError("Düzen tuvalden taşıyor")

    big = canvas * ss
    img = Image.new("L", (big, big), bg)
    dr = ImageDraw.Draw(img)
    c0 = big / 2.0

    for x, y, d in circles:
        cx = c0 + (x + shift[0]) * ss
        cy = c0 + (y + shift[1]) * ss
        r = d * ss / 2.0
        # PIL kutuyu kapsayıcı çizer (çap +1 px); 0.5 px içeri alarak düzeltilir.
        dr.ellipse([cx - r + 0.5, cy - r + 0.5, cx + r - 0.5, cy + r - 0.5], fill=fg)

    return img.resize((canvas, canvas), Image.Resampling.LANCZOS) if ss > 1 else img


def condition_list():
    conds = [("tek", None, NEAR_GAP)]
    conds += [(f"r{r}", r, NEAR_GAP) for r in RATIOS]
    conds += [("uzak_r0.4", 0.4, FAR_GAP), ("uzak_r2.0", 2.0, FAR_GAP)]
    return conds


def check_all():
    """Tüm test koşulları x karşılaştırma boyutları x en uç kaymalar
    (köşeler dahil) x dönüşler için taşma kontrolü.
    """
    worst = {}
    for setup_name, p in [("insan", HUMAN), ("model", MODEL)]:
        s = p["max_shift"]
        shifts = [(dx, dy) for dx in (-s, 0.0, s) for dy in (-s, 0.0, s)]
        for name, r, g in condition_list():
            for c in COMPARISON:
                for k in range(72):
                    rot = k * math.pi / 36.0
                    circ = layout(p["center_d"] * c, r, g, ref_d=p["center_d"], rotation=rot)
                    for sh in shifts:
                        assert fits(circ, p["canvas"], sh), (setup_name, name, c, rot, sh)
            worst[(setup_name, name)] = extent(
                layout(p["center_d"] * max(COMPARISON), r, g, ref_d=p["center_d"])
            )
    return worst


if __name__ == "__main__":
    for (s, name), e in check_all().items():
        print(f"{s:6s} {name:10s} dış sınır {e:6.1f} px")
    print("Tüm koşullar tuval içinde.")