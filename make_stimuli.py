import math 
import csv
import random
from pathlib import Path

from PIL import Image

from stimuli import draw_config, HUMAN, NEAR_GAP, COMPARISON, condition_list

KLASOR = Path(__file__).resolve().parent


def make_pair(ratio, gap, comparison, context_side, rng):
    """Bir deneme görseli: bir yarıda bağlamlı merkez, diğerinde tek daire."""
    canvas = HUMAN["canvas"]      # 560
    D = HUMAN["center_d"]         # 60
    s = HUMAN["max_shift"]        # 10 px

    # Rastgele dönüş ve konum kayması
    rot = rng.uniform(0.0, 2.0 * math.pi)
    ctx_shift = (rng.uniform(-s, s), rng.uniform(-s, s))
    cmp_shift = (rng.uniform(-s, s), rng.uniform(-s, s))

    # İki yarıyı ayrı ayrı çiz
    ctx = draw_config(canvas, D, ratio=ratio, gap=gap, ref_d=D,
                      rotation=rot, shift=ctx_shift)
    cmp = draw_config(canvas, D * comparison, ratio=None, shift=cmp_shift)

    # 1120 x 560 beyaz tuval, iki yarıyı yapıştır
    pair = Image.new("L", (2 * canvas, canvas), 255)
    if context_side == "left":
        left, right = ctx, cmp
    else:
        left, right = cmp, ctx
    pair.paste(left, (0, 0))
    pair.paste(right, (canvas, 0))
    return pair, rot, ctx_shift, cmp_shift


PNG_DIR = KLASOR / "stimuli" / "png"


CSV_PATH = KLASOR / "stimuli" / "stimuli.csv"

FIELDS = [
    "image", "condition", "ratio", "gap", "comparison", "comparison_d",
    "center_d", "context_side", "corrAns", "rotation_deg",
    "ctx_shift_x", "ctx_shift_y", "cmp_shift_x", "cmp_shift_y",
]


def main():
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    for eski in PNG_DIR.glob("*.png"):
        eski.unlink()

    rng = random.Random(2026)
    rows = []
    for name, r, g in condition_list():
        for c in COMPARISON:
            for side in ("left", "right"):
                img, rot, cs, ms = make_pair(r, g, c, side, rng)
                fname = f"{name}_c{c:.2f}_{side[0].upper()}.png"
                img.save(PNG_DIR / fname)

                # Doğru cevap: fiziksel olarak büyük dairenin tarafı
                if math.isclose(c, 1.0):
                    corr = ""                      # eşit boyut, doğru cevap yok
                elif c < 1.0:
                    corr = side                    # bağlamlı daire büyük
                else:
                    corr = "right" if side == "left" else "left"  # karşılaştırma büyük

                rows.append({
                    "image": f"stimuli/png/{fname}",
                    "condition": name,
                    "ratio": "" if r is None else r,
                    "gap": g,
                    "comparison": f"{c:.2f}",
                    "comparison_d": f"{HUMAN['center_d'] * c:.1f}",
                    "center_d": HUMAN["center_d"],
                    "context_side": side,
                    "corrAns": corr,
                    "rotation_deg": f"{math.degrees(rot):.1f}",
                    "ctx_shift_x": f"{cs[0]:.2f}", "ctx_shift_y": f"{cs[1]:.2f}",
                    "cmp_shift_x": f"{ms[0]:.2f}", "cmp_shift_y": f"{ms[1]:.2f}",
                })

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(len(rows), "görsel ->", PNG_DIR)
    print("CSV ->", CSV_PATH)

if __name__ == "__main__":
    main()