"""
Simüle insan verisi üretir (PsychoPy Builder çıktısına benzer CSV'ler).

Amaç: Gerçek veri gelmeden fit kodunu test etmek. Doğru PSE'leri biz
belirlediğimiz için fit kodunun onları geri bulup bulmadığını ölçebiliriz.

Kullanım:
    python analysis/sim_data.py                 # 18 katılımcı -> data/sim/
    python analysis/sim_data.py --n 20 --seed 3

Her katılımcı için bir CSV yazılır: data/sim/P01_sim.csv, P02_sim.csv ...
Ayrıca doğru değerler data/sim/truth.csv dosyasına yazılır.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ---- Tasarım (PDF Bölüm 4 ile aynı) ---------------------------------------
COMPARISON = [0.84, 0.92, 1.00, 1.08, 1.16]   # karşılaştırma / D
REPS = 6                                       # her (koşul, boyut) için tekrar
CENTER_D = 60                                  # px, yalnızca bilgi amaçlı

# Koşul adı, oran, uzak mı, varsayılan "gerçek" grup PSE'si.
# PSE > 1: bağlamdaki merkez daha BÜYÜK algılanıyor (küçük çevre daireler).
# Adlar make_stimuli.py'deki adlardan farklıysa sorun değil: fit kodu
# CSV'de hangi koşul adı varsa onu kullanır.
CONDITIONS = [
    # name,      ratio, far,  true_pse
    ("single",   None,  False, 1.000),
    ("r0.4",     0.4,   False, 1.060),
    ("r0.7",     0.7,   False, 1.030),
    ("r1.0",     1.0,   False, 1.005),
    ("r1.5",     1.5,   False, 0.970),
    ("r2.0",     2.0,   False, 0.940),
    ("far0.4",   0.4,   True,  1.020),
    ("far2.0",   2.0,   True,  0.985),
]

# Dikkat (catch) denemeleri: tek merkez, karşılaştırma çok farklı
CATCH = [0.6, 1.4]
CATCH_REPS = 3          # 2 boyut x 2 taraf x 3 = 12 deneme

# Katılımcılar arası değişkenlik
SD_PSE_LOG = 0.02       # bireysel PSE sapması (log birim, ~%2)
SLOPE_LOG = 0.045       # psikometrik genişlik s (log birim). JND ~ %5
SD_SLOPE = 0.010
LAPSE = 0.02            # dikkatsizlik oranı


def p_choose_comparison(c, pse, s, lapse):
    """Karşılaştırma dairesinin 'daha büyük' seçilme olasılığı."""
    core = 1.0 / (1.0 + np.exp(-(np.log(c) - np.log(pse)) / s))
    return lapse / 2 + (1 - lapse) * core


def simulate_participant(pid, rng, true_pse=None, reps=REPS, lapse=LAPSE,
                         slope=None, add_catch=True, timeout_rate=0.01):
    """Tek katılımcının deneme tablosunu döndürür (PsychoPy sütun adlarıyla)."""
    s = slope if slope is not None else max(0.02, rng.normal(SLOPE_LOG, SD_SLOPE))
    rows = []
    truth = []
    for name, ratio, far, grp_pse in CONDITIONS:
        pse = grp_pse if true_pse is None else true_pse[name]
        pse_i = float(np.exp(np.log(pse) + rng.normal(0, SD_PSE_LOG))) if true_pse is None else pse
        truth.append({"participant": pid, "condition": name, "true_pse": pse_i, "true_s": s})
        for c in COMPARISON:
            for side in ("left", "right"):
                for _ in range(reps // 2):
                    rows.append(_trial(pid, name, ratio, far, c, side, pse_i, s, lapse, rng, timeout_rate))
    if add_catch:
        for c in CATCH:
            for side in ("left", "right"):
                for _ in range(CATCH_REPS):
                    rows.append(_trial(pid, "catch", None, False, c, side, 1.0, s, lapse, rng, timeout_rate))
    df = pd.DataFrame(rows).sample(frac=1, random_state=int(rng.integers(1e9))).reset_index(drop=True)
    df["trials.thisN"] = np.arange(len(df))
    return df, pd.DataFrame(truth)


def _trial(pid, name, ratio, far, c, side, pse, s, lapse, rng, timeout_rate):
    cmp_side = "right" if side == "left" else "left"
    if np.isclose(c, 1.0):
        corr = ""
    else:
        corr = side if c < 1.0 else cmp_side
    if rng.random() < timeout_rate:
        key, rt = None, None                      # 3 s içinde yanıt yok
    else:
        chose_cmp = rng.random() < p_choose_comparison(c, pse, s, lapse)
        key = cmp_side if chose_cmp else side
        rt = float(np.clip(rng.lognormal(np.log(0.65), 0.35), 0.12, 2.99))
    return {
        "participant": pid,
        "condition": name,
        "ratio": "" if ratio is None else ratio,
        "far": far,
        "comparison": c,
        "comparison_d": round(CENTER_D * c, 1),
        "center_d": CENTER_D,
        "context_side": side,
        "corrAns": corr,
        "resp.keys": key,
        "resp.rt": rt,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=18, help="katılımcı sayısı")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="data/sim")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    truths = []
    for i in range(1, args.n + 1):
        pid = f"P{i:02d}"
        df, truth = simulate_participant(pid, rng)
        df.to_csv(out / f"{pid}_sim.csv", index=False)
        truths.append(truth)
    pd.concat(truths).to_csv(out / "truth.csv", index=False)
    print(f"{args.n} katılımcı -> {out}/  (doğru değerler: {out}/truth.csv)")


if __name__ == "__main__":
    main()
