"""
P1 "bitti" ölçütü: fit kodu simüle verideki PSE'leri ±%1 içinde buluyor mu?

Test 1 (asıl ölçüt): Çok sayıda deneme, gürültüsüz katılımcı.
    Fit kodunda bir hata yoksa PSE'ler ±%1 içinde çıkmalı.
Test 2 (gerçekçi): 18 katılımcı x 30 deneme/koşul. Bireysel PSE'ler
    gürültülü olur; GRUP ortalaması gerçek değere yakın olmalı (±%1).

Kullanım:  python analysis/check_fit.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import sim_data as sim
import fit_pse as fit

TOL = 1.0  # yüzde puanı


def test_large_n():
    rng = np.random.default_rng(0)
    truth = {name: pse for name, _, _, pse in sim.CONDITIONS}
    df, _ = sim.simulate_participant("BIG", rng, true_pse=truth, reps=400,
                                     slope=0.045, add_catch=False, timeout_rate=0)
    df["key"] = df["resp.keys"]
    df["chose_cmp"] = (df["key"] != df["context_side"]).astype(float)
    res = fit.fit_participant(df)
    res["true_pse"] = res["condition"].map(truth)
    res["err_pct"] = (res["pse"] / res["true_pse"] - 1) * 100
    return res[["condition", "true_pse", "pse", "err_pct"]], res["s"].iloc[0]


def test_realistic(n=18, seed=1, tmp=Path("data/_check")):
    rng = np.random.default_rng(seed)
    tmp.mkdir(parents=True, exist_ok=True)
    truths = []
    for i in range(1, n + 1):
        df, t = sim.simulate_participant(f"P{i:02d}", rng)
        df.to_csv(tmp / f"P{i:02d}.csv", index=False)
        truths.append(t)
    truth = pd.concat(truths)
    clean, _ = fit.apply_exclusions(fit.load_folder(tmp))
    per_pp = fit.fit_all(clean).merge(truth, on=["participant", "condition"])
    per_pp["true_shift"] = (per_pp["true_pse"] - 1) * 100
    g = per_pp.groupby("condition")[["true_shift", "pse_shift_pct"]].mean()
    g["err"] = g["pse_shift_pct"] - g["true_shift"]
    indiv_err = (per_pp["pse_shift_pct"] - per_pp["true_shift"]).abs()
    for f in tmp.glob("*.csv"):
        f.unlink()
    tmp.rmdir()
    return g, indiv_err


def main():
    ok = True
    res, s = test_large_n()
    print("Test 1 — büyük n (koşul başına 2000 deneme), gerçek s = 0.045")
    print(res.round(4).to_string(index=False))
    print(f"tahmini s = {s:.4f}")
    t1 = (res["err_pct"].abs() <= TOL).all()
    print("SONUÇ:", "GEÇTİ" if t1 else "KALDI", f"(en büyük hata %{res['err_pct'].abs().max():.2f})\n")
    ok &= t1

    g, indiv = test_realistic()
    print("Test 2 — gerçekçi: 18 katılımcı x 30 deneme/koşul (PSE kayması, %)")
    print(g.round(2).to_string())
    t2 = (g["err"].abs() <= TOL).all()
    print("SONUÇ:", "GEÇTİ" if t2 else "KALDI", f"(grup ortalamasında en büyük hata {g['err'].abs().max():.2f} puan)")
    print(f"Bilgi: bireysel PSE hatası medyan {indiv.median():.2f}, %90'lık {indiv.quantile(.9):.2f} puan "
          "(30 denemeyle normal; bu yüzden analiz grup düzeyinde yapılıyor)")
    ok &= t2

    print("\nGENEL:", "P1 ölçütü sağlandı ✔" if ok else "Bir test kaldı ✘")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
