"""
Model 2AFC sonuçlarını toplu özetler: her rejim ve seed için PSE, sonra
seed'ler arası ortalama ± SD.

Kullanım:
    py analysis\\model_summary.py results
    py analysis\\model_summary.py results --desen "*_siratesti_aralik.csv"

Dosya adları: model_2afc_<REJİM>_seed<N>_... .csv
Sütunlar: condition, comparison, p  (p = P(bağlamlı merkez büyük seçildi))
Çıktılar (aynı klasöre):
    model_pse_uzun.csv   rejim, seed, koşul başına PSE / sansür
    model_pse_tablo.csv  Kişi 1'in tablosu biçiminde (koşul x rejim-seed)
    model_pse_ozet.csv   rejim x koşul: ortalama ± SD, aralık, sansür sayısı
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import fit_pse as fit

NAME = re.compile(r"model_2afc_(?P<rejim>.+?)_seed(?P<seed>\d+)")
ORDER = ["tek", "r0.4", "r0.7", "r1.0", "r1.5", "r2.0", "uzak_r0.4", "uzak_r2.0"]


def crossing(agg):
    """Model-bağımsız PSE: p(karşılaştırma)'nın 0.5'i ilk geçtiği nokta
    (log karşılaştırma uzayında doğrusal ara değer). Birden çok geçiş = eğri monoton değil."""
    out = {}
    for c, g in agg.groupby("condition"):
        g = g.sort_values("comparison")
        x, y = np.log(g["comparison"].to_numpy()), (g["k"] / g["n"]).to_numpy()
        idx = np.where(np.diff(np.sign(y - 0.5)) != 0)[0]
        if len(idx) == 0:
            out[c] = (np.nan, 0)
            continue
        j = idx[0]
        xc = x[j] + (0.5 - y[j]) * (x[j + 1] - x[j]) / (y[j + 1] - y[j])
        out[c] = (float(np.exp(xc)), len(idx))
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    folder = Path(args[0] if args else "results")
    pattern = "*_siratesti_aralik.csv"
    if "--desen" in sys.argv:
        pattern = sys.argv[sys.argv.index("--desen") + 1]
        args = [a for a in args if a != pattern]
    files = sorted(f for f in folder.glob(pattern) if not f.stem.endswith("_pse"))
    if not files:
        raise SystemExit(f"{folder} içinde '{pattern}' ile eşleşen dosya yok.")

    rows = []
    for f in files:
        m = NAME.search(f.stem)
        if not m:
            print(f"Atlandı (ad biçimi tanınmadı): {f.name}")
            continue
        agg, _ = fit.load_model(f)
        res = fit.fit_agg(agg, lapse_max=0.0, shared_slope=False)
        res["rejim"], res["seed"] = m["rejim"], int(m["seed"])
        cr = crossing(agg)
        res["pse_gecis"] = res["condition"].map(lambda c: cr[c][0])
        res["gecis_sayisi"] = res["condition"].map(lambda c: cr[c][1])
        # Modelin kendi tek-daire temel çizgisine göre kayma (%): yanılsama = bağlama bağlı sapma
        base = res.loc[res["condition"] == "tek", "pse"]
        base = float(base.iloc[0]) if len(base) and pd.notna(base.iloc[0]) else np.nan
        res["kayma_tek_gore_pct"] = (res["pse"] / base - 1) * 100
        res["aralik"] = f"{agg['comparison'].min():.2f}-{agg['comparison'].max():.2f}"
        rows.append(res)
        print(f"Okundu: {f.name}  ({agg['comparison'].nunique()} düzey, {res['aralik'].iloc[0]})")
    long = pd.concat(rows, ignore_index=True)
    long["goster"] = np.where(long["censored"] != "", long["censored"],
                              long["pse"].map(lambda v: f"{v:.3f}"))

    order = [c for c in ORDER if c in long["condition"].unique()] + \
            sorted(set(long["condition"]) - set(ORDER))
    wide = long.pivot_table(index="condition", columns=["rejim", "seed"],
                            values="goster", aggfunc="first").reindex(order)

    summ = []
    for (rej, cond), g in long.groupby(["rejim", "condition"]):
        ok = g["pse"].dropna()
        n_up = g["censored"].str.startswith(">").sum()
        n_dn = g["censored"].str.startswith("<").sum()
        if len(ok) >= 2:
            txt = f"{ok.mean():.3f} ± {ok.std(ddof=1):.3f}"
        elif len(ok) == 1:
            txt = f"{ok.iloc[0]:.3f} (tek seed)"
        else:
            txt = "—"
        if n_up or n_dn:
            lim = g.loc[g["censored"] != "", "censored"].iloc[0]
            txt += f"  [{n_up + n_dn}/{len(g)} seed {lim}]"
        summ.append({"rejim": rej, "condition": cond, "n_seed": len(g), "n_olculen": len(ok),
                     "ortalama": ok.mean() if len(ok) else np.nan,
                     "sd": ok.std(ddof=1) if len(ok) >= 2 else np.nan,
                     "min": ok.min() if len(ok) else np.nan,
                     "max": ok.max() if len(ok) else np.nan,
                     "kayma_pct": (ok.mean() - 1) * 100 if len(ok) else np.nan,
                     "kayma_tek_gore_pct": g["kayma_tek_gore_pct"].mean(),
                     "kayma_tek_gore_sd": g["kayma_tek_gore_pct"].std(ddof=1),
                     "fit_gecis_max_fark": (g["pse"] - g["pse_gecis"]).abs().max(),
                     "n_ust_sansur": n_up, "n_alt_sansur": n_dn, "ozet": txt})
    summ = pd.DataFrame(summ)
    show = summ.pivot(index="condition", columns="rejim", values="ozet").reindex(order)

    long.drop(columns="goster").to_csv(folder / "model_pse_uzun.csv", index=False)
    wide.to_csv(folder / "model_pse_tablo.csv")
    summ.to_csv(folder / "model_pse_ozet.csv", index=False)

    pd.set_option("display.width", 200)
    print("\nSeed başına PSE (fit):")
    print(wide.to_string())
    print("\nSeed'ler arası ortalama ± SD  [aralık dışı seed sayısı]:")
    print(show.to_string())
    rel = summ.pivot(index="condition", columns="rejim", values="kayma_tek_gore_pct").reindex(order)
    print("\nModelin kendi 'tek' PSE'sine göre kayma (%, seed ortalaması):")
    print(rel.round(1).to_string())
    multi = long[long["gecis_sayisi"] > 1]
    if len(multi):
        print("\nUYARI: eğrisi 0.5'i birden çok kez geçen (monoton olmayan) koşullar:")
        print(multi[["rejim", "seed", "condition", "gecis_sayisi"]].to_string(index=False))
    d = (long["pse"] - long["pse_gecis"]).abs()
    print(f"\nFit PSE ile 0.5-geçiş PSE'si arasındaki en büyük fark: {d.max():.3f} "
          f"({long.loc[d.idxmax(), 'rejim']} seed {long.loc[d.idxmax(), 'seed']}, {long.loc[d.idxmax(), 'condition']})")
    print("\nNot: Aralık dışı (sansürlü) seed'ler ortalamaya katılmaz; yalnızca sayılır.")
    print(f"\n-> {folder/'model_pse_uzun.csv'}, model_pse_tablo.csv, model_pse_ozet.csv")


if __name__ == "__main__":
    main()
