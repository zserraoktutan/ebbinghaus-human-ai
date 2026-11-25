"""
Psikometrik fit: her katılımcı ve her koşul için PSE (öznel eşitlik noktası).

Model (ortak eğimli lojistik, PDF Bölüm 5):
    P(karşılaştırma seçildi) = λ/2 + (1-λ) * σ( (log c - log PSE_k) / s )
      c      : karşılaştırma boyutu / D  (0.84 ... 1.16)
      PSE_k  : k koşulundaki PSE (koşula özgü)
      s      : genişlik, katılımcının tüm koşullarında ORTAK
      λ      : dikkatsizlik (lapse) oranı, 0–0.10 arası
Her koşulda kişi başı yalnızca 30 deneme olduğu için eğimi ortak tutmak
PSE tahminlerini kararlı yapar.

PSE kayması (%) = (PSE - 1) * 100.  Pozitif: merkez daha büyük algılanıyor.

Kullanım:
    python analysis/fit_pse.py data/sim            # simüle veri
    python analysis/fit_pse.py data/raw            # gerçek PsychoPy verisi
Çıktılar: <klasör>/pse_by_participant.csv, <klasör>/pse_group.csv,
          <klasör>/excluded.csv, <klasör>/psychometric.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# PsychoPy'deki klavye bileşeninin adı farklıysa yalnızca burayı değiştirin
KEY_COL = "resp.keys"
RT_COL = "resp.rt"
MAIN_LOOP = "trials"   # Builder'daki ana döngünün adı

# Önceden belirlenmiş dışlama ölçütleri (PDF Bölüm 4)
MIN_CATCH_ACC = 0.80
MAX_TIMEOUT = 0.10
MIN_RT = 0.150   # saniye


# ---------------------------------------------------------------------------
# Veri okuma ve temizleme
# ---------------------------------------------------------------------------
def load_folder(folder):
    """Klasördeki tüm katılımcı CSV'lerini tek tabloda birleştirir."""
    files = sorted(p for p in Path(folder).glob("*.csv")
                   if p.name not in {"truth.csv", "pse_by_participant.csv",
                                     "pse_group.csv", "excluded.csv"})
    if not files:
        raise SystemExit(f"{folder} içinde CSV yok.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # Yalnızca ana döngünün (adı 'trials') satırları; alıştırma döngüsü
    # ('practice') ve talimat satırları atılır.
    if MAIN_LOOP + ".thisN" in df.columns:
        df = df[df[MAIN_LOOP + ".thisN"].notna()]
    df = df[df["condition"].notna()].copy()
    df["key"] = df[KEY_COL].replace({"None": np.nan, "": np.nan})
    df["rt"] = pd.to_numeric(df[RT_COL], errors="coerce")
    df["timeout"] = df["key"].isna()
    other = {"left": "right", "right": "left"}
    df["chose_cmp"] = (df["key"] == df["context_side"].map(other)).astype(float)
    df.loc[df["timeout"], "chose_cmp"] = np.nan
    return df


def apply_exclusions(df):
    """Katılımcı ve deneme dışlamalarını uygular. (temiz_veri, rapor) döndürür."""
    report = []
    for pid, d in df.groupby("participant"):
        catch = d[d["condition"] == "catch"]
        catch_ok = catch[~catch["timeout"]]
        acc = (catch_ok["key"] == catch_ok["corrAns"]).mean() if len(catch_ok) else np.nan
        # Çevreli dikkat denemeleri (ratio dolu): "hep bağlamlı tarafa bas" stratejisini yakalar
        r = pd.to_numeric(catch_ok["ratio"], errors="coerce") if "ratio" in catch_ok else pd.Series(dtype=float)
        ctx = catch_ok[r.notna().to_numpy()] if len(r) else catch_ok.iloc[0:0]
        ctx_acc = (ctx["key"] == ctx["corrAns"]).mean() if len(ctx) else np.nan
        to = d["timeout"].mean()
        reasons = []
        if len(catch) and acc < MIN_CATCH_ACC:
            reasons.append(f"dikkat doğruluğu %{acc*100:.0f}")
        if to > MAX_TIMEOUT:
            reasons.append(f"zaman aşımı %{to*100:.0f}")
        report.append({"participant": pid, "catch_acc": acc, "catch_cevreli_acc": ctx_acc,
                       "timeout_rate": to,
                       "excluded": bool(reasons), "reason": "; ".join(reasons)})
    report = pd.DataFrame(report)
    keep = report.loc[~report["excluded"], "participant"]
    clean = df[df["participant"].isin(keep)
               & (df["condition"] != "catch")
               & ~df["timeout"]
               & (df["rt"] >= MIN_RT)].copy()
    return clean, report


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
def _p(logc, logpse, s, lapse):
    z = np.clip((logc - logpse) / s, -50, 50)
    return lapse / 2 + (1 - lapse) / (1 + np.exp(-z))


def censor_status(agg):
    """Koşul başına PSE ölçüm aralığının içinde mi?
    agg: condition, comparison, k (karşılaştırma seçilen), n
    En büyük karşılaştırmada bile karşılaştırma <%50 seçildiyse PSE > üst sınır;
    en küçükte bile >%50 seçildiyse PSE < alt sınır. Bu durumda PSE tahmin
    EDİLEMEZ, yalnızca bir sınır raporlanır."""
    out = {}
    lo, hi = agg["comparison"].min(), agg["comparison"].max()
    for c, g in agg.groupby("condition"):
        g = g.sort_values("comparison")
        p_lo = g["k"].iloc[0] / g["n"].iloc[0]
        p_hi = g["k"].iloc[-1] / g["n"].iloc[-1]
        if p_hi < 0.5:
            out[c] = f">{hi:.2f}"
        elif p_lo > 0.5:
            out[c] = f"<{lo:.2f}"
        else:
            out[c] = ""
    return out


def fit_agg(agg, lapse_max=0.10, shared_slope=True):
    """Ortak eğimli fit, gruplanmış veri üzerinde.
    agg: condition, comparison, k, n  (k kesirli olabilir: model olasılıkları).
    Aralık dışı (sansürlü) koşullar fit'e katılmaz, PSE'leri NaN olur."""
    if not shared_slope:   # model: her koşula kendi eğimi (deneme sayısı sorun değil)
        parts = [fit_agg(g, lapse_max, True) for _, g in agg.groupby("condition")]
        return pd.concat(parts, ignore_index=True)
    cens = censor_status(agg)
    conds = sorted(agg["condition"].unique())
    ok = [c for c in conds if not cens[c]]
    fit = agg[agg["condition"].isin(ok)]
    pse_map, s, lapse, conv = {}, np.nan, np.nan, False
    if ok:
        idx = fit["condition"].map({c: i for i, c in enumerate(ok)}).to_numpy()
        logc = np.log(fit["comparison"].to_numpy(float))
        k = fit["k"].to_numpy(float)
        n = fit["n"].to_numpy(float)
        K = len(ok)

        def nll(theta):
            p = np.clip(_p(logc, theta[:K][idx], np.exp(theta[K]), theta[K + 1]), 1e-12, 1 - 1e-12)
            return -np.sum(k * np.log(p) + (n - k) * np.log(1 - p))

        x0 = np.r_[np.zeros(K), np.log(0.05), min(0.02, lapse_max)]
        bounds = ([(np.log(0.5), np.log(2.0))] * K
                  + [(np.log(0.001), np.log(0.5)), (0.0, lapse_max)])
        res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
        pse_map = dict(zip(ok, np.exp(res.x[:K])))
        s, lapse, conv = np.exp(res.x[K]), res.x[K + 1], res.success

    out = pd.DataFrame({"condition": conds})
    out["pse"] = out["condition"].map(pse_map)
    out["pse_shift_pct"] = (out["pse"] - 1) * 100
    out["censored"] = out["condition"].map(cens)   # "", ">1.16" ya da "<0.84"
    out["n_trials"] = agg.groupby("condition")["n"].sum().reindex(conds).to_numpy()
    out["s"], out["lapse"], out["converged"] = s, lapse, conv
    # Aralık içinde ama yine de sınıra yakın (güvenilmez) tahminler
    out["at_bound"] = (out["pse"] < 0.80) | (out["pse"] > 1.25)
    return out


def fit_participant(d):
    """Tek katılımcının deneme verisi -> koşul başına PSE."""
    agg = (d.groupby(["condition", "comparison"])["chose_cmp"]
             .agg(k="sum", n="size").reset_index())
    return fit_agg(agg)


def fit_all(clean):
    parts = []
    for pid, d in clean.groupby("participant"):
        f = fit_participant(d)
        f.insert(0, "participant", pid)
        parts.append(f)
    return pd.concat(parts, ignore_index=True)


def group_summary(per_pp):
    """Grup ortalaması yalnızca aralık içindeki PSE'lerden; sansürlüler ayrıca sayılır."""
    rows = []
    for c, g in per_pp.groupby("condition"):
        v = g["pse_shift_pct"].dropna()
        rows.append({"condition": c, "n": len(v),
                     "n_above": (g["censored"].str.startswith(">")).sum(),
                     "n_below": (g["censored"].str.startswith("<")).sum(),
                     "mean_shift_pct": v.mean(), "sd": v.std()})
    out = pd.DataFrame(rows)
    out["sem"] = out["sd"] / np.sqrt(out["n"])
    out["ci95_low"] = out["mean_shift_pct"] - 1.96 * out["sem"]
    out["ci95_high"] = out["mean_shift_pct"] + 1.96 * out["sem"]
    return out


def ratio_slopes(per_pp, clean):
    """Katılımcı başına PSE kayması ~ log(oran) eğimi (yakın koşullar) + t-testi."""
    from scipy.stats import ttest_1samp
    cols = [c for c in ("condition", "ratio", "far", "gap") if c in clean.columns]
    info = clean.drop_duplicates("condition")[cols].copy()
    info["ratio"] = pd.to_numeric(info["ratio"], errors="coerce")
    has_ratio = info["ratio"].notna()
    if "far" in info:                       # simüle veri
        far = info["far"].astype(str) == "True"
    elif "gap" in info:                     # stimuli.csv: yakın koşulların gap'i en sık olan
        near_gap = info.loc[has_ratio, "gap"].mode().iloc[0]
        far = info["gap"] != near_gap
    else:
        far = info["condition"].str.contains("far|uzak", case=False)
    near = info.loc[has_ratio & ~far, ["condition", "ratio"]]
    m = per_pp.merge(near, on="condition")
    m["log_ratio"] = np.log(m["ratio"].astype(float))
    m = m.dropna(subset=["pse_shift_pct"])
    m = m.groupby("participant").filter(lambda x: len(x) >= 2)
    if m.empty:
        return pd.Series(dtype=float), None
    slopes = m.groupby("participant").apply(
        lambda x: np.polyfit(x["log_ratio"], x["pse_shift_pct"], 1)[0], include_groups=False)
    t = ttest_1samp(slopes, 0)
    return slopes, t


def plot(agg, per_pp, path, ylabel="P(karşılaştırma daha büyük)"):
    """agg: condition, comparison, k, n (tüm katılımcılar birleşik)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conds = sorted(agg["condition"].unique())
    fig, axes = plt.subplots(2, (len(conds) + 1) // 2, figsize=(14, 6), sharey=True)
    xs = np.linspace(0.82, 1.18, 200)
    for ax, c in zip(axes.flat, conds):
        g = agg[agg["condition"] == c].groupby("comparison")[["k", "n"]].sum()
        ax.plot(g.index, g["k"] / g["n"], "o", color="#333")
        ax.axhline(0.5, color="#aaa", lw=0.8)
        f = per_pp[(per_pp["condition"] == c) & per_pp["pse"].notna()]
        n_cens = (per_pp[per_pp["condition"] == c]["censored"] != "").sum()
        if len(f):
            pse = np.exp(np.log(f["pse"]).mean())
            s, lapse = f["s"].median(), f["lapse"].median()
            ax.plot(xs, _p(np.log(xs), np.log(pse), s, lapse), color="#2a6fdb")
            ax.axvline(pse, ls="--", color="#2a6fdb", lw=1)
            title = f"{c}  (PSE {pse:.3f})"
        else:
            title = f"{c}  (PSE {per_pp[per_pp['condition'] == c]['censored'].iloc[0]})"
        if n_cens and len(f):
            title += f"  [{n_cens} aralık dışı]"
        ax.set_title(title, fontsize=10)
    for ax in axes.flat[len(conds):]:
        ax.axis("off")
    fig.supxlabel("karşılaştırma / D")
    fig.supylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def load_model(path, p_is_cmp=False):
    """Model 2AFC çıktısı: condition, comparison, p.
    Varsayılan: p = P(model BAĞLAMLI merkezi daha büyük seçti).
    (tek koşulunda p, karşılaştırma büyüdükçe 1'den 0'a iniyorsa doğru varsayım.)
    p karşılaştırmanın seçilme olasılığıysa --p-karsilastirma kullanın."""
    m = pd.read_csv(path)
    need = {"condition", "comparison", "p"}
    if not need <= set(m.columns):
        raise SystemExit(f"Model dosyasında {sorted(need)} sütunları gerekli. Var olanlar: {list(m.columns)}")
    p_cmp = m["p"] if p_is_cmp else 1 - m["p"]
    agg = pd.DataFrame({"condition": m["condition"], "comparison": m["comparison"].astype(float),
                        "k": p_cmp.astype(float), "n": 1.0})
    return agg, m


def main_model(path, p_is_cmp):
    path = Path(path)
    agg, m = load_model(path, p_is_cmp)
    res = fit_agg(agg, lapse_max=0.0, shared_slope=False)  # model: lapse yok, koşul başına eğim
    res.insert(0, "participant", path.stem)
    out = path.with_name(path.stem + "_pse.csv")
    res.to_csv(out, index=False)
    plot(agg, res, path.with_name(path.stem + "_psychometric.png"))
    print(f"Model: {path.name}")
    if "bias" in m:
        print(f"İki görüntü sırası arasındaki en büyük fark (|bias|): {m['bias'].abs().max():.3g}")
    show = res[["condition", "pse", "pse_shift_pct", "censored"]].copy()
    print(show.round(3).to_string(index=False))
    lo, hi = agg["comparison"].min(), agg["comparison"].max()
    cens = res[res["censored"] != ""]
    if len(cens):
        print(f"\n{len(cens)} koşulda PSE ölçüm aralığının ({lo:.2f}–{hi:.2f}) DIŞINDA:")
        for _, r in cens.iterrows():
            yon = ("merkez KÜÇÜK algılanıyor (kayma < -%{:.0f})".format((1 - lo) * 100)
                   if r["censored"].startswith("<")
                   else "merkez BÜYÜK algılanıyor (kayma > +%{:.0f})".format((hi - 1) * 100))
            print(f"  {r['condition']:<10} PSE {r['censored']}  -> {yon}")
        print("Bu koşullarda PSE sayısı yok; yalnızca yön ve alt sınır raporlanabilir.\n"
              "Gerçek PSE için modeli daha geniş bir karşılaştırma aralığında test edin.")
    print(f"\n-> {out}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--model" in sys.argv:
        return main_model(args[0], "--p-karsilastirma" in sys.argv)
    test_mode = "--hepsi" in sys.argv   # deneme için: katılımcı dışlamayı atla
    folder = Path(args[0] if args else "data/sim")
    df = load_folder(folder)
    clean, report = apply_exclusions(df)
    if test_mode and report["excluded"].any():
        print("--hepsi: dışlama ölçütleri yalnızca raporlanıyor, kimse çıkarılmıyor.")
        print(report[report["excluded"]].to_string(index=False), "\n")
        report["excluded"] = False
        clean = df[(df["condition"] != "catch") & ~df["timeout"] & (df["rt"] >= MIN_RT)].copy()
    if clean.empty:
        print("Analize kalan katılımcı yok. Dışlanma nedenleri:")
        print(report.to_string(index=False))
        print("\nDeneme verisiyle test ediyorsanız komutun sonuna --hepsi ekleyin.")
        sys.exit(1)
    per_pp = fit_all(clean)
    grp = group_summary(per_pp)

    per_pp.to_csv(folder / "pse_by_participant.csv", index=False)
    grp.to_csv(folder / "pse_group.csv", index=False)
    report.to_csv(folder / "excluded.csv", index=False)
    agg = (clean.groupby(["condition", "comparison"])["chose_cmp"]
             .agg(k="sum", n="size").reset_index())
    plot(agg, per_pp, folder / "psychometric.png")

    print(f"Katılımcı: {report.shape[0]}, dışlanan: {report['excluded'].sum()}")
    if report["excluded"].any():
        print(report[report["excluded"]].to_string(index=False))
    print("\nGrup PSE kayması (%):")
    print(grp.round(2).to_string(index=False))
    cens = per_pp[per_pp["censored"] != ""]
    if len(cens):
        print("\nUYARI: PSE'si ölçüm aralığının dışında kalan (sansürlü) fit'ler "
              "(grup ortalamasına katılmadı, n_above / n_below sütunlarında sayıldı):")
        print(cens[["participant", "condition", "censored"]].to_string(index=False))
    bad = per_pp[(per_pp["at_bound"] | ~per_pp["converged"]) & (per_pp["censored"] == "")]
    if len(bad):
        print("\nUYARI: sınıra dayanan / yakınsamayan fit'ler:")
        print(bad.to_string(index=False))
    slopes, t = ratio_slopes(per_pp, clean)
    if t is None or len(slopes) < 2:
        print("\nPSE ~ log(oran) eğimi: t-testi için en az 2 katılımcı gerekli.")
    else:
        print(f"\nPSE ~ log(oran) eğimi: ort. {slopes.mean():.2f} %/log-birim, "
              f"t({len(slopes)-1}) = {t.statistic:.2f}, p = {t.pvalue:.2g}")


if __name__ == "__main__":
    main()
