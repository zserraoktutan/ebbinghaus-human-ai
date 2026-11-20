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
    # PsychoPy alıştırma/talimat satırlarında condition boş olur: at
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
        to = d["timeout"].mean()
        reasons = []
        if len(catch) and acc < MIN_CATCH_ACC:
            reasons.append(f"dikkat doğruluğu %{acc*100:.0f}")
        if to > MAX_TIMEOUT:
            reasons.append(f"zaman aşımı %{to*100:.0f}")
        report.append({"participant": pid, "catch_acc": acc, "timeout_rate": to,
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


def fit_participant(d):
    """Tek katılımcı için ortak eğimli fit. Koşul başına PSE döndürür."""
    conds = sorted(d["condition"].unique())
    idx = d["condition"].map({c: i for i, c in enumerate(conds)}).to_numpy()
    logc = np.log(d["comparison"].to_numpy(float))
    y = d["chose_cmp"].to_numpy(float)
    K = len(conds)

    def nll(theta):
        logpse = theta[:K][idx]
        s = np.exp(theta[K])
        lapse = theta[K + 1]
        p = np.clip(_p(logc, logpse, s, lapse), 1e-9, 1 - 1e-9)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    x0 = np.r_[np.zeros(K), np.log(0.05), 0.02]
    bounds = [(np.log(0.5), np.log(2.0))] * K + [(np.log(0.005), np.log(0.5)), (0.0, 0.10)]
    res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
    pse = np.exp(res.x[:K])
    out = pd.DataFrame({"condition": conds, "pse": pse,
                        "pse_shift_pct": (pse - 1) * 100,
                        "n_trials": d.groupby("condition").size().reindex(conds).to_numpy()})
    out["s"] = np.exp(res.x[K])
    out["lapse"] = res.x[K + 1]
    out["converged"] = res.success
    # Sınıra dayanan PSE güvenilmez: test aralığının (0.84–1.16) çok dışında
    out["at_bound"] = (pse < 0.80) | (pse > 1.25)
    return out


def fit_all(clean):
    parts = []
    for pid, d in clean.groupby("participant"):
        f = fit_participant(d)
        f.insert(0, "participant", pid)
        parts.append(f)
    return pd.concat(parts, ignore_index=True)


def group_summary(per_pp):
    g = per_pp.groupby("condition")["pse_shift_pct"]
    out = pd.DataFrame({"n": g.size(), "mean_shift_pct": g.mean(), "sd": g.std()})
    out["sem"] = out["sd"] / np.sqrt(out["n"])
    out["ci95_low"] = out["mean_shift_pct"] - 1.96 * out["sem"]
    out["ci95_high"] = out["mean_shift_pct"] + 1.96 * out["sem"]
    return out.reset_index()


def ratio_slopes(per_pp, clean):
    """Katılımcı başına PSE kayması ~ log(oran) eğimi (yakın koşullar) + t-testi."""
    from scipy.stats import ttest_1samp
    info = clean.drop_duplicates("condition")[["condition", "ratio", "far"]]
    m = per_pp.merge(info, on="condition")
    m = m[(m["far"].astype(str) != "True") & m["ratio"].notna() & (m["ratio"].astype(str) != "")]
    m["log_ratio"] = np.log(m["ratio"].astype(float))
    slopes = m.groupby("participant").apply(
        lambda x: np.polyfit(x["log_ratio"], x["pse_shift_pct"], 1)[0], include_groups=False)
    t = ttest_1samp(slopes, 0)
    return slopes, t


def plot(clean, per_pp, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conds = sorted(clean["condition"].unique())
    fig, axes = plt.subplots(2, (len(conds) + 1) // 2, figsize=(14, 6), sharey=True)
    xs = np.linspace(0.82, 1.18, 200)
    for ax, c in zip(axes.flat, conds):
        d = clean[clean["condition"] == c]
        prop = d.groupby("comparison")["chose_cmp"].mean()
        ax.plot(prop.index, prop.values, "o", color="#333")
        f = per_pp[per_pp["condition"] == c]
        pse = np.exp(np.log(f["pse"]).mean())
        s, lapse = f["s"].median(), f["lapse"].median()
        ax.plot(xs, _p(np.log(xs), np.log(pse), s, lapse), color="#2a6fdb")
        ax.axvline(pse, ls="--", color="#2a6fdb", lw=1)
        ax.axhline(0.5, color="#aaa", lw=0.8)
        ax.set_title(f"{c}  (PSE {pse:.3f})", fontsize=10)
    for ax in axes.flat[len(conds):]:
        ax.axis("off")
    fig.supxlabel("karşılaştırma / D")
    fig.supylabel("P(karşılaştırma daha büyük)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "data/sim")
    df = load_folder(folder)
    clean, report = apply_exclusions(df)
    per_pp = fit_all(clean)
    grp = group_summary(per_pp)

    per_pp.to_csv(folder / "pse_by_participant.csv", index=False)
    grp.to_csv(folder / "pse_group.csv", index=False)
    report.to_csv(folder / "excluded.csv", index=False)
    plot(clean, per_pp, folder / "psychometric.png")

    print(f"Katılımcı: {report.shape[0]}, dışlanan: {report['excluded'].sum()}")
    if report["excluded"].any():
        print(report[report["excluded"]].to_string(index=False))
    print("\nGrup PSE kayması (%):")
    print(grp.round(2).to_string(index=False))
    bad = per_pp[per_pp["at_bound"] | ~per_pp["converged"]]
    if len(bad):
        print("\nUYARI: sınıra dayanan / yakınsamayan fit'ler:")
        print(bad.to_string(index=False))
    slopes, t = ratio_slopes(per_pp, clean)
    print(f"\nPSE ~ log(oran) eğimi: ort. {slopes.mean():.2f} %/log-birim, "
          f"t({len(slopes)-1}) = {t.statistic:.2f}, p = {t.pvalue:.2g}")


if __name__ == "__main__":
    main()
