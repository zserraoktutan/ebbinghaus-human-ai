"""Regresyon eğitim döngüsü (plan §5: Adam 1e-3, batch 64, 10 epoch,
20.000 eğitim görseli, 2.000 doğrulama görseli, en iyi doğrulama kaybındaki ağırlık).

Öğrenme oranı: 1e-3'ten 1e-5'e kosinüs azalma (adım başına). İlk deneme eğitiminde
sabit 1e-3 ile doğrulama hatası epoch'tan epoch'a %0,5 ile %3,8 arasında sıçradı;
azalan öğrenme oranı son epoch'ları kararlı hale getirir ve "en iyi epoch" seçiminin
şansa bağlı olmasını önler.

Hız notu: Görsel üretimi (4x çizim + LANCZOS küçültme) Colab CPU'sunda yavaştır
(~130 s / 20.000 görsel). Bu yüzden her eğitimin başında görseller BİR KEZ üretilip
uint8 olarak GPU belleğinde tutulur (~1 GB) ve 10 epoch boyunca bu havuz karıştırılarak
kullanılır. Böylece bir eğitim ~22 dk yerine ~4 dk sürer.

Kullanım (Colab):
    from model.train import train_regression
    net, hist = train_regression("A", seed=0, out_dir=SONUC)
"""
import csv
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import stimuli as st
from model import data
from model.cnn import RegressionNet

DEFAULTS = dict(epochs=10, n_train=20_000, n_val=2_000, batch_size=64, lr=1e-3,
                lr_min=1e-5, num_workers=2, amp=True)


def set_seed(seed):
    """Ağırlık başlatma ve karıştırma sırası seed'e bağlı. (Görseller zaten
    EbbinghausDataset içinde seed'e bağlı üretiliyor.)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_name(task, regime, seed):
    return f"model_{task}_{regime}_seed{seed}"


# ---------------------------------------------------------------- veri havuzu
def build_pool(regime, split, n, seed, num_workers=2, device="cpu", log=print):
    """Görselleri bir kez üretir. Döndürür: X (n x 1 x 224 x 224, uint8), Y (n, float32).
    Görseller 0..255 basamaklı olduğu için uint8'e çevirme kayıpsızdır."""
    ds = data.EbbinghausDataset(regime, split, n=n, seed=seed)
    dl = DataLoader(ds, batch_size=250, shuffle=False, num_workers=num_workers)
    X = torch.empty((n, 1, data.CANVAS, data.CANVAS), dtype=torch.uint8)
    Y = torch.empty(n, dtype=torch.float32)
    t0, i = time.time(), 0
    for x, y in dl:
        b = y.numel()
        X[i:i + b] = (x * 255).round_().to(torch.uint8)
        Y[i:i + b] = y
        i += b
    log(f"  {split:5s} havuzu: {n} görsel, {time.time() - t0:.0f} s")
    return X.to(device), Y.to(device)


def iterate(X, Y, batch_size, perm=None):
    """Havuzdan batch'ler: uint8 → float (daire 1, zemin 0)."""
    n = Y.numel()
    idx = perm if perm is not None else torch.arange(n, device=Y.device)
    for s in range(0, n - batch_size + 1 if perm is not None else n, batch_size):
        j = idx[s:s + batch_size]
        yield X[j].float().div_(255), Y[j]


@torch.no_grad()
def evaluate(model, X, Y, batch_size=500):
    """Ortalama MSE ve ortalama mutlak yüzde hata (|tahmin − y| / y · 100)."""
    model.eval()
    se, ape = 0.0, 0.0
    for x, y in iterate(X, Y, batch_size):
        p = model(x).float()
        se += ((p - y) ** 2).sum().item()
        ape += ((p - y).abs() / y).sum().item() * 100
    return se / Y.numel(), ape / Y.numel()


# ---------------------------------------------------------------- eğitim
def train_regression(regime, seed, out_dir, log=print, **kw):
    """Bir regresyon modelini eğitir; en iyi ağırlıkları ve epoch kaydını out_dir'e yazar.
    Döndürür: (en iyi ağırlıklı model, geçmiş listesi)."""
    cfg = {**DEFAULTS, **kw}
    device = get_device()
    use_amp = cfg["amp"] and device.type == "cuda"
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    name = run_name("regresyon", regime, seed)
    ckpt = os.path.join(out_dir, name + ".pt")
    log_csv = os.path.join(out_dir, name + "_log.csv")

    log(f"[{name}] cihaz={device.type}  amp={use_amp}  epoch={cfg['epochs']}  "
        f"train={cfg['n_train']}  val={cfg['n_val']}  batch={cfg['batch_size']}  lr={cfg['lr']}")
    Xtr, Ytr = build_pool(regime, "train", cfg["n_train"], seed, cfg["num_workers"], device, log)
    Xva, Yva = build_pool(regime, "val", cfg["n_val"], seed, cfg["num_workers"], device, log)

    model = RegressionNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = nn.MSELoss()
    g = torch.Generator(device="cpu").manual_seed(seed)
    steps = cfg["epochs"] * (cfg["n_train"] // cfg["batch_size"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=cfg["lr_min"])

    best, hist = math.inf, []
    for ep in range(cfg["epochs"]):
        t0 = time.time()
        model.train()
        perm = torch.randperm(cfg["n_train"], generator=g).to(device)
        tot, cnt = 0.0, 0
        for x, y in iterate(Xtr, Ytr, cfg["batch_size"], perm):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                p = model(x)
            loss = loss_fn(p.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * y.numel()
            cnt += y.numel()
        val_mse, val_mape = evaluate(model, Xva, Yva)
        is_best = val_mse < best
        if is_best:
            best = val_mse
            torch.save({"state_dict": model.state_dict(), "epoch": ep + 1, "val_mse": val_mse,
                        "regime": regime, "seed": seed, "cfg": cfg}, ckpt)
        row = dict(epoch=ep + 1, lr=opt.param_groups[0]["lr"], train_mse=tot / cnt,
                   val_mse=val_mse, val_mape=val_mape, best=int(is_best),
                   sec=round(time.time() - t0, 1))
        hist.append(row)
        log(f"  epoch {ep + 1:2d}/{cfg['epochs']}  lr {row['lr']:.1e}  train MSE {row['train_mse']:.5f}  "
            f"val MSE {val_mse:.5f}  val hata %{val_mape:.2f}  ({row['sec']} s){'  *' if is_best else ''}")
        with open(log_csv, "w", newline="") as f:     # her epoch'ta yaz: oturum koparsa kayıt kalsın
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerows(hist)

    model.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])
    model.eval()
    del Xtr, Ytr, Xva, Yva
    if device.type == "cuda":
        torch.cuda.empty_cache()
    log(f"[{name}] en iyi val MSE {best:.5f} → {ckpt}")
    return model, hist


def load_regression(path, device=None):
    device = device or get_device()
    ck = torch.load(path, map_location=device)
    model = RegressionNet().to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


@torch.no_grad()
def single_circle_check(model, n_per_size=100, seed=999, threshold=3.0):
    """Plan §5 doğrulama testi: tek dairede ortalama mutlak hata < %3.
    Görseller test ile aynı yoldan (stimuli.draw_config) ve insanla aynı
    karşılaştırma boyutlarında (D × 0.84 … 1.16) üretilir."""
    device = next(model.parameters()).device
    model.eval()
    rng = np.random.default_rng(seed)
    per_size = {}
    for c in st.COMPARISON:
        xs = []
        for _ in range(n_per_size):
            shift = tuple(rng.uniform(-data.MAX_SHIFT, data.MAX_SHIFT, size=2))
            img = st.draw_config(canvas=data.CANVAS, center_d=c * data.D, ratio=None, shift=shift)
            xs.append(data.to_array(img)[None])
        x = torch.from_numpy(np.stack(xs)).to(device)
        p = model(x).float().cpu().numpy()
        per_size[c] = dict(mean_pred=float(p.mean()), mape=float(np.mean(np.abs(p - c) / c) * 100))
    overall = float(np.mean([v["mape"] for v in per_size.values()]))
    return overall, overall < threshold, per_size
