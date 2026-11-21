"""Regresyon eğitim döngüsü (plan §5: Adam 1e-3, batch 64, 10 epoch,
epoch başına 20.000 görsel, doğrulama 2.000, en iyi doğrulama kaybındaki ağırlık).

Kullanım (Colab):
    from model.train import train_regression
    model, hist = train_regression("A", seed=0, out_dir=SONUC)
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

DEFAULTS = dict(epochs=10, n_train=20_000, n_val=2_000, batch_size=64, lr=1e-3, num_workers=2)


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


def _loaders(regime, seed, n_train, n_val, batch_size, num_workers, device):
    train_ds = data.EbbinghausDataset(regime, "train", n=n_train, seed=seed)
    val_ds = data.EbbinghausDataset(regime, "val", n=n_val, seed=seed)
    g = torch.Generator().manual_seed(seed)
    common = dict(num_workers=num_workers, pin_memory=(device.type == "cuda"))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                          generator=g, **common)
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False, **common)
    return train_ds, train_dl, val_dl


@torch.no_grad()
def evaluate(model, loader, device):
    """Ortalama MSE ve ortalama mutlak yüzde hata (|tahmin − y| / y · 100)."""
    model.eval()
    se, ape, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        p = model(x)
        se += ((p - y) ** 2).sum().item()
        ape += ((p - y).abs() / y).sum().item() * 100
        n += y.numel()
    return se / n, ape / n


def train_regression(regime, seed, out_dir, log=print, **kw):
    """Bir regresyon modelini eğitir; en iyi ağırlıkları ve epoch kaydını out_dir'e yazar.
    Döndürür: (en iyi ağırlıklı model, geçmiş listesi)."""
    cfg = {**DEFAULTS, **kw}
    device = get_device()
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    name = run_name("regresyon", regime, seed)
    ckpt = os.path.join(out_dir, name + ".pt")
    log_csv = os.path.join(out_dir, name + "_log.csv")

    model = RegressionNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.MSELoss()
    train_ds, train_dl, val_dl = _loaders(regime, seed, cfg["n_train"], cfg["n_val"],
                                          cfg["batch_size"], cfg["num_workers"], device)

    log(f"[{name}] cihaz={device.type}  epoch={cfg['epochs']}  "
        f"train={cfg['n_train']}  val={cfg['n_val']}  batch={cfg['batch_size']}  lr={cfg['lr']}")
    best, hist = math.inf, []
    for ep in range(cfg["epochs"]):
        t0 = time.time()
        train_ds.set_epoch(ep)                        # her epoch yeni görseller
        model.train()
        tot, cnt = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * y.numel()
            cnt += y.numel()
        val_mse, val_mape = evaluate(model, val_dl, device)
        is_best = val_mse < best
        if is_best:
            best = val_mse
            torch.save({"state_dict": model.state_dict(), "epoch": ep, "val_mse": val_mse,
                        "regime": regime, "seed": seed, "cfg": cfg}, ckpt)
        row = dict(epoch=ep + 1, train_mse=tot / cnt, val_mse=val_mse, val_mape=val_mape,
                   best=int(is_best), sec=round(time.time() - t0, 1))
        hist.append(row)
        log(f"  epoch {ep + 1:2d}/{cfg['epochs']}  train MSE {row['train_mse']:.5f}  "
            f"val MSE {val_mse:.5f}  val hata %{val_mape:.2f}  ({row['sec']} s){'  *' if is_best else ''}")
        with open(log_csv, "w", newline="") as f:     # her epoch'ta yaz: oturum koparsa kayıt kalsın
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerows(hist)

    model.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])
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
        p = model(x).cpu().numpy()
        per_size[c] = dict(mean_pred=float(p.mean()), mape=float(np.mean(np.abs(p - c) / c) * 100))
    overall = float(np.mean([v["mape"] for v in per_size.values()]))
    return overall, overall < threshold, per_size
