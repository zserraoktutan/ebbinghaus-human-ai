"""2AFC (siyam) eğitim döngüsü.

Görseller train.build_pool ile bir kez üretilir (regresyonla aynı havuz);
her epoch'ta rastgele ikili eşleştirilir. Etiket: 1 = sol merkez daha büyük.
Ayarlar regresyonla aynıdır (train.DEFAULTS, kosinüs azalma).
"""
import csv
import math
import os
import time

import torch
import torch.nn as nn

from model.cnn import TwoAFCNet
from model.train import DEFAULTS, build_pool, get_device, run_name, set_seed


def make_pairs(Y, g, min_diff=0.01):
    """Havuzdaki görselleri rastgele ikili eşleştirir.

    Y: (n,) merkez çapı / D.  g: CPU rastgele sayı üreteci (seed'e bağlı).
    min_diff: |çap farkı| bundan küçük çiftler atılır (D cinsinden).
    Döndürür: sol indeksler, sağ indeksler, etiketler (1.0 = sol büyük).
    """
    y = Y.cpu()
    n = y.numel()
    left = torch.randperm(n, generator=g)
    right = torch.randperm(n, generator=g)

    diff = y[left] - y[right]
    keep = diff.abs() >= min_diff
    left, right, diff = left[keep], right[keep], diff[keep]

    labels = (diff > 0).float()
    return left.to(Y.device), right.to(Y.device), labels.to(Y.device)


def batches(X, left, right, labels, batch_size, drop_last=True):
    """Çiftleri batch'lere böler: (sol görseller, sağ görseller, etiketler)."""
    n = labels.numel()
    stop = n - batch_size + 1 if drop_last else n
    for s in range(0, stop, batch_size):
        i = left[s:s + batch_size]
        j = right[s:s + batch_size]
        yield X[i].float().div_(255), X[j].float().div_(255), labels[s:s + batch_size]


@torch.no_grad()
def evaluate_2afc(model, X, left, right, labels, batch_size=250):
    """Ortalama BCE kaybı ve doğruluk (%)."""
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    tot_loss, correct = 0.0, 0
    for a, b, y in batches(X, left, right, labels, batch_size, drop_last=False):
        logit = model(a, b).float()
        tot_loss += loss_fn(logit, y).item()
        correct += ((logit > 0).float() == y).sum().item()
    n = labels.numel()
    return tot_loss / n, correct / n * 100


def train_2afc(regime, seed, out_dir, log=print, **kw):
    """Bir 2AFC modelini eğitir; en iyi ağırlıkları ve epoch kaydını out_dir'e yazar.
    Döndürür: (en iyi ağırlıklı model, geçmiş listesi)."""
    cfg = {**DEFAULTS, **kw}
    device = get_device()
    use_amp = cfg["amp"] and device.type == "cuda"
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    name = run_name("2afc", regime, seed)
    ckpt = os.path.join(out_dir, name + ".pt")
    log_csv = os.path.join(out_dir, name + "_log.csv")

    log(f"[{name}] cihaz={device.type}  amp={use_amp}  epoch={cfg['epochs']}  "
        f"train={cfg['n_train']}  val={cfg['n_val']}  batch={cfg['batch_size']}  lr={cfg['lr']}")
    Xtr, Ytr = build_pool(regime, "train", cfg["n_train"], seed, cfg["num_workers"], device, log)
    Xva, Yva = build_pool(regime, "val", cfg["n_val"], seed, cfg["num_workers"], device, log)

    # Doğrulama çiftleri BİR KEZ, ayrı bir üreteçle: her epoch aynı çiftlerle ölçülür
    g_val = torch.Generator().manual_seed(seed + 10_000)
    va_left, va_right, va_lab = make_pairs(Yva, g_val)
    g = torch.Generator().manual_seed(seed)      # eğitim çiftleri için

    model = TwoAFCNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = nn.BCEWithLogitsLoss()
    steps = cfg["epochs"] * (cfg["n_train"] // cfg["batch_size"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=cfg["lr_min"])

    best, hist = math.inf, []
    for ep in range(cfg["epochs"]):
        t0 = time.time()
        model.train()
        left, right, labels = make_pairs(Ytr, g)   # her epoch yeni çiftler
        tot, correct, cnt = 0.0, 0, 0
        for a, b, y in batches(Xtr, left, right, labels, cfg["batch_size"]):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logit = model(a, b)
            loss = loss_fn(logit.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * y.numel()
            correct += ((logit.float() > 0).float() == y).sum().item()
            cnt += y.numel()

        val_loss, val_acc = evaluate_2afc(model, Xva, va_left, va_right, va_lab)
        is_best = val_loss < best
        if is_best:
            best = val_loss
            torch.save({"state_dict": model.state_dict(), "epoch": ep + 1, "val_loss": val_loss,
                        "val_acc": val_acc, "regime": regime, "seed": seed, "cfg": cfg}, ckpt)
        row = dict(epoch=ep + 1, lr=opt.param_groups[0]["lr"], train_loss=tot / cnt,
                   train_acc=correct / cnt * 100, val_loss=val_loss, val_acc=val_acc,
                   best=int(is_best), sec=round(time.time() - t0, 1))
        hist.append(row)
        log(f"  epoch {ep + 1:2d}/{cfg['epochs']}  lr {row['lr']:.1e}  "
            f"train kayıp {row['train_loss']:.4f} (%{row['train_acc']:.1f})  "
            f"val kayıp {val_loss:.4f} (%{val_acc:.1f})  ({row['sec']} s){'  *' if is_best else ''}")
        with open(log_csv, "w", newline="") as f:    # her epoch'ta yaz: oturum koparsa kayıt kalsın
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerows(hist)

    model.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])
    model.eval()
    del Xtr, Ytr, Xva, Yva
    if device.type == "cuda":
        torch.cuda.empty_cache()
    log(f"[{name}] en iyi val kayıp {best:.4f} → {ckpt}")
    return model, hist


def load_2afc(path, device=None):
    """Kaydedilmiş bir 2AFC modelini yükler."""
    device = device or get_device()
    ck = torch.load(path, map_location=device)
    model = TwoAFCNet().to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck