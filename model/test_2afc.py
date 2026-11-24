"""2AFC modeli için iki sıra testi.

Her (koşul, karşılaştırma) hücresi için bağlamlı merkez (D) ve tek daire (D × c)
iki sırayla verilir:
  sıra 1: (bağlamlı, tek)  → p1 = P(bağlamlı büyük)
  sıra 2: (tek, bağlamlı)  → p2 = 1 − P(tek büyük)
p = (p1 + p2) / 2 sıra yanlılığından arındırılmış olasılık; bias = p1 − p2.
"""
import numpy as np
import torch

import stimuli as st
from model import data



@torch.no_grad()
def order_test(model, n_per_cell=50, seed=2026, comparisons=None):
    """comparisons: test edilecek karşılaştırma boyutları (D cinsinden).
    Verilmezse insan deneyiyle aynı 5 düzey (stimuli.COMPARISON) kullanılır."""
    cs = st.COMPARISON if comparisons is None else comparisons
    device = next(model.parameters()).device
    model.eval()
    rng = np.random.default_rng(seed)
    D, s = data.D, data.MAX_SHIFT
    rows = []
    for name, r, g in st.condition_list():
        for c in cs:
            ctx, cmp = [], []
            for _ in range(n_per_cell):
                rot = rng.uniform(0, 2 * np.pi)
                sh1 = tuple(rng.uniform(-s, s, size=2))
                sh2 = tuple(rng.uniform(-s, s, size=2))
                ctx.append(data.to_array(st.draw_config(data.CANVAS, D, ratio=r, gap=g, ref_d=D,
                                                        rotation=rot, shift=sh1))[None])
                cmp.append(data.to_array(st.draw_config(data.CANVAS, D * c, ratio=None,
                                                        shift=sh2))[None])
            a = torch.from_numpy(np.stack(ctx)).to(device)
            b = torch.from_numpy(np.stack(cmp)).to(device)

            p1 = torch.sigmoid(model(a, b)).mean().item()      # bağlamlı solda
            p2 = 1 - torch.sigmoid(model(b, a)).mean().item()  # bağlamlı sağda
            rows.append(dict(condition=name, comparison=float(c),
                             p1=p1, p2=p2, p=(p1 + p2) / 2, bias=p1 - p2))
    return rows