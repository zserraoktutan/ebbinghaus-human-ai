"""Kendi CNN'imiz: ortak gövde + regresyon kafası (+ 2AFC için siyam kafa).

Gövde: 224x224 gri girdi → 5 blok [Conv3x3 → BN → ReLU → MaxPool2]
       kanallar 32-64-128-128-256 → 7x7x256 → global ortalama havuzlama → 256.
Alıcı alan (son blok): 94 px.  Parametre: ≈ 0,55 M (regresyon modeli).
"""
import torch
import torch.nn as nn

CHANNELS = (32, 64, 128, 128, 256)


def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),  # BN varken bias gereksiz
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class Encoder(nn.Module):
    """Ortak gövde. Regresyon ve 2AFC modelleri aynı gövdeyi kullanır."""

    def __init__(self, channels=CHANNELS):
        super().__init__()
        blocks, c_in = [], 1
        for c in channels:
            blocks.append(conv_block(c_in, c))
            c_in = c
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = c_in

    def forward(self, x):                     # x: B x 1 x 224 x 224
        return self.pool(self.features(x)).flatten(1)   # B x out_dim


class RegressionNet(nn.Module):
    """Tek görüntü → merkez çapı / D."""

    def __init__(self, channels=CHANNELS, hidden=64):
        super().__init__()
        self.encoder = Encoder(channels)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.out_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        nn.init.constant_(self.head[-1].bias, 1.0)    # etiketlerin ortalaması 1.0

    def forward(self, x):
        return self.head(self.encoder(x)).squeeze(1)  # B


class TwoAFCNet(nn.Module):
    """Siyam 2AFC: iki görüntü aynı gövdeden geçer → "sol daha büyük" logit'i.
    (Eğitim döngüsü sonraki pencerede yazılacak.)"""

    def __init__(self, channels=CHANNELS, hidden=64):
        super().__init__()
        self.encoder = Encoder(channels)
        d = self.encoder.out_dim
        self.head = nn.Sequential(nn.Linear(2 * d, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1))

    def forward(self, left, right):
        a, b = self.encoder(left), self.encoder(right)
        return self.head(torch.cat([a, b], dim=1)).squeeze(1)   # logit; sigmoid → P(sol büyük)


# ---------------------------------------------------------------- yardımcılar
def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def receptive_field(n_blocks=len(CHANNELS)):
    """Kuramsal alıcı alan: her blok conv3 (s=1) + maxpool2 (s=2)."""
    rf, jump = 1, 1
    for _ in range(n_blocks):
        rf += 2 * jump          # conv 3x3
        rf += 1 * jump          # maxpool 2x2
        jump *= 2
    return rf


def receptive_field_empirical(encoder, size=224, n=16, seed=0):
    """Son blok haritasının ortadaki birimi girdide hangi piksellere duyarlı?
    n rastgele girdide gradyanı sıfır olmayan piksellerin birleşiminin genişliği.
    (MaxPool her pencerede tek piksele gradyan geçirdiği için tek girdi yetmez.)
    Sonuç kuramsal değeri (94) aşmamalı ve ona yakın olmalı."""
    was_training = encoder.training
    encoder.eval()
    p = next(encoder.parameters())
    gen = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 1, size, size, generator=gen).to(p.device).requires_grad_(True)
    fmap = encoder.features(x)                      # n x C x h x w
    h, w = fmap.shape[-2:]
    fmap[:, :, h // 2, w // 2].sum().backward()
    g = (x.grad[:, 0] != 0).any(0)                  # size x size
    rows = torch.nonzero(g.any(1)).flatten()
    cols = torch.nonzero(g.any(0)).flatten()
    encoder.train(was_training)
    return int(rows.max() - rows.min() + 1), int(cols.max() - cols.min() + 1)
