import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])
ana = df[df["trials.thisN"].notna()]          # alıştırmayı dışarıda bırak

print("Ana deneme sayısı:", len(ana))          # 252 olmalı (84 x 3)
print("Cevapsız:", ana["resp.keys"].isna().sum())

catch = ana[ana["condition"] == "catch"]
print(f"Dikkat denemeleri: {int(catch['resp.corr'].sum())}/{len(catch)} doğru")

print(f"Ortalama tepki süresi: {ana['resp.rt'].mean():.2f} s")