import pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

# ---------- порог, чтобы серия считалась «хорошей»
SMAPE_THRESHOLD = 30    # %
MAE_THRESHOLD   = 1     # шт.

# ---------- пути к файлам
root = Path(r"E:\projects_PY\N-Beats")
metrics_path = root / "result" / "series_metrics.csv"
preds_path   = root / "result" / "predictions_horizons.csv"

fig_dir = root / "figures" / "good_ru"
fig_dir.mkdir(parents=True, exist_ok=True)

# ---------- загрузка данных
metrics = pd.read_csv(metrics_path)
preds   = pd.read_csv(preds_path)

# ---------- фильтруем «хорошие» позиции
mask = (metrics["SMAPE"] < SMAPE_THRESHOLD) & (metrics["MAE"] < MAE_THRESHOLD)
metrics_g = metrics[mask].copy()
preds_g   = preds.merge(metrics_g[["series_id"]], on="series_id")

print(f"Отобрано {len(metrics_g)} товаров из {len(metrics)} "
      f"({len(metrics_g)/len(metrics):.1%})")

# ---------- 1. гистограммы MAE, SMAPE, WAPE
for col, rus in [("MAE", "MAE"),
                 ("SMAPE", "SMAPE, %"),
                 ("WAPE", "WAPE")]:
    plt.figure()
    plt.hist(metrics_g[col].dropna(), bins=30)
    plt.xlabel(rus)
    plt.ylabel("Количество товаров")
    plt.title(f"Гистограмма {rus} ")
    plt.tight_layout()
    plt.savefig(fig_dir / f"hist_{col.lower()}_good.png")
    plt.close()

# ---------- 2. box‑plot трёх метрик
plt.figure()
plt.boxplot([metrics_g["MAE"], metrics_g["SMAPE"], metrics_g["WAPE"]],
            vert=False, labels=["MAE", "SMAPE, %", "WAPE"])
plt.title("Box‑plot MAE / SMAPE / WAPE ")
plt.tight_layout()
plt.savefig(fig_dir / "boxplot_metrics_good.png")
plt.close()

# ---------- 3. scatter MAE vs SMAPE
plt.figure()
plt.scatter(metrics_g["MAE"], metrics_g["SMAPE"], s=10)
plt.xlabel("MAE, шт.")
plt.ylabel("SMAPE, %")
plt.title("MAE vs SMAPE ")
plt.tight_layout()
plt.savefig(fig_dir / "scatter_mae_smape_good.png")
plt.close()

# ---------- 4. Lorenz‑кривая SMAPE
sorted_s = np.sort(metrics_g["SMAPE"].values)
cum = np.arange(1, len(sorted_s)+1) / len(sorted_s)
plt.figure()
plt.plot(sorted_s, cum)
plt.xlabel("Порог SMAPE, %")
plt.ylabel("Накопительная доля серий")
plt.title("Кривая Лоренца для SMAPE ")
plt.tight_layout()
plt.savefig(fig_dir / "lorenz_smape_good.png")
plt.close()

# ---------- 5. гистограммы прогнозов 1 неделя / 6 мес
for col, rus in [("pred_1w", "Прогноз за 1 неделю, шт."),
                 ("pred_6m", "Прогноз за 6 месяцев, шт.")]:
    plt.figure()
    plt.hist(preds_g[col].dropna(), bins=30)
    plt.xlabel(rus)
    plt.ylabel("Количество товаров")
    plt.title(f"Гистограмма {rus.lower()} ")
    plt.tight_layout()
    plt.savefig(fig_dir / f"hist_{col}_good.png")
    plt.close()

# ---------- 6. доля объёма по горизонтам
tot = preds_g[["pred_1w","pred_3w","pred_3m","pred_6m"]].sum()
share = tot / tot["pred_6m"]
plt.figure()
plt.bar(share.index, share.values)
plt.ylabel("Доля от объёма за 6 месяцев")
plt.title("Распределение объёма продаж по горизонтам ")
plt.tight_layout()
plt.savefig(fig_dir / "stacked_volume_share_good.png")
plt.close()

# ---------- 7. scatter 6‑мес. объём vs SMAPE
merged = preds_g.merge(metrics_g[["series_id","SMAPE","MAE"]], on="series_id")
plt.figure()
plt.scatter(merged["pred_6m"], merged["SMAPE"],
            s=5 + merged["MAE"]*5)
plt.xlabel("Прогноз на 6 мес., шт.")
plt.ylabel("SMAPE, %")
plt.title("6‑мес. объём vs SMAPE (размер = MAE)\n")
plt.tight_layout()
plt.savefig(fig_dir / "scatter_6m_smape_good.png")
plt.close()

# ---------- 8. Pareto для топ‑30
N = 30
top = preds_g.nlargest(N, "pred_6m").sort_values("pred_6m", ascending=False)
cum = top["pred_6m"].cumsum() / top["pred_6m"].sum()*100
fig, ax1 = plt.subplots()
ax1.bar(range(N), top["pred_6m"])
ax1.set_xlabel("Ранг товара")
ax1.set_ylabel("Продажи за 6 мес., шт.")
ax2 = ax1.twinx()
ax2.plot(range(N), cum, marker="o")
ax2.set_ylabel("Накопительная доля, %")
plt.title(f"Диаграмма Парето топ‑{N} ")
fig.tight_layout()
plt.savefig(fig_dir / "pareto_top30_6m_good.png")
plt.close()

print("Графики сохранены в:", fig_dir)
