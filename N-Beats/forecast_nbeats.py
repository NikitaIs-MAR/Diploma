# forecast_nbeats.py — darts 0.35.0
import logging, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from darts import TimeSeries
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mae, smape  # WAPE и Pinball — вручную

# ---------- 1. константы
INPUT_CHUNK, OUTPUT_CHUNK = 52, 13
VAL_LEN, MIN_TOTAL, FREQ = 65, 130, "W-MON"

CSV_PATH = Path(r"E:\projects_PY\ready_for_training\augmented_ready.csv")

ROOT       = Path(__file__).resolve().parent
MODEL_PKL  = ROOT / "model" / "nbeats_best.pkl"
RESULT_DIR = ROOT / "result"; RESULT_DIR.mkdir(exist_ok=True)
OUT_CSV    = RESULT_DIR / "predictions_horizons.csv"
MET_CSV    = RESULT_DIR / "series_metrics.csv"

# горизонты (в неделях) → названия столбцов
HORIZONS = {1: "pred_1w", 3: "pred_3w", 13: "pred_3m", 26: "pred_6m"}
FUT_HORIZON = max(HORIZONS.keys())          # 26 недель
FUT_COVARIATE_HORIZON = max(0, FUT_HORIZON - OUTPUT_CHUNK)

# ---------- 2. сервис
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# ---------- 3. загрузка данных
df = (
    pd.read_csv(CSV_PATH, parse_dates=["Месяц"], dtype={"Количество": np.float32})
      .sort_values(["series_id", "Месяц"])
)

# ---------- 4. загрузка модели
model = NBEATSModel.load(str(MODEL_PKL))
logger.info("Модель загружена из %s", MODEL_PKL)

pred_rows, met_rows = [], []
freq_offset = to_offset(FREQ)   # W‑MON → DateOffset(weeks=1)

for sid, g in df.groupby("series_id", sort=False):
    # --- TimeSeries + фильтр длины
    ts = TimeSeries.from_dataframe(
        g, time_col="Месяц", value_cols="Количество",
        freq=FREQ, fill_missing_dates=False
    )
    if len(ts) < MIN_TOTAL:
        continue

    # --- train / val split
    tr, va = ts[:-VAL_LEN], ts[-VAL_LEN:]

    sc = Scaler()
    tr_n, va_n = sc.fit_transform(tr), sc.transform(va)

    # --- полный индекс ковариат
    val_end = va.end_time()
    if FUT_COVARIATE_HORIZON > 0:
        fut_idx = pd.date_range(
            start=val_end + freq_offset,
            periods=FUT_COVARIATE_HORIZON,
            freq=FREQ
        )
        full_idx = tr.time_index.union(va.time_index).union(fut_idx)
    else:
        full_idx = tr.time_index.union(va.time_index)

    m_full = full_idx.month
    cov_full = TimeSeries.from_times_and_values(
        full_idx,
        np.column_stack([
            np.sin(2 * np.pi * m_full / 12),
            np.cos(2 * np.pi * m_full / 12),
        ]).astype(np.float32),
        ["month_sin", "month_cos"]
    )

    cov_ho  = cov_full.slice(tr.start_time(), va.end_time())
    cov_fut = cov_full.slice(tr.start_time(), full_idx[-1])

    # ---------- hold‑out метрики
    pr_val = model.predict(
        len(va_n), series=tr_n, past_covariates=cov_ho, show_warnings=False
    )
    y_true, y_pred = sc.inverse_transform(va_n), sc.inverse_transform(pr_val)

    err = np.abs(y_pred.values() - y_true.values())
    wape_val = err.sum() / np.abs(y_true.values()).sum()

    e = y_true.values() - y_pred.values()
    pinb_val = np.mean(np.maximum(0.5 * e, (0.5 - 1) * e))

    met_rows.append({
        "series_id":    sid,
        "MAE":          round(mae(y_true, y_pred), 3),
        "SMAPE":        round(smape(y_true, y_pred), 3),
        "WAPE":         round(wape_val, 3),
        "Pinball_q0.5": round(pinb_val, 3),
    })

    # ---------- прогноз на 26 недель и накопление
    history = TimeSeries.concatenate(tr_n, va_n)
    pr_fut = model.predict(
        FUT_HORIZON, series=history, past_covariates=cov_fut, show_warnings=False
    )
    fut_pred = sc.inverse_transform(pr_fut)

    vals = fut_pred.values().flatten()   # помесячные (26)
    cums = np.cumsum(vals)               # накопленные

    row = {"series_id": sid,
           "pred_1w": round(float(cums[0]), 3),
           "pred_3w": round(float(cums[2]), 3),
           "pred_3m": round(float(cums[12]), 3),
           "pred_6m": round(float(cums[25]), 3)}
    pred_rows.append(row)

# ---------- 5. сохранение
pd.DataFrame(pred_rows).to_csv(OUT_CSV, index=False)
pd.DataFrame(met_rows ).to_csv(MET_CSV, index=False)
logger.info("Прогнозы → %s", OUT_CSV)
logger.info("Метрики  → %s", MET_CSV)
