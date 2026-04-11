"""
data_loader.py — Load, clean, aggregate, and prepare sequences from EV.csv.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader

from config import (
    DATA_PATH, LOOKBACK, HORIZON, TARGET,
    WEATHER_FEATURES, TRAIN_RATIO, VAL_RATIO,
    TEST_START_DATE, RANDOM_SEED, BATCH_SIZE
)

np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Raw load & clean
# ─────────────────────────────────────────────────────────────────────────────
def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=["Location Name"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Daily aggregation & feature engineering
# ─────────────────────────────────────────────────────────────────────────────
def build_daily(df: pd.DataFrame) -> pd.DataFrame:
    # Numeric columns for EV side
    ev_agg = (
        df.groupby("Date")
        .agg(
            energy_sum    = (TARGET,                     "sum"),
            charge_dur    = ("Charge Duration (min)",    "mean"),
            connected_dur = ("Connected Duration (min)", "mean"),
            n_sessions    = (TARGET,                     "count"),
            station_id_enc= ("Station ID",               lambda x: x.nunique()),
        )
        .reset_index()
    )

    # Weather: take the mean across stations on each date
    wdf = df[["Date"] + WEATHER_FEATURES].groupby("Date").mean().reset_index()

    daily = ev_agg.merge(wdf, on="Date", how="inner").sort_values("Date").reset_index(drop=True)

    # ── Temporal encodings (4 features) ──────────────────────────────────────
    daily["day_of_week"]  = daily["Date"].dt.dayofweek
    daily["month"]        = daily["Date"].dt.month
    daily["week_of_year"] = daily["Date"].dt.isocalendar().week.astype(int)
    daily["is_weekend"]   = (daily["day_of_week"] >= 5).astype(int)

    return daily


# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature matrix
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "energy_sum",        # target also used as lag feature
    "charge_dur",
    "connected_dur",
    "n_sessions",
    "station_id_enc",
    "day_of_week",
    "month",
    "week_of_year",
    "is_weekend",
    "tmpf",
    "relh",
    "feel",
    "sped",
    "p01m",
    "snowdepth",
]  # 15 features — matches paper Table I / F=15


def prepare_arrays(daily: pd.DataFrame):
    """Return (X_all, y_all, dates, scaler_y)."""
    X_raw = daily[FEATURE_COLS].values.astype(np.float32)
    y_raw = daily["energy_sum"].values.astype(np.float32)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X_raw)
    y_scaled = scaler_y.fit_transform(y_raw.reshape(-1, 1)).ravel()

    X_seq, y_seq, date_seq = [], [], []
    for i in range(LOOKBACK, len(X_scaled) - HORIZON + 1):
        X_seq.append(X_scaled[i - LOOKBACK: i])
        y_seq.append(y_raw[i: i + HORIZON])   # keep target in original scale
        date_seq.append(daily["Date"].iloc[i])

    return (
        np.array(X_seq, dtype=np.float32),
        np.array(y_seq, dtype=np.float32),
        np.array(date_seq),
        scaler_X,
        scaler_y,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Chronological split
# ─────────────────────────────────────────────────────────────────────────────
def split_data(X, y, dates):
    test_mask  = np.array([str(d)[:10] >= TEST_START_DATE for d in dates])
    train_mask = ~test_mask

    n_train_val = train_mask.sum()
    n_train     = int(n_train_val * TRAIN_RATIO / (TRAIN_RATIO + VAL_RATIO))

    idx = np.where(train_mask)[0]
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:]
    test_idx  = np.where(test_mask)[0]

    return (
        X[train_idx], y[train_idx], dates[train_idx],
        X[val_idx],   y[val_idx],   dates[val_idx],
        X[test_idx],  y[test_idx],  dates[test_idx],
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. PyTorch Dataset / DataLoader
# ─────────────────────────────────────────────────────────────────────────────
class EVDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_loaders(X_tr, y_tr, X_val, y_val, X_te, y_te):
    tr_ds  = EVDataset(X_tr,  y_tr)
    val_ds = EVDataset(X_val, y_val)
    te_ds  = EVDataset(X_te,  y_te)

    tr_dl  = DataLoader(tr_ds,  batch_size=BATCH_SIZE, shuffle=False)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    te_dl  = DataLoader(te_ds,  batch_size=BATCH_SIZE, shuffle=False)
    return tr_dl, val_dl, te_dl


# ─────────────────────────────────────────────────────────────────────────────
# 6. Convenience: build everything in one call
# ─────────────────────────────────────────────────────────────────────────────
def build_dataset():
    raw   = load_raw()
    daily = build_daily(raw)
    X, y, dates, scaler_X, scaler_y = prepare_arrays(daily)
    (X_tr, y_tr, d_tr,
     X_val, y_val, d_val,
     X_te, y_te, d_te) = split_data(X, y, dates)

    loaders = get_loaders(X_tr, y_tr, X_val, y_val, X_te, y_te)

    print(f"[data] daily rows: {len(daily)} | "
          f"train: {len(X_tr)} | val: {len(X_val)} | test: {len(X_te)}")

    return dict(
        daily=daily,
        X_tr=X_tr, y_tr=y_tr, d_tr=d_tr,
        X_val=X_val, y_val=y_val, d_val=d_val,
        X_te=X_te, y_te=y_te, d_te=d_te,
        loaders=loaders,
        scaler_X=scaler_X, scaler_y=scaler_y,
    )
