"""
evaluate.py — Six evaluation metrics: R², RMSE, MAE, MAPE, sMAPE, PSNR.
"""
import numpy as np
import pandas as pd


def r2(y_true, y_pred) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred, eps: float = 1e-8) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100)


def smape(y_true, y_pred, eps: float = 1e-8) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2 + eps
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def psnr(y_true, y_pred) -> float:
    mse_val = np.mean((y_true - y_pred) ** 2)
    max_val = np.max(y_true)
    if mse_val == 0:
        return float("inf")
    return float(10 * np.log10(max_val ** 2 / mse_val))


def compute_all(y_true, y_pred) -> dict:
    return {
        "R2":    round(r2(y_true, y_pred),   3),
        "RMSE":  round(rmse(y_true, y_pred), 3),
        "MAE":   round(mae(y_true, y_pred),  3),
        "MAPE":  round(mape(y_true, y_pred), 2),
        "sMAPE": round(smape(y_true, y_pred),2),
        "PSNR":  round(psnr(y_true, y_pred), 2),
    }


def build_performance_table(y_true, results: dict) -> pd.DataFrame:
    """
    results: {model_name: {y_pred: ndarray, ...}, ...}
    Returns a DataFrame matching paper Table II.
    """
    order = ["LinearRegression", "RandomForest", "LSTM", "TransformerLSTM"]
    labels = {
        "LinearRegression": "Linear Regression",
        "RandomForest":     "Random Forest",
        "LSTM":             "LSTM",
        "TransformerLSTM":  "Transformer-LSTM*",
    }
    rows = []
    for key in order:
        m = compute_all(y_true, results[key]["y_pred"])
        m["Model"] = labels[key]
        rows.append(m)

    df = pd.DataFrame(rows)[["Model", "R2", "RMSE", "MAE", "MAPE", "sMAPE", "PSNR"]]
    return df
