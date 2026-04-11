"""
main.py — End-to-end pipeline runner from project root.

Run: python main.py
"""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_loader import build_dataset
from evaluate import build_performance_table, r2, rmse
from figures import (
    fig1_framework,
    fig2_architecture,
    fig3_forecast,
    fig4_model_comparison,
    fig5_loss_curves,
    fig6_scatter,
    fig7_error_dist,
    fig8_optimization,
    fig9_cost_analysis,
    fig10_weather,
)
from milp_optimizer import run_optimization
from tables import table1_features, table2_performance, table3_optimization
from train import train_all


def main() -> None:
    print("=" * 60)
    print("  EV Charging Demand Forecasting - Full Pipeline")
    print("=" * 60)

    print("\n[1/5] Loading and preprocessing data ...")
    data = build_dataset()

    print("\n[2/5] Training models ...")
    results = train_all(data)

    print("\n[3/5] Evaluating models ...")
    y_true = data["y_te"].ravel()
    perf_df = build_performance_table(y_true, results)
    print(perf_df.to_string(index=False))

    print("\n[4/5] Running MILP optimization ...")
    opt = run_optimization()

    print("\n[5/5] Generating figures and tables ...")
    fig1_framework()
    fig2_architecture()
    fig3_forecast(
        y_true,
        results["TransformerLSTM"]["y_pred"],
        results["LSTM"]["y_pred"],
        data["d_te"],
    )
    fig4_model_comparison(perf_df)
    fig5_loss_curves(
        results["TransformerLSTM"]["tr_hist"],
        results["TransformerLSTM"]["val_hist"],
        results["LSTM"]["tr_hist"],
        results["LSTM"]["val_hist"],
    )
    fig6_scatter(
        y_true,
        results["TransformerLSTM"]["y_pred"],
        results["LSTM"]["y_pred"],
        r2(y_true, results["TransformerLSTM"]["y_pred"]),
        rmse(y_true, results["TransformerLSTM"]["y_pred"]),
        r2(y_true, results["LSTM"]["y_pred"]),
        rmse(y_true, results["LSTM"]["y_pred"]),
    )
    fig7_error_dist(y_true, results)
    fig8_optimization(opt["no_opt_load"], opt["opt_load"])
    fig9_cost_analysis(opt["no_opt_load"], opt["opt_load"], opt["tariff"])
    fig10_weather(data["daily"])

    table1_features()
    table2_performance(perf_df)
    table3_optimization(
        opt["no_opt_load"],
        opt["opt_load"],
        opt["no_opt_wait"],
        opt["opt_wait"],
        opt["tariff"],
    )

    out_dir = os.path.join(ROOT_DIR, "figures_output")
    print("\nAll outputs saved to:", out_dir)
    print("Done.\n")


if __name__ == "__main__":
    main()
