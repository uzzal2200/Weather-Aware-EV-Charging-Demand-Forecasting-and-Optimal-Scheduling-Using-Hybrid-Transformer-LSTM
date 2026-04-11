"""
pipeline.py — Modular end-to-end orchestration for forecasting + optimization.

This module keeps workflow logic reusable and testable, while `main.py`
stays a thin execution entrypoint.
"""
import os

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


def run_pipeline(verbose: bool = True) -> dict:
    if verbose:
        _print_header()

    # 1) Data
    if verbose:
        print("\n[1/5] Loading & preprocessing data ...")
    data = build_dataset()

    # 2) Train
    if verbose:
        print("\n[2/5] Training models ...")
    results = train_all(data)

    # 3) Evaluate
    if verbose:
        print("\n[3/5] Evaluating models ...")
    y_true = data["y_te"].ravel()
    perf_df = build_performance_table(y_true, results)
    if verbose:
        print(perf_df.to_string(index=False))

    # 4) Optimization
    if verbose:
        print("\n[4/5] Running MILP optimization ...")
    opt = run_optimization()

    # 5) Visuals + tables
    if verbose:
        print("\n[5/5] Generating figures and tables ...")
    _generate_outputs(data, results, perf_df, opt)

    if verbose:
        out_dir = os.path.join(os.path.dirname(__file__), "..", "figures_output")
        print("\n\u2713 All figures saved to:", out_dir)
        print("\u2713 Done.\n")

    return {
        "data": data,
        "results": results,
        "performance": perf_df,
        "optimization": opt,
    }


def _generate_outputs(data: dict, results: dict, perf_df, opt: dict) -> None:
    y_true = data["y_te"].ravel()

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


def _print_header() -> None:
    print("=" * 60)
    print("  EV Charging Demand Forecasting - Full Pipeline")
    print("=" * 60)
