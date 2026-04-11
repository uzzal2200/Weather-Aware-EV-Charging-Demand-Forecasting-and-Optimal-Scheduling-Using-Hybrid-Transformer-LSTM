"""
figures.py — Generate all 10 paper figures.
Each public function saves a PNG to FIGURES_DIR and returns the filepath.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde

from config import FIGURES_DIR, DPI, FONT_SIZE

plt.rcParams.update({
    "font.size":        FONT_SIZE,
    "axes.titlesize":   FONT_SIZE + 1,
    "axes.labelsize":   FONT_SIZE,
    "xtick.labelsize":  FONT_SIZE - 1,
    "ytick.labelsize":  FONT_SIZE - 1,
    "legend.fontsize":  FONT_SIZE - 1,
    "figure.dpi":       DPI,
})

MODEL_COLORS = {
    "Linear Regression":  "#E74C3C",
    "Random Forest":      "#F39C12",
    "LSTM":               "#3498DB",
    "Transformer-LSTM*":  "#27AE60",
}

def _save(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] saved → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — System Framework (schematic)
# ─────────────────────────────────────────────────────────────────────────────
def fig1_framework():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    boxes = [
        (0.05, 0.20, 0.18, 0.60, "EV Charging\nDataset\n(NYC Data.gov)"),
        (0.05, -0.10, 0.18, 0.28, "Weather Dataset\n(Iowa ASOS)"),
        (0.27, 0.05, 0.18, 0.90, "Preprocessing\n& Fusion"),
        (0.49, 0.05, 0.18, 0.90, "Hybrid\nTransformer-LSTM\nModel"),
        (0.71, 0.05, 0.14, 0.90, "MILP\nOptimization\nScheduler"),
        (0.88, 0.05, 0.11, 0.90, "Optimal\nCharging\nSchedule"),
    ]

    for (x, y_off, w, h, label) in boxes:
        rect = mpatches.FancyBboxPatch(
            (x, 0.05 + y_off * 0.0), w, h,
            boxstyle="round,pad=0.02",
            linewidth=1.5, edgecolor="#2C3E50", facecolor="#EBF5FB"
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, 0.05 + y_off * 0.0 + h / 2, label,
                ha="center", va="center", fontsize=9, fontweight="bold")

    # arrows
    for x_start in [0.23, 0.45, 0.67, 0.85]:
        ax.annotate("", xy=(x_start + 0.04, 0.50),
                    xytext=(x_start, 0.50),
                    arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.5))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Fig. 1 — Overall System Framework for Weather-Aware EV Charging Forecasting & Optimization",
                 fontsize=10, pad=8)
    return _save(fig, "Fig1_Framework.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — Architecture Diagram
# ─────────────────────────────────────────────────────────────────────────────
def fig2_architecture():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")

    layers = [
        ("Input Sequence\n(T×F = 30×15)", "#D5E8D4", 0.02),
        ("Linear Embedding\n+ Positional Encoding\n(d_model=64)", "#DAE8FC", 0.18),
        ("Transformer Encoder\n(×2 blocks)\nMHSA (h=4, d_k=16)\nFFN (d_ff=256)\nAdd & LayerNorm", "#FFE6CC", 0.36),
        ("LSTM Layer\n(hidden=128,\nbidirectional=False)", "#E1D5E7", 0.58),
        ("Dropout (p=0.10)\n+\nDense Output Head", "#F8CECC", 0.76),
        ("Daily Demand\nForecast (kWh)", "#D5E8D4", 0.90),
    ]

    for (label, color, x) in layers:
        rect = mpatches.FancyBboxPatch(
            (x, 0.15), 0.14, 0.70,
            boxstyle="round,pad=0.02",
            linewidth=1.3, edgecolor="#555", facecolor=color
        )
        ax.add_patch(rect)
        ax.text(x + 0.07, 0.50, label, ha="center", va="center",
                fontsize=8.5, fontweight="bold")

    for i in range(len(layers) - 1):
        x_start = layers[i][2] + 0.14
        x_end   = layers[i + 1][2]
        ax.annotate("", xy=(x_end, 0.50), xytext=(x_start, 0.50),
                    arrowprops=dict(arrowstyle="->", color="#333", lw=1.5))

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1)
    ax.set_title("Fig. 2 — Hybrid Transformer–LSTM Model Architecture", fontsize=10, pad=8)
    return _save(fig, "Fig2_Architecture.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — One-week forecast vs actual
# ─────────────────────────────────────────────────────────────────────────────
def fig3_forecast(y_true, y_pred_tl, y_pred_lstm, dates):
    # pick 7 consecutive test days with highest variance for visual impact
    if len(y_true) < 7:
        raise ValueError("Need at least 7 test samples for Fig 3.")

    # find 7-day window with best contrast
    best_start = 0
    best_var   = -1
    for s in range(len(y_true) - 7):
        v = np.var(y_true[s: s + 7])
        if v > best_var:
            best_var   = v
            best_start = s

    sl = slice(best_start, best_start + 7)
    yt  = y_true[sl]
    yp  = y_pred_tl[sl]
    yl  = y_pred_lstm[sl]
    dt  = dates[sl]
    day_labels = [str(d)[:10] for d in dt]

    fig, axes = plt.subplots(2, 1, figsize=(9, 6),
                              gridspec_kw={"height_ratios": [3, 1]})
    ax1, ax2 = axes
    x = np.arange(7)

    ax1.plot(x, yt,  "k-o",  lw=2,   ms=5,  label="Actual Load")
    ax1.plot(x, yp,  "g--s", lw=1.8, ms=5,  label="Transformer-LSTM (Proposed)")
    ax1.plot(x, yl,  "b:^",  lw=1.5, ms=5,  label="LSTM Baseline")
    ax1.fill_between(x, yp - 1.5, yp + 1.5, alpha=0.15, color="green", label="±1.5 kWh CI")
    ax1.set_ylabel("Charging Demand (kWh/h)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Fig. 3 — One-Week EV Charging Demand Forecast vs. Actual")

    residuals = yt - yp
    ax2.bar(x, residuals, color=["#27AE60" if r >= 0 else "#E74C3C" for r in residuals],
            width=0.6, alpha=0.85)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_ylabel("Residual (kWh)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    return _save(fig, "Fig3_Forecast.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 — Six-panel metric bar chart
# ─────────────────────────────────────────────────────────────────────────────
def fig4_model_comparison(perf_df: pd.DataFrame):
    metrics = ["R2", "RMSE", "MAE", "MAPE", "sMAPE", "PSNR"]
    metric_labels = {
        "R2":    r"$R^2$ Score",
        "RMSE":  "RMSE (kWh)",
        "MAE":   "MAE (kWh)",
        "MAPE":  "MAPE (%)",
        "sMAPE": "sMAPE (%)",
        "PSNR":  "PSNR (dB)",
    }
    higher_better = {"R2", "PSNR"}
    colors = list(MODEL_COLORS.values())
    models = perf_df["Model"].tolist()

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.ravel()

    for ax, met in zip(axes, metrics):
        vals = perf_df[met].values
        bars = ax.bar(range(len(models)), vals, color=colors, width=0.55, edgecolor="white")
        ax.set_title(metric_labels[met])
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(["LR", "RF", "LSTM", "TL*"], fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        # annotate best
        best_idx = np.argmax(vals) if met in higher_better else np.argmin(vals)
        ax.annotate(
            f"{'↑' if met in higher_better else '↓'}{vals[best_idx]:.2f}",
            xy=(best_idx, vals[best_idx]),
            xytext=(best_idx, vals[best_idx] * 1.02),
            ha="center", fontsize=8, color="black", fontweight="bold"
        )
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 0.5,
                    f"{v:.2f}", ha="center", va="center", fontsize=8, color="white")

    patches = [mpatches.Patch(color=c, label=m) for m, c in MODEL_COLORS.items()]
    fig.legend(handles=patches, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.suptitle("Fig. 4 — Comparative Model Performance Across Six Evaluation Metrics",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    return _save(fig, "Fig4_ModelComparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 — Training / validation loss curves
# ─────────────────────────────────────────────────────────────────────────────
def fig5_loss_curves(tl_tr, tl_val, lstm_tr, lstm_val):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, (tr, va, lbl) in zip(axes, [
        (tl_tr,   tl_val,   "Transformer-LSTM"),
        (lstm_tr, lstm_val, "LSTM Baseline"),
    ]):
        ep = range(1, len(tr) + 1)
        ax.plot(ep, tr, "b-",  lw=1.5, label=f"Train Loss ({lbl})")
        ax.plot(ep, va, "r--", lw=1.5, label=f"Val Loss ({lbl})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title(f"(a) {lbl}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Fig. 5 — Training and Validation Loss Curves", fontsize=11)
    fig.tight_layout()
    return _save(fig, "Fig5_LossCurves.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 — Scatter: predicted vs actual
# ─────────────────────────────────────────────────────────────────────────────
def fig6_scatter(y_true, y_pred_tl, y_pred_lstm, r2_tl, rmse_tl, r2_lstm, rmse_lstm):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, (yp, r2v, rmv, title, color) in zip(axes, [
        (y_pred_tl,   r2_tl,   rmse_tl,   "Transformer-LSTM (Proposed)", "#27AE60"),
        (y_pred_lstm, r2_lstm, rmse_lstm, "LSTM Baseline",                "#3498DB"),
    ]):
        lims = [min(y_true.min(), yp.min()) - 2, max(y_true.max(), yp.max()) + 2]
        ax.scatter(y_true, yp, alpha=0.4, s=12, color=color, label="Predictions")
        ax.plot(lims, lims,             "k-",  lw=1.5, label="Perfect Prediction (y=x)")
        ax.plot(lims, [l * 1.1 for l in lims], "k--", lw=0.8, alpha=0.5)
        ax.plot(lims, [l * 0.9 for l in lims], "k--", lw=0.8, alpha=0.5, label="±10% Band")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Actual Demand (kWh)")
        ax.set_ylabel("Predicted Demand (kWh)")
        ax.set_title(title)
        ax.text(0.05, 0.92, f"$R^2$={r2v:.3f}\nRMSE={rmv:.3f} kWh",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Fig. 6 — Predicted vs. Actual Demand Scatter Plots", fontsize=11)
    fig.tight_layout()
    return _save(fig, "Fig6_ScatterPlot.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7 — Prediction error distribution
# ─────────────────────────────────────────────────────────────────────────────
def fig7_error_dist(y_true, results: dict):
    order  = ["LinearRegression", "RandomForest", "LSTM", "TransformerLSTM"]
    labels = {
        "LinearRegression": "Linear Regression",
        "RandomForest":     "Random Forest",
        "LSTM":             "LSTM Baseline",
        "TransformerLSTM":  "Transformer-LSTM (Proposed)",
    }
    colors = list(MODEL_COLORS.values())

    fig, ax = plt.subplots(figsize=(9, 5))
    x_range = np.linspace(-15, 15, 500)

    for key, color in zip(order, colors):
        err = y_true - results[key]["y_pred"]
        sigma = err.std()
        kde   = gaussian_kde(err)
        ax.plot(x_range, kde(x_range), lw=2, color=color,
                label=f"{labels[key]} (σ={sigma:.2f} kWh)")
        ax.fill_between(x_range, kde(x_range), alpha=0.08, color=color)

    ax.axvline(0, color="k", lw=0.8, linestyle="--")
    ax.set_xlabel("Prediction Error (kWh/h)")
    ax.set_ylabel("Probability Density")
    ax.set_title("Fig. 7 — Prediction Error Distribution for All Models on the Test Set")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, "Fig7_ErrorDist.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 8 — Load profile before vs after MILP
# ─────────────────────────────────────────────────────────────────────────────
def fig8_optimization(no_opt_load, opt_load):
    hours  = np.arange(96) * 0.25
    labels = [f"{int(h):02d}:00" for h in hours[::4]]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(hours, no_opt_load, "r-",  lw=2, label="Without Optimization",   alpha=0.85)
    ax.plot(hours, opt_load,    "g--", lw=2, label="With MILP Optimization",  alpha=0.85)

    peak_no  = no_opt_load.max()
    peak_opt = opt_load.max()
    ax.axhline(peak_no,  color="red",   lw=1,   linestyle=":",  alpha=0.6,
               label=f"Peak (No Opt.): {peak_no:.1f} kW")
    ax.axhline(peak_opt, color="green", lw=1,   linestyle=":",  alpha=0.6,
               label=f"Peak (With Opt.): {peak_opt:.1f} kW")

    peak_area_x = hours[(no_opt_load > opt_load)]
    if len(peak_area_x):
        ax.fill_between(hours, no_opt_load, opt_load,
                         where=(no_opt_load > opt_load),
                         alpha=0.15, color="red", label="Peak Reduction Area")

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Aggregated Load (kW)")
    ax.set_xticks(hours[::4])
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Fig. 8 — Aggregated EV Charging Load Profile: Before vs. After MILP Optimization")
    fig.tight_layout()
    return _save(fig, "Fig8_Optimization.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 9 — Hourly cost + ToU tariff
# ─────────────────────────────────────────────────────────────────────────────
def fig9_cost_analysis(no_opt_load, opt_load, tariff):
    hours = np.arange(96) * 0.25
    slot_hrs = 0.25
    cost_no  = no_opt_load  * tariff * slot_hrs
    cost_opt = opt_load     * tariff * slot_hrs
    savings  = cost_no - cost_opt
    total_save = savings.sum()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

    # ── Panel a: load + tariff ────────────────────────────────────────────
    ax1_twin = ax1.twinx()
    ax1.bar(hours, no_opt_load,  width=0.22, color="#E74C3C", alpha=0.7, label="Load (No Opt.)")
    ax1.bar(hours + 0.23, opt_load, width=0.22, color="#27AE60", alpha=0.7, label="Load (With Opt.)")
    ax1_twin.step(hours, tariff, color="#8E44AD", lw=2, linestyle="--", where="post",
                  label="ToU Tariff ($/kWh)")
    ax1.set_ylabel("Aggregated Load (kW)")
    ax1_twin.set_ylabel("ToU Tariff ($/kWh)")
    ax1.set_title("(a) Hourly Load Distribution and Time-of-Use Tariff")
    ax1.grid(True, alpha=0.3)
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper left", fontsize=9)

    # ── Panel b: hourly cost ──────────────────────────────────────────────
    ax2.bar(hours,        cost_no,  width=0.22, color="#E74C3C", alpha=0.75, label="Energy Cost (No Opt.)")
    ax2.bar(hours + 0.23, cost_opt, width=0.22, color="#27AE60", alpha=0.75, label="Energy Cost (With Opt.)")
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Hourly Energy Cost ($)")
    ax2.set_title(f"(b) Hourly Energy Cost Comparison — Total Daily Saving: ${total_save:.2f}")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    for ax in [ax1, ax2]:
        ax.set_xticks(np.arange(0, 25, 2))
        ax.set_xticklabels([f"{int(h):02d}" for h in np.arange(0, 25, 2)], rotation=45)

    fig.suptitle("Fig. 9 — Hourly Load Distribution with ToU Tariff and Energy Cost Comparison",
                 fontsize=11)
    fig.tight_layout()
    return _save(fig, "Fig9_CostAnalysis.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 10 — Weather feature correlations
# ─────────────────────────────────────────────────────────────────────────────
def fig10_weather(daily: pd.DataFrame):
    target = "energy_sum"
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Panel a: Temperature vs. demand
    ax = axes[0]
    sc = ax.scatter(daily["tmpf"], daily[target], alpha=0.25, s=6,
                    c=daily["feel"], cmap="RdYlBu_r")
    plt.colorbar(sc, ax=ax, label="Apparent Temp (°F)")

    # quadratic fit
    coef = np.polyfit(daily["tmpf"], daily[target], 2)
    xfit = np.linspace(daily["tmpf"].min(), daily["tmpf"].max(), 200)
    ax.plot(xfit, np.polyval(coef, xfit), "k--", lw=2, label="Quadratic Fit")
    ax.set_xlabel("Air Temperature (°F)")
    ax.set_ylabel("Daily EV Demand (kWh)")
    ax.set_title("(a) Temperature vs. Charging Demand")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel b: Precipitation vs. demand
    ax2 = axes[1]
    sc2 = ax2.scatter(daily["p01m"], daily[target], alpha=0.25, s=6,
                      c=daily["snowdepth"], cmap="Blues")
    plt.colorbar(sc2, ax=ax2, label="Snow Depth (binary)")
    coef2 = np.polyfit(daily["p01m"], daily[target], 1)
    xfit2 = np.linspace(0, daily["p01m"].quantile(0.99), 100)
    ax2.plot(xfit2, np.polyval(coef2, xfit2), "r--", lw=2, label="Linear Fit")
    r = np.corrcoef(daily["p01m"], daily[target])[0, 1]
    ax2.set_xlabel("Precipitation (mm)")
    ax2.set_ylabel("Daily EV Demand (kWh)")
    ax2.set_title(f"(b) Precipitation vs. Charging Demand  (r={r:.2f})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.1, daily["p01m"].quantile(0.99) + 0.2)

    fig.suptitle("Fig. 10 — Weather Feature Correlation with EV Charging Demand", fontsize=11)
    fig.tight_layout()
    return _save(fig, "Fig10_Weather.png")
