import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import load_model

from src.config import (
    BATTERY_CAPACITY_WH,
    DATA_FILE,
    DEVICE_CONSUMPTION_W,
    MODEL_SAVE_PATH,
    PLANT_AREA_M2,
    TRAIN_TEST_SPLIT,
)
from src.data_preprocessing import DataPreprocessor


FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


@dataclass
class EvalBundle:
    df: object
    y_actual: np.ndarray
    y_pred: np.ndarray
    errors: np.ndarray
    mae: float
    bias: float
    mape: float
    accuracy: float
    r2: float
    catastrophic_fails: int
    range_labels: list
    range_mae: list
    range_bias: list
    range_counts: list
    max_actual: float


def save_fig(filename: str):
    path = os.path.join(FIG_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def _load_models():
    models = []
    for i in range(3):
        p = os.path.join("saved_model", "ensemble", f"model_{i}.h5")
        if os.path.exists(p):
            models.append(load_model(p, compile=False))

    if models:
        return models

    if os.path.exists(MODEL_SAVE_PATH):
        return [load_model(MODEL_SAVE_PATH, compile=False)]

    raise FileNotFoundError("No saved model found in saved_model/ensemble or MODEL_SAVE_PATH")


def build_eval_bundle() -> EvalBundle:
    pre = DataPreprocessor(DATA_FILE)
    X, y = pre.full_preprocess()
    _, X_test, _, y_test = pre.prepare_train_test_split(X, y, split_ratio=TRAIN_TEST_SPLIT)

    models = _load_models()
    pred_list = [m.predict(X_test, verbose=0).reshape(-1) for m in models]
    y_pred_scaled = np.mean(np.vstack(pred_list), axis=0)

    y_actual = pre.scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)
    y_pred = pre.scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(-1)
    errors = y_pred - y_actual

    mae = float(mean_absolute_error(y_actual, y_pred))
    bias = float(np.mean(errors))  # predicted - actual
    r2 = float(r2_score(y_actual, y_pred))

    mape_mask = y_actual > 10.0
    if mape_mask.sum() > 0:
        mape = float(np.mean(np.abs((y_actual[mape_mask] - y_pred[mape_mask]) / y_actual[mape_mask])) * 100.0)
    else:
        mape = 0.0
    accuracy = 100.0 - mape

    catastrophic_fails = int(np.sum(np.abs(errors) > np.maximum(np.abs(y_actual), 1e-6)))

    bins = [(0, 30), (30, 100), (100, 500), (500, np.inf)]
    labels = ["0-30", "30-100", "100-500", "500+"]
    range_mae, range_bias, range_counts = [], [], []

    for (lo, hi), label in zip(bins, labels):
        m = (y_actual >= lo) & (y_actual < hi)
        if m.sum() == 0:
            range_mae.append(0.0)
            range_bias.append(0.0)
            range_counts.append(0)
            continue
        range_mae.append(float(np.mean(np.abs(errors[m]))))
        range_bias.append(float(np.mean(errors[m])))
        range_counts.append(int(m.sum()))

    return EvalBundle(
        df=pre.df,
        y_actual=y_actual,
        y_pred=y_pred,
        errors=errors,
        mae=mae,
        bias=bias,
        mape=mape,
        accuracy=accuracy,
        r2=r2,
        catastrophic_fails=catastrophic_fails,
        range_labels=labels,
        range_mae=range_mae,
        range_bias=range_bias,
        range_counts=range_counts,
        max_actual=float(np.max(y_actual)),
    )


def graph_1_training_history(bundle: EvalBundle):
    # Derived diagnostic curves from prediction errors (keeps this plot data-driven).
    errs = np.abs(bundle.errors)
    x = np.arange(1, len(errs) + 1)
    rolling = np.convolve(errs, np.ones(300) / 300, mode="same")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(x, errs, color="#4E79A7", alpha=0.25, linewidth=0.8, label="|Error|")
    ax1.plot(x, rolling, color="#E15759", linewidth=2.0, label="Rolling mean |Error| (300)")
    ax1.axhline(bundle.mae, color="#59A14F", linestyle="--", linewidth=1.8,
                label=f"Overall MAE = {bundle.mae:.2f}")
    ax1.set_xlabel("Test sample index", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Absolute Error (W/m²)", fontsize=11, fontweight="bold")
    ax1.set_title("(a) Error Trace", fontsize=13, fontweight="bold")
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=9)

    q = np.linspace(0, 1, 200)
    quant = np.quantile(errs, q)
    ax2.plot(q * 100, quant, color="#F28E2B", linewidth=2.2)
    ax2.set_xlabel("Percentile (%)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Absolute Error (W/m²)", fontsize=11, fontweight="bold")
    ax2.set_title("(b) Error Quantile Curve", fontsize=13, fontweight="bold")
    ax2.grid(alpha=0.25)

    save_fig("training_history.png")


def graph_2_predicted_vs_actual(bundle: EvalBundle):
    n = min(2000, len(bundle.y_actual))
    idx = np.linspace(0, len(bundle.y_actual) - 1, n).astype(int)
    actual = bundle.y_actual[idx]
    predicted = bundle.y_pred[idx]
    errors = predicted - actual
    timesteps = np.arange(n)

    fig = plt.figure(figsize=(16, 12))

    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(timesteps, actual, "b-", label="Actual", linewidth=1.2, alpha=0.75)
    ax1.plot(timesteps, predicted, "r--", label="Predicted", linewidth=1.2, alpha=0.75)
    ax1.set_xlabel("Sample Index", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Solar Irradiance (W/m²)", fontsize=11, fontweight="bold")
    ax1.set_title("(a) Predicted vs Actual Timeline", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    ax2 = plt.subplot(2, 2, 2)
    lim = max(bundle.max_actual, float(np.max(predicted)))
    ax2.scatter(actual, predicted, alpha=0.35, s=12, c="#1f77b4", edgecolors="none")
    ax2.plot([0, lim], [0, lim], "r--", linewidth=2, label="Perfect Prediction")
    ax2.set_xlabel("Actual Irradiance (W/m²)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Predicted Irradiance (W/m²)", fontsize=11, fontweight="bold")
    ax2.set_title("(b) Scatter Plot with Perfect Line", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, lim)
    ax2.set_ylim(0, lim)
    ax2.text(0.05 * lim, 0.90 * lim, f"R² = {bundle.r2:.3f}", fontsize=11,
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax3 = plt.subplot(2, 2, 3)
    ax3.hist(errors, bins=60, density=True, alpha=0.7, color="#4E79A7", edgecolor="black")
    mu, sigma = errors.mean(), errors.std()
    x = np.linspace(errors.min(), errors.max(), 200)
    if sigma > 1e-9:
        gaussian = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        ax3.plot(x, gaussian, "r-", linewidth=2, label=f"Gaussian fit (mu={mu:.2f}, sigma={sigma:.2f})")
    ax3.axvline(bundle.bias, color="green", linestyle="--", linewidth=2,
                label=f"Mean bias (pred-actual) = {bundle.bias:.2f}")
    ax3.set_xlabel("Prediction Error (W/m²)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax3.set_title("(c) Error Distribution", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    ax4 = plt.subplot(2, 2, 4)
    window = 120
    rolling_mae = np.array([
        np.mean(np.abs(errors[max(0, i - window): i + 1])) for i in range(len(errors))
    ])
    ax4.plot(timesteps, np.abs(errors), "b.", alpha=0.25, markersize=2.5, label="|Error|")
    ax4.plot(timesteps, rolling_mae, "r-", linewidth=2, label=f"Rolling MAE (window={window})")
    ax4.axhline(bundle.mae, color="green", linestyle="--", linewidth=2,
                label=f"Overall MAE = {bundle.mae:.2f}")
    ax4.set_xlabel("Sample Index", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Absolute Error (W/m²)", fontsize=11, fontweight="bold")
    ax4.set_title("(d) Temporal Error Stability", fontsize=13, fontweight="bold")
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)

    save_fig("predicted_vs_actual.png")


def graph_3_energy_dashboard(bundle: EvalBundle):
    day_str = "01/26/2019"
    day_df = bundle.df[bundle.df.index.strftime("%m/%d/%Y") == day_str].copy()
    if day_df.empty:
        raise ValueError(f"No rows found for {day_str} in dataset")

    minutes = np.arange(len(day_df))
    hours = minutes / 60.0
    irradiance = day_df["Irradiance"].values
    harvested_power = irradiance * PLANT_AREA_M2
    consumption = np.ones_like(harvested_power) * DEVICE_CONSUMPTION_W

    battery = np.zeros_like(harvested_power)
    for i in range(1, len(harvested_power)):
        dt = 1 / 60
        battery[i] = min(
            BATTERY_CAPACITY_WH,
            max(0.0, battery[i - 1] + harvested_power[i] * dt - consumption[i] * dt),
        )

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    ax1.fill_between(hours, 0, irradiance, alpha=0.7, color="orange", label="Solar Irradiance")
    ax1.set_ylabel("Irradiance (W/m²)", fontsize=12, fontweight="bold")
    ax1.set_title("Energy Harvesting Dashboard - January 26, 2019", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=11)
    ax1.grid(alpha=0.3)

    ax2.fill_between(hours, 0, harvested_power, alpha=0.7, color="green", label="Harvested Power")
    ax2.plot(hours, consumption, "r--", linewidth=2, label="Device Consumption")
    ax2.set_ylabel("Power (W)", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=11)
    ax2.grid(alpha=0.3)

    ax3.fill_between(hours, 0, battery, alpha=0.7, color="blue", label="Battery Level")
    ax3.axhline(BATTERY_CAPACITY_WH, color="red", linestyle="--", linewidth=2, alpha=0.6,
                label="Max Capacity")
    ax3.set_xlabel("Time (hours)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Energy (Wh)", fontsize=12, fontweight="bold")
    ax3.legend(loc="upper right", fontsize=11)
    ax3.grid(alpha=0.3)
    ax3.set_xlim(0, max(hours) if len(hours) else 24)

    save_fig("dashboard_01-26-2019.png")


def graph_4_comprehensive_comparison(bundle: EvalBundle):
    methods = [
        "EWMA", "WCMA", "Pro-\nEnergy", "Mod\nPro-Energy",
        "EENA\n(NARNET)", "PADC-MAC\n(NARNET)", "Our CNN\n(100-MAPE)",
    ]

    # Literature-reported ranges and reproducible model metric for this repository.
    accuracy_mid = [10.0, 15.0, 16.0, 20.0, 99.62, 80.0, bundle.accuracy]
    accuracy_low = [5.0, 10.0, 7.0, 11.0, 99.62, 71.0, bundle.accuracy]
    accuracy_high = [15.0, 20.0, 25.0, 29.0, 99.62, 89.0, bundle.accuracy]

    fig, ax = plt.subplots(figsize=(14, 8))
    x_pos = np.arange(len(methods))
    ax.plot(x_pos, accuracy_low, color="#d62728", marker="o", markersize=7,
            linewidth=2.2, label="Low Irradiance")
    ax.plot(x_pos, accuracy_mid, color="#1f77b4", marker="s", markersize=7,
            linewidth=2.2, label="Midpoint Accuracy")
    ax.plot(x_pos, accuracy_high, color="#2ca02c", marker="^", markersize=7,
            linewidth=2.2, label="High Irradiance")

    for i, acc in enumerate(accuracy_mid):
        ax.text(x_pos[i], acc + 2.5, f"{acc:.2f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.set_ylabel("Prediction Accuracy (%)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Prediction Method", fontsize=14, fontweight="bold")
    ax.set_title("Comprehensive Accuracy Comparison: Traditional vs ML vs CNN",
                 fontsize=16, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(fontsize=10, loc="upper left", ncol=2)

    ax.text(
        0.99,
        0.02,
        f"Our CNN point is computed from repository artifacts: accuracy = 100-MAPE = {bundle.accuracy:.2f}%",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    save_fig("comprehensive_comparison.png")


def graph_5_per_range_performance(bundle: EvalBundle):
    ranges = [f"{r}\n(W/m²)" for r in bundle.range_labels]
    mae_values = bundle.range_mae
    bias_values = bundle.range_bias
    samples = bundle.range_counts

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x_pos = np.arange(len(ranges))
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]
    bars1 = ax1.bar(x_pos, mae_values, color=colors, alpha=0.85, edgecolor="black", linewidth=1.2)

    for bar, mae, n in zip(bars1, mae_values, samples):
        ax1.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 1,
                 f"{mae:.2f}\n(n={n})", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax1.set_ylabel("Mean Absolute Error (W/m²)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Irradiance Range", fontsize=12, fontweight="bold")
    ax1.set_title("(a) MAE Performance by Range", fontsize=14, fontweight="bold")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(ranges, fontsize=10)
    ax1.grid(axis="y", alpha=0.3)
    ax1.axhline(bundle.mae, color="green", linestyle="--", linewidth=2,
                label=f"Overall MAE ({bundle.mae:.2f} W/m²)")
    ax1.legend(fontsize=9)

    bars2 = ax2.bar(x_pos, bias_values, color=colors, alpha=0.85, edgecolor="black", linewidth=1.2)
    for bar, bias in zip(bars2, bias_values):
        ax2.text(bar.get_x() + bar.get_width() / 2.0,
                 bar.get_height() + (0.7 if bias >= 0 else -1.5),
                 f"{bias:+.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax2.set_ylabel("Mean Bias (pred-actual, W/m²)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Irradiance Range", fontsize=12, fontweight="bold")
    ax2.set_title("(b) Bias Consistency Across Ranges", fontsize=14, fontweight="bold")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(ranges, fontsize=10)
    ax2.grid(axis="y", alpha=0.3)
    ax2.axhline(bundle.bias, color="green", linestyle="--", linewidth=2,
                label=f"Overall bias ({bundle.bias:+.2f} W/m²)")
    ax2.legend(fontsize=9)

    save_fig("per_range_performance.png")


def graph_6_ablation_study(bundle: EvalBundle):
    # Historical iteration values retained from project logs for components,
    # final point updated from reproducible saved-model evaluation.
    configurations = ["Baseline\nCNN", "+ Weighted\nLoss", "+ Data\nAugment", "+ Batch\nNorm", "Final\n(Reprod)"]
    mae_values = [52.3, 41.2, 38.7, 36.1, bundle.mae]
    bias_values = [-7.78, 12.47, 8.34, 6.98, bundle.bias]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    x_pos = np.arange(len(configurations))
    colors = ["#E63946", "#F77F00", "#FCBF49", "#06D6A0", "#118AB2"]

    bars1 = ax1.bar(x_pos, mae_values, color=colors, alpha=0.85, edgecolor="black", linewidth=1.2)
    for i, (bar, mae) in enumerate(zip(bars1, mae_values)):
        ax1.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 1,
                 f"{mae:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        if i > 0:
            improvement = mae_values[i - 1] - mae
            ax1.annotate(f"{improvement:+.2f}",
                         xy=(x_pos[i] - 0.45, (mae_values[i - 1] + mae) / 2),
                         fontsize=9, color="darkred", fontweight="bold")

    ax1.set_ylabel("Mean Absolute Error (W/m²)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Configuration", fontsize=12, fontweight="bold")
    ax1.set_title("(a) MAE Across Iterative Configurations", fontsize=14, fontweight="bold")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(configurations, fontsize=10)
    ax1.grid(axis="y", alpha=0.3)

    bars2 = ax2.bar(x_pos, bias_values, color=colors, alpha=0.85, edgecolor="black", linewidth=1.2)
    for bar, bias in zip(bars2, bias_values):
        ax2.text(bar.get_x() + bar.get_width() / 2.0,
                 bar.get_height() + (0.5 if bias >= 0 else -1.0),
                 f"{bias:+.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax2.set_ylabel("Mean Bias (pred-actual, W/m²)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Configuration", fontsize=12, fontweight="bold")
    ax2.set_title("(b) Bias Trajectory", fontsize=14, fontweight="bold")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(configurations, fontsize=10)
    ax2.grid(axis="y", alpha=0.3)

    baseline = mae_values[0]
    final = mae_values[-1]
    rel = ((baseline - final) / baseline) * 100.0
    ax1.text(0.5, 0.96, f"Baseline to final change: {baseline - final:.2f} W/m² ({rel:.1f}%)",
             transform=ax1.transAxes, fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
             ha="center", va="top")

    save_fig("ablation_study.png")


def graph_7_iteration_comparison(bundle: EvalBundle):
    metrics = ["|Mean\nBias|", "MAE", "Range\nCoverage", "Failures"]
    iter1_values = [7.78, 52.3, 200, 150]
    iter2_values = [12.47, 41.2, 1200, 20]
    iter3_values = [abs(bundle.bias), bundle.mae, round(bundle.max_actual), bundle.catastrophic_fails]

    iter1_norm = [iter1_values[0], iter1_values[1], iter1_values[2] / 14.0, iter1_values[3] / 10.0]
    iter2_norm = [iter2_values[0], iter2_values[1], iter2_values[2] / 14.0, iter2_values[3] / 10.0]
    iter3_norm = [iter3_values[0], iter3_values[1], iter3_values[2] / 14.0, iter3_values[3] / 10.0]

    x = np.arange(len(metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 7))

    bars1 = ax.bar(x - width, iter1_norm, width, label="Iteration 1 (Baseline)",
                   color="#FF6B6B", alpha=0.85, edgecolor="black")
    bars2 = ax.bar(x, iter2_norm, width, label="Iteration 2 (Improved)",
                   color="#FFA500", alpha=0.85, edgecolor="black")
    bars3 = ax.bar(x + width, iter3_norm, width, label="Iteration 3 (Reproducible)",
                   color="#4ECDC4", alpha=0.85, edgecolor="black")

    ax.set_ylabel("Value (normalized)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Metric", fontsize=12, fontweight="bold")
    ax.set_title("Three-Iteration Model Evolution", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    for i, (b1, b2, b3) in enumerate(zip(bars1, bars2, bars3)):
        ax.text(b1.get_x() + b1.get_width() / 2, b1.get_height() + 1.8, f"{iter1_values[i]:.0f}", ha="center", fontsize=9)
        ax.text(b2.get_x() + b2.get_width() / 2, b2.get_height() + 1.8, f"{iter2_values[i]:.0f}", ha="center", fontsize=9)
        ax.text(b3.get_x() + b3.get_width() / 2, b3.get_height() + 1.8, f"{iter3_values[i]:.0f}", ha="center", fontsize=9, fontweight="bold")

    save_fig("iteration_comparison.png")


def main():
    bundle = build_eval_bundle()
    print(
        f"Computed metrics: MAE={bundle.mae:.4f}, bias(pred-actual)={bundle.bias:+.4f}, "
        f"MAPE={bundle.mape:.2f}%, accuracy(100-MAPE)={bundle.accuracy:.2f}%"
    )
    graph_1_training_history(bundle)
    graph_2_predicted_vs_actual(bundle)
    graph_3_energy_dashboard(bundle)
    graph_4_comprehensive_comparison(bundle)
    graph_5_per_range_performance(bundle)
    graph_6_ablation_study(bundle)
    graph_7_iteration_comparison(bundle)
    print("All graphs generated successfully.")


if __name__ == "__main__":
    main()
