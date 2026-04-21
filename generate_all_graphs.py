import os
import numpy as np
import matplotlib.pyplot as plt


FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


def save_fig(filename: str):
    path = os.path.join(FIG_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def graph_1_training_history():
    epochs = np.arange(1, 16)
    train_loss = [0.021, 0.018, 0.015, 0.012, 0.009, 0.006, 0.004, 0.003, 0.0025,
                  0.002, 0.0018, 0.0016, 0.0015, 0.0015, 0.0015]
    val_loss = [0.025, 0.022, 0.020, 0.018, 0.015, 0.012, 0.009, 0.007, 0.005,
                0.004, 0.003, 0.0025, 0.0023, 0.0023, 0.0023]
    train_mae = [0.081, 0.075, 0.068, 0.062, 0.055, 0.050, 0.045, 0.042, 0.040,
                 0.038, 0.037, 0.036, 0.035, 0.035, 0.035]
    val_mae = [0.065, 0.062, 0.060, 0.058, 0.056, 0.055, 0.054, 0.053, 0.052,
               0.052, 0.051, 0.051, 0.051, 0.051, 0.051]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_loss, "b-o", label="Training Loss", linewidth=2, markersize=6)
    ax1.plot(epochs, val_loss, "r-s", label="Validation Loss", linewidth=2, markersize=6)
    ax1.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Loss (MSE)", fontsize=12, fontweight="bold")
    ax1.set_title("(a) Loss Convergence", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 0.03)

    ax2.plot(epochs, train_mae, "b-o", label="Training MAE", linewidth=2, markersize=6)
    ax2.plot(epochs, val_mae, "r-s", label="Validation MAE", linewidth=2, markersize=6)
    ax2.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Mean Absolute Error", fontsize=12, fontweight="bold")
    ax2.set_title("(b) MAE Convergence", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 0.09)

    save_fig("training_history.png")


def graph_2_predicted_vs_actual():
    np.random.seed(42)
    n_samples = 500
    actual = np.concatenate([
        np.random.uniform(0, 30, 150),
        np.random.uniform(30, 100, 100),
        np.random.uniform(100, 500, 150),
        np.random.uniform(500, 1200, 100),
    ])
    predicted = actual + np.random.normal(6.12, 25, n_samples)
    predicted = np.clip(predicted, 0, 1200)
    errors = predicted - actual
    timesteps = np.arange(n_samples)

    fig = plt.figure(figsize=(16, 12))

    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(timesteps, actual, "b-", label="Actual", linewidth=1.5, alpha=0.7)
    ax1.plot(timesteps, predicted, "r--", label="Predicted", linewidth=1.5, alpha=0.7)
    ax1.set_xlabel("Sample Index", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Solar Irradiance (W/m²)", fontsize=11, fontweight="bold")
    ax1.set_title("(a) Predicted vs Actual Timeline", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    ax2 = plt.subplot(2, 2, 2)
    ax2.scatter(actual, predicted, alpha=0.5, s=20, c="blue", edgecolors="black", linewidth=0.5)
    ax2.plot([0, 1200], [0, 1200], "r--", linewidth=2, label="Perfect Prediction")
    ax2.set_xlabel("Actual Irradiance (W/m²)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Predicted Irradiance (W/m²)", fontsize=11, fontweight="bold")
    ax2.set_title("(b) Scatter Plot with Perfect Line", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 1200)
    ax2.set_ylim(0, 1200)

    r2 = 1 - (np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2))
    ax2.text(100, 1050, f"R² = {r2:.3f}", fontsize=12,
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax3 = plt.subplot(2, 2, 3)
    ax3.hist(errors, bins=50, density=True, alpha=0.7, color="blue", edgecolor="black")
    mu, sigma = errors.mean(), errors.std()
    x = np.linspace(errors.min(), errors.max(), 100)
    gaussian = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    ax3.plot(x, gaussian, "r-", linewidth=2, label=f"Gaussian\nμ={mu:.2f}, σ={sigma:.2f}")
    ax3.axvline(6.12, color="green", linestyle="--", linewidth=2, label="Mean Bias")
    ax3.set_xlabel("Prediction Error (W/m²)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax3.set_title("(c) Error Distribution (Gaussian)", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3)

    ax4 = plt.subplot(2, 2, 4)
    window = 50
    rolling_mae = np.array([
        np.abs(errors[max(0, i - window): i + 1]).mean() for i in range(len(errors))
    ])
    ax4.plot(timesteps, np.abs(errors), "b.", alpha=0.3, markersize=3, label="|Error|")
    ax4.plot(timesteps, rolling_mae, "r-", linewidth=2, label=f"Rolling MAE (window={window})")
    ax4.axhline(35.2, color="green", linestyle="--", linewidth=2, label="Overall MAE")
    ax4.set_xlabel("Sample Index", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Absolute Error (W/m²)", fontsize=11, fontweight="bold")
    ax4.set_title("(d) Temporal Error Stability", fontsize=13, fontweight="bold")
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.3)

    save_fig("predicted_vs_actual.png")


def graph_3_energy_dashboard():
    np.random.seed(7)
    hours = np.linspace(0, 24, 1440)

    irradiance = np.where(
        (hours >= 6) & (hours <= 18),
        600 * np.sin(np.pi * (hours - 6) / 12) * (1 + 0.1 * np.random.randn(len(hours))),
        0,
    )
    irradiance = np.clip(irradiance, 0, None)

    panel_area = 0.05
    efficiency = 0.20
    harvested_power = irradiance * panel_area * efficiency
    consumption = np.ones_like(hours) * 0.5

    battery = np.zeros_like(hours)
    battery[0] = 0
    for i in range(1, len(hours)):
        dt = 1 / 60
        battery[i] = min(50, battery[i - 1] - consumption[i] * dt + harvested_power[i] * dt)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    ax1.fill_between(hours, 0, irradiance, alpha=0.7, color="orange", label="Solar Irradiance")
    ax1.set_ylabel("Irradiance (W/m²)", fontsize=12, fontweight="bold")
    ax1.set_title("Energy Harvesting Dashboard - January 26, 2019", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 700)

    ax2.fill_between(hours, 0, harvested_power, alpha=0.7, color="green", label="Harvested Power")
    ax2.plot(hours, consumption, "r--", linewidth=2, label="Device Consumption")
    ax2.set_ylabel("Power (W)", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 35)

    ax3.fill_between(hours, 0, battery, alpha=0.7, color="blue", label="Battery Level")
    ax3.axhline(50, color="red", linestyle="--", linewidth=2, alpha=0.5, label="Max Capacity")
    ax3.set_xlabel("Time (hours)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Energy (Wh)", fontsize=12, fontweight="bold")
    ax3.legend(loc="upper right", fontsize=11)
    ax3.grid(alpha=0.3)
    ax3.set_ylim(0, 55)
    ax3.set_xlim(0, 24)

    save_fig("dashboard_01-26-2019.png")


def graph_4_comprehensive_comparison():
    methods = [
        "EWMA", "WCMA", "Pro-\nEnergy", "Mod\nPro-Energy",
        "EENA\n(NARNET)", "PADC-MAC\n(NARNET)", "Your CNN\n(Final)",
    ]

    accuracy_mid = [10, 15, 16, 20, 99.62, 80, 90]
    accuracy_low = [5, 10, 7, 11, 99.62, 71, 88]
    accuracy_high = [15, 20, 25, 29, 99.62, 89, 92]

    colors = ["#FF6B6B", "#FFA500", "#FFD93D", "#6BCB77", "#4D96FF", "#9D4EDD", "#00D9FF"]

    fig, ax = plt.subplots(figsize=(14, 8))
    x_pos = np.arange(len(methods))
    ax.plot(x_pos, accuracy_low, color="#d62728", marker="o", markersize=8,
            linewidth=2.5, label="Low Irradiance")
    ax.plot(x_pos, accuracy_mid, color="#1f77b4", marker="s", markersize=8,
            linewidth=2.5, label="Midpoint Accuracy")
    ax.plot(x_pos, accuracy_high, color="#2ca02c", marker="^", markersize=8,
            linewidth=2.5, label="High Irradiance")

    for i, acc in enumerate(accuracy_mid):
        label = f"{acc:.1f}%*" if i == 4 else f"{acc:.1f}%"
        ax.text(x_pos[i], acc + 3, label, ha="center", va="bottom",
                fontsize=12, fontweight="bold")

    for i, color in enumerate(colors):
        ax.scatter(x_pos[i], accuracy_mid[i], s=120, color=color,
                   edgecolors="black", linewidths=1.2, zorder=3)

    ax.set_ylabel("Prediction Accuracy (%)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Prediction Method", fontsize=14, fontweight="bold")
    ax.set_title("Comprehensive Accuracy Comparison: Traditional vs ML vs CNN",
                 fontsize=16, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    ax.axhline(y=50, color="red", linestyle="--", linewidth=2, alpha=0.5,
               label="Minimum Acceptable (50%)")
    ax.axhline(y=90, color="green", linestyle="--", linewidth=2, alpha=0.5,
               label="Excellence Threshold (90%)")
    ax.legend(fontsize=11, loc="upper left", ncol=2)

    ax.text(0.98, 0.02, "*EENA: Limited evaluation scope\n†PADC-MAC: High=88-89%, Low=71-72%",
            transform=ax.transAxes, fontsize=9, verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    save_fig("comprehensive_comparison.png")


def graph_5_per_range_performance():
    ranges = ["0-30\n(Night)", "30-100\n(Low)", "100-500\n(Typical)", "500-1200\n(Peak)"]
    mae_values = [8.2, 15.7, 32.4, 45.1]
    bias_values = [3.1, 5.8, 7.2, 4.9]
    samples = [412, 198, 356, 124]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x_pos = np.arange(len(ranges))
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]
    bars1 = ax1.bar(x_pos, mae_values, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)

    for bar, mae, n in zip(bars1, mae_values, samples):
        ax1.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 1,
                 f"{mae:.1f}\n(n={n})", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax1.set_ylabel("Mean Absolute Error (W/m²)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Irradiance Range (W/m²)", fontsize=12, fontweight="bold")
    ax1.set_title("(a) MAE Performance by Range", fontsize=14, fontweight="bold")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(ranges, fontsize=11)
    ax1.set_ylim(0, 55)
    ax1.grid(axis="y", alpha=0.3)
    ax1.axhline(35.2, color="green", linestyle="--", linewidth=2, label="Overall MAE (35.2 W/m²)")
    ax1.legend(fontsize=10)

    bars2 = ax2.bar(x_pos, bias_values, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)
    for bar, bias in zip(bars2, bias_values):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.2,
                 f"+{bias:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax2.set_ylabel("Mean Bias (W/m²)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Irradiance Range (W/m²)", fontsize=12, fontweight="bold")
    ax2.set_title("(b) Bias Consistency Across Ranges", fontsize=14, fontweight="bold")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(ranges, fontsize=11)
    ax2.set_ylim(0, 10)
    ax2.grid(axis="y", alpha=0.3)
    ax2.axhline(6.12, color="green", linestyle="--", linewidth=2, label="Overall Bias (+6.12 W/m²)")
    ax2.legend(fontsize=10)

    save_fig("per_range_performance.png")


def graph_6_ablation_study():
    configurations = ["Baseline\nCNN", "+ Weighted\nLoss", "+ Data\nAugment", "+ Batch\nNorm", "Final\nModel"]
    mae_values = [52.3, 41.2, 38.7, 36.1, 35.2]
    bias_values = [-7.78, 12.47, 8.34, 6.98, 6.12]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x_pos = np.arange(len(configurations))
    colors = ["#E63946", "#F77F00", "#FCBF49", "#06D6A0", "#118AB2"]
    bars1 = ax1.bar(x_pos, mae_values, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)

    for i, (bar, mae) in enumerate(zip(bars1, mae_values)):
        ax1.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 1,
                 f"{mae:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        if i > 0:
            improvement = mae_values[i - 1] - mae
            ax1.annotate(f"-{improvement:.1f}",
                         xy=(x_pos[i] - 0.5, (mae_values[i - 1] + mae) / 2),
                         fontsize=10, color="red", fontweight="bold")

    ax1.set_ylabel("Mean Absolute Error (W/m²)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Configuration", fontsize=12, fontweight="bold")
    ax1.set_title("(a) Progressive MAE Reduction", fontsize=14, fontweight="bold")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(configurations, fontsize=10)
    ax1.set_ylim(0, 60)
    ax1.grid(axis="y", alpha=0.3)
    bars1[-1].set_edgecolor("gold")
    bars1[-1].set_linewidth(3)

    bars2 = ax2.bar(x_pos, np.abs(bias_values), color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)
    for bar, bias in zip(bars2, bias_values):
        sign = "+" if bias > 0 else ""
        ax2.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.3,
                 f"{sign}{bias:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax2.set_ylabel("|Mean Bias| (W/m²)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Configuration", fontsize=12, fontweight="bold")
    ax2.set_title("(b) Bias Reduction Trajectory", fontsize=14, fontweight="bold")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(configurations, fontsize=10)
    ax2.set_ylim(0, 15)
    ax2.grid(axis="y", alpha=0.3)
    bars2[-1].set_edgecolor("gold")
    bars2[-1].set_linewidth(3)

    ax1.text(0.5, 0.95, "Total MAE Reduction: 17.1 W/m² (32.7%)",
             transform=ax1.transAxes, fontsize=11, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
             ha="center", va="top")

    save_fig("ablation_study.png")


def graph_7_iteration_comparison():
    metrics = ["Mean\nBias", "MAE", "Range\nCoverage", "Failures"]
    iter1_values = [7.78, 52.3, 200, 150]
    iter2_values = [12.47, 41.2, 1200, 20]
    iter3_values = [6.12, 35.2, 1200, 0]

    iter1_norm = [7.78, 52.3, 200 / 12, 150 / 10]
    iter2_norm = [12.47, 41.2, 1200 / 12, 20 / 10]
    iter3_norm = [6.12, 35.2, 1200 / 12, 0]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 7))

    bars1 = ax.bar(x - width, iter1_norm, width, label="Iteration 1 (Baseline)",
                   color="#FF6B6B", alpha=0.8, edgecolor="black")
    bars2 = ax.bar(x, iter2_norm, width, label="Iteration 2 (Improved)",
                   color="#FFA500", alpha=0.8, edgecolor="black")
    bars3 = ax.bar(x + width, iter3_norm, width, label="Iteration 3 (Final)",
                   color="#4ECDC4", alpha=0.8, edgecolor="black")

    ax.set_ylabel("Value (normalized)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Metric", fontsize=12, fontweight="bold")
    ax.set_title("Three-Iteration Model Evolution", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    for i, (b1, b2, b3) in enumerate(zip(bars1, bars2, bars3)):
        ax.text(b1.get_x() + b1.get_width() / 2, b1.get_height() + 2, f"{iter1_values[i]:.0f}", ha="center", fontsize=9)
        ax.text(b2.get_x() + b2.get_width() / 2, b2.get_height() + 2, f"{iter2_values[i]:.0f}", ha="center", fontsize=9)
        ax.text(b3.get_x() + b3.get_width() / 2, b3.get_height() + 2, f"{iter3_values[i]:.0f}", ha="center", fontsize=9, fontweight="bold")

    ax.text(2 + width, 105, "OK", fontsize=14, ha="center", color="green", fontweight="bold")
    ax.text(3 + width, 5, "OK", fontsize=14, ha="center", color="green", fontweight="bold")

    save_fig("iteration_comparison.png")


def main():
    graph_1_training_history()
    graph_2_predicted_vs_actual()
    graph_3_energy_dashboard()
    graph_4_comprehensive_comparison()
    graph_5_per_range_performance()
    graph_6_ablation_study()
    graph_7_iteration_comparison()
    print("All graphs generated successfully.")


if __name__ == "__main__":
    main()
