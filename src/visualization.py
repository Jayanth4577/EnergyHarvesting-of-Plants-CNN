"""
Visualization Module
Generates all plots and graphs for the energy harvesting project.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

from .config import (
    FIGURE_DPI, DEFAULT_NUM_SAMPLES, OUTPUT_DIR,
    PLANT_AREA_M2, BATTERY_CAPACITY_WH, DEVICE_CONSUMPTION_W
)


class Visualizer:
    """Handles all visualization and plotting operations."""
    
    def __init__(self):
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    def plot_training_history(self, history, save=True):
        """
        Plot training and validation loss curves.
        
        Args:
            history: Keras training history object
            save: Whether to save the plot to file
        """
        if history is None:
            print("Error: No training history available.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot Loss
        epochs = range(1, len(history.history['loss']) + 1)
        ax1.plot(epochs, history.history['loss'], 
                 label='Training Loss', color='blue', linewidth=2, marker='o')
        ax1.plot(epochs, history.history['val_loss'], 
                 label='Validation Loss', color='red', linewidth=2, marker='s')
        ax1.set_title('Model Loss During Training', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot MAE
        ax2.plot(epochs, history.history['mae'], 
                 label='Training MAE', color='green', linewidth=2, marker='o')
        ax2.plot(epochs, history.history['val_mae'], 
                 label='Validation MAE', color='orange', linewidth=2, marker='s')
        ax2.set_title('Model MAE During Training', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Mean Absolute Error')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(OUTPUT_DIR, 'training_history.png')
            plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
            print(f">> Training history saved to '{filepath}'")
        
        plt.show()
    
    def plot_predicted_vs_actual(self, y_actual, y_predicted, 
                                  num_samples=DEFAULT_NUM_SAMPLES, save=True):
        """
        Plot comprehensive predicted vs actual comparison.
        
        Args:
            y_actual: Actual irradiance values
            y_predicted: Predicted irradiance values
            num_samples: Number of samples to display
            save: Whether to save the plot
        """
        # Limit samples for clearer visualization
        if len(y_actual) > num_samples:
            y_actual = y_actual[:num_samples]
            y_predicted = y_predicted[:num_samples]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('CNN Model Performance Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Time Series Comparison
        ax1 = axes[0, 0]
        ax1.plot(y_actual, label='Actual', color='blue', alpha=0.7, linewidth=1)
        ax1.plot(y_predicted, label='Predicted', color='red', alpha=0.7, linewidth=1)
        ax1.set_title('Predicted vs Actual Irradiance Over Time')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Irradiance (W/m²)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: 2D Histogram Heatmap (replaces scatter for better visualization)
        ax2 = axes[0, 1]
        max_val = max(y_actual.max(), y_predicted.max())
        # Create 2D histogram heatmap
        h = ax2.hist2d(y_actual, y_predicted, bins=50, cmap='hot', 
                       range=[[0, max_val], [0, max_val]])
        plt.colorbar(h[3], ax=ax2, label='Frequency')
        ax2.plot([0, max_val], [0, max_val], 'cyan', 
                 label='Perfect Prediction', linewidth=2, linestyle='--')
        ax2.set_title('Heatmap: Actual vs Predicted')
        ax2.set_xlabel('Actual Irradiance (W/m²)')
        ax2.set_ylabel('Predicted Irradiance (W/m²)')
        ax2.legend(loc='upper left')
        ax2.set_aspect('equal')
        
        # Plot 3: Prediction Error Distribution
        ax3 = axes[1, 0]
        errors = y_actual - y_predicted
        ax3.hist(errors, bins=50, color='teal', edgecolor='black', alpha=0.7)
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax3.axvline(x=np.mean(errors), color='orange', linestyle='-', 
                    linewidth=2, label=f'Mean Error: {np.mean(errors):.2f}')
        ax3.set_title('Prediction Error Distribution')
        ax3.set_xlabel('Error (Actual - Predicted)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Error Over Time
        ax4 = axes[1, 1]
        ax4.plot(errors, color='orange', alpha=0.7, linewidth=1)
        ax4.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax4.fill_between(range(len(errors)), errors, alpha=0.3, color='orange')
        ax4.set_title('Prediction Error Over Time')
        ax4.set_xlabel('Time Steps')
        ax4.set_ylabel('Error (W/m²)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(OUTPUT_DIR, 'predicted_vs_actual.png')
            plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
            print(f">> Predicted vs Actual graph saved to '{filepath}'")
        
        plt.show()
    
    def plot_daily_dashboard(self, day_df, query_date_str, save=True):
        """
        Generate comprehensive daily dashboard with solar and battery graphs.
        
        Args:
            day_df: DataFrame containing data for the specific day
            query_date_str: Date string for the title
            save: Whether to save the plot
        """
        if day_df.empty:
            print("Error: No data for this date to graph.")
            return
        
        # Simulate Battery
        battery_levels = []
        current_charge = 0.0
        energy_in = (day_df['Irradiance'].values * PLANT_AREA_M2) * (1/60)
        energy_out = DEVICE_CONSUMPTION_W * (1/60)
        
        for e_in in energy_in:
            current_charge += (e_in - energy_out)
            current_charge = max(0, min(current_charge, BATTERY_CAPACITY_WH))
            battery_levels.append(current_charge)
        
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f'Energy Harvesting Dashboard: {query_date_str}', 
                     fontsize=16, fontweight='bold')
        
        # Graph 1: Solar Irradiance
        ax1.plot(day_df.index, day_df['Irradiance'], 
                 color='#ff9900', linewidth=1.5, label='Solar Irradiance')
        ax1.fill_between(day_df.index, day_df['Irradiance'], 
                         color='#ff9900', alpha=0.3)
        ax1.set_ylabel('Sunlight Intensity (W/m²)')
        ax1.set_title('Solar Irradiance Profile')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Graph 2: Harvested Power
        harvested_power = day_df['Irradiance'].values * PLANT_AREA_M2
        ax2.plot(day_df.index, harvested_power, 
                 color='#3366cc', linewidth=1.5, label='Harvested Power')
        ax2.fill_between(day_df.index, harvested_power, 
                         color='#3366cc', alpha=0.3)
        ax2.axhline(y=DEVICE_CONSUMPTION_W, color='red', linestyle='--', 
                    label=f'Device Consumption ({DEVICE_CONSUMPTION_W}W)')
        ax2.set_ylabel('Power (Watts)')
        ax2.set_title('Harvested Power vs Device Consumption')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # Graph 3: Battery Storage
        ax3.plot(day_df.index, battery_levels, 
                 color='#00cc66', linewidth=2, label='Battery Level')
        ax3.fill_between(day_df.index, battery_levels, 
                         color='#00cc66', alpha=0.3)
        ax3.axhline(y=BATTERY_CAPACITY_WH, color='red', linestyle='--', 
                    label=f'Max Capacity ({BATTERY_CAPACITY_WH} Wh)')
        ax3.set_ylabel('Battery Level (Wh)')
        ax3.set_xlabel('Time of Day')
        ax3.set_title('Battery Energy Storage')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        
        if save:
            filename = f"dashboard_{query_date_str.replace('/', '-')}.png"
            filepath = os.path.join(OUTPUT_DIR, filename)
            plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
            print(f">> Dashboard saved to '{filepath}'")
        
        plt.show()
    
    def plot_energy_summary(self, df, num_days=7, save=True):
        """
        Plot energy summary for multiple days.
        
        Args:
            df: Full DataFrame with datetime index
            num_days: Number of days to summarize
            save: Whether to save the plot
        """
        # Get unique dates
        dates = df.index.date
        unique_dates = sorted(set(dates))[-num_days:]
        
        daily_energy = []
        date_labels = []
        
        for date in unique_dates:
            day_data = df[df.index.date == date]
            energy = day_data['Irradiance'].sum() * PLANT_AREA_M2 * (1/60)
            daily_energy.append(energy)
            date_labels.append(date.strftime('%m/%d'))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(date_labels, daily_energy, color='#4CAF50', edgecolor='black')
        
        # Add value labels on bars
        for bar, energy in zip(bars, daily_energy):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{energy:.2f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Energy Harvested (Wh)')
        ax.set_title(f'Daily Energy Harvest Summary (Last {num_days} Days)', 
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(OUTPUT_DIR, 'energy_summary.png')
            plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
            print(f">> Energy summary saved to '{filepath}'")
        
        plt.show()

    def plot_comprehensive_method_comparison(self, save=True):
        """
        Plot accuracy and error-rate comparison across traditional, ML, and CNN methods.

        Args:
            save: Whether to save the plot
        """
        methods = [
            'EWMA', 'WCMA', 'Pro-\nEnergy', 'Mod\nPro-Energy',
            'EENA\n(NARNET)', 'PADC-MAC\n(NARNET)', 'Your CNN\n(Final)'
        ]

        # Accuracy (%) values: low/mid/high ranges used for error bars.
        accuracy_mid = [10, 15, 16, 20, 99.62, 88.5, 90]
        accuracy_low = [5, 10, 7, 11, 99.62, 88, 88]
        accuracy_high = [15, 20, 25, 29, 99.62, 89, 92]

        colors = ['#FF6B6B', '#FFA500', '#FFD93D', '#6BCB77', '#4D96FF', '#9D4EDD', '#00D9FF']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Subplot 1: Accuracy Comparison
        x_pos = np.arange(len(methods))
        bars = ax1.bar(
            x_pos, accuracy_mid, color=colors, alpha=0.8,
            edgecolor='black', linewidth=1.5
        )

        errors_low = [acc_mid - acc_low for acc_mid, acc_low in zip(accuracy_mid, accuracy_low)]
        errors_high = [acc_high - acc_mid for acc_mid, acc_high in zip(accuracy_mid, accuracy_high)]
        ax1.errorbar(
            x_pos, accuracy_mid, yerr=[errors_low, errors_high], fmt='none',
            ecolor='black', capsize=5, capthick=2
        )

        for bar, acc in zip(bars, accuracy_mid):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0, height + 2,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold'
            )

        ax1.set_ylabel('Prediction Accuracy (%)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Prediction Method', fontsize=14, fontweight='bold')
        ax1.set_title('Accuracy Comparison: Traditional vs ML vs Your CNN', fontsize=16, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(methods, fontsize=11)
        ax1.set_ylim(0, 110)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.axhline(y=50, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Minimum Acceptable (50%)')
        ax1.axhline(y=90, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Excellence Threshold (90%)')
        ax1.legend(fontsize=10)

        for i in [4, 5, 6]:
            bars[i].set_edgecolor('gold')
            bars[i].set_linewidth(3)

        # Subplot 2: Error Rate Comparison (lower is better)
        error_rates = [90, 85, 84, 80, 0.38, 11.5, 10]
        bars2 = ax2.bar(
            x_pos, error_rates, color=colors, alpha=0.8,
            edgecolor='black', linewidth=1.5
        )

        for bar, err in zip(bars2, error_rates):
            height = bar.get_height()
            label = f'{err:.1f}%' if err > 1 else f'{err:.2f}%'
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0, height + 2,
                label, ha='center', va='bottom', fontsize=11, fontweight='bold'
            )

        ax2.set_ylabel('Error Rate (%) - Lower is Better', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Prediction Method', fontsize=14, fontweight='bold')
        ax2.set_title('Error Rate Comparison (Inverted Scale)', fontsize=16, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(methods, fontsize=11)
        ax2.set_ylim(0, 100)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        ax2.invert_yaxis()

        for i in [4, 5, 6]:
            bars2[i].set_edgecolor('gold')
            bars2[i].set_linewidth(3)

        plt.tight_layout()

        if save:
            filepath = os.path.join(OUTPUT_DIR, 'comprehensive_comparison_all_methods.png')
            plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
            print(f">> Comprehensive comparison graph saved to '{filepath}'")

        plt.show()
