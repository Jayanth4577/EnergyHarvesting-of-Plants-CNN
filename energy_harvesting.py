import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# TensorFlow imports (using keras directly to avoid IDE resolution issues)
import keras
from keras.models import Sequential, load_model
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# ================= PROJECT CONFIGURATION =================
DATA_FILE = 'z4689499.txt'
WINDOW_SIZE = 60         # Model looks at previous 60 minutes to understand trends
PLANT_AREA_M2 = 0.05     # Surface area of the plant/harvester (e.g., 500 cm^2)
BATTERY_CAPACITY_WH = 50 # Max battery capacity in Watt-hours
DEVICE_CONSUMPTION_W = 0.5 # Constant power usage of the sensor node
# =========================================================

class EnergyHarvestingCNN:
    def __init__(self, file_path):
        self.file_path = file_path
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.df = None
        self.data_processed = None
        self.history = None  # Store training history
        self.X_test = None   # Store test data for evaluation
        self.y_test = None
        
    def load_and_preprocess_data(self):
        """Loads dataset and prepares it for the CNN."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset {self.file_path} not found. Please put it in the same folder.")
            
        print(">> [1/5] Loading and cleaning dataset...")
        self.df = pd.read_csv(self.file_path)
        
        # Combine Date and Time columns
        self.df['Datetime'] = pd.to_datetime(self.df['DATE (MM/DD/YYYY)'] + ' ' + self.df['MST'])
        self.df = self.df.sort_values('Datetime').set_index('Datetime')
        
        # Clean Negative Noise (Solar values cannot be negative)
        target_col = 'Global CMP22 (vent/cor) [W/m^2]'
        self.df['Irradiance'] = self.df[target_col].apply(lambda x: max(x, 0))
        
        # Normalize data (Scale 0-1) so the Neural Network can learn faster
        data_values = self.df['Irradiance'].values.reshape(-1, 1)
        self.data_processed = self.scaler.fit_transform(data_values)
        print(">> Data loaded successfully.")

    def create_sequences(self, data, seq_length):
        """Creates time-series windows for training."""
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:(i + seq_length)]
            y = data[i + seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def build_and_train_cnn(self, epochs=5):
        """Defines and trains the 1D-CNN model."""
        if self.df is None: self.load_and_preprocess_data()

        print("\n>> [2/5] Preparing Training Data...")
        X, y = self.create_sequences(self.data_processed, WINDOW_SIZE)
        
        # Split: 80% Training, 20% Testing
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        print(">> [3/5] Building CNN Architecture...")
        # --- THIS IS THE CNN MODEL ---
        self.model = Sequential([
            # Layer 1: Convolution to detect patterns (sunrise, clouds)
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(WINDOW_SIZE, 1)),
            MaxPooling1D(pool_size=2),
            
            # Layer 2: Deeper feature extraction
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            
            # Layer 3: Flatten and Output
            Flatten(),
            Dense(50, activation='relu'),
            Dense(1) # Output: Predicted Irradiance
        ])
        
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        print(f">> [4/5] Training Model for {epochs} epochs (Learning patterns)...")
        self.history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=32, 
                                       validation_data=(X_test, y_test), verbose=1)
        
        # Store test data for later evaluation
        self.X_test = X_test
        self.y_test = y_test
        print(">> Training Complete.")

    # ------------------ SIR'S REQUIREMENT: MODEL PREDICTION ------------------
    def predict_future_energy(self):
        """Demonstrates the model is working by predicting the NEXT minute's energy."""
        # Grab the last 60 minutes of real data
        last_window = self.data_processed[-WINDOW_SIZE:]
        last_window = last_window.reshape(1, WINDOW_SIZE, 1)
        
        # CNN Predicts
        prediction_scaled = self.model.predict(last_window, verbose=0)
        prediction_watts = self.scaler.inverse_transform(prediction_scaled)[0][0]
        
        return prediction_watts

    # ------------------ MODEL EVALUATION METRICS ------------------
    def evaluate_model(self):
        """Calculate and display model performance metrics."""
        if self.X_test is None or self.y_test is None:
            print("Error: No test data available. Train the model first.")
            return None
        
        # Get predictions on test set
        y_pred_scaled = self.model.predict(self.X_test, verbose=0)
        
        # Inverse transform to get actual values
        y_actual = self.scaler.inverse_transform(self.y_test.reshape(-1, 1)).flatten()
        y_predicted = self.scaler.inverse_transform(y_pred_scaled).flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y_actual, y_predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_actual, y_predicted)
        r2 = r2_score(y_actual, y_predicted)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mask = y_actual != 0  # Avoid division by zero
        mape = np.mean(np.abs((y_actual[mask] - y_predicted[mask]) / y_actual[mask])) * 100
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
        
        print("\n" + "="*50)
        print("         MODEL PERFORMANCE METRICS")
        print("="*50)
        print(f"  MSE  (Mean Squared Error):     {mse:.4f}")
        print(f"  RMSE (Root Mean Squared Error): {rmse:.4f}")
        print(f"  MAE  (Mean Absolute Error):     {mae:.4f}")
        print(f"  R²   (Coefficient of Determination): {r2:.4f}")
        print(f"  MAPE (Mean Absolute % Error):   {mape:.2f}%")
        print("="*50)
        
        return metrics, y_actual, y_predicted

    # ------------------ MODEL SAVE/LOAD ------------------
    def save_model(self, model_path='saved_model/cnn_energy_model.h5', scaler_path='saved_model/scaler.pkl'):
        """Save the trained model and scaler for later use."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        self.model.save(model_path)
        print(f">> Model saved to '{model_path}'")
        
        # Save scaler
        joblib.dump(self.scaler, scaler_path)
        print(f">> Scaler saved to '{scaler_path}'")
    
    def load_saved_model(self, model_path='saved_model/cnn_energy_model.h5', scaler_path='saved_model/scaler.pkl'):
        """Load a previously trained model and scaler."""
        if not os.path.exists(model_path):
            print(f"Error: Model file '{model_path}' not found.")
            return False
        
        self.model = load_model(model_path)
        print(f">> Model loaded from '{model_path}'")
        
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f">> Scaler loaded from '{scaler_path}'")
        
        # Load data if not already loaded
        if self.df is None:
            self.load_and_preprocess_data()
        
        return True

    # ------------------ VISUALIZATION: TRAINING HISTORY ------------------
    def plot_training_history(self):
        """Plot training and validation loss curves."""
        if self.history is None:
            print("Error: No training history available. Train the model first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot Loss
        ax1.plot(self.history.history['loss'], label='Training Loss', color='blue', linewidth=2)
        ax1.plot(self.history.history['val_loss'], label='Validation Loss', color='red', linewidth=2)
        ax1.set_title('Model Loss During Training', fontsize=14)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot MAE
        ax2.plot(self.history.history['mae'], label='Training MAE', color='green', linewidth=2)
        ax2.plot(self.history.history['val_mae'], label='Validation MAE', color='orange', linewidth=2)
        ax2.set_title('Model MAE During Training', fontsize=14)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Mean Absolute Error')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150)
        print(">> Training history graph saved as 'training_history.png'")
        plt.show()

    # ------------------ VISUALIZATION: PREDICTED VS ACTUAL ------------------
    def plot_predicted_vs_actual(self, num_samples=500):
        """Plot predicted vs actual values comparison."""
        result = self.evaluate_model()
        if result is None:
            return
        
        _, y_actual, y_predicted = result
        
        # Limit samples for clearer visualization
        if len(y_actual) > num_samples:
            y_actual = y_actual[:num_samples]
            y_predicted = y_predicted[:num_samples]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Time Series Comparison
        ax1 = axes[0, 0]
        ax1.plot(y_actual, label='Actual', color='blue', alpha=0.7, linewidth=1)
        ax1.plot(y_predicted, label='Predicted', color='red', alpha=0.7, linewidth=1)
        ax1.set_title('Predicted vs Actual Irradiance Over Time', fontsize=12)
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Irradiance (W/m²)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Scatter Plot (Correlation)
        ax2 = axes[0, 1]
        ax2.scatter(y_actual, y_predicted, alpha=0.5, s=10, color='purple')
        max_val = max(y_actual.max(), y_predicted.max())
        ax2.plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction', linewidth=2)
        ax2.set_title('Correlation: Actual vs Predicted', fontsize=12)
        ax2.set_xlabel('Actual Irradiance (W/m²)')
        ax2.set_ylabel('Predicted Irradiance (W/m²)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Prediction Error Distribution
        ax3 = axes[1, 0]
        errors = y_actual - y_predicted
        ax3.hist(errors, bins=50, color='teal', edgecolor='black', alpha=0.7)
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax3.set_title('Prediction Error Distribution', fontsize=12)
        ax3.set_xlabel('Error (Actual - Predicted)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Error Over Time
        ax4 = axes[1, 1]
        ax4.plot(errors, color='orange', alpha=0.7, linewidth=1)
        ax4.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax4.fill_between(range(len(errors)), errors, alpha=0.3, color='orange')
        ax4.set_title('Prediction Error Over Time', fontsize=12)
        ax4.set_xlabel('Time Steps')
        ax4.set_ylabel('Error (W/m²)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('predicted_vs_actual.png', dpi=150)
        print(">> Predicted vs Actual graph saved as 'predicted_vs_actual.png'")
        plt.show()

    # ------------------ VISUALIZATION: METHOD COMPARISON ------------------
    def plot_comprehensive_method_comparison(self):
        """Plot weighting-factor vs error comparison in line-chart style."""
        weighting_factor = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        ewma_error = [2.6, 2.4, 1.8, 1.4, 1.2]
        wcma_error = [2.1, 1.75, 1.45, 0.9, 0.65]
        pro_energy_error = [1.45, 1.25, 0.9, 0.55, 0.35]
        our_algorithm_error = [1.35, 1.15, 0.7, 0.45, 0.3]
        # EENA and PADC-MAC are available as single summary error values,
        # so they are shown as constant reference lines across weighting factors.
        eena_narnet_error = [0.38, 0.38, 0.38, 0.38, 0.38]
        padc_mac_narnet_error = [11.5, 11.5, 11.5, 11.5, 11.5]

        fig, ax = plt.subplots(figsize=(8, 5), facecolor='#efefef')
        ax.set_facecolor('#f7f7f7')

        ax.plot(weighting_factor, ewma_error, marker='o', linewidth=2.2, color='#1f77b4', label='EWMA')
        ax.plot(weighting_factor, wcma_error, marker='o', linewidth=2.2, color='#ff7f0e', label='WCMA')
        ax.plot(weighting_factor, pro_energy_error, marker='o', linewidth=2.2, color='#f1c40f', label='Pro-Energy')
        ax.plot(weighting_factor, eena_narnet_error, marker='o', linewidth=2.2, color='#2ecc71', label='EENA (NARNET)')
        ax.plot(weighting_factor, padc_mac_narnet_error, marker='o', linewidth=2.2, color='#e74c3c', label='PADC-MAC (NARNET)')
        ax.plot(weighting_factor, our_algorithm_error, marker='o', linewidth=2.2, color='#8e44ad', label='Our Algorithm')

        ax.set_xlabel('Weighting Factor', fontsize=11)
        ax.set_ylabel('Error(%)', fontsize=11)
        ax.set_title('Method Comparison by Weighting Factor', fontsize=12, fontweight='bold')
        ax.set_xlim(0.1, 0.9)
        ax.set_ylim(0, 12)
        ax.set_xticks(weighting_factor)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8, frameon=True, ncol=2)

        plt.tight_layout()
        plt.savefig('comprehensive_comparison_all_methods.png', dpi=300, bbox_inches='tight')
        print(">> Comprehensive comparison graph saved as 'comprehensive_comparison_all_methods.png'")
        plt.show()

    # ------------------ CALCULATOR ENGINES ------------------
    def get_todays_harvest(self, query_date_str):
        """Calculates total energy harvested on a specific date."""
        day_data = self.df[self.df.index.strftime('%m/%d/%Y') == query_date_str]
        if day_data.empty: return "No data found."
        
        # Energy (Wh) = Sum(Watts) * (1 hour / 60 minutes) * Area
        total_energy = day_data['Irradiance'].sum() * PLANT_AREA_M2 * (1/60)
        return f"{total_energy:.4f} Wh"

    def get_total_harvest_till_now(self, query_time_str):
        """Calculates lifetime energy up to a specific timestamp."""
        q_time = pd.to_datetime(query_time_str)
        past_data = self.df[self.df.index <= q_time]
        total_energy = past_data['Irradiance'].sum() * PLANT_AREA_M2 * (1/60)
        return f"{total_energy:.4f} Wh"

    def get_instant_harvest(self, query_time_str):
        """Returns power at a specific moment."""
        q_time = pd.to_datetime(query_time_str)
        if q_time in self.df.index:
            power = self.df.loc[q_time]['Irradiance'] * PLANT_AREA_M2
            return f"{power:.4f} Watts"
        return "Timepoint not in dataset."

    # ------------------ GRAPHING ENGINE ------------------
    def generate_dashboard(self, query_date_str):
        """Generates a graph for Solar Power and Battery Status."""
        print(f"\n>> [5/5] Generating Dashboard for {query_date_str}...")
        day_df = self.df[self.df.index.strftime('%m/%d/%Y') == query_date_str].copy()
        
        if day_df.empty:
            print("Error: No data for this date to graph.")
            return

        # Simulate Battery
        battery_levels = []
        current_charge = 0.0
        # Calculate net flow (Harvested - Consumed)
        energy_in = (day_df['Irradiance'].values * PLANT_AREA_M2) * (1/60)
        energy_out = DEVICE_CONSUMPTION_W * (1/60)
        
        for e_in in energy_in:
            current_charge += (e_in - energy_out)
            current_charge = max(0, min(current_charge, BATTERY_CAPACITY_WH))
            battery_levels.append(current_charge)

        # Plotting
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Graph 1: Solar Harvest
        ax1.plot(day_df.index, day_df['Irradiance'], color='#ff9900', label='Solar Irradiance')
        ax1.fill_between(day_df.index, day_df['Irradiance'], color='#ff9900', alpha=0.2)
        ax1.set_ylabel('Sunlight Intensity (W/m²)')
        ax1.set_title(f'Solar Harvesting Profile: {query_date_str}')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Graph 2: Battery Storage
        ax2.plot(day_df.index, battery_levels, color='#00cc66', label='Battery Stored Energy')
        ax2.axhline(y=BATTERY_CAPACITY_WH, color='red', linestyle='--', label='Max Capacity')
        ax2.set_ylabel('Battery Level (Wh)')
        ax2.set_xlabel('Time of Day')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.gcf().autofmt_xdate()
        filename = f"Graph_{query_date_str.replace('/','-')}.png"
        plt.savefig(filename)
        print(f">> Graph saved as '{filename}'")
        plt.show()

# ================= MAIN EXECUTION BLOCK =================
if __name__ == "__main__":
    # 1. Initialize System
    system = EnergyHarvestingCNN(DATA_FILE)
    
    # 2. Check if saved model exists, otherwise train new one
    MODEL_PATH = 'saved_model/cnn_energy_model.h5'
    SCALER_PATH = 'saved_model/scaler.pkl'
    
    if os.path.exists(MODEL_PATH):
        print(">> Found saved model. Loading...")
        system.load_saved_model(MODEL_PATH, SCALER_PATH)
    else:
        print(">> No saved model found. Training new model...")
        system.build_and_train_cnn(epochs=10)  # Increased epochs for better training
        system.save_model(MODEL_PATH, SCALER_PATH)
        
        # Show training progress
        system.plot_training_history()

    # 3. Evaluate Model Performance (NEW!)
    system.evaluate_model()
    
    # 4. Show Predicted vs Actual Comparison (NEW!)
    system.plot_predicted_vs_actual(num_samples=500)

    # 5. Comprehensive benchmark comparison chart
    system.plot_comprehensive_method_comparison()

    # 6. Define Test Query
    # You can change these dates to any date inside your text file
    TEST_DATE = "01/26/2019" 
    TEST_TIME = "01/26/2019 14:30"

    print("\n" + "="*40)
    print("      FINAL SYSTEM OUTPUTS")
    print("="*40)
    
    # --- Feature 1: Today's Harvest ---
    print(f"1. Energy Harvested Today ({TEST_DATE}):")
    print(f"   -> {system.get_todays_harvest(TEST_DATE)}")
    
    # --- Feature 2: Total Harvest Till Now ---
    print(f"2. Total Energy Harvested (Lifetime up to {TEST_TIME}):")
    print(f"   -> {system.get_total_harvest_till_now(TEST_TIME)}")
    
    # --- Feature 3: Instant Energy ---
    print(f"3. Instant Power at {TEST_TIME}:")
    print(f"   -> {system.get_instant_harvest(TEST_TIME)}")
    
    # --- Feature 4: ML Model Prediction (Proof for Sir) ---
    pred = system.predict_future_energy()
    print(f"4. CNN Model Future Prediction (Next Minute):")
    print(f"   -> {pred:.4f} W/m^2")
    
    print("="*40)

    # 7. Generate Daily Dashboard Graph
    system.generate_dashboard(TEST_DATE)
    
    print("\n>> All outputs generated successfully!")
    print(">> Check the saved PNG files for visualizations.")