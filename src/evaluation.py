"""
Model Evaluation Module
Calculates and displays model performance metrics.

Features:
- Standard metrics (MSE, RMSE, MAE, R², MAPE)
- Bias calibration: measures and corrects systematic prediction bias
"""

import numpy as np
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .config import BIAS_SAVE_PATH


class ModelEvaluator:
    """Evaluates CNN model performance with various metrics."""
    
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
        self.metrics = None
        self.y_actual = None
        self.y_predicted = None
        self.bias_offset = 0.0   # calibrated bias correction
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test data and calculate metrics.
        
        Args:
            X_test: Test input sequences
            y_test: Test target values (scaled)
            
        Returns:
            Dictionary of metrics
        """
        # Get predictions on test set
        y_pred_scaled = self.model.predict(X_test)
        
        # Inverse transform to get actual values
        self.y_actual = self.scaler.inverse_transform(
            y_test.reshape(-1, 1)
        ).flatten()
        self.y_predicted = self.scaler.inverse_transform(
            y_pred_scaled
        ).flatten()
        
        # Calculate metrics
        mse = mean_squared_error(self.y_actual, self.y_predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_actual, self.y_predicted)
        r2 = r2_score(self.y_actual, self.y_predicted)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        # Only include values > 10 W/m² (exclude night-time near-zero values)
        # Near-zero values cause inflated MAPE due to division
        threshold = 10.0  # Minimum irradiance to include in MAPE
        mask = self.y_actual > threshold
        if mask.sum() > 0:
            mape = np.mean(
                np.abs((self.y_actual[mask] - self.y_predicted[mask]) / self.y_actual[mask])
            ) * 100
        else:
            mape = 0.0
        
        self.metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
        
        return self.metrics
    
    def print_metrics(self):
        """Display metrics in a formatted table."""
        if self.metrics is None:
            print("Error: No metrics available. Run evaluate() first.")
            return
        
        print("\n" + "="*55)
        print("           MODEL PERFORMANCE METRICS")
        print("="*55)
        print(f"  MSE  (Mean Squared Error):          {self.metrics['MSE']:.4f}")
        print(f"  RMSE (Root Mean Squared Error):     {self.metrics['RMSE']:.4f}")
        print(f"  MAE  (Mean Absolute Error):         {self.metrics['MAE']:.4f}")
        print(f"  R²   (Coefficient of Determination): {self.metrics['R2']:.4f}")
        print(f"  MAPE (Mean Absolute % Error):       {self.metrics['MAPE']:.2f}%")
        print("="*55)
        
        # Interpretation
        print("\n>> INTERPRETATION:")
        if self.metrics['R2'] > 0.9:
            print("   ✅ Excellent! R² > 0.9 means model explains >90% of variance")
        elif self.metrics['R2'] > 0.7:
            print("   ✓ Good. R² > 0.7 indicates a reliable model")
        else:
            print("   ⚠ Consider increasing epochs or tuning hyperparameters")
        
        if self.metrics['MAPE'] < 10:
            print("   ✅ MAPE < 10% indicates highly accurate predictions")
        elif self.metrics['MAPE'] < 20:
            print("   ✓ MAPE < 20% is acceptable for this application")
        else:
            print("   ⚠ MAPE > 20% - model may need improvement")
        
        print()
    
    def get_predictions(self):
        """Return actual and predicted values."""
        return self.y_actual, self.y_predicted
    
    def get_errors(self):
        """Calculate prediction errors."""
        if self.y_actual is None or self.y_predicted is None:
            return None
        return self.y_actual - self.y_predicted
    
    # ── Bias Calibration ─────────────────────────
    def calibrate_bias(self):
        """
        Compute the mean prediction bias on the test set and store it.
        bias_offset = mean(actual - predicted)
        Positive offset means model under-predicts on average.
        """
        if self.y_actual is None:
            print(">> Run evaluate() before calibrating bias.")
            return
        
        errors = self.y_actual - self.y_predicted
        self.bias_offset = float(np.mean(errors))
        
        print(f"\n>> Bias Calibration:")
        print(f"   Raw mean bias: {self.bias_offset:+.4f} W/m²")
        print(f"   (Positive = model under-predicts, Negative = model over-predicts)")
        print(f"   All future predictions will be corrected by {self.bias_offset:+.4f} W/m²")
        
        return self.bias_offset
    
    def save_bias(self, path=BIAS_SAVE_PATH):
        """Save bias offset to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.bias_offset, path)
        print(f">> Bias offset saved to '{path}'")
    
    def load_bias(self, path=BIAS_SAVE_PATH):
        """Load bias offset from disk."""
        if os.path.exists(path):
            self.bias_offset = joblib.load(path)
            print(f">> Bias offset loaded: {self.bias_offset:+.4f} W/m²")
        else:
            self.bias_offset = 0.0
            print(">> No saved bias found, using 0.")
        return self.bias_offset
    
    def apply_bias_correction(self, predictions):
        """Apply bias correction to raw predictions (real-scale W/m²)."""
        corrected = predictions + self.bias_offset
        return np.maximum(corrected, 0)  # can't go negative
