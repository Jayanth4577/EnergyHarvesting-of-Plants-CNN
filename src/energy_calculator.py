"""
Energy Calculator Module
Handles all energy harvesting calculations and predictions.
"""

import pandas as pd
import numpy as np

from .config import PLANT_AREA_M2, WINDOW_SIZE


class EnergyCalculator:
    """Calculates energy harvesting metrics from irradiance data."""
    
    def __init__(self, df, model=None, scaler=None, data_processed=None,
                 data_features=None, bias_offset=0.0):
        """
        Initialize calculator with data and optional model.
        
        Args:
            df: DataFrame with datetime index and 'Irradiance' column
            model: Trained CNN model (optional, for predictions)
            scaler: Fitted scaler (optional, for predictions)
            data_processed: Normalized irradiance data (optional)
            data_features: 2-feature array [irradiance, rate-of-change] (for CNN input)
            bias_offset: Bias correction value from calibration (W/m²)
        """
        self.df = df
        self.model = model
        self.scaler = scaler
        self.data_processed = data_processed
        self.data_features = data_features
        self.bias_offset = bias_offset
    
    def get_todays_harvest(self, query_date_str):
        """
        Calculate total energy harvested on a specific date.
        
        Args:
            query_date_str: Date string in format 'MM/DD/YYYY'
            
        Returns:
            Energy harvested in Watt-hours
        """
        day_data = self.df[self.df.index.strftime('%m/%d/%Y') == query_date_str]
        
        if day_data.empty:
            return None, "No data found for this date."
        
        # Energy (Wh) = Sum(W/m² × Area) × (1 hour / 60 minutes)
        total_energy = day_data['Irradiance'].sum() * PLANT_AREA_M2 * (1/60)
        
        return total_energy, f"{total_energy:.4f} Wh"
    
    def get_total_harvest_till_now(self, query_time_str):
        """
        Calculate cumulative energy harvested up to a specific time.
        
        Args:
            query_time_str: Datetime string (e.g., 'MM/DD/YYYY HH:MM')
            
        Returns:
            Total energy harvested in Watt-hours
        """
        q_time = pd.to_datetime(query_time_str)
        past_data = self.df[self.df.index <= q_time]
        
        if past_data.empty:
            return None, "No data found before this time."
        
        total_energy = past_data['Irradiance'].sum() * PLANT_AREA_M2 * (1/60)
        
        return total_energy, f"{total_energy:.4f} Wh"
    
    def get_instant_power(self, query_time_str):
        """
        Get instantaneous power at a specific timestamp.
        
        Args:
            query_time_str: Datetime string (e.g., 'MM/DD/YYYY HH:MM')
            
        Returns:
            Power in Watts at that moment
        """
        q_time = pd.to_datetime(query_time_str)
        
        # Try exact match
        if q_time in self.df.index:
            irradiance = self.df.loc[q_time]['Irradiance']
            power = irradiance * PLANT_AREA_M2
            return power, f"{power:.4f} Watts"
        
        # Try nearest time
        try:
            nearest_idx = self.df.index.get_indexer([q_time], method='nearest')[0]
            if nearest_idx >= 0:
                irradiance = self.df.iloc[nearest_idx]['Irradiance']
                power = irradiance * PLANT_AREA_M2
                actual_time = self.df.index[nearest_idx]
                return power, f"{power:.4f} Watts (nearest: {actual_time})"
        except:
            pass
        
        return None, "Timepoint not in dataset."
    
    def predict_future_energy(self):
        """
        Use CNN model to predict the next minute's irradiance.
        
        Returns:
            Predicted irradiance in W/m²
        """
        if self.model is None or self.scaler is None:
            return None, "Model or scaler not available."
        
        if self.data_processed is None or self.data_features is None:
            return None, "Processed data not available."
        
        # Grab the last WINDOW_SIZE minutes of 2-feature data
        last_window = self.data_features[-WINDOW_SIZE:]
        last_window = last_window.reshape(1, WINDOW_SIZE, 2)
        
        # CNN Predicts
        prediction_scaled = self.model.predict(last_window, verbose=0)
        prediction_watts = self.scaler.inverse_transform(prediction_scaled)[0][0]
        
        # Apply bias calibration
        prediction_watts = prediction_watts + self.bias_offset
        
        # Ensure non-negative
        prediction_watts = max(0, prediction_watts)
        
        # Post-Processing Safety Net (physics rule)
        # Solar irradiance is physically zero at night; force prediction to 0
        # if the latest timestamp falls outside daylight hours (before 6 AM or after 8 PM).
        last_hour = self.df.index[-1].hour
        if last_hour < 6 or last_hour >= 20:
            prediction_watts = 0.0
        
        return prediction_watts, f"{prediction_watts:.4f} W/m²"
    
    def predict_future_power(self):
        """
        Predict next minute's harvested power (in Watts).
        
        Returns:
            Predicted power in Watts
        """
        irradiance, _ = self.predict_future_energy()
        
        if irradiance is None:
            return None, "Prediction failed."
        
        power = irradiance * PLANT_AREA_M2
        return power, f"{power:.4f} Watts"
    
    def get_daily_statistics(self, query_date_str):
        """
        Get comprehensive statistics for a specific day.
        
        Args:
            query_date_str: Date string in format 'MM/DD/YYYY'
            
        Returns:
            Dictionary with daily statistics
        """
        day_data = self.df[self.df.index.strftime('%m/%d/%Y') == query_date_str]
        
        if day_data.empty:
            return None
        
        irradiance = day_data['Irradiance']
        power = irradiance * PLANT_AREA_M2
        
        stats = {
            'date': query_date_str,
            'total_energy_wh': irradiance.sum() * PLANT_AREA_M2 * (1/60),
            'peak_irradiance': irradiance.max(),
            'peak_power': power.max(),
            'avg_irradiance': irradiance.mean(),
            'avg_power': power.mean(),
            'daylight_hours': len(irradiance[irradiance > 10]) / 60,  # Hours with >10 W/m²
            'num_readings': len(irradiance)
        }
        
        return stats
    
    def print_daily_report(self, query_date_str):
        """Print a formatted daily report."""
        stats = self.get_daily_statistics(query_date_str)
        
        if stats is None:
            print(f"No data available for {query_date_str}")
            return
        
        print("\n" + "="*55)
        print(f"      DAILY ENERGY REPORT: {stats['date']}")
        print("="*55)
        print(f"  Total Energy Harvested:    {stats['total_energy_wh']:.4f} Wh")
        print(f"  Peak Irradiance:           {stats['peak_irradiance']:.2f} W/m²")
        print(f"  Peak Power:                {stats['peak_power']:.4f} W")
        print(f"  Average Irradiance:        {stats['avg_irradiance']:.2f} W/m²")
        print(f"  Average Power:             {stats['avg_power']:.4f} W")
        print(f"  Effective Daylight Hours:  {stats['daylight_hours']:.2f} hours")
        print(f"  Data Points:               {stats['num_readings']}")
        print("="*55 + "\n")
