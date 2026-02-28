"""
Data Preprocessing Module
Handles loading, cleaning, and preparing data for the CNN model.

Features:
- Hour-based nighttime zeroing (physical rule: no sun → irradiance = 0)
- Threshold-based noise mask- Rate-of-change as 2nd input feature (helps model learn transitions)- Low-light data augmentation (oversamples dawn/dusk transitions)
- Stratified train/test split (ensures both sets see day + night)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os

from .config import (
    DATE_COLUMN, TIME_COLUMN, TARGET_COLUMN, WINDOW_SIZE
)


class DataPreprocessor:
    """Handles all data loading and preprocessing operations."""
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.df = None
        self.data_processed = None
        self.data_features = None  # 2-feature array: [irradiance, rate-of-change]
        
    def load_data(self):
        """Load raw data from file."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"Dataset '{self.file_path}' not found. "
                "Please put it in the same folder."
            )
        
        print(">> Loading dataset...")
        self.df = pd.read_csv(self.file_path)
        print(f">> Loaded {len(self.df)} records.")
        return self.df
    
    def clean_data(self):
        """Clean and preprocess the raw data."""
        if self.df is None:
            self.load_data()
        
        print(">> Cleaning data...")
        
        # Combine Date and Time columns into datetime
        self.df['Datetime'] = pd.to_datetime(
            self.df[DATE_COLUMN] + ' ' + self.df[TIME_COLUMN]
        )
        self.df = self.df.sort_values('Datetime').set_index('Datetime')
        
        # Clean Negative Noise (Solar values cannot be negative)
        self.df['Irradiance'] = self.df[TARGET_COLUMN].apply(lambda x: max(x, 0))
        
        # Nighttime Noise Mask (in-memory only; z4689499.txt is NOT modified)
        # Two-pronged approach:
        #   1) Hour-based: zero out anything before 6 AM or after 8 PM (sun is down)
        #   2) Threshold-based: any remaining reading < 10 W/m² is sensor noise
        hour = self.df.index.hour
        self.df.loc[(hour < 6) | (hour >= 20), 'Irradiance'] = 0.0
        self.df['Irradiance'] = self.df['Irradiance'].where(
            self.df['Irradiance'] >= 10.0, 0.0
        )
        
        print(">> Data cleaned successfully (nighttime zeroed, noise removed).")
        return self.df
    
    def normalize_data(self):
        """Normalize data to 0-1 range and compute rate-of-change feature."""
        if self.df is None:
            self.clean_data()
        
        print(">> Normalizing data...")
        data_values = self.df['Irradiance'].values.reshape(-1, 1)
        self.data_processed = self.scaler.fit_transform(data_values)
        
        # Compute rate-of-change (temporal gradient) as 2nd feature
        # This tells the model HOW FAST irradiance is changing
        irr_flat = self.data_processed.flatten()
        delta = np.diff(irr_flat, prepend=irr_flat[0])  # first delta = 0
        delta = delta.reshape(-1, 1)
        
        # Stack: column 0 = irradiance, column 1 = rate-of-change
        self.data_features = np.concatenate(
            [self.data_processed, delta], axis=1
        )
        
        print(f">> Normalization complete. Features shape: {self.data_features.shape}")
        print(">> Feature 0: irradiance (normalized), Feature 1: rate-of-change")
        return self.data_processed
    
    def create_sequences(self, seq_length=WINDOW_SIZE):
        """
        Creates time-series windows for training.
        Uses 2-feature input (irradiance + rate-of-change).
        
        Args:
            seq_length: Number of past time steps to use as input
            
        Returns:
            X: Input sequences (shape: samples, seq_length, 2)
            y: Target values (shape: samples, 1)  [irradiance only]
        """
        if self.data_features is None:
            self.normalize_data()
        
        print(f">> Creating sequences with window size {seq_length} (2 features)...")
        
        xs, ys = [], []
        for i in range(len(self.data_features) - seq_length):
            x = self.data_features[i:(i + seq_length)]   # shape: (seq_length, 2)
            y = self.data_processed[i + seq_length]       # target: irradiance only
            xs.append(x)
            ys.append(y)
        
        X = np.array(xs)
        y = np.array(ys)
        
        print(f">> Created {len(X)} sequences. X shape: {X.shape}, y shape: {y.shape}")
        return X, y
    
    def prepare_train_test_split(self, X, y, split_ratio=0.8, shuffle=False):
        """
        Stratified train/test split.
        Ensures both train and test contain proportional amounts of
        night (y≈0), low-light, and full-sun samples.
        Falls back to sequential split if stratification isn't possible.
        """
        # Create irradiance bins for stratification
        # 0 = night, 1 = low-light, 2 = medium, 3 = high
        bins = np.digitize(y.flatten(), bins=[0.001, 0.05, 0.3])
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=(1 - split_ratio),
                shuffle=True,
                stratify=bins,
                random_state=42
            )
            split_type = "stratified"
        except ValueError:
            # Fallback to sequential if bins are too small
            split = int(len(X) * split_ratio)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            split_type = "sequential (stratification fallback)"
        
        print(f">> Train set: {len(X_train)} samples ({split_type} split)")
        print(f">> Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def augment_low_light(self, X, y, threshold=0.05, augment_factor=1):
        """
        Low-light data augmentation.
        Duplicates sequences where the target is in the low-light range
        (normalised value < threshold) with small Gaussian noise added.
        This forces the model to see more dawn/dusk/night transitions.
        
        Args:
            X: Input sequences
            y: Target values
            threshold: Normalised low-light threshold
            augment_factor: How many copies to add
            
        Returns:
            Augmented X, y arrays
        """
        low_mask = y.flatten() < threshold
        n_low = low_mask.sum()
        
        if n_low == 0:
            print(">> No low-light samples to augment.")
            return X, y
        
        X_low = X[low_mask]
        y_low = y[low_mask]
        
        X_aug_list = [X]
        y_aug_list = [y]
        
        for _ in range(augment_factor):
            noise = np.random.normal(0, 0.005, X_low.shape).astype(np.float32)
            X_noisy = np.clip(X_low + noise, 0, 1)
            X_aug_list.append(X_noisy)
            y_aug_list.append(y_low)
        
        X_aug = np.concatenate(X_aug_list, axis=0)
        y_aug = np.concatenate(y_aug_list, axis=0)
        
        # Shuffle augmented data
        idx = np.random.permutation(len(X_aug))
        X_aug = X_aug[idx]
        y_aug = y_aug[idx]
        
        print(f">> Low-light augmentation: {n_low} low-light samples × "
              f"{augment_factor} = +{n_low * augment_factor} synthetic samples")
        print(f">> Total training samples after augmentation: {len(X_aug)}")
        
        return X_aug, y_aug
    
    def inverse_transform(self, data):
        """Convert normalized data back to original scale."""
        return self.scaler.inverse_transform(data.reshape(-1, 1)).flatten()
    
    def get_last_window(self, seq_length=WINDOW_SIZE):
        """Get the last sequence for prediction (2-feature)."""
        if self.data_features is None:
            self.normalize_data()
        
        last_window = self.data_features[-seq_length:]  # shape: (seq_length, 2)
        return last_window.reshape(1, seq_length, 2)
    
    def full_preprocess(self):
        """Run the complete preprocessing pipeline."""
        print("\n" + "="*50)
        print("       DATA PREPROCESSING PIPELINE")
        print("="*50)
        
        self.load_data()
        self.clean_data()
        self.normalize_data()
        X, y = self.create_sequences()
        
        print("="*50 + "\n")
        return X, y
