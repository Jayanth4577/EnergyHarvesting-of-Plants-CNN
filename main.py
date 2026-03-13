"""
Energy Harvesting CNN - Main Entry Point
=========================================
ML-Based Energy Harvesting from Plants Using CNN

Pipeline:
1. Data preprocessing (nighttime zeroing, noise mask)
2. Stratified train/test split + low-light augmentation
3. Ensemble CNN training (with attention, weighted loss, ReLU clip)
4. Bias calibration
5. Model evaluation
6. Energy calculations
7. Visualization and dashboards

Author: Student Project
Date: 2026
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    DATA_FILE, MODEL_SAVE_PATH, SCALER_SAVE_PATH,
    DEFAULT_EPOCHS, TRAIN_TEST_SPLIT, ENSEMBLE_DIR, ENSEMBLE_SIZE,
    BIAS_SAVE_PATH
)
from src.data_preprocessing import DataPreprocessor
from src.model import CNNModel, EnsembleCNN
from src.evaluation import ModelEvaluator
from src.visualization import Visualizer
from src.energy_calculator import EnergyCalculator


def main():
    """Main execution function."""
    
    print("\n" + "="*60)
    print("   ML-BASED ENERGY HARVESTING FROM PLANTS USING CNN")
    print("="*60 + "\n")
    
    # ===================== STEP 1: DATA PREPROCESSING =====================
    print(">> STEP 1: Data Preprocessing")
    print("-" * 40)
    
    preprocessor = DataPreprocessor(DATA_FILE)
    X, y = preprocessor.full_preprocess()
    
    # Stratified split (ensures train & test both see night + day)
    X_train, X_test, y_train, y_test = preprocessor.prepare_train_test_split(
        X, y, split_ratio=TRAIN_TEST_SPLIT
    )
    
    # Low-light augmentation on training set only
    X_train, y_train = preprocessor.augment_low_light(X_train, y_train)
    
    # ===================== STEP 2: MODEL TRAINING/LOADING =====================
    print("\n>> STEP 2: CNN Ensemble Model")
    print("-" * 40)
    
    ensemble = EnsembleCNN(n_models=ENSEMBLE_SIZE)
    visualizer = Visualizer()
    
    # Check if saved ensemble exists
    ensemble_exists = os.path.exists(ENSEMBLE_DIR) and all(
        os.path.exists(os.path.join(ENSEMBLE_DIR, f'model_{i}.h5'))
        for i in range(ENSEMBLE_SIZE)
    )
    
    if ensemble_exists:
        print(">> Found saved ensemble. Loading...")
        loaded_scaler = ensemble.load(ENSEMBLE_DIR, SCALER_SAVE_PATH)
        if loaded_scaler is not None:
            preprocessor.scaler = loaded_scaler
    else:
        print(f">> No saved ensemble found. Training {ENSEMBLE_SIZE} models...")
        ensemble.build_and_train(X_train, y_train, X_test, y_test,
                                 epochs=DEFAULT_EPOCHS)
        
        # Save ensemble
        ensemble.save(ENSEMBLE_DIR, preprocessor.scaler, SCALER_SAVE_PATH)
        
        # Also save the first model as the single-model fallback
        ensemble.models[0].save(MODEL_SAVE_PATH, preprocessor.scaler, SCALER_SAVE_PATH)
        
        # Plot training history of first model
        visualizer.plot_training_history(ensemble.get_training_history())
    
    # ===================== STEP 3: MODEL EVALUATION + BIAS CALIBRATION =====================
    print("\n>> STEP 3: Model Evaluation & Bias Calibration")
    print("-" * 40)
    
    # Use ensemble predictions for evaluation
    y_pred_scaled = ensemble.predict(X_test)
    
    evaluator = ModelEvaluator(None, preprocessor.scaler)
    # Manually set predictions (ensemble, not single model)
    evaluator.y_actual = preprocessor.scaler.inverse_transform(
        y_test.reshape(-1, 1)).flatten()
    evaluator.y_predicted = preprocessor.scaler.inverse_transform(
        y_pred_scaled.reshape(-1, 1)).flatten()
    
    # Compute standard metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import numpy as np
    
    mse = mean_squared_error(evaluator.y_actual, evaluator.y_predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(evaluator.y_actual, evaluator.y_predicted)
    r2 = r2_score(evaluator.y_actual, evaluator.y_predicted)
    
    mask = evaluator.y_actual > 10.0
    if mask.sum() > 0:
        mape = np.mean(np.abs(
            (evaluator.y_actual[mask] - evaluator.y_predicted[mask]) /
            evaluator.y_actual[mask]
        )) * 100
    else:
        mape = 0.0
    
    evaluator.metrics = {
        'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape
    }
    evaluator.print_metrics()
    
    # Bias calibration
    bias = evaluator.calibrate_bias()
    if not ensemble_exists:
        evaluator.save_bias(BIAS_SAVE_PATH)
    else:
        evaluator.load_bias(BIAS_SAVE_PATH)
    
    # Plot predicted vs actual
    y_actual, y_predicted = evaluator.get_predictions()
    visualizer.plot_predicted_vs_actual(y_actual, y_predicted)
    
    # ===================== STEP 4: ENERGY CALCULATIONS =====================
    print("\n>> STEP 4: Energy Calculations")
    print("-" * 40)
    
    # Initialize calculator with ensemble model for predictions
    # Use the first model in the ensemble for single-step prediction
    calculator = EnergyCalculator(
        df=preprocessor.df,
        model=ensemble.models[0].model,
        scaler=preprocessor.scaler,
        data_processed=preprocessor.data_processed,
        data_features=preprocessor.data_features,
        bias_offset=evaluator.bias_offset
    )
    
    # ===================== USER INPUT FOR DATE/TIME =====================
    print("\n" + "="*50)
    print("         SELECT DATE AND TIME")
    print("="*50)
    
    # Show available date range from dataset
    min_date = preprocessor.df.index.min().strftime('%m/%d/%Y')
    max_date = preprocessor.df.index.max().strftime('%m/%d/%Y')
    print(f"\n>> Available date range: {min_date} to {max_date}")
    
    # Get user input for date
    print("\n>> Enter date (MM/DD/YYYY format) or press Enter for default (01/26/2019):")
    user_date = input("   Date: ").strip()
    TEST_DATE = user_date if user_date else "01/26/2019"
    
    # Get user input for time
    print(f"\n>> Enter time (MM/DD/YYYY HH:MM format) or press Enter for default ({TEST_DATE} 14:30):")
    user_time = input("   Time: ").strip()
    TEST_TIME = user_time if user_time else f"{TEST_DATE} 14:30"
    
    print(f"\n>> Selected Date: {TEST_DATE}")
    print(f">> Selected Time: {TEST_TIME}")
    
    print("\n" + "="*50)
    print("         ENERGY HARVESTING RESULTS")
    print("="*50)
    
    # Feature 1: Today's Harvest
    energy, energy_str = calculator.get_todays_harvest(TEST_DATE)
    print(f"\n1. Energy Harvested Today ({TEST_DATE}):")
    print(f"   → {energy_str}")
    
    # Feature 2: Total Harvest Till Now
    total, total_str = calculator.get_total_harvest_till_now(TEST_TIME)
    print(f"\n2. Total Energy Harvested (up to {TEST_TIME}):")
    print(f"   → {total_str}")
    
    # Feature 3: Instant Power
    power, power_str = calculator.get_instant_power(TEST_TIME)
    print(f"\n3. Instant Power at {TEST_TIME}:")
    print(f"   → {power_str}")
    
    # Feature 4: CNN Prediction
    pred, pred_str = calculator.predict_future_energy()
    print(f"\n4. CNN Model Prediction (Next Minute):")
    print(f"   → {pred_str}")
    
    pred_power, pred_power_str = calculator.predict_future_power()
    print(f"   → Harvested Power: {pred_power_str}")
    
    print("\n" + "="*50)
    
    # Daily detailed report
    calculator.print_daily_report(TEST_DATE)
    
    # ===================== STEP 5: VISUALIZATION =====================
    print("\n>> STEP 5: Generating Visualizations")
    print("-" * 40)
    
    # Generate daily dashboard
    day_df = preprocessor.df[
        preprocessor.df.index.strftime('%m/%d/%Y') == TEST_DATE
    ].copy()
    visualizer.plot_daily_dashboard(day_df, TEST_DATE)
    
    # Generate energy summary (last 7 days)
    visualizer.plot_energy_summary(preprocessor.df, num_days=7)

    # Generate comprehensive benchmark comparison chart
    visualizer.plot_comprehensive_method_comparison()
    
    # ===================== COMPLETE =====================
    print("\n" + "="*60)
    print("   ✅ ALL OUTPUTS GENERATED SUCCESSFULLY!")
    print("="*60)
    print("\n>> Output files saved in 'outputs/' folder:")
    print("   • training_history.png")
    print("   • predicted_vs_actual.png")
    print("   • dashboard_*.png")
    print("   • energy_summary.png")
    print("   • comprehensive_comparison_all_methods.png")
    print("\n>> Saved model in 'saved_model/' folder:")
    print("   • cnn_energy_model.h5")
    print("   • scaler.pkl")
    print("\n")


if __name__ == "__main__":
    main()
