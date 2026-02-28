"""
Configuration file for Energy Harvesting CNN Project
All project constants and settings are defined here.
"""

# ================= DATASET CONFIGURATION =================
DATA_FILE = 'z4689499.txt'
DATE_COLUMN = 'DATE (MM/DD/YYYY)'
TIME_COLUMN = 'MST'
TARGET_COLUMN = 'Global CMP22 (vent/cor) [W/m^2]'

# ================= MODEL CONFIGURATION =================
WINDOW_SIZE = 60          # Model looks at previous 60 minutes to understand trends
TRAIN_TEST_SPLIT = 0.8    # 80% training, 20% testing
BATCH_SIZE = 64           # Larger batch for better generalization
DEFAULT_EPOCHS = 100      # Early Stopping decides when to stop

# CNN Architecture
CONV1_FILTERS = 32        # First conv layer filters
CONV1_KERNEL = 5          # Wider kernel to capture multi-hour patterns
CONV2_FILTERS = 16        # Second conv layer filters  
CONV2_KERNEL = 3          # Standard kernel size
POOL_SIZE = 2
DENSE_UNITS = 32          # Dense layer neurons
ATTENTION_UNITS = 16      # Attention layer dimensionality

# Training Hyperparameters
INITIAL_LEARNING_RATE = 0.0002  # Balanced learning rate
LOW_LIGHT_WEIGHT = 2.0         # Mild weight for low-light (reduced from 5.0 to cut bias)
LOW_LIGHT_THRESHOLD = 0.05     # Normalized threshold (~74 W/m² out of ~1474 max)
HUBER_DELTA = 0.1              # Huber loss delta (normalized); linear beyond this
NUM_FEATURES = 2               # Irradiance + rate-of-change

# Ensemble Configuration
ENSEMBLE_SIZE = 3              # Number of models in ensemble

# Bias Calibration
BIAS_SAVE_PATH = 'saved_model/bias_offset.pkl'  # Saved bias correction value

# ================= ENERGY HARVESTING CONFIGURATION =================
PLANT_AREA_M2 = 0.05      # Surface area of the plant/harvester (e.g., 500 cm^2)
BATTERY_CAPACITY_WH = 50  # Max battery capacity in Watt-hours
DEVICE_CONSUMPTION_W = 0.5  # Constant power usage of the sensor node

# ================= FILE PATHS =================
MODEL_SAVE_PATH = 'saved_model/cnn_energy_model.h5'
SCALER_SAVE_PATH = 'saved_model/scaler.pkl'
ENSEMBLE_DIR = 'saved_model/ensemble'     # Directory for ensemble sub-models
OUTPUT_DIR = 'outputs'

# ================= VISUALIZATION SETTINGS =================
FIGURE_DPI = 150
DEFAULT_NUM_SAMPLES = 500
