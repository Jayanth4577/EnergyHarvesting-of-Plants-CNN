"""
CNN Model Module
Defines and handles the 1D-CNN architecture for energy prediction.

Features:
- Weighted Huber loss (symmetric, robust to spikes, mild low-light weighting)
- Self-attention layer for temporal focus
- 2-feature input: irradiance + rate-of-change
- Output ReLU clipping (predictions can never be negative)
- Balanced Dropout (0.3) to avoid underfitting
- Ensemble support (train N independent models, average predictions)
"""

import os
import joblib
import numpy as np

# TensorFlow / Keras imports
import keras
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import (
    Input, Conv1D, MaxPooling1D, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling1D, Activation,
    Multiply, Permute, RepeatVector, Flatten, Lambda
)
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam

from .config import (
    WINDOW_SIZE, CONV1_FILTERS, CONV1_KERNEL, CONV2_FILTERS,
    CONV2_KERNEL, POOL_SIZE, DENSE_UNITS, ATTENTION_UNITS,
    BATCH_SIZE, DEFAULT_EPOCHS, MODEL_SAVE_PATH, SCALER_SAVE_PATH,
    INITIAL_LEARNING_RATE, LOW_LIGHT_WEIGHT, LOW_LIGHT_THRESHOLD,
    HUBER_DELTA, NUM_FEATURES,
    ENSEMBLE_SIZE, ENSEMBLE_DIR, BIAS_SAVE_PATH
)


# ──────────────────────────────────────────────
#  Custom weighted Huber loss (replaces weighted_mse)
# ───────────────────────────────────────────────
def weighted_huber(y_true, y_pred):
    """
    Weighted Huber loss – symmetric and robust to large error spikes.
    - Quadratic for small errors (|e| < HUBER_DELTA) → precise on normal samples
    - Linear for large errors (|e| >= HUBER_DELTA) → doesn’t chase outlier spikes
    - LOW_LIGHT_WEIGHT× more penalty for low-light samples (mild, to avoid bias)
    """
    error = y_true - y_pred
    abs_error = tf.abs(error)
    delta = HUBER_DELTA

    # Huber loss per sample
    quadratic = tf.minimum(abs_error, delta)
    linear = abs_error - quadratic
    huber = 0.5 * tf.square(quadratic) + delta * linear

    # Mild low-light weighting
    weights = tf.where(
        y_true < LOW_LIGHT_THRESHOLD,
        LOW_LIGHT_WEIGHT,   # mildly heavier penalty for low-light
        1.0                 # normal penalty for daytime
    )
    return tf.reduce_mean(weights * huber)


# ──────────────────────────────────────────────
#  Attention block (squeeze-excite style on time axis)
# ──────────────────────────────────────────────
def temporal_attention_block(x, name_prefix='attn'):
    """
    Learns which time steps matter most.
    Produces a weight for each time step, then element-wise multiplies.
    """
    features = x.shape[2]

    # Compute attention scores per time step
    a = Dense(ATTENTION_UNITS, activation='tanh',
              name=f'{name_prefix}_dense1')(x)                       # (B, T, A)
    a = Dense(1, activation='linear',
              name=f'{name_prefix}_dense2')(a)                        # (B, T, 1)
    a = Flatten(name=f'{name_prefix}_flatten')(a)                     # (B, T)
    a = Activation('softmax', name=f'{name_prefix}_softmax')(a)       # (B, T)
    a = RepeatVector(features)(a)                                     # (B, F, T)
    a = Permute((2, 1), name=f'{name_prefix}_permute')(a)             # (B, T, F)

    # Weight the original input
    out = Multiply(name=f'{name_prefix}_mul')([x, a])
    return out


class CNNModel:
    """1D Convolutional Neural Network for time-series prediction."""

    def __init__(self):
        self.model = None
        self.history = None

    # ──────────────────────────────────────────
    #  Build (Functional API for attention)
    # ──────────────────────────────────────────
    def build(self, input_shape=(WINDOW_SIZE, NUM_FEATURES)):
        """
        Architecture:
          Input(60, 2) → Conv1D(32) → BN → Pool → Drop(0.3)
                → Conv1D(16) → BN → Pool → Drop(0.3)
                → **Temporal Attention**
                → GlobalAvgPool → Dense(32) → BN → Drop(0.3)
                → Dense(1) → ReLU clip (output ≥ 0)

        Input features: [irradiance, rate-of-change]
        Loss: Weighted Huber (symmetric, robust to spikes)
        """
        print("\n>> Building CNN Architecture (Attention + Weighted Huber Loss)...")
        print(f">> Input shape: {input_shape}  (irradiance + rate-of-change)")

        inp = Input(shape=input_shape, name='input')

        # --- Conv Block 1 ---
        x = Conv1D(CONV1_FILTERS, CONV1_KERNEL, padding='same',
                    activation='relu', kernel_regularizer=l2(0.001),
                    name='conv1d_pattern_detection')(inp)
        x = BatchNormalization(name='bn_1')(x)
        x = MaxPooling1D(pool_size=POOL_SIZE, name='maxpool_1')(x)
        x = Dropout(0.3, name='dropout_1')(x)

        # --- Conv Block 2 ---
        x = Conv1D(CONV2_FILTERS, CONV2_KERNEL, padding='same',
                    activation='relu', kernel_regularizer=l2(0.001),
                    name='conv1d_feature_extraction')(x)
        x = BatchNormalization(name='bn_2')(x)
        x = MaxPooling1D(pool_size=POOL_SIZE, name='maxpool_2')(x)
        x = Dropout(0.3, name='dropout_2')(x)

        # --- Temporal Attention ---
        x = temporal_attention_block(x, name_prefix='attn')

        # --- Head ---
        x = GlobalAveragePooling1D(name='global_avg_pool')(x)
        x = Dense(DENSE_UNITS, activation='relu',
                  kernel_regularizer=l2(0.001), name='dense_hidden')(x)
        x = BatchNormalization(name='bn_3')(x)
        x = Dropout(0.3, name='dropout_3')(x)

        # Output with ReLU clipping – predictions can NEVER go negative
        x = Dense(1, name='raw_output')(x)
        out = Activation('relu', name='clipped_output')(x)

        self.model = Model(inputs=inp, outputs=out, name='CNN_Attention')

        optimizer = Adam(learning_rate=INITIAL_LEARNING_RATE)
        self.model.compile(
            optimizer=optimizer,
            loss=weighted_huber,
            metrics=['mae']
        )

        print(">> CNN Model built successfully!")
        self.print_summary()
        return self.model

    def print_summary(self):
        print("\n" + "-" * 50)
        print("MODEL ARCHITECTURE SUMMARY:")
        print("-" * 50)
        self.model.summary()
        print("-" * 50 + "\n")

    # ──────────────────────────────────────────
    #  Train
    # ──────────────────────────────────────────
    def train(self, X_train, y_train, X_test, y_test, epochs=DEFAULT_EPOCHS):
        if self.model is None:
            self.build()

        print(f"\n>> Training Model for up to {epochs} epochs (with Early Stopping)...")
        print(f">> Initial Learning Rate: {INITIAL_LEARNING_RATE}")
        print(f">> Using weighted Huber loss (low-light weight = {LOW_LIGHT_WEIGHT}×, delta = {HUBER_DELTA})")
        print(">> (Learning patterns from solar irradiance data)\n")

        callbacks = [
            EarlyStopping(
                monitor='val_loss', patience=10,
                restore_best_weights=True, verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss', factor=0.5,
                patience=5, min_lr=1e-6, verbose=1
            )
        ]

        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=BATCH_SIZE,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )

        print("\n>> Training Complete!")
        return self.history

    # ──────────────────────────────────────────
    #  Predict
    # ──────────────────────────────────────────
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not built or loaded.")
        return self.model.predict(X, verbose=0)

    # ──────────────────────────────────────────
    #  Save / Load
    # ──────────────────────────────────────────
    def save(self, model_path=MODEL_SAVE_PATH, scaler=None,
             scaler_path=SCALER_SAVE_PATH):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        print(f">> Model saved to '{model_path}'")
        if scaler is not None:
            joblib.dump(scaler, scaler_path)
            print(f">> Scaler saved to '{scaler_path}'")

    def load(self, model_path=MODEL_SAVE_PATH, scaler_path=SCALER_SAVE_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found.")

        self.model = load_model(
            model_path, compile=False,
            custom_objects={'weighted_huber': weighted_huber}
        )
        optimizer = Adam(learning_rate=INITIAL_LEARNING_RATE)
        self.model.compile(optimizer=optimizer, loss=weighted_huber, metrics=['mae'])
        print(f">> Model loaded from '{model_path}'")

        scaler = None
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print(f">> Scaler loaded from '{scaler_path}'")
        return scaler

    def get_training_history(self):
        return self.history


# ══════════════════════════════════════════════
#  Ensemble wrapper
# ══════════════════════════════════════════════
class EnsembleCNN:
    """Train ENSEMBLE_SIZE independent CNNModels and average their predictions."""

    def __init__(self, n_models=ENSEMBLE_SIZE):
        self.n_models = n_models
        self.models = []
        self.histories = []

    def build_and_train(self, X_train, y_train, X_test, y_test,
                        epochs=DEFAULT_EPOCHS):
        print(f"\n>> Building Ensemble of {self.n_models} models...")
        for i in range(self.n_models):
            print(f"\n{'='*50}")
            print(f"   ENSEMBLE MODEL {i+1} / {self.n_models}")
            print(f"{'='*50}")
            m = CNNModel()
            m.build()
            m.train(X_train, y_train, X_test, y_test, epochs=epochs)
            self.models.append(m)
            self.histories.append(m.get_training_history())
        print(f"\n>> Ensemble training complete ({self.n_models} models).")

    def predict(self, X):
        preds = np.stack([m.predict(X) for m in self.models], axis=0)
        return np.mean(preds, axis=0)

    def save(self, ensemble_dir=ENSEMBLE_DIR, scaler=None,
             scaler_path=SCALER_SAVE_PATH):
        os.makedirs(ensemble_dir, exist_ok=True)
        for i, m in enumerate(self.models):
            path = os.path.join(ensemble_dir, f'model_{i}.h5')
            m.save(model_path=path)
        if scaler is not None:
            joblib.dump(scaler, scaler_path)
            print(f">> Scaler saved to '{scaler_path}'")

    def load(self, ensemble_dir=ENSEMBLE_DIR, scaler_path=SCALER_SAVE_PATH):
        self.models = []
        for i in range(self.n_models):
            path = os.path.join(ensemble_dir, f'model_{i}.h5')
            if not os.path.exists(path):
                raise FileNotFoundError(f"Ensemble model '{path}' not found.")
            m = CNNModel()
            m.load(model_path=path, scaler_path=scaler_path)
            self.models.append(m)
        print(f">> Loaded ensemble of {self.n_models} models.")

        scaler = None
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        return scaler

    def get_training_history(self):
        """Return the history of the first model (for plotting)."""
        return self.histories[0] if self.histories else None
