"""
Train LSTM Model for HDP Temporal Prediction
Uses same synthetic longitudinal data as Random Forest models
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import os

print("=" * 80)
print("LSTM MODEL TRAINING FOR HDP TEMPORAL PREDICTION")
print("=" * 80)

# Load the same synthetic data used for Random Forest models
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def generate_synthetic_longitudinal(n_patients=2500, seq_len=20):
    patient_ids = np.arange(n_patients)
    age = np.random.normal(28, 5, n_patients)
    bmi = np.random.normal(26, 4, n_patients)
    prior_hypertension = np.random.binomial(1, 0.18, n_patients)
    diabetes = np.random.binomial(1, 0.1, n_patients)

    risk_base = 0.1 + 0.02 * (age - 25) + 0.05 * (bmi - 24) + 0.25 * prior_hypertension + 0.2 * diabetes
    risk_base = np.clip(risk_base, 0.02, 0.92)

    X_list = []
    y_class = []
    y_proba = []

    for i in range(n_patients):
        rr = risk_base[i]
        event_happens = np.random.rand() < rr
        if event_happens:
            event_time = np.random.randint(5, seq_len - 1)
        else:
            event_time = seq_len + np.random.randint(1, 6)

        sbp = np.linspace(110, 118, seq_len) + np.random.normal(0, 5, seq_len)
        dbp = np.linspace(70, 76, seq_len) + np.random.normal(0, 3, seq_len)
        mapv = sbp * 0.4 + dbp * 0.6

        if event_happens:
            sbp[event_time:] += np.linspace(0, 30, seq_len - event_time)
            dbp[event_time:] += np.linspace(0, 20, seq_len - event_time)

        bmi_ts = bmi[i] + np.random.normal(0, 0.25, seq_len)
        weight_change = np.linspace(0, 4, seq_len) + np.random.normal(0, 0.5, seq_len)
        oliguria = np.random.binomial(1, 0.05 + 0.5*event_happens, seq_len)
        proteinuria = np.random.binomial(1, 0.03 + 0.6*event_happens, seq_len)

        extra = np.vstack([sbp, dbp, mapv, bmi_ts, weight_change, oliguria, proteinuria]).T
        X_list.append(extra)

        y_class.append(1 if event_happens else 0)
        
        # Probability curve: ramps up to event, stays high after
        proba_seq = np.ones(seq_len) * (rr / 2)  # Base probability
        if event_happens:
            proba_seq[event_time:] = np.linspace(rr, 1.0, seq_len - event_time)
        y_proba.append(proba_seq)

    X = np.stack(X_list)
    y_class = np.array(y_class)
    y_proba = np.stack(y_proba)

    return X, y_class, y_proba


print("\n[1] GENERATING SYNTHETIC DATA...")
np.random.seed(42)
seq_len = 20
X, y_class, y_proba = generate_synthetic_longitudinal(n_patients=2500, seq_len=seq_len)
print(f"[OK] Generated {X.shape[0]} sequences of length {X.shape[1]} with {X.shape[2]} features")
print(f"[OK] y_proba shape: {y_proba.shape}")

# Scale the data
print("\n[2] PREPROCESSING DATA...")
n_patients, T, n_features = X.shape
X_2d = X.reshape(-1, n_features)
scaler = StandardScaler().fit(X_2d)
X_scaled = scaler.transform(X_2d).reshape(n_patients, T, n_features)
print(f"[OK] Data scaled")

# Split data: 70% train, 10% val, 20% test
print("\n[3] SPLITTING DATA (70/10/20)...")
X_trainval, X_test, y_proba_trainval, y_proba_test = train_test_split(
    X_scaled, y_proba, test_size=0.2, random_state=42
)
X_train, X_val, y_proba_train, y_proba_val = train_test_split(
    X_trainval, y_proba_trainval, test_size=0.125, random_state=42
)
print(f"[OK] Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

# Build LSTM model
print("\n[4] BUILDING LSTM MODEL...")
model = keras.Sequential([
    layers.LSTM(64, activation='relu', input_shape=(seq_len, n_features), return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(32, activation='relu', return_sequences=True),
    layers.Dropout(0.2),
    layers.TimeDistributed(layers.Dense(16, activation='relu')),
    layers.TimeDistributed(layers.Dense(1, activation='sigmoid'))  # Sigmoid for probability output
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)
print(f"[OK] Model built")
print(model.summary())

# Train model
print("\n[5] TRAINING LSTM MODEL...")
history = model.fit(
    X_train, y_proba_train,
    validation_data=(X_val, y_proba_val),
    epochs=20,
    batch_size=32,
    verbose=1
)

# Evaluate
print("\n[6] EVALUATING MODEL...")
test_loss, test_mae = model.evaluate(X_test, y_proba_test, verbose=0)
print(f"[OK] Test Loss (MSE): {test_loss:.4f}")
print(f"[OK] Test MAE: {test_mae:.4f}")

# Save model
print("\n[7] SAVING MODEL...")
os.makedirs('artifacts', exist_ok=True)
model.save('artifacts/hdp_lstm_model.h5')
np.save('artifacts/lstm_scaler_mean.npy', scaler.mean_)
np.save('artifacts/lstm_scaler_scale.npy', scaler.scale_)
print(f"[OK] Model saved to artifacts/hdp_lstm_model.h5")
print(f"[OK] Scaler saved")

print("\n" + "=" * 80)
print("LSTM TRAINING COMPLETE")
print("=" * 80)
