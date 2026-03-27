import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import os

np.random.seed(42)

def generate_synthetic_longitudinal(n_patients=2500, seq_len=20):
    # Create patient-based synthetic data for longitudinal risk prediction
    patient_ids = np.arange(n_patients)
    age = np.random.normal(28, 5, n_patients)  # pregnant age
    bmi = np.random.normal(26, 4, n_patients)
    prior_hypertension = np.random.binomial(1, 0.18, n_patients)
    diabetes = np.random.binomial(1, 0.1, n_patients)

    # risk score baseline
    risk_base = 0.1 + 0.02 * (age - 25) + 0.05 * (bmi - 24) + 0.25 * prior_hypertension + 0.2 * diabetes
    risk_base = np.clip(risk_base, 0.02, 0.92)

    X_list = []
    y_class = []
    y_time = []

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
        y_time.append(event_time if event_happens else seq_len + 5)

    X = np.stack(X_list)
    y_class = np.array(y_class)
    y_time = np.array(y_time, dtype=np.float32)

    return X, y_class, y_time


def build_model(seq_len=20, n_features=7):
    inp = Input(shape=(seq_len, n_features), name='patient_sequence')
    x = Bidirectional(LSTM(64, return_sequences=True, activation='tanh'))(inp)
    x = Dropout(0.22)(x)
    x = Bidirectional(LSTM(32, activation='tanh'))(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)

    out_risk = Dense(1, activation='sigmoid', name='risk')(x)
    out_time = Dense(1, activation='relu', name='time_to_event')(x)

    model = Model(inputs=inp, outputs=[out_risk, out_time])
    model.compile(
        optimizer='adam',
        loss={'risk': 'binary_crossentropy', 'time_to_event': 'mse'},
        loss_weights={'risk': 1.0, 'time_to_event': 0.75},
        metrics={'risk': ['accuracy'], 'time_to_event': ['mse']}
    )
    return model


def main():
    seq_len = 20
    X, y_class, y_time = generate_synthetic_longitudinal(n_patients=2500, seq_len=seq_len)

    # scale features per time step combined
    n_patients, T, n_features = X.shape
    X_2d = X.reshape(-1, n_features)
    scaler = StandardScaler().fit(X_2d)
    X_scaled = scaler.transform(X_2d).reshape(n_patients, T, n_features)

    X_trainval, X_test, y_class_trainval, y_class_test, y_time_trainval, y_time_test = train_test_split(
        X_scaled, y_class, y_time, test_size=0.2, random_state=42, stratify=y_class)

    X_train, X_val, y_class_train, y_class_val, y_time_train, y_time_val = train_test_split(
        X_trainval, y_class_trainval, y_time_trainval, test_size=0.125, random_state=42, stratify=y_class_trainval)
    # 0.125 of 80% is 10% => 70/20/10

    model = build_model(seq_len=seq_len, n_features=n_features)
    print(model.summary())

    es = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)

    history = model.fit(
        X_train,
        {'risk': y_class_train, 'time_to_event': y_time_train},
        validation_data=(X_val, {'risk': y_class_val, 'time_to_event': y_time_val}),
        epochs=90,
        batch_size=64,
        callbacks=[es],
        verbose=2
    )

    results = model.evaluate(X_test, {'risk': y_class_test, 'time_to_event': y_time_test}, verbose=2)
    print('Test set results:', results)

    os.makedirs('artifacts', exist_ok=True)
    model.save('artifacts/hdp_lstm_model.keras')
    np.save('artifacts/risk_scaler_mean.npy', scaler.mean_)
    np.save('artifacts/risk_scaler_scale.npy', scaler.scale_)

    # save train/val/test distribution info
    pd.DataFrame({'y_class': y_class_test, 'y_time': y_time_test}).to_csv('artifacts/test_labels.csv', index=False)
    print('Saved model and artifacts.')

if __name__ == '__main__':
    main()
