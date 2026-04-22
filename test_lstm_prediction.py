import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

print("Testing LSTM model loading and prediction...")

# Test loading the model
try:
    lstm_model = keras.models.load_model('artifacts/hdp_lstm_model.h5', compile=False)
    print("✅ LSTM model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading LSTM: {e}")
    exit(1)

# Create test input
seq_len = 20
n_features = 7

# Create synthetic input 
np.random.seed(42)
sbp_seq = np.linspace(115, 130, seq_len) + np.random.normal(0, 2, seq_len)
dbp_seq = np.linspace(70, 85, seq_len) + np.random.normal(0, 1.5, seq_len)
map_seq = 0.4 * sbp_seq + 0.6 * dbp_seq
bmi_seq = np.full(seq_len, 26.5) + np.linspace(0, 1.2, seq_len)
weight_seq = np.linspace(0, 2.0, seq_len)
oliguria = np.where(np.random.rand(seq_len) < 0.05, 1, 0)
proteinuria = np.where(np.random.rand(seq_len) < 0.03, 1, 0)

input_data = np.vstack([sbp_seq, dbp_seq, map_seq, bmi_seq, weight_seq, oliguria, proteinuria]).T
input_model = input_data.reshape(1, seq_len, n_features)

# Load scaler
mean_ = np.load('artifacts/risk_scaler_mean.npy')
scale_ = np.load('artifacts/risk_scaler_scale.npy')
scaler = StandardScaler()
scaler.mean_ = mean_
scaler.scale_ = scale_
scaler.var_ = scale_**2
scaler.n_features_in_ = 7

input_scaled = scaler.transform(input_model.reshape(-1, 7)).reshape(1, seq_len, 7)

print(f"Input shape: {input_scaled.shape}")
print(f"LSTM model input shape: {lstm_model.input_shape}")
print(f"LSTM model output shape: {lstm_model.output_shape}")

# Make prediction
try:
    lstm_pred = lstm_model.predict(input_scaled, verbose=0)
    print(f"✅ Prediction successful!")
    print(f"Output shape: {lstm_pred.shape}")
    
    pred_lstm_temporal = lstm_pred[0].flatten()
    print(f"Flattened output shape: {pred_lstm_temporal.shape}")
    print(f"Output values: {pred_lstm_temporal}")
    print(f"Output range: {pred_lstm_temporal.min():.4f} - {pred_lstm_temporal.max():.4f}")
    
    if pred_lstm_temporal is not None and len(pred_lstm_temporal) > 0:
        print("✅ LSTM output ready for visualization!")
    else:
        print("❌ LSTM output is empty or None!")
        
except Exception as e:
    print(f"❌ Error during prediction: {e}")
    import traceback
    traceback.print_exc()
