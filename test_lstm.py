import os
from pathlib import Path

# Check if LSTM model exists
lstm_path = "artifacts/hdp_lstm_model.h5"
exists = os.path.exists(lstm_path)
print(f"LSTM model exists: {exists}")

if exists:
    size = os.path.getsize(lstm_path)
    print(f"LSTM model size: {size} bytes")
    
    # Try to load it
    try:
        from tensorflow import keras
        model = keras.models.load_model(lstm_path)
        print(f"LSTM model loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
    except Exception as e:
        print(f"ERROR loading LSTM: {e}")
else:
    print("LSTM model file NOT FOUND!")
    print("Available files in artifacts/:")
    if os.path.exists("artifacts"):
        for f in os.listdir("artifacts"):
            print(f"  - {f}")
