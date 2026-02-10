import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import argparse

# ================================
# Command-line arguments
# ================================
parser = argparse.ArgumentParser(description="Test renewable energy prediction models")
parser.add_argument("--base-dir", default=r"C:\Users\chris\OneDrive\Desktop\renewable energy prediction", 
                    help="Base directory for models and results")
parser.add_argument("--data-path", default=r"C:\Users\chris\OneDrive\Documents\renewable_energy\cleaned_energy_dataset.csv", 
                    help="Path to the dataset")
parser.add_argument("--time-steps", type=int, default=12, help="Number of time steps for sequences")
args = parser.parse_args()

# ================================
# Directories and configurations
# ================================
base_dir = args.base_dir
data_path = args.data_path
time_steps = args.time_steps
models_dir = os.path.join(base_dir, "models", "trained")
min_rows_initial = 30
min_rows_preprocessed = 20
min_sequences = 15
min_non_zero = 10  # Minimum non-zero values required for an energy type

# ================================
# Load and validate dataset
# ================================
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at: {data_path}")

df_original = pd.read_csv(data_path)
required_cols = ["date", "state_name", "wind_energy", "solar_energy", "other_renewable_energy"]
if not all(col in df_original.columns for col in required_cols):
    raise ValueError(f"Dataset missing required columns: {set(required_cols) - set(df_original.columns)}")

df_original["date"] = pd.to_datetime(df_original["date"], errors="coerce", dayfirst=True)
if df_original["date"].isna().any():
    raise ValueError("Some dates could not be parsed. Check date format in dataset.")

# ================================
# Preprocessing functions
# ================================
def preprocess_data(df):
    df = df.copy()
    df["YEAR"] = df["date"].dt.year
    df["MONTH"] = df["date"].dt.month
    df["QUARTER"] = df["MONTH"].apply(lambda x: (x - 1)//3 + 1)
    df["SIN_MONTH"] = np.sin(2 * np.pi * df["MONTH"] / 12)
    df["COS_MONTH"] = np.cos(2 * np.pi * df["MONTH"] / 12)
    return df

def detect_and_handle_outliers(data):
    lower, upper = np.percentile(data, [1, 99])
    return np.clip(data, lower, upper)

def add_lag_rolling_features(df, col, lags=[1, 3, 6, 12], windows=[3, 6]):
    df = df.copy()
    for lag in lags:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
    for window in windows:
        df[f"{col}_rolling_mean{window}"] = df[col].rolling(window=window).mean()
        df[f"{col}_rolling_std{window}"] = df[col].rolling(window=window).std()
    return df

def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps + 1):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps - 1])
    return np.array(Xs), np.array(ys)

def smape(y_true, y_pred):
    denominator = np.abs(y_true) + np.abs(y_pred) + 1e-5
    diff = np.abs(y_pred - y_true)
    return np.mean(np.where(denominator > 0, diff / denominator, 0)) * 100

# ================================
# Model testing function
# ================================
def test_model(state, energy_type, df_preprocessed, time_steps):
    print(f"\nTesting {state} - {energy_type}")

    # Filter data for the state
    df_state = df_preprocessed[df_preprocessed["state_name"] == state].sort_values("date").reset_index(drop=True)
    if len(df_state) < min_rows_initial or energy_type not in df_state:
        print(f"Skipping {state} - {energy_type}: insufficient data or missing column")
        return None

    # Check for sufficient non-zero values
    non_zero_count = (df_state[energy_type] > 0).sum()
    if non_zero_count < min_non_zero:
        print(f"Skipping {state} - {energy_type}: only {non_zero_count} non-zero values")
        return None

    # Use only the energy column as feature (assuming model was trained this way)
    X = df_state[[energy_type]].values
    y = df_state[energy_type].values
    dates = df_state["date"].values

    # Handle outliers
    X = detect_and_handle_outliers(X)

    # Scaling
    scaler_X = RobustScaler().fit(X)
    scaler_y = StandardScaler().fit(y.reshape(-1, 1))
    X_scaled = scaler_X.transform(X)
    y_scaled = scaler_y.transform(y.reshape(-1, 1)).flatten()

    # Create sequences
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)
    if len(X_seq) < min_sequences:
        print(f"Skipping {state} - {energy_type}: insufficient sequences ({len(X_seq)})")
        return None

    # Split test set (last 15%)
    n = len(X_seq)
    val_end = int(0.85 * n)
    X_test = X_seq[val_end:]
    y_test = y_seq[val_end:]
    test_dates = dates[val_end + time_steps - 1:]

    # Load model
    model_dir = os.path.join(models_dir, state, energy_type)
    lstm_model = os.path.join(model_dir, "best_model_LSTM.h5")
    cnn_model = os.path.join(model_dir, "best_model_CNN.h5")

    if os.path.exists(lstm_model):
        model_path = lstm_model
    elif os.path.exists(cnn_model):
        model_path = cnn_model
    else:
        print(f"No model found for {state} - {energy_type}")
        return None

    try:
        model = load_model(model_path)
        print(f"Loaded model: {model_path}")
        print(f"Model input shape: {model.input_shape}")
    except Exception as e:
        print(f"Failed to load model for {state} - {energy_type}: {e}")
        return None

    # Verify feature compatibility
    expected_features = model.input_shape[-1]
    actual_features = X_test.shape[-1]
    print(f"Expected features: {expected_features}, Actual features: {actual_features}")
    if actual_features != expected_features:
        print(f"Feature mismatch for {state} - {energy_type}. Expected {expected_features}, got {actual_features}. Skipping.")
        return None

    # Predict
    try:
        y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    except Exception as e:
        print(f"Prediction failed for {state} - {energy_type}: {e}")
        return None

    if np.any(np.isnan(y_pred_scaled)) or np.any(np.isinf(y_pred_scaled)):
        print(f"Invalid predictions for {state} - {energy_type}. Skipping metrics.")
        return None

    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred = np.maximum(y_pred, 0)

    # Metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    smape_val = smape(y_true, y_pred)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(test_dates, y_true, label="Actual", marker="o")
    plt.plot(test_dates, y_pred, label="Predicted", marker="x")
    plt.title(f"{state} - {energy_type} Actual vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Energy Value")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = os.path.join(base_dir, "results", "plots", f"{state}_{energy_type}_test.png")
    try:
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)
        plt.close()
    except Exception as e:
        print(f"Failed to save plot for {state} - {energy_type}: {e}")

    return {
        "State": state,
        "Energy_Type": energy_type,
        "MAE": mae,
        "RMSE": rmse,
        "SMAPE": smape_val,
        "R2": r2,
        "Test_Samples": len(y_test)
    }

# ================================
# Main Execution
# ================================
df_preprocessed = preprocess_data(df_original)
states = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
energy_types = ["wind_energy", "solar_energy", "other_renewable_energy"]
results = []

for state in states:
    for energy_type in energy_types:
        result = test_model(state, energy_type, df_preprocessed, time_steps)
        if result:
            results.append(result)
            print(f"{state}-{energy_type}: MAE={result['MAE']:.2f}, RMSE={result['RMSE']:.2f}, "
                  f"SMAPE={result['SMAPE']:.2f}%, R2={result['R2']:.2f}, Test Samples={result['Test_Samples']}")

# Save all results
if results:
    results_df = pd.DataFrame(results)
    output_path = os.path.join(base_dir, "results", "metrics", "test_accuracy_results.csv")
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"\n✅ All results saved to {output_path}")
    except Exception as e:
        print(f"Failed to save results to {output_path}: {e}")
else:
    print("\nNo results to save. Check data or model availability.")