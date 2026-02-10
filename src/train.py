import pandas as pd
import numpy as np
import os, math, warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Input
warnings.filterwarnings("ignore")

# ==============================
# 1. Define Paths
# ==============================
MODEL_SAVE_PATH = r"C:\Users\chris\OneDrive\Desktop\renewable energy prediction\models\trained1"
RESULTS_LOG_PATH = r"C:\Users\chris\OneDrive\Desktop\renewable energy prediction\results\logs1"
PLOTS_PATH = r"C:\Users\chris\OneDrive\Desktop\renewable energy prediction\results\plots1"

# Create directories if they don't exist
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(RESULTS_LOG_PATH, exist_ok=True)
os.makedirs(PLOTS_PATH, exist_ok=True)

# ==============================
# 2. Load and Prepare Data
# ==============================
file_name = r"c:\Users\chris\OneDrive\Documents\renewable_energy\cleaned_energy_dataset.csv"
df = pd.read_csv(file_name)
df.columns = df.columns.str.lower().str.strip()

# Handle missing column
if "other_renewable_energy" not in df.columns:
    df["other_renewable_energy"] = 0
energy_types = ["solar_energy", "wind_energy", "other_renewable_energy"]

# Ensure date column
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
elif "year" in df.columns:
    df["date"] = pd.to_datetime(df["year"].astype(str) + "-01-01", errors="coerce")

# ==============================
# 3. Helper Functions
# ==============================
def create_sequences(data, seq_len=10):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)

def evaluate_metrics(actual, pred):
    rmse = math.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    mape = np.mean(np.abs((actual - pred) / (actual + 1e-10))) * 100
    return rmse, mae, r2, mape

def plot_predictions(actual, pred, state, energy_col, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label="Actual", color="blue", marker="o")
    plt.plot(pred, label="Predicted", color="red", linestyle="--", marker="x")
    plt.title(f"{state} - {energy_col} ({model_name})")
    plt.xlabel("Time Step")
    plt.ylabel("Energy Value")
    plt.legend()
    plt.grid(True)
    # Sanitize state and energy_col for filename
    safe_state = state.replace(" ", "_")
    safe_energy_col = energy_col.replace(" ", "_")
    plot_filename = os.path.join(PLOTS_PATH, f"{safe_state}{safe_energy_col}{model_name}.png")
    plt.savefig(plot_filename)
    plt.close()

# ----- LSTM Model -----
def lstm_model(X_train, Y_train):
    model = Sequential([
        Input(shape=(X_train.shape[1], 1)),        # <-- This ensures the InputLayer has correct shape!
        LSTM(50, return_sequences=False),
        Dense(1)
    ])
    # No need to call model.build() here!
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, Y_train, epochs=20, batch_size=16, verbose=0)
    return model

# ----- CNN Model -----
def cnn_model(X_train, Y_train, seq_len):
    model = Sequential([
        Input(shape=(seq_len, 1)),                 # <-- Use Input layer for serialization!
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Conv1D(filters=32, kernel_size=2, activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, Y_train, epochs=50, batch_size=8, verbose=0)
    return model

# ==============================
# 4. Process Each State
# ==============================
results = []
for state in df["state_name"].unique():
    print(f"\n==============================\nProcessing State: {state}\n==============================")
    state_df = df[df["state_name"] == state].sort_values(by="date")
    for energy_col in energy_types:
        data = state_df[energy_col].fillna(0).values.reshape(-1, 1)
        if len(data) < 20:
            print(f"⚠ Skipping {state} - {energy_col} (insufficient data)")
            continue
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)
        # ----------------------------
        # Split into train and test
        # ----------------------------
        seq_len = 10
        X, Y = create_sequences(data_scaled, seq_len)
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        Y_train, Y_test = Y[:split], Y[split:]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        # ============================
        # Run LSTM Model
        # ============================
        lstm = lstm_model(X_train, Y_train)
        lstm_pred = lstm.predict(X_test, verbose=0)
        lstm_pred = scaler.inverse_transform(lstm_pred)
        lstm_actual = scaler.inverse_transform(Y_test.reshape(-1, 1))
        lstm_rmse, lstm_mae, lstm_r2, lstm_mape = evaluate_metrics(lstm_actual, lstm_pred)
        # ============================
        # Run CNN Model
        # ============================
        cnn = cnn_model(X_train, Y_train, seq_len)
        cnn_pred = cnn.predict(X_test, verbose=0)
        cnn_pred = scaler.inverse_transform(cnn_pred)
        cnn_actual = scaler.inverse_transform(Y_test.reshape(-1, 1))
        cnn_rmse, cnn_mae, cnn_r2, cnn_mape = evaluate_metrics(cnn_actual, cnn_pred)
        # ============================
        # Compare and Choose Best
        # ============================
        # Create directory for saving models
        model_save_dir = os.path.join(MODEL_SAVE_PATH, state, energy_col)
        os.makedirs(model_save_dir, exist_ok=True)
        if cnn_r2 > lstm_r2:
            best_model = "CNN"
            best_rmse, best_mae, best_r2, best_mape = cnn_rmse, cnn_mae, cnn_r2, cnn_mape
            # Save CNN model
            cnn.save(os.path.join(model_save_dir, "best_model_CNN.h5"))
            # Plot CNN predictions
            plot_predictions(cnn_actual, cnn_pred, state, energy_col, "CNN")
        else:
            best_model = "LSTM"
            best_rmse, best_mae, best_r2, best_mape = lstm_rmse, lstm_mae, lstm_r2, lstm_mape
            # Save LSTM model
            lstm.save(os.path.join(model_save_dir, "best_model_LSTM.h5"))
            # Plot LSTM predictions
            plot_predictions(lstm_actual, lstm_pred, state, energy_col, "LSTM")
        results.append({
            "State": state,
            "Energy_Type": energy_col,
            "Best_Model": best_model,
            "Best_R2": best_r2,
            "Best_RMSE": best_rmse,
            "Best_MAE": best_mae,
            "Best_MAPE": best_mape,
            "LSTM_R2": lstm_r2,
            "LSTM_RMSE": lstm_rmse,
            "LSTM_MAE": lstm_mae,
            "LSTM_MAPE": lstm_mape,
            "CNN_R2": cnn_r2,
            "CNN_RMSE": cnn_rmse,
            "CNN_MAE": cnn_mae,
            "CNN_MAPE": cnn_mape
        })
        print(f"\n📊 {state} - {energy_col}:")
        print(f" LSTM → R2={lstm_r2:.3f}, RMSE={lstm_rmse:.3f}, MAE={lstm_mae:.3f}, MAPE={lstm_mape:.2f}%")
        print(f" CNN → R2={cnn_r2:.3f}, RMSE={cnn_rmse:.3f}, MAE={cnn_mae:.3f}, MAPE={cnn_mape:.2f}%")
        print(f"✅ Best Model: {best_model} (Saved to {model_save_dir})")
        print(f"📈 Plot saved to {PLOTS_PATH}")

# ==============================
# 5. Save Final Comparison
# ==============================
results_df = pd.DataFrame(results)
results_csv_path = os.path.join(RESULTS_LOG_PATH, "state_energy_model_comparison.csv")
results_df.to_csv(results_csv_path, index=False)
print("\n\n===== ✅ FINAL SUMMARY =====")
print(results_df)
print(f"\n✅ Comparison saved to '{results_csv_path}'")