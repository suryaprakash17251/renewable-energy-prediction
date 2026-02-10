import numpy as np
import tensorflow as tf
import json, os, math
import matplotlib.pyplot as plt

# ==================================================
# USER CONFIG
# ==================================================
STATE = "Andhra_Pradesh"       # e.g. "All_India", "Andhra_Pradesh"
ENERGY_TYPE = "solar_energy"   # "solar_energy", "wind_energy", "other_renewable_energy"
SEQ_LEN = 12                   # Same as during training

BASE_MODEL_PATH = r"C:\Users\chris\OneDrive\Desktop\renewable energy prediction\models\trained1"
PARAM_FEATURE_PATH = r"C:\Users\chris\OneDrive\Desktop\renewable energy prediction\param_data\scaler_features.json"
PARAM_STATE_PATH = r"C:\Users\chris\OneDrive\Desktop\renewable energy prediction\param_data\scaler_params_per_state.json"

# ==================================================
# 1. Locate model files
# ==================================================
state_dir = os.path.join(BASE_MODEL_PATH, STATE, ENERGY_TYPE)
if os.path.exists(os.path.join(state_dir, "best_model_LSTM_tfjs")):
    model_dir = os.path.join(state_dir, "best_model_LSTM_tfjs")
elif os.path.exists(os.path.join(state_dir, "best_model_CNN_tfjs")):
    model_dir = os.path.join(state_dir, "best_model_CNN_tfjs")
else:
    raise FileNotFoundError(f"No TFJS model found for {STATE} – {ENERGY_TYPE}")

model_json_path = os.path.join(model_dir, "model.json")
model_bin_path = os.path.join(model_dir, "group1-shard1of1.bin")

if not os.path.exists(model_json_path) or not os.path.exists(model_bin_path):
    raise FileNotFoundError(f"Missing TFJS model files in {model_dir}")

print(f"🔹 Loading TFJS model from {model_dir}")

# ==================================================
# 2. Load model.json manually and reconstruct Keras model
# ==================================================
with open(model_json_path, "r") as f:
    model_json = json.load(f)

# Extract architecture
model_config = model_json.get("modelTopology")
weights_manifest = model_json.get("weightsManifest")[0]["weights"]

# Load binary weights
with open(model_bin_path, "rb") as f:
    weights_data = f.read()

# Convert binary data to float32
import io
import struct
weights_buffer = np.frombuffer(weights_data, dtype=np.float32)

# Load model from JSON (TensorFlow 2.x handles this safely)
model = tf.keras.models.model_from_json(json.dumps(model_config))

# Because TFJS stores weights as one big binary, we reload weights by shape:
offset = 0
for layer, w in zip(model.weights, weights_manifest):
    shape = w["shape"]
    size = np.prod(shape)
    array = weights_buffer[offset:offset+size].reshape(shape)
    offset += size
    layer.assign(array)
print("✅ Model loaded successfully from TFJS JSON + BIN files")

# ==================================================
# 3. Load scaling parameters
# ==================================================
with open(PARAM_FEATURE_PATH, "r") as f:
    feat_params = json.load(f)
with open(PARAM_STATE_PATH, "r") as f:
    state_params = json.load(f)

# ==================================================
# 4. Helper functions
# ==================================================
def scale_value(value, vmin, vmax):
    return (value - vmin) / (vmax - vmin + 1e-10)

def inverse_scale(value, vmin, vmax):
    return value * (vmax - vmin + 1e-10) + vmin

def get_seasonal_features(month, year, feat_params):
    quarter = math.ceil(month / 3)
    sin_m = math.sin(2 * math.pi * month / 12)
    cos_m = math.cos(2 * math.pi * month / 12)
    year_norm = scale_value(year, 2010, 2025)
    return [
        scale_value(month, feat_params["month"]["min"], feat_params["month"]["max"]),
        scale_value(quarter, feat_params["quarter"]["min"], feat_params["quarter"]["max"]),
        scale_value(sin_m, feat_params["sin"]["min"], feat_params["sin"]["max"]),
        scale_value(cos_m, feat_params["cos"]["min"], feat_params["cos"]["max"]),
        scale_value(year_norm, feat_params["year_norm"]["min"], feat_params["year_norm"]["max"])
    ]

# ==================================================
# 5. Prepare base input sequence
# ==================================================
if STATE not in state_params or ENERGY_TYPE not in state_params[STATE]:
    raise ValueError(f"No scaling params found for {STATE} – {ENERGY_TYPE}")

state_min = state_params[STATE][ENERGY_TYPE]["min"]
state_max = state_params[STATE][ENERGY_TYPE]["max"]

base_energy = scale_value((state_min + state_max) / 2, state_min, state_max)
base_seq = []
year_start, month_start = 2025, 1
for i in range(SEQ_LEN):
    month = (month_start + i - 1) % 12 + 1
    year = year_start if month_start + i <= 12 else year_start + 1
    base_seq.append([base_energy] + get_seasonal_features(month, year, feat_params))
base_seq = np.array([base_seq])

# ==================================================
# 6. Predict next 5 years (60 months)
# ==================================================
print("🔹 Predicting next 5 years...")
preds_scaled = []
current_seq = base_seq.copy()

for step in range(60):
    pred_scaled = model.predict(current_seq, verbose=0)[0][0]
    preds_scaled.append(pred_scaled)

    next_month = (month_start + SEQ_LEN + step - 1) % 12 + 1
    next_year = year_start + ((month_start + SEQ_LEN + step - 1) // 12)
    next_input = [pred_scaled] + get_seasonal_features(next_month, next_year, feat_params)
    current_seq = np.append(current_seq[:, 1:, :], [[next_input]], axis=1)

preds = [inverse_scale(v, state_min, state_max) for v in preds_scaled]

# ==================================================
# 7. Plot results
# ==================================================
years, months = [], []
cy, cm = year_start, month_start
for _ in range(60):
    years.append(cy)
    months.append(cm)
    cm += 1
    if cm > 12:
        cm = 1
        cy += 1
x_labels = [f"{m:02d}/{y}" for m, y in zip(months, years)]

plt.figure(figsize=(12, 6))
plt.plot(x_labels, preds, color='red', marker='o', label=f"{STATE} – {ENERGY_TYPE}")
plt.title(f"{STATE} – {ENERGY_TYPE.replace('_', ' ').title()} (Predicted 2026–2030)")
plt.xlabel("Month/Year")
plt.ylabel("Predicted Energy Value")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
