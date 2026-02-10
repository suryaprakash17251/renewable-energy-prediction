```python
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import logging
import json
from datetime import datetime
import base64

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model and dataset paths
MODEL_BASE_PATH = "C:/Users/chris/OneDrive/Desktop/renewable_energy_prediction/models/trained"
DATASET_PATH = "C:/Users/chris/OneDrive/Desktop/renewable energy prediction/data/processed.csv"
ENERGY_TYPES = {
    "wind_energy": "best_model_LSTM.h5",
    "solar_energy": "best_model_LSTM.h5",
    "other_renewable_energy": "best_model_CNN.h5"
}

# Load dataset
try:
    dataset = pd.read_csv(DATASET_PATH)
    dataset['date'] = pd.to_datetime(dataset['date'], format='%d-%m-%Y')
    logger.info("Dataset loaded successfully from: %s", DATASET_PATH)
except Exception as e:
    logger.error(f"Failed to load dataset: {e}")
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# State options
STATES = [
    "All India", "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chandigarh",
    "Chhattisgarh", "Delhi", "Goa", "Gujarat", "Haryana", "Himachal Pradesh",
    "Jammu And Kashmir", "Jharkhand", "Karnataka", "Kerala", "Ladakh", "Madhya Pradesh",
    "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Puducherry",
    "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana",
    "The Dadra And Nagar Haveli And Daman And Diu", "Tripura", "Uttar Pradesh",
    "Uttarakhand", "West Bengal"
]

def load_model(state, energy_type):
    state_clean = state.replace(' ', '_')
    model_path = os.path.join(MODEL_BASE_PATH, state_clean, energy_type, ENERGY_TYPES[energy_type])
    logger.debug(f"Attempting to load model from: {model_path}")
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        raise FileNotFoundError(f"Model not found at {model_path}")
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Model loaded successfully for {state}, {energy_type}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model at {model_path}: {e}")
        raise

def get_historical_data(state, energy_type):
    logger.debug(f"Fetching historical data for {state}, {energy_type}")
    state_data = dataset[dataset['state_name'] == state]
    if state_data.empty:
        logger.error(f"No data found for state: {state}")
        raise ValueError(f"No data found for state: {state}")
    values = state_data[energy_type].values[-30:]  # Last 30 days
    if len(values) < 30:
        logger.error(f"Insufficient data for {state}: {len(values)} values")
        raise ValueError(f"Insufficient data for {state}: {len(values)} values")
    if np.all(values == 0):
        logger.warning(f"All values are zero for {state}, {energy_type}")
    return values

def generate_predictions(state1, state2, energy_type, start_year, end_year):
    try:
        predictions = []
        model1 = load_model(state1, energy_type)
        input_data = get_historical_data(state1, energy_type)
        input_data = input_data.reshape((1, input_data.shape[0], 1))
        logger.debug(f"Input data shape for {state1}: {input_data.shape}")

        current_value = dataset[dataset['state_name'] == state1][energy_type].iloc[-1]
        logger.debug(f"Current value for {state1}, {energy_type}: {current_value}")

        for year in range(start_year, end_year + 1):
            try:
                pred = model1.predict(input_data, verbose=0)
                pred_value = float(pred[0][0])
                percent_change = ((pred_value - current_value) / current_value) * 100 if current_value != 0 else 0
                predictions.append({"year": year, "value": max(0, pred_value), "percentChange": percent_change})
                input_data = np.roll(input_data, -1)
                input_data[0, -1, 0] = pred_value
                logger.debug(f"Prediction for {state1}, year {year}: {pred_value}")
            except Exception as e:
                logger.error(f"Prediction failed for {state1}, year {year}: {e}")
                raise ValueError(f"Prediction failed for {state1}, year {year}: {e}")

        if state2:
            model2 = load_model(state2, energy_type)
            input_data2 = get_historical_data(state2, energy_type)
            input_data2 = input_data2.reshape((1, input_data2.shape[0], 1))
            try:
                pred2 = model2.predict(input_data2, verbose=0)
                pred_value2 = float(pred2[0][0])
                percent_change2 = ((pred_value2 - current_value) / current_value) * 100 if current_value != 0 else 0
                predictions = [
                    {"year": end_year, "value": max(0, predictions[-1]["value"]), "percentChange": predictions[-1]["percentChange"]},
                    {"year": end_year, "value": max(0, pred_value2), "percentChange": percent_change2}
                ]
                logger.debug(f"Prediction for {state2}, year {end_year}: {pred_value2}")
            except Exception as e:
                logger.error(f"Prediction failed for {state2}, year {end_year}: {e}")
                raise ValueError(f"Prediction failed for {state2}, year {end_year}: {e}")

        final_value = predictions[-1]["value"]
        percent_change = ((final_value - current_value) / current_value) * 100 if current_value != 0 else 0

        return predictions, percent_change
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise

# Streamlit UI
st.set_page_config(page_title="Renewable Energy Prediction", layout="wide")

# Theme toggle
theme = st.session_state.get('theme', 'light')
if st.button("Toggle Theme", key="theme_toggle"):
    theme = 'dark' if theme == 'light' else 'light'
    st.session_state.theme = theme

# Apply theme
theme_styles = """
    <style>
        body, .stApp { 
            background-color: %s; 
            color: %s; 
        }
        .stSelectbox, .stNumberInput, .stButton>button { 
            background-color: %s; 
            color: %s; 
            border: 1px solid %s; 
        }
        .stButton>button:hover { 
            background-color: %s; 
        }
        .error { 
            background-color: %s; 
            color: %s; 
            padding: 10px; 
            border-radius: 5px; 
        }
        .chart-container { 
            background-color: %s; 
            padding: 10px; 
            border-radius: 5px; 
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); 
            height: 400px; 
        }
    </style>
"""
light_theme = theme_styles % ("#ffffff", "#1f2937", "#f3f4f6", "#1f2937", "#d1d5db", "#2563eb", "#fee2e2", "#b91c1c", "#f3f4f6")
dark_theme = theme_styles % ("#1f2937", "#f3f4f6", "#374151", "#f3f4f6", "#4b5563", "#3b82f6", "#7f1d1d", "#f87171", "#374151")
st.markdown(light_theme if theme == 'light' else dark_theme, unsafe_allow_html=True)

st.title("Renewable Energy Prediction")

# Input form
with st.form(key="prediction_form"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        state1 = st.selectbox("State 1", STATES, index=STATES.index("Andhra Pradesh"))
    with col2:
        state2 = st.selectbox("State 2 (Optional)", [""] + STATES, index=0)
    with col3:
        energy_type = st.selectbox("Energy Type", ["wind_energy", "solar_energy", "other_renewable_energy"], format_func=lambda x: x.replace('_', ' ').title())
    with col4:
        graph_type = st.selectbox("Graph Type", ["bar", "line", "scatter"])
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        start_year = st.number_input("Start Year", min_value=2020, max_value=2050, value=2020)
    with col6:
        end_year = st.number_input("End Year", min_value=2020, max_value=2050, value=2030)
    with col7:
        x_axis = st.selectbox("X-Axis", ["years", "states"])
    with col8:
        y_axis = st.selectbox("Y-Axis", ["energy", "percentChange"], format_func=lambda x: "Energy (MW)" if x == "energy" else "Percent Change (%)")
    submit_button = st.form_submit_button("Generate Prediction")

# Initialize session state for saved graphs
if 'saved_graphs' not in st.session_state:
    st.session_state.saved_graphs = []

# Handle form submission
if submit_button:
    if start_year > end_year:
        st.error("Start year must be less than or equal to end year")
        logger.error("Invalid year range")
    else:
        with st.spinner("Generating predictions..."):
            try:
                predictions, percent_change = generate_predictions(state1, state2, energy_type, start_year, end_year)
                
                # Prepare Chart.js configuration
                labels = [str(p["year"]) for p in predictions] if not state2 else [state1, state2]
                data = [p["percentChange"] if y_axis == "percentChange" else p["value"] for p in predictions]
                
                chart_config = {
                    "type": "line" if graph_type == "scatter" else graph_type,
                    "data": {
                        "labels": labels,
                        "datasets": [{
                            "label": f"{energy_type.replace('_', ' ')} ({state1}{f' vs {state2}' if state2 else ''})",
                            "data": data,
                            "backgroundColor": "rgba(59, 130, 246, 0.5)",
                            "borderColor": "rgba(59, 130, 246, 1)",
                            "borderWidth": 1,
                            "pointRadius": 5 if graph_type == "scatter" else 0
                        }]
                    },
                    "options": {
                        "scales": {
                            "x": {"title": {"display": True, "text": "Years" if x_axis == "years" else "States"}, "type": "category"},
                            "y": {"title": {"display": True, "text": "Energy (MW)" if y_axis == "energy" else "Percent Change (%)"}, "beginAtZero": True}
                        },
                        "responsive": True,
                        "maintainAspectRatio": False,
                        "plugins": {"legend": {"display": True}, "tooltip": {"enabled": True}}
                    }
                }

                # Render Chart.js using HTML
                chart_html = f"""
                    <div class="chart-container">
                        <canvas id="energyChart"></canvas>
                    </div>
                    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
                    <script>
                        document.addEventListener('DOMContentLoaded', () => {{
                            const ctx = document.getElementById('energyChart').getContext('2d');
                            new Chart(ctx, {json.dumps(chart_config)});
                        }});
                    </script>
                """
                st.components.v1.html(chart_html, height=400)

                # Display percent change
                st.markdown(f"<div class='percent-change'>Percent Change from Current: {percent_change:.2f}%</div>", unsafe_allow_html=True)

                # Save graph button
                if st.button("Save Graph"):
                    graph_data = {
                        "id": int(datetime.now().timestamp() * 1000),
                        "state1": state1,
                        "state2": state2,
                        "energy_type": energy_type,
                        "start_year": start_year,
                        "end_year": end_year,
                        "graph_type": graph_type,
                        "x_axis": x_axis,
                        "y_axis": y_axis,
                        "chart_config": chart_config
                    }
                    st.session_state.saved_graphs.append(graph_data)
                    logger.info("Graph saved: %s", graph_data)

            except Exception as e:
                st.error(f"Error: {str(e)}")
                logger.error(f"Prediction error: {str(e)}")

# Display saved graphs
st.subheader("Saved Graphs")
if not st.session_state.saved_graphs:
    st.write("No saved graphs yet.")
else:
    cols = st.columns(3)
    for idx, graph in enumerate(st.session_state.saved_graphs):
        with cols[idx % 3]:
            chart_html = f"""
                <div class="chart-container">
                    <canvas id="savedChart{graph['id']}"></canvas>
                </div>
                <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
                <script>
                    document.addEventListener('DOMContentLoaded', () => {{
                        const ctx = document.getElementById('savedChart{graph['id']}').getContext('2d');
                        new Chart(ctx, {json.dumps(graph['chart_config'])});
                    }});
                </script>
                <p>{graph['state1']}{f" vs {graph['state2']}" if graph['state2'] else ''} - {graph['energy_type'].replace('_', ' ')} ({graph['start_year']}-{graph['end_year']})</p>
            """
            st.components.v1.html(chart_html, height=300)
            if st.button("Delete", key=f"delete_{graph['id']}"):
                st.session_state.saved_graphs = [g for g in st.session_state.saved_graphs if g['id'] != graph['id']]
                st.experimental_rerun()
```

### Setup Instructions
1. **File Structure**:
   - Place `app.py` in `C:/Users/chris/OneDrive/Desktop/renewable_energy_prediction/`.
   - Ensure `processed.csv` is at `C:/Users/chris/OneDrive/Desktop/renewable energy prediction/data/processed.csv`.
   - Verify model files, e.g., `C:/Users/chris/OneDrive/Desktop/renewable_energy_prediction/models/trained/Andhra_Pradesh/solar_energy/best_model_LSTM.h5`.

2. **Run the App**:
   - Activate your Python environment:
     ```bash
     .\venv\Scripts\activate
     ```
   - Install dependencies:
     ```bash
     pip install streamlit==1.31.0 tensorflow==2.12.0 numpy==1.24.3 pandas==2.0.3
     ```
   - Run the Streamlit app:
     ```bash
     streamlit run app.py
     ```
   - Open the provided URL (e.g., `http://localhost:8501`) in your browser.

3. **Expected Output**:
   - A Streamlit UI with a form for selecting state1, state2 (optional), energy type, graph type, start/end years, x-axis, and y-axis.
   - A "Generate Prediction" button that triggers your models to produce a Chart.js graph and percent change.
   - A "Save Graph" button to save the chart, displayed in the "Saved Graphs" section.
   - A theme toggle button for light/dark mode.

### Ensuring the Generate Button Works
- **Button Functionality**: The button triggers `generate_predictions`, which:
  - Loads your pre-trained model for the selected state and energy type.
  - Retrieves the last 30 days of data from `processed.csv`.
  - Generates predictions for the specified years (or compares two states if state2 is selected).
  - Renders a Chart.js graph and displays percent change.
- **Error Handling**: Displays user-friendly errors (e.g., “Model not found” or “Insufficient data”) and logs detailed errors to the terminal.
- **Streamlit Advantage**: Streamlit’s simpler rendering reduces white screen risks, with automatic error display.

### Debugging
If the button doesn’t work or you see a blank page:
1. **Browser Console** (F12, Console tab):
   - Check for Chart.js CDN errors (`https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js`).
   - Look for JavaScript errors in the HTML component.
2. **Terminal Logs**:
   - Verify `Dataset loaded successfully`.
   - Check for errors like:
     - `Failed to load dataset`: Ensure `processed.csv` exists and has correct columns.
     - `Model not found at ...`: Verify model paths.
     - `Insufficient data for ...`: Ensure 30+ days of data.
3. **Test Predictions**:
   - Select “Andhra Pradesh,” “solar_energy,” 2020–2030, “line” graph, and click “Generate Prediction.”
   - Check logs for `Prediction for Andhra_Pradesh, year ...: ...`.
4. **Streamlit Server**:
   - Ensure `streamlit run app.py` starts without errors.
   - Test `http://localhost:8501` in Chrome, Firefox, or Edge.

### Clarifications Needed
To ensure the button works with your models:
1. **Model Input/Output**: Confirm your models expect `(1, 30, 1)` input and output a single value. If different, share the exact input shape and output format.
2. **Dataset Columns**: Verify `processed.csv` has `date`, `state_name`, `wind_energy`, `solar_energy`, `other_renewable_energy`. If different, share column names.
3. **State Names**: Confirm state names in `processed.csv` match the `STATES` list (e.g., “Andhra Pradesh”). If not, share a sample state name.

### If Issues Persist
- Share terminal logs after clicking the button.
- Share browser console errors (F12, Console tab).
- Confirm dataset and model paths.
- Test with a minimal dataset if needed (I can provide a template).

This Streamlit app prioritizes the "Generate Prediction" button, using your pre-trained models and dataset exclusively. I’vestr