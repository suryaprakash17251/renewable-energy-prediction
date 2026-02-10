
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import logging
import json

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
MODEL_BASE_PATH = "C:/Users/chris/OneDrive/Desktop/renewable energy prediction/models/trained"
DATASET_PATH = "C:/Users/chris/OneDrive/Desktop/renewable energy prediction/data/processed.csv"
ENERGY_TYPES = ["wind_energy", "solar_energy", "other_renewable_energy"]

# Load dataset
try:
    dataset = pd.read_csv(DATASET_PATH)
    dataset['date'] = pd.to_datetime(dataset['date'], format='%d-%m-%Y')
    logger.info("Dataset loaded successfully from: %s", DATASET_PATH)
except Exception as e:
    logger.error(f"Failed to load dataset: {e}")
    st.error(f"Failed to load dataset: {e}. Check file path and format.")
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
    model_names = [f"best_model_LSTM.h5", f"best_model_CNN.h5"]
    model_path = None
    for model_name in model_names:
        path = os.path.join(MODEL_BASE_PATH, state_clean, energy_type, model_name)
        if os.path.exists(path):
            model_path = path
            break
    if not model_path:
        logger.error(f"No model found for {state}, {energy_type} at {MODEL_BASE_PATH}/{state_clean}/{energy_type}/")
        raise FileNotFoundError(f"No model found for {state}, {energy_type}. Expected best_model_LSTM.h5 or best_model_CNN.h5 at {MODEL_BASE_PATH}/{state_clean}/{energy_type}/")
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model at {model_path}: {e}")
        raise

def get_historical_data(state, energy_type):
    logger.debug(f"Fetching data for {state}, {energy_type}")
    state_data = dataset[dataset['state_name'] == state]
    if state_data.empty:
        logger.error(f"No data for state: {state}")
        raise ValueError(f"No data for state: {state}")
    values = state_data[energy_type].values[-30:]
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
        input_data = input_data.reshape((1, 30, 1))
        logger.debug(f"Input shape for {state1}: {input_data.shape}")

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
            input_data2 = input_data2.reshape((1, 30, 1))
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
if st.button("Toggle Theme"):
    theme = 'dark' if theme == 'light' else 'light'
    st.session_state.theme = theme

# Apply theme
theme_styles = """
    <style>
        .stApp { background-color: %s; color: %s; }
        .stSelectbox, .stNumberInput, .stButton>button { background-color: %s; color: %s; border: 1px solid %s; }
        .stButton>button:hover { background-color: %s; }
        .error { background-color: %s; color: %s; padding: 10px; border-radius: 5px; }
        .chart-container { background-color: %s; padding: 10px; border-radius: 5px; height: 400px; }
    </style>
"""
light_theme = theme_styles % ("#ffffff", "#1f2937", "#f3f4f6", "#1f2937", "#d1d5db", "#2563eb", "#fee2e2", "#b91c1c", "#f3f4f6")
dark_theme = theme_styles % ("#1f2937", "#f3f4f6", "#374151", "#f3f4f6", "#4b5563", "#3b82f6", "#7f1d1d", "#f87171", "#374151")
st.markdown(light_theme if theme == 'light' else dark_theme, unsafe_allow_html=True)

st.title("Renewable Energy Prediction")

# Input form
with st.form(key="prediction_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        state1 = st.selectbox("State 1", STATES, index=STATES.index("Andhra Pradesh"))
    with col2:
        state2 = st.selectbox("State 2 (Optional)", [""] + STATES, index=0)
    with col3:
        energy_type = st.selectbox("Energy Type", ENERGY_TYPES, format_func=lambda x: x.replace('_', ' ').title())
    col4, col5, col6 = st.columns(3)
    with col4:
        start_year = st.number_input("Start Year", min_value=2020, max_value=2050, value=2020)
    with col5:
        end_year = st.number_input("End Year", min_value=2020, max_value=2050, value=2030)
    with col6:
        graph_type = st.selectbox("Graph Type", ["bar", "line", "scatter"])
    submit_button = st.form_submit_button("Generate Prediction", type="primary")

# Initialize session state
if 'saved_graphs' not in st.session_state:
    st.session_state.saved_graphs = []

# Handle prediction
if submit_button:
    if start_year > end_year:
        st.error("Start year must be less than or equal to end year")
        logger.error("Invalid year range")
    else:
        with st.spinner("Generating predictions..."):
            try:
                predictions, percent_change = generate_predictions(state1, state2, energy_type, start_year, end_year)
                if not predictions:
                    st.error("No predictions generated")
                    logger.error("Empty predictions")
                    st.stop()

                # Chart.js config
                labels = [str(p["year"]) for p in predictions] if not state2 else [state1, state2]
                data = [p["value"] for p in predictions]
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
                            "x": {"title": {"display": True, "text": "Years" if not state2 else "States"}, "type": "category"},
                            "y": {"title": {"display": True, "text": "Energy (MW)"}, "beginAtZero": True}
                        },
                        "responsive": True,
                        "maintainAspectRatio": False
                    }
                }

                # Render chart
                chart_id = f"chart_{len(st.session_state.saved_graphs)}"
                chart_html = f"""
                    <div class="chart-container">
                        <canvas id="{chart_id}"></canvas>
                    </div>
                    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
                    <script>
                        const ctx = document.getElementById('{chart_id}').getContext('2d');
                        new Chart(ctx, {json.dumps(chart_config)});
                    </script>
                """
                st.components.v1.html(chart_html, height=400)

                # Display percent change
                st.markdown(f"<div class='percent-change'>Percent Change from Current: {percent_change:.2f}%</div>", unsafe_allow_html=True)

                # Save graph button
                if st.button("Save Graph"):
                    st.session_state.saved_graphs.append({
                        "id": len(st.session_state.saved_graphs),
                        "state1": state1,
                        "state2": state2,
                        "energy_type": energy_type,
                        "start_year": start_year,
                        "end_year": end_year,
                        "chart_config": chart_config
                    })
                    logger.info("Graph saved")
                    st.experimental_rerun()

            except Exception as e:
                st.error(f"Error: {str(e)}")
                logger.error(f"Prediction error: {str(e)}")

# Saved graphs
st.subheader("Saved Graphs")
if not st.session_state.saved_graphs:
    st.write("No saved graphs yet.")
else:
    for graph in st.session_state.saved_graphs:
        chart_id = f"saved_chart_{graph['id']}"
        chart_html = f"""
            <div class="chart-container">
                <canvas id="{chart_id}"></canvas>
            </div>
            <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
            <script>
                const ctx = document.getElementById('{chart_id}').getContext('2d');
                new Chart(ctx, {json.dumps(graph['chart_config'])});
            </script>
            <p>{graph['state1']}{f" vs {graph['state2']}" if graph['state2'] else ''} - {graph['energy_type'].replace('_', ' ')} ({graph['start_year']}-{graph['end_year']})</p>
        """
        st.components.v1.html(chart_html, height=300)
        if st.button("Delete", key=f"delete_{graph['id']}"):
            st.session_state.saved_graphs = [g for g in st.session_state.saved_graphs if g['id'] != graph['id']]
            st.experimental_rerun()