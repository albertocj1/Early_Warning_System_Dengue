import pandas as pd
import numpy as np
import tensorflow as tf
import streamlit as st
import joblib
import requests
import datetime
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# --- Suppress sklearn version mismatch warnings ---
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# --- Your WeatherAPI key (directly used) ---
API_KEY = "9c8585dd43864b27a66224931251910"

# --- Model paths ---
MODEL_PATH = "Model/dengue_classification_model.keras"
SCALER_PATH = "Model/scaler_classification.pkl"

# --- Load model and scaler ---
try:
    model_classification = tf.keras.models.load_model(MODEL_PATH)
    scaler_classification = joblib.load(SCALER_PATH)
except Exception as e:
    st.error(f"⚠️ Error loading model or scaler: {e}")
    st.stop()

# --- Feature columns expected by the model ---
FEATURE_COLUMNS = [
    'RAINFALL', 'TMAX', 'TMIN', 'TMEAN', 'RH', 'SUNSHINE', 'POPULATION',
    'LAND AREA', 'POP_DENSITY', 'RAINFALL_lag1', 'RAINFALL_lag2',
    'RAINFALL_lag3', 'RAINFALL_lag4', 'TMAX_lag1', 'TMAX_lag2', 'TMAX_lag3',
    'TMAX_lag4', 'TMIN_lag1', 'TMIN_lag2', 'TMIN_lag3', 'TMIN_lag4',
    'TMEAN_lag1', 'TMEAN_lag2', 'TMEAN_lag3', 'TMEAN_lag4', 'RH_lag1',
    'RH_lag2', 'RH_lag3', 'RH_lag4', 'SUNSHINE_lag1', 'SUNSHINE_lag2',
    'SUNSHINE_lag3', 'SUNSHINE_lag4', 'RAINFALL_roll2_mean',
    'RAINFALL_roll4_mean', 'RAINFALL_roll2_sum', 'RAINFALL_roll4_sum',
    'TMEAN_roll2_mean', 'TMEAN_roll4_mean', 'TMEAN_roll2_sum',
    'TMEAN_roll4_sum', 'RH_roll2_mean', 'RH_roll4_mean', 'RH_roll2_sum',
    'RH_roll4_sum', 'INCIDENCE_per_100k', 'YEAR_WEEK_numerical'
]

# --- Fetch Weather Data from WeatherAPI ---
def fetch_weather_data(location: str):
    """Fetch 7-day weather forecast data from WeatherAPI."""
    url = f"http://api.weatherapi.com/v1/forecast.json?key={API_KEY}&q={location}&days=7"
    response = requests.get(url)
    if response.status_code != 200:
        st.error(f"❌ Failed to fetch weather data: {response.text}")
        return None
    return response.json()

# --- Process weather data into model-ready format ---
def process_weather_data(weather_data):
    forecast_days = weather_data['forecast']['forecastday']
    rainfall = [day['day']['totalprecip_mm'] for day in forecast_days]
    tmax = [day['day']['maxtemp_c'] for day in forecast_days]
    tmin = [day['day']['mintemp_c'] for day in forecast_days]
    tmean = [(hi + lo) / 2 for hi, lo in zip(tmax, tmin)]
    rh = [day['day']['avghumidity'] for day in forecast_days]
    sunshine = [day['day']['daily_chance_of_sunshine'] for day in forecast_days]

    week_data = {
        'RAINFALL': np.mean(rainfall),
        'TMAX': np.mean(tmax),
        'TMIN': np.mean(tmin),
        'TMEAN': np.mean(tmean),
        'RH': np.mean(rh),
        'SUNSHINE': np.mean(sunshine),
        'YEAR_WEEK_numerical': int(datetime.date.today().strftime("%Y%W")),
        # Static socio-demographic placeholders (can be updated for your city)
        'POPULATION': 150000,
        'LAND AREA': 100,
        'POP_DENSITY': 1500,
        'INCIDENCE_per_100k': 0
    }

    # Fill lag and rolling features with 0 (for live weekly prediction)
    for col in FEATURE_COLUMNS:
        if col not in week_data:
            week_data[col] = 0

    df = pd.DataFrame([week_data])
    df = df[FEATURE_COLUMNS]
    df[FEATURE_COLUMNS] = scaler_classification.transform(df[FEATURE_COLUMNS])
    reshaped = df.values.reshape((df.shape[0], 1, df.shape[1]))
    return reshaped, week_data

# --- Streamlit UI ---
st.set_page_config(page_title="Dengue Early Warning System", page_icon="🦠")
st.title("🦠 Weekly Dengue Early Warning System")
st.markdown("Predict dengue risk based on **real-time 7-day weather forecasts** using WeatherAPI and deep learning.")

# --- User input ---
location = st.text_input("📍 Enter City or Province (e.g., Manila, Cebu, Davao):", "Manila")

if st.button("🔍 Generate Weekly Prediction"):
    with st.spinner("Fetching weather data and predicting dengue risk..."):
        weather_data = fetch_weather_data(location)
        if weather_data:
            processed_input, week_summary = process_weather_data(weather_data)
            prediction = model_classification.predict(processed_input)
            predicted_labels = (prediction > 0.5).astype(int)
            risk_labels = ['Low', 'Moderate', 'High', 'Very High']
            predicted_risk_levels = [risk_labels[i] for i, label in enumerate(predicted_labels[0]) if label == 1]

            # --- Display results ---
            st.subheader("🌦 Weekly Weather Summary")
            st.dataframe(pd.DataFrame([week_summary]))

            st.subheader("📊 Predicted Dengue Risk Level")
            if predicted_risk_levels:
                for r in predicted_risk_levels:
                    if r == "Low":
                        st.success(f"🟢 **Risk Level:** {r}")
                    elif r == "Moderate":
                        st.info(f"🟡 **Risk Level:** {r}")
                    elif r == "High":
                        st.warning(f"🟠 **Risk Level:** {r}")
                    else:
                        st.error(f"🔴 **Risk Level:** {r}")
            else:
                st.warning("No risk level detected (model output below threshold).")

            st.caption("⚙️ Powered by WeatherAPI and Deep Learning (CNN-LSTM).")
