import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import joblib
import tensorflow as tf
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# -------------------------------------
# ⚙️ CONFIG
# -------------------------------------
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
st.set_page_config(page_title="🦟 Dengue Weekly Early Warning System", page_icon="🦠", layout="wide")

# -------------------------------------
# 🔑 API KEYS
# -------------------------------------
WEATHER_API_KEY = "9c8585dd43864b27a66224931251910"

# -------------------------------------
# 📦 MODEL PATHS
# -------------------------------------
MODEL_PATH = "Model/dengue_classification_model.keras"
SCALER_PATH = "Model/scaler_classification.pkl"

# -------------------------------------
# 🌍 STATIC CITY DATA
# -------------------------------------
CITY_DATA = {
    "MANILA CITY": {"lat": 14.6, "lon": 120.98, "land_area": 24.98, "pop_2015": 1780148, "pop_2020": 1846513},
    "QUEZON CITY": {"lat": 14.676, "lon": 121.0437, "land_area": 171.71, "pop_2015": 2936116, "pop_2020": 2960048},
    "CALOOCAN CITY": {"lat": 14.65, "lon": 120.97, "land_area": 55.8, "pop_2015": 1583978, "pop_2020": 1661584},
    "LAS PINAS CITY": {"lat": 14.45, "lon": 120.98, "land_area": 32.69, "pop_2015": 588894, "pop_2020": 606293},
    "MAKATI CITY": {"lat": 14.55, "lon": 121.03, "land_area": 21.57, "pop_2015": 582602, "pop_2020": 629616},
    "MALABON CITY": {"lat": 14.67, "lon": 120.96, "land_area": 15.71, "pop_2015": 365525, "pop_2020": 380522},
    "MANDALUYONG CITY": {"lat": 14.58, "lon": 121.04, "land_area": 9.29, "pop_2015": 386276, "pop_2020": 425758},
    "MARIKINA CITY": {"lat": 14.65, "lon": 121.1, "land_area": 21.52, "pop_2015": 450741, "pop_2020": 456059},
    "MUNTINLUPA CITY": {"lat": 14.38, "lon": 121.04, "land_area": 39.75, "pop_2015": 504509, "pop_2020": 543445},
    "NAVOTAS CITY": {"lat": 14.67, "lon": 120.95, "land_area": 8.94, "pop_2015": 249463, "pop_2020": 247543},
    "PARANAQUE CITY": {"lat": 14.48, "lon": 121.02, "land_area": 46.57, "pop_2015": 665822, "pop_2020": 689992},
    "PASAY CITY": {"lat": 14.55, "lon": 121.0, "land_area": 55.8, "pop_2015": 416522, "pop_2020": 440656},
    "PASIG CITY": {"lat": 14.58, "lon": 121.08, "land_area": 48.46, "pop_2015": 755300, "pop_2020": 803159},
    "PATEROS": {"lat": 14.55, "lon": 121.07, "land_area": 10.4, "pop_2015": 63840, "pop_2020": 65227},
    "SAN JUAN CITY": {"lat": 14.6, "lon": 121.03, "land_area": 5.95, "pop_2015": 122180, "pop_2020": 126347},
    "TAGUIG CITY": {"lat": 14.52, "lon": 121.05, "land_area": 45.21, "pop_2015": 804915, "pop_2020": 886722},
    "VALENZUELA CITY": {"lat": 14.7, "lon": 120.97, "land_area": 47.02, "pop_2015": 620422, "pop_2020": 714978},
}


# -------------------------------------
# 🧠 LOAD MODEL + SCALER
# -------------------------------------
@st.cache_resource
def load_model_and_scaler():
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model_classification, scaler_classification = load_model_and_scaler()

# -------------------------------------
# ☀️ FETCH SUNSHINE DATA FROM METEOMATICS
# -------------------------------------
def fetch_sunshine_meteomatics(lat=14.6, lon=120.98, days=7):
    username = "nationaluniversity-manila_alberto_christianjoshua"
    password = "l1898PFZcsuDiKEMOhM0"

    start_date = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = start_date + datetime.timedelta(days=days - 1)
    start_str = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")

    parameter = "sunshine_duration_24h:min"
    url = f"https://api.meteomatics.com/{start_str}--{end_str}:P1D/{parameter}/{lat},{lon}/json"

    r = requests.get(url, auth=(username, password), timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"Meteomatics error {r.status_code}: {r.text}")

    data = r.json()
    records = []
    for item in data["data"][0]["coordinates"][0]["dates"]:
        date = item["date"].split("T")[0]
        minutes = item["value"]
        hours = round(minutes / 60, 2)
        records.append({"DATE": date, "SUNSHINE": hours})
    return pd.DataFrame(records)

# -------------------------------------
# 🌦️ FETCH WEATHER FROM WEATHERAPI
# -------------------------------------
def fetch_weather_forecast(city: str, days: int = 7):
    url = f"http://api.weatherapi.com/v1/forecast.json?key={WEATHER_API_KEY}&q={city}&days={days}"
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"WeatherAPI error {r.status_code}: {r.text}")
    data = r.json()

    weather_records = []
    for day in data["forecast"]["forecastday"]:
        date = day["date"]
        f = day["day"]
        weather_records.append({
            "DATE": date,
            "RAINFALL": f["totalprecip_mm"],
            "TMAX": f["maxtemp_c"],
            "TMIN": f["mintemp_c"],
            "TMEAN": f["avgtemp_c"],
            "RH": f["avghumidity"],
        })

    weather_df = pd.DataFrame(weather_records)
    city_info = CITY_DATA.get(city, {"lat": 14.6, "lon": 120.98})
    sunshine_df = fetch_sunshine_meteomatics(city_info["lat"], city_info["lon"], days)
    merged_df = pd.merge(weather_df, sunshine_df, on="DATE", how="left")
    return merged_df

# -------------------------------------
# 📈 FEATURE ENGINEERING
# -------------------------------------
def add_lag_and_rolling(df):
    df = df.sort_values("DATE").reset_index(drop=True)
    numeric_cols = ["RAINFALL", "TMAX", "TMIN", "TMEAN", "RH", "SUNSHINE"]

    for col in numeric_cols:
        for lag in range(1, 5):
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    for col in ["RAINFALL", "TMEAN", "RH"]:
        df[f"{col}_roll2_mean"] = df[col].rolling(2).mean()
        df[f"{col}_roll4_mean"] = df[col].rolling(4).mean()
        df[f"{col}_roll2_sum"] = df[col].rolling(2).sum()
        df[f"{col}_roll4_sum"] = df[col].rolling(4).sum()

    return df.dropna().reset_index(drop=True)

# -------------------------------------
# 🧮 POPULATION PROJECTION
# -------------------------------------
def project_population(city, target_year=2025):
    info = CITY_DATA[city]
    p2015, p2020 = info["pop_2015"], info["pop_2020"]
    growth_rate = (p2020 / p2015) ** (1 / 5) - 1
    return int(p2020 * ((1 + growth_rate) ** (target_year - 2020)))

# -------------------------------------
# 🧩 PREDICTION DATA PIPELINE
# -------------------------------------
def prepare_data(df, city, recent_cases):
    pop = project_population(city)
    area = CITY_DATA[city]["land_area"]

    df["CITY"] = city
    df["LAND AREA"] = area
    df["POPULATION"] = pop
    df["POP_DENSITY"] = pop / area
    df["YEAR_WEEK_numerical"] = df["DATE"].apply(
        lambda x: int(pd.to_datetime(x).isocalendar().year * 100 + pd.to_datetime(x).isocalendar().week)
    )

    df = add_lag_and_rolling(df)

    # --- Manual dengue case inputs ---
    df["CASES_lag1"] = recent_cases[0]
    df["CASES_lag2"] = recent_cases[1]
    df["CASES_lag3"] = recent_cases[2]
    df["CASES_lag4"] = recent_cases[3]

    df["CASES_roll2_mean"] = np.mean(recent_cases[:2])
    df["CASES_roll4_mean"] = np.mean(recent_cases)
    df["CASES_roll2_sum"] = np.sum(recent_cases[:2])
    df["CASES_roll4_sum"] = np.sum(recent_cases)

    feature_order = [
        'RAINFALL', 'TMAX', 'TMIN', 'TMEAN', 'RH', 'SUNSHINE',
        'POPULATION', 'LAND AREA', 'POP_DENSITY',
        'CASES_lag1', 'CASES_lag2', 'CASES_lag3', 'CASES_lag4',
        'RAINFALL_lag1', 'RAINFALL_lag2', 'RAINFALL_lag3', 'RAINFALL_lag4',
        'TMAX_lag1', 'TMAX_lag2', 'TMAX_lag3', 'TMAX_lag4',
        'TMIN_lag1', 'TMIN_lag2', 'TMIN_lag3', 'TMIN_lag4',
        'TMEAN_lag1', 'TMEAN_lag2', 'TMEAN_lag3', 'TMEAN_lag4',
        'RH_lag1', 'RH_lag2', 'RH_lag3', 'RH_lag4',
        'SUNSHINE_lag1', 'SUNSHINE_lag2', 'SUNSHINE_lag3', 'SUNSHINE_lag4',
        'CASES_roll2_mean', 'CASES_roll4_mean', 'CASES_roll2_sum', 'CASES_roll4_sum',
        'RAINFALL_roll2_mean', 'RAINFALL_roll4_mean', 'RAINFALL_roll2_sum', 'RAINFALL_roll4_sum',
        'TMEAN_roll2_mean', 'TMEAN_roll4_mean', 'TMEAN_roll2_sum', 'TMEAN_roll4_sum',
        'RH_roll2_mean', 'RH_roll4_mean', 'RH_roll2_sum', 'RH_roll4_sum',
        'YEAR_WEEK_numerical'
    ]

    # ✅ Ensure all required columns exist and in correct order
    for col in feature_order:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_order]

    return df

# -------------------------------------
# 🖥️ STREAMLIT APP
# -------------------------------------
st.title("🦠 Weekly Dengue Early Warning System")
st.markdown("Predicts dengue risk using WeatherAPI + Meteomatics sunshine + lag & rolling features.")

city = st.selectbox("🏙️ Select City", list(CITY_DATA.keys()))

# --- Manual dengue case input section ---
st.markdown("### 🧮 Enter Recent Weekly Dengue Cases")
col1, col2, col3, col4 = st.columns(4)
case_week1 = col1.number_input("Week -1 (Most Recent)", min_value=0, step=1, value=10)
case_week2 = col2.number_input("Week -2", min_value=0, step=1, value=8)
case_week3 = col3.number_input("Week -3", min_value=0, step=1, value=6)
case_week4 = col4.number_input("Week -4", min_value=0, step=1, value=5)
recent_cases = [case_week1, case_week2, case_week3, case_week4]

if st.button("Run Weekly Prediction"):
    try:
        with st.spinner("Fetching weather and sunshine data..."):
            weather_df = fetch_weather_forecast(city)
        st.success("✅ Data fetched successfully!")
        st.dataframe(weather_df)

        with st.spinner("Preparing model features..."):
            df_ready = prepare_data(weather_df, city, recent_cases)

            # ✅ Align and scale safely
            X = df_ready.select_dtypes(include=[np.number])
            X = X.replace([np.inf, -np.inf], 0).fillna(0)
            X_scaled = scaler_classification.transform(X)
            X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

        with st.spinner("Running model predictions..."):
            preds = model_classification.predict(X_scaled)
            labels = ["Low", "Moderate", "High", "Very High"]
            df_ready["Predicted_Risk"] = [labels[np.argmax(p)] for p in preds]

        st.subheader("📊 Weekly Dengue Risk Forecast")
        st.dataframe(df_ready[["YEAR_WEEK_numerical", "Predicted_Risk"]])

        csv = df_ready.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Full Results as CSV",
            data=csv,
            file_name=f"{city}_dengue_forecast.csv",
            mime="text/csv"
        )

        st.line_chart(df_ready.set_index("YEAR_WEEK_numerical")["Predicted_Risk"].astype("category").cat.codes)

    except Exception as e:
        st.error(f"❌ Error: {e}")
