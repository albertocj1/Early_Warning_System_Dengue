import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import joblib
from sklearn.preprocessing import StandardScaler

# ----------------------------
# 🔑 WeatherAPI Configuration
# ----------------------------
WEATHER_API_KEY = "9c8585dd43864b27a66224931251910"
BASE_URL = "https://api.weatherapi.com/v1"

# ----------------------------
# 🧭 Static Data
# ----------------------------
CITY_DATA = {
    "MANILA CITY": {"land_area": 24.98, "pop_2015": 1780148, "pop_2020": 1846513},
    "QUEZON CITY": {"land_area": 171.71, "pop_2015": 2936116, "pop_2020": 2960048},
    "CALOOCAN CITY": {"land_area": 55.8, "pop_2015": 1583978, "pop_2020": 1661584},
    "LAS PINAS CITY": {"land_area": 32.69, "pop_2015": 588894, "pop_2020": 606293},
    "MAKATI CITY": {"land_area": 21.57, "pop_2015": 582602, "pop_2020": 629616},
    "MALABON CITY": {"land_area": 15.71, "pop_2015": 365525, "pop_2020": 380522},
    "MANDALUYONG CITY": {"land_area": 9.29, "pop_2015": 386276, "pop_2020": 425758},
    "MARIKINA CITY": {"land_area": 21.52, "pop_2015": 450741, "pop_2020": 456059},
    "MUNTINLUPA CITY": {"land_area": 39.75, "pop_2015": 504509, "pop_2020": 543445},
    "NAVOTAS CITY": {"land_area": 8.94, "pop_2015": 249463, "pop_2020": 247543},
    "PARANAQUE CITY": {"land_area": 46.57, "pop_2015": 665822, "pop_2020": 689992},
    "PASAY CITY": {"land_area": 55.8, "pop_2015": 416522, "pop_2020": 440656},
    "PASIG CITY": {"land_area": 48.46, "pop_2015": 755300, "pop_2020": 803159},
    "PATEROS": {"land_area": 10.4, "pop_2015": 63840, "pop_2020": 65227},
    "SAN JUAN CITY": {"land_area": 5.95, "pop_2015": 122180, "pop_2020": 126347},
    "TAGUIG CITY": {"land_area": 45.21, "pop_2015": 804915, "pop_2020": 886722},
    "VALENZUELA CITY": {"land_area": 47.02, "pop_2015": 620422, "pop_2020": 714978},
}

# ----------------------------
# ⚙️ Load Model
# ----------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("dengue_model.pkl")
        return model
    except:
        st.error("Model not found. Please upload dengue_model.pkl")
        return None


# ----------------------------
# 🌦️ WeatherAPI Fetch Function
# ----------------------------
def fetch_weather_data(city):
    url = f"{BASE_URL}/forecast.json?key={WEATHER_API_KEY}&q={city}&days=7"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        forecast_days = data["forecast"]["forecastday"]

        records = []
        for day in forecast_days:
            date = day["date"]
            avg_temp = day["day"]["avgtemp_c"]
            max_temp = day["day"]["maxtemp_c"]
            min_temp = day["day"]["mintemp_c"]
            rainfall = day["day"]["totalprecip_mm"]
            humidity = day["day"]["avghumidity"]
            sunshine = 100 - day["day"]["daily_chance_of_rain"]  # chance of sun (approx)

            records.append({
                "DATE": date,
                "RAINFALL": rainfall,
                "TMAX": max_temp,
                "TMIN": min_temp,
                "TMEAN": avg_temp,
                "RH": humidity,
                "SUNSHINE": sunshine,
            })

        return pd.DataFrame(records)
    else:
        st.error(f"WeatherAPI Error {response.status_code}: {response.text}")
        return None


# ----------------------------
# 🧮 Feature Engineering
# ----------------------------
def add_lag_features(df, column, lags=4):
    for lag in range(1, lags + 1):
        df[f"{column}_lag{lag}"] = df[column].shift(lag)
    return df


def add_rolling_features(df, column, window=3):
    df[f"{column}_roll_mean"] = df[column].rolling(window=window).mean()
    df[f"{column}_roll_std"] = df[column].rolling(window=window).std()
    return df


# ----------------------------
# 🧩 Prediction Pipeline
# ----------------------------
def prepare_data(df, city):
    city_info = CITY_DATA[city]
    df["CITY"] = city
    df["LAND AREA"] = city_info["land_area"]

    # Estimate 2025 population via linear interpolation
    pop_2015, pop_2020 = city_info["pop_2015"], city_info["pop_2020"]
    growth_rate = (pop_2020 - pop_2015) / 5
    pop_2025 = pop_2020 + growth_rate
    df["POPULATION"] = pop_2025
    df["POP_DENSITY"] = pop_2025 / city_info["land_area"]

    # Add lag and rolling features
    for col in ["RAINFALL", "TMEAN", "RH"]:
        df = add_lag_features(df, col)
        df = add_rolling_features(df, col)

    # Drop NaN after lag creation
    df = df.dropna().reset_index(drop=True)
    return df


# ----------------------------
# 🖥️ Streamlit Interface
# ----------------------------
st.set_page_config(page_title="Weekly Dengue Early Warning System", layout="wide")

st.title("🦟 Weekly Early Warning System for Dengue")
st.markdown("This app predicts weekly dengue risk using weather and population data from WeatherAPI.com")

city = st.selectbox("Select City", list(CITY_DATA.keys()))

if st.button("Run Prediction"):
    st.info("Fetching weather data...")
    weather_df = fetch_weather_data(city)

    if weather_df is not None:
        st.write("📅 7-Day Weather Forecast:")
        st.dataframe(weather_df)

        st.info("Preparing data and features...")
        data_ready = prepare_data(weather_df, city)

        model = load_model()
        if model:
            st.success("✅ Model loaded successfully.")
            X = data_ready.select_dtypes(include=[np.number])

            preds = model.predict(X)
            data_ready["Predicted_Risk"] = preds

            st.subheader("📊 Weekly Dengue Early Warning:")
            st.dataframe(data_ready[["DATE", "CITY", "Predicted_Risk"]])

            st.line_chart(data_ready.set_index("DATE")["Predicted_Risk"])
