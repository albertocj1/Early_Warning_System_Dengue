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
st.set_page_config(
    page_title="🦟 Dengue Weekly Early Warning System",
    page_icon="🦠",
    layout="wide"
)

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
# ☀️ FETCH SUNSHINE FROM OPEN-METEO
# -------------------------------------
def fetch_sunshine_open_meteo(lat, lon, days=7):
    start_date = datetime.date.today()
    end_date = start_date + datetime.timedelta(days=days - 1)

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}"
        f"&longitude={lon}"
        f"&daily=sunshine_duration"
        f"&timezone=Asia/Manila"
        f"&start_date={start_date}"
        f"&end_date={end_date}"
    )

    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"Open-Meteo error {r.status_code}: {r.text}")

    data = r.json()
    dates = data["daily"]["time"]
    sunshine = data["daily"]["sunshine_duration"]

    return pd.DataFrame({
        "DATE": dates,
        "SUNSHINE": [round(s / 3600, 2) if s else 0 for s in sunshine]
    })

# -------------------------------------
# 🌦️ FETCH WEATHER FROM WEATHERAPI
# -------------------------------------
def fetch_weather_forecast(city, days=7):
    url = f"http://api.weatherapi.com/v1/forecast.json?key={WEATHER_API_KEY}&q={city}&days={days}"
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"WeatherAPI error {r.status_code}: {r.text}")

    data = r.json()

    weather_df = pd.DataFrame([
        {
            "DATE": d["date"],
            "RAINFALL": d["day"]["totalprecip_mm"],
            "TMAX": d["day"]["maxtemp_c"],
            "TMIN": d["day"]["mintemp_c"],
            "TMEAN": d["day"]["avgtemp_c"],
            "RH": d["day"]["avghumidity"],
        }
        for d in data["forecast"]["forecastday"]
    ])

    info = CITY_DATA[city]
    sunshine_df = fetch_sunshine_open_meteo(info["lat"], info["lon"], days)

    return pd.merge(weather_df, sunshine_df, on="DATE", how="left")

# -------------------------------------
# 📈 FEATURE ENGINEERING
# -------------------------------------
def add_lag_and_rolling(df):
    df = df.sort_values("DATE").reset_index(drop=True)
    cols = ["RAINFALL", "TMAX", "TMIN", "TMEAN", "RH", "SUNSHINE"]

    for c in cols:
        for l in range(1, 5):
            df[f"{c}_lag{l}"] = df[c].shift(l)

    for c in ["RAINFALL", "TMEAN", "RH"]:
        df[f"{c}_roll2_mean"] = df[c].rolling(2).mean()
        df[f"{c}_roll4_mean"] = df[c].rolling(4).mean()
        df[f"{c}_roll2_sum"] = df[c].rolling(2).sum()
        df[f"{c}_roll4_sum"] = df[c].rolling(4).sum()

    return df.dropna().reset_index(drop=True)

# -------------------------------------
# 🧮 POPULATION PROJECTION
# -------------------------------------
def project_population(city, target_year=2025):
    info = CITY_DATA[city]
    growth = (info["pop_2020"] / info["pop_2015"]) ** (1 / 5) - 1
    return int(info["pop_2020"] * ((1 + growth) ** (target_year - 2020)))

# -------------------------------------
# 🧩 PREPARE MODEL INPUT
# -------------------------------------
def prepare_data(df, city, recent_cases):
    pop = project_population(city)
    area = CITY_DATA[city]["land_area"]

    df["POPULATION"] = pop
    df["LAND AREA"] = area
    df["POP_DENSITY"] = pop / area
    df["YEAR_WEEK_numerical"] = df["DATE"].apply(
        lambda x: int(pd.to_datetime(x).isocalendar().year * 100 +
                      pd.to_datetime(x).isocalendar().week)
    )

    df = add_lag_and_rolling(df)

    df["CASES_lag1"], df["CASES_lag2"], df["CASES_lag3"], df["CASES_lag4"] = recent_cases
    df["CASES_roll2_mean"] = np.mean(recent_cases[:2])
    df["CASES_roll4_mean"] = np.mean(recent_cases)
    df["CASES_roll2_sum"] = np.sum(recent_cases[:2])
    df["CASES_roll4_sum"] = np.sum(recent_cases)

    X = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], 0).fillna(0)
    X_scaled = scaler_classification.transform(X)
    return df, X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# -------------------------------------
# 🖥️ STREAMLIT UI
# -------------------------------------
st.title("🦠 Weekly Dengue Early Warning System")

city = st.selectbox("🏙️ Select City", list(CITY_DATA.keys()))

st.markdown("### 🧮 Enter Recent Weekly Dengue Cases")
cols = st.columns(4)
recent_cases = [
    cols[0].number_input("Week -1", min_value=0, value=10),
    cols[1].number_input("Week -2", min_value=0, value=8),
    cols[2].number_input("Week -3", min_value=0, value=6),
    cols[3].number_input("Week -4", min_value=0, value=5),
]

if st.button("Run Weekly Prediction"):
    with st.spinner("Fetching data..."):
        weather_df = fetch_weather_forecast(city)

    df_ready, X = prepare_data(weather_df, city, recent_cases)
    preds = model_classification.predict(X)

    labels = ["Low", "Moderate", "High", "Very High"]
    df_ready["Predicted_Risk"] = [labels[np.argmax(p)] for p in preds]

    st.subheader("📊 Dengue Risk Forecast")
    st.dataframe(df_ready[["DATE", "Predicted_Risk"]])

    st.download_button(
        "📥 Download CSV",
        df_ready.to_csv(index=False),
        f"{city}_dengue_forecast.csv",
        "text/csv"
    )
