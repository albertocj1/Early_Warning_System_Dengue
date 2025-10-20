# streamlit_ews.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import joblib
import tensorflow as tf
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# ---------- CONFIG ----------
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
st.set_page_config(page_title="Dengue Weekly Early Warning System", page_icon="🦠", layout="wide")

# ---------- API KEYS ----------
OPENWEATHER_API_KEY = "9c8585dd43864b27a66224931251910"
METEOMATICS_USER = "nationaluniversity-manila_alberto_christianjoshua"
METEOMATICS_PASS = "l1898PFZcsuDiKEMOhM0"

# ---------- MODEL PATHS ----------
MODEL_PATH = "Model/dengue_classification_model.keras"
SCALER_PATH = "Model/scaler_classification.pkl"

# ---------- STATIC CITY DATA ----------
LAND_AREA = {
    "MANILA CITY": 24.98, "QUEZON CITY": 171.71, "CALOOCON CITY": 55.8,
    "LAS PINAS CITY": 32.69, "MAKATI CITY": 21.57, "MALABON CITY": 15.71,
    "MANDALUYONG CITY": 9.29, "MARIKINA CITY": 21.52, "MUNTINLUPA CITY": 39.75,
    "NAVOTAS CITY": 8.94, "PARANAQUE CITY": 46.57, "PASAY CITY": 55.8,
    "PASIG CITY": 48.46, "PATEROS": 10.4, "SAN JUAN CITY": 5.95,
    "TAGUIG CITY": 45.21, "VALENZUELA CITY": 47.02
}

POP_2015_2020 = {
    "MANILA CITY": (1780148, 1846513),
    "MANDALUYONG CITY": (386276, 425758),
    "MARIKINA CITY": (450741, 456059),
    "PASIG CITY": (755300, 803159),
    "QUEZON CITY": (2936116, 2960048),
    "SAN JUAN CITY": (122180, 126347),
    "CALOOCON CITY": (1583978, 1661584),
    "MALABON CITY": (365525, 380522),
    "NAVOTAS CITY": (249463, 247543),
    "VALENZUELA CITY": (620422, 714978),
    "LAS PINAS CITY": (588894, 606293),
    "MAKATI CITY": (582602, 629616),
    "MUNTINLUPA CITY": (504509, 543445),
    "PARANAQUE CITY": (665822, 689992),
    "PASAY CITY": (416522, 440656),
    "PATEROS": (63840, 65227),
    "TAGUIG CITY": (804915, 886722)
}

# ---------- FUNCTIONS ----------
def project_population_to_year(city: str, target_year: int = None):
    if target_year is None:
        current_year = datetime.date.today().year
        cycle = 5 * ((current_year + 4) // 5)
        target_year = cycle
    city_key = city.upper()
    if city_key not in POP_2015_2020:
        return 100000
    p2015, p2020 = POP_2015_2020[city_key]
    cagr = (p2020 / p2015) ** (1 / 5) - 1
    years_to_project = target_year - 2020
    return int(p2020 * ((1 + cagr) ** years_to_project))

def get_population_density(city: str, population: int):
    land = LAND_AREA.get(city.upper(), np.nan)
    return population / land if land > 0 else np.nan

def yr_week_from_date(dt: datetime.date):
    iso = dt.isocalendar()
    return iso[0] * 100 + iso[1]

def geocode_city(city: str):
    url = f"http://api.weatherapi.com/v1/search.json?key={OPENWEATHER_API_KEY}&q={city}"
    r = requests.get(url, timeout=10)
    data = r.json()
    if not data:
        raise ValueError("City not found.")
    return data[0]["lat"], data[0]["lon"]

def fetch_weatherapi_daily(lat: float, lon: float, days: int = 28):
    """Fetch daily data (rainfall, temp, humidity) for the past X days."""
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date)
    records = []

    for d in dates:
        url = f"http://api.weatherapi.com/v1/history.json?key={OPENWEATHER_API_KEY}&q={lat},{lon}&dt={d.strftime('%Y-%m-%d')}"
        r = requests.get(url, timeout=10)
        data = r.json()
        if "forecast" in data:
            f = data["forecast"]["forecastday"][0]["day"]
            records.append({
                "date": d,
                "RAINFALL": f["totalprecip_mm"],
                "TMAX": f["maxtemp_c"],
                "TMIN": f["mintemp_c"],
                "TMEAN": (f["maxtemp_c"] + f["mintemp_c"]) / 2,
                "RH": f["avghumidity"]
            })
    return pd.DataFrame(records)

def fetch_sunshine_meteomatics(lat: float, lon: float, days: int = 28):
    start_date = (datetime.date.today() - datetime.timedelta(days=days)).strftime("%Y-%m-%dT00:00:00Z")
    end_date = datetime.date.today().strftime("%Y-%m-%dT00:00:00Z")
    parameter = "sunshine_duration_24h:min"
    url = f"https://api.meteomatics.com/{start_date}--{end_date}:P1D/{parameter}/{lat},{lon}/json"
    response = requests.get(url, auth=(METEOMATICS_USER, METEOMATICS_PASS))
    if response.status_code != 200:
        raise RuntimeError(f"Meteomatics error {response.status_code}: {response.text}")
    data = response.json()
    records = [
        {"date": datetime.datetime.fromisoformat(d["date"][:-1]).date(),
         "SUNSHINE": d["value"] / 60}  # convert min → hr
        for d in data["data"][0]["coordinates"][0]["dates"]
    ]
    return pd.DataFrame(records)

def generate_lag_roll_features(df, columns):
    df = df.sort_values("date").reset_index(drop=True)
    for col in columns:
        for lag in range(1, 5):
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
        df[f"{col}_roll2"] = df[col].rolling(2).mean()
        df[f"{col}_roll4"] = df[col].rolling(4).mean()
    return df.dropna().reset_index(drop=True)

@st.cache_resource
def load_model_and_scaler():
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

# ---------- LOAD MODEL ----------
model_classification, scaler_classification = load_model_and_scaler()

# ---------- UI ----------
st.title("🦠 Dengue Weekly Early Warning System")
st.markdown("Uses WeatherAPI + Meteomatics with lag & rolling features for weekly dengue risk prediction.")

city = st.selectbox("Select City", sorted(list(LAND_AREA.keys())), index=0)
incidence = st.number_input("Current INCIDENCE_per_100k (if known)", value=0.0, format="%.4f")

if st.button("Generate Weekly Early Warning"):
    with st.spinner("Fetching weather data and computing features..."):
        try:
            lat, lon = geocode_city(city)

            # Fetch 4 weeks of data
            weather_df = fetch_weatherapi_daily(lat, lon, days=28)
            sunshine_df = fetch_sunshine_meteomatics(lat, lon, days=28)

            # Merge datasets
            df = pd.merge(weather_df, sunshine_df, on="date", how="inner")

            # Add lag & rolling features
            feature_cols = ["RAINFALL", "TMAX", "TMIN", "TMEAN", "RH", "SUNSHINE"]
            df = generate_lag_roll_features(df, feature_cols)

            # Use latest week
            latest = df.iloc[-1].to_dict()

            # Add static features
            pop_proj = project_population_to_year(city)
            land = LAND_AREA.get(city.upper(), 0)
            pop_density = get_population_density(city, pop_proj)
            latest["POPULATION"] = pop_proj
            latest["LAND AREA"] = land
            latest["POP_DENSITY"] = pop_density
            latest["INCIDENCE_per_100k"] = incidence
            latest["YEAR_WEEK_numerical"] = int(yr_week_from_date(datetime.date.today()))

            input_df = pd.DataFrame([latest])

            # Scale and reshape
            scaled = scaler_classification.transform(input_df)
            x = scaled.reshape((1, 1, scaled.shape[1]))

            # Predict
            probs = model_classification.predict(x)
            class_labels = ["Low", "Moderate", "High", "Very High"]
            probs_flat = probs[0]
            predicted = class_labels[int(np.argmax(probs_flat))]

            # ---------- DISPLAY ----------
            st.subheader("📊 Latest Week Data (with lags/rolls)")
            st.dataframe(input_df.T.rename(columns={0: "Value"}))

            st.subheader("🔍 Predicted Dengue Risk Level")
            if predicted == "Low":
                st.success(f"🟢 {predicted}")
            elif predicted == "Moderate":
                st.info(f"🟡 {predicted}")
            elif predicted == "High":
                st.warning(f"🟠 {predicted}")
            else:
                st.error(f"🔴 {predicted}")

        except Exception as e:
            st.error(f"Failed: {e}")
