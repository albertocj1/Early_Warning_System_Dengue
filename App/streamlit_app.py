import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# ================================================================
# APP TITLE
# ================================================================
st.set_page_config(page_title="Weekly Early Warning System for Dengue", layout="wide")
st.title("🦟 Weekly Early Warning System for Dengue (Metro Manila)")
st.caption("Automated integration of static and weekly weather data using WeatherAPI.com")

# ================================================================
# STATIC DATA
# ================================================================
land_area_data = {
    "CITY": ["MANILA CITY", "QUEZON CITY", "CALOOCAN CITY", "LAS PINAS CITY", "MAKATI CITY",
             "MALABON CITY", "MANDALUYONG CITY", "MARIKINA CITY", "MUNTINLUPA CITY",
             "NAVOTAS CITY", "PARANAQUE CITY", "PASAY CITY", "PASIG CITY", "PATEROS",
             "SAN JUAN CITY", "TAGUIG CITY", "VALENZUELA CITY"],
    "LAND AREA": [24.98, 171.71, 55.8, 32.69, 21.57, 15.71, 9.29, 21.52, 39.75, 8.94,
                  46.57, 55.8, 48.46, 10.4, 5.95, 45.21, 47.02]
}

pop_data = {
    "CITY": ["MANILA CITY", "QUEZON CITY", "CALOOCAN CITY", "LAS PINAS CITY", "MAKATI CITY",
             "MALABON CITY", "MANDALUYONG CITY", "MARIKINA CITY", "MUNTINLUPA CITY",
             "NAVOTAS CITY", "PARANAQUE CITY", "PASAY CITY", "PASIG CITY", "PATEROS",
             "SAN JUAN CITY", "TAGUIG CITY", "VALENZUELA CITY"],
    "POP_2015": [1780148, 2936116, 1583978, 588894, 582602, 365525, 386276, 450741, 504509,
                 249463, 665822, 416522, 755300, 63840, 122180, 804915, 620422],
    "POP_2020": [1846513, 2960048, 1661584, 606293, 629616, 380522, 425758, 456059, 543445,
                 247543, 689992, 440656, 803159, 65227, 126347, 886722, 714978]
}

df_land = pd.DataFrame(land_area_data)
df_pop = pd.DataFrame(pop_data)
df_static = pd.merge(df_land, df_pop, on="CITY")

# ================================================================
# WEATHER API CONFIG
# ================================================================
st.header("🌦️ Weather Data Fetching")
API_KEY = "9c8585dd43864b27a66224931251910"  # WeatherAPI.com key
base_url = "https://api.weatherapi.com/v1/forecast.json"

def get_weekly_weather(city):
    """Fetch 7-day weather data from WeatherAPI.com and compute weekly averages."""
    try:
        params = {
            "key": API_KEY,
            "q": city + ", Philippines",
            "days": 7,
            "aqi": "no",
            "alerts": "no"
        }
        response = requests.get(base_url, params=params)
        data = response.json()

        if "error" in data:
            st.warning(f"⚠️ {city}: {data['error']['message']}")
            return None

        forecast = data.get("forecast", {}).get("forecastday", [])
        if not forecast:
            return None

        df = pd.DataFrame([{
            "DATE": f["date"],
            "RAINFALL": f["day"]["totalprecip_mm"],
            "TMAX": f["day"]["maxtemp_c"],
            "TMIN": f["day"]["mintemp_c"],
            "TMEAN": f["day"]["avgtemp_c"],
            "RH": f["day"]["avghumidity"],
            "SUNSHINE": f["day"]["daily_chance_of_sunshine"]
        } for f in forecast])

        # Compute weekly averages
        summary = df.mean(numeric_only=True).to_dict()
        summary["CITY"] = city
        summary["YR-WEEK"] = datetime.now().strftime("%Y-W%U")
        return summary
    except Exception as e:
        st.error(f"Error fetching {city}: {e}")
        return None

# ================================================================
# MAIN ACTION
# ================================================================
if st.button("🔄 Fetch Weekly Weather Data"):
    st.info("Fetching 7-day forecast data for Metro Manila cities. Please wait...")
    weather_data = [get_weekly_weather(city) for city in df_static["CITY"]]
    weather_df = pd.DataFrame([d for d in weather_data if d])

    if not weather_df.empty:
        # Merge with static data
        merged_df = pd.merge(df_static, weather_df, on="CITY")
        merged_df["POP_DENSITY"] = merged_df["POP_2020"] / merged_df["LAND AREA"]

        st.success("✅ Weekly weather data fetched and merged successfully!")
        st.subheader("📊 Combined Data (Static + Weather Features)")
        st.dataframe(merged_df, use_container_width=True)

        # Save and allow download
        csv = merged_df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Weekly Data (CSV)",
            data=csv,
            file_name="weekly_dengue_input.csv",
            mime="text/csv"
        )

        st.caption("Tip: This CSV can be directly used for your dengue prediction model input.")
    else:
        st.warning("No weather data fetched. Check API key or network connection.")

# ================================================================
# FOOTER
# ================================================================
st.markdown("""
---
**Developed by:** Christian Joshua Q. Alberto  
**Data Sources:**  
- [WeatherAPI.com](https://www.weatherapi.com/) for weather data  
- Philippine Statistics Authority (PSA) for population and land area data  
---
""")
