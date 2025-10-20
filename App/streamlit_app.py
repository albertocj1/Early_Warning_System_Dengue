import pandas as pd
import numpy as np
import tensorflow as tf
import streamlit as st
import joblib
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# --- Suppress sklearn version mismatch warnings (safe for StandardScaler) ---
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# --- Paths to model and scaler ---
MODEL_PATH = "Model/dengue_classification_model.keras"
SCALER_PATH = "Model/scaler_classification.pkl"

# --- Load model and scaler ---
try:
    model_classification = tf.keras.models.load_model(MODEL_PATH)
    scaler_classification = joblib.load(SCALER_PATH)
    st.success("✅ Model and Scaler loaded successfully!")
except FileNotFoundError:
    st.error("❌ Model or scaler file not found. Ensure both exist in the /Model directory.")
    st.stop()
except Exception as e:
    st.error(f"⚠️ Error loading model or scaler: {e}")
    st.stop()

# --- Define the model's expected input features ---
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

# --- Preprocessing Function ---
def preprocess_input(input_df: pd.DataFrame):
    """Preprocess user input for the dengue risk prediction model."""
    
    # Convert YEAR_WEEK (e.g., 2023-W40) to a numeric representation (YYYYWW)
    if 'YEAR_WEEK' in input_df.columns:
        def convert_year_week_to_numerical(week_str):
            try:
                year, week = week_str.split('-W')
                return int(year) * 100 + int(week)
            except:
                return np.nan

        input_df['YEAR_WEEK_numerical'] = input_df['YEAR_WEEK'].apply(convert_year_week_to_numerical)
        input_df.drop(columns=['YEAR_WEEK'], inplace=True, errors='ignore')

        # ✅ Fixed assignment to avoid chained assignment warning
        if input_df['YEAR_WEEK_numerical'].isnull().any():
            input_df['YEAR_WEEK_numerical'] = input_df['YEAR_WEEK_numerical'].fillna(
                input_df['YEAR_WEEK_numerical'].mean()
            )

    # Ensure all required features are present
    for col in FEATURE_COLUMNS:
        if col not in input_df.columns:
            input_df[col] = 0

    # Enforce column order
    input_df = input_df[FEATURE_COLUMNS]

    # Scale features using the loaded scaler
    try:
        input_df[FEATURE_COLUMNS] = scaler_classification.transform(input_df[FEATURE_COLUMNS])
    except Exception as e:
        st.warning("⚠️ Scaling may not be applied properly. Check your scaler version or data format.")
        st.write(e)

    # Reshape for CNN-LSTM model input: (samples, timesteps, features)
    input_reshaped = input_df.values.reshape((input_df.shape[0], 1, input_df.shape[1]))
    return input_reshaped

# --- Streamlit Interface ---
st.title("🦠 Dengue Risk Level Prediction")
st.markdown("Enter the feature values below to predict the **Dengue Risk Level** for a given week.")

input_data = {}
st.subheader("🌤 Environmental and Socio-Demographic Inputs")

# Input fields for all features except YEAR_WEEK_numerical (user enters YEAR_WEEK)
for feature in FEATURE_COLUMNS:
    if feature == 'YEAR_WEEK_numerical':
        year_week_str = st.text_input("YEAR_WEEK (e.g., 2023-W40)", value="2023-W40")
        input_data['YEAR_WEEK'] = year_week_str
    else:
        input_data[feature] = st.number_input(feature, value=0.0, format="%.4f")

# --- Prediction Button ---
if st.button("🔍 Predict Risk Level"):
    try:
        # Convert user input to DataFrame
        input_df = pd.DataFrame([input_data])
        processed_input = preprocess_input(input_df)

        # Run model prediction
        prediction_probabilities = model_classification.predict(processed_input)
        predicted_labels = (prediction_probabilities > 0.5).astype(int)

        # Define class labels (ensure same order as training output)
        risk_labels = ['Low', 'Moderate', 'High', 'Very High']

        # Display prediction results
        st.subheader("📊 Prediction Results:")
        predicted_risk_levels = [risk_labels[i] for i, label in enumerate(predicted_labels[0]) if label == 1]

        if predicted_risk_levels:
            st.success("### Predicted Risk Level(s):")
            for risk in predicted_risk_levels:
                st.write(f"- **{risk}**")
        else:
            st.warning("No clear risk level detected above threshold.")
            st.write("**Raw Prediction Probabilities:**")
            for i, label in enumerate(risk_labels):
                st.write(f"{label}: {prediction_probabilities[0][i]:.4f}")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
