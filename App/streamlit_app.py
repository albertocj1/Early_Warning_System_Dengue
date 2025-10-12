import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model
# Ensure the path matches where you saved the model
model = tf.keras.models.load_model('Model/dengue_classification_model.keras')

# Load the scaler used during training
# Assuming you saved the scaler as 'scaler_classification.pkl' during data preprocessing
# If you didn't save it, you'll need to regenerate and save it or incorporate scaling differently
try:
    with open('scaler_classification.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("Scaler file not found. Please ensure 'scaler_classification.pkl' is available.")
    st.stop() # Stop the app if the scaler is not found

# Define the expected feature columns (in the same order as during training)
# This list should match the columns in your X_classification DataFrame after preprocessing
# You can get this list by running X_classification.columns.tolist() in your notebook
feature_columns = ['DEATHS', 'RAINFALL', 'TMAX', 'TMIN', 'TMEAN', 'RH', 'SUNSHINE',
                   'POPULATION', 'LAND AREA', 'POP_DENSITY', 'CASES_lag1', 'CASES_lag2',
                   'CASES_lag3', 'CASES_lag4', 'DEATHS_lag1', 'DEATHS_lag2', 'DEATHS_lag3',
                   'DEATHS_lag4', 'RAINFALL_lag1', 'RAINFALL_lag2', 'RAINFALL_lag3',
                   'RAINFALL_lag4', 'TMAX_lag1', 'TMAX_lag2', 'TMAX_lag3', 'TMAX_lag4',
                   'TMIN_lag1', 'TMIN_lag2', 'TMIN_lag3', 'TMIN_lag4', 'TMEAN_lag1',
                   'TMEAN_lag2', 'TMEAN_lag3', 'TMEAN_lag4', 'RH_lag1', 'RH_lag2',
                   'RH_lag3', 'RH_lag4', 'SUNSHINE_lag1', 'SUNSHINE_lag2', 'SUNSHINE_lag3',
                   'SUNSHINE_lag4', 'CASES_roll2_mean', 'CASES_roll4_mean',
                   'CASES_roll2_sum', 'CASES_roll4_sum', 'RAINFALL_roll2_mean',
                   'RAINFALL_roll4_mean', 'RAINFALL_roll2_sum', 'RAINFALL_roll4_sum',
                   'TMEAN_roll2_mean', 'TMEAN_roll4_mean', 'TMEAN_roll2_sum',
                   'TMEAN_roll4_sum', 'RH_roll2_mean', 'RH_roll4_mean', 'RH_roll2_sum',
                   'RH_roll4_sum', 'INCIDENCE_per_100k', 'YEAR_WEEK_numerical',
                   'CITY_CALOOCAN CITY', 'CITY_LAS PINAS CITY', 'CITY_MAKATI CITY',
                   'CITY_MALABON CITY', 'CITY_MANDALUYONG CITY', 'CITY_MANILA CITY',
                   'CITY_MARIKINA CITY', 'CITY_MUNTINLUPA CITY', 'CITY_NAVOTAS CITY',
                   'CITY_PARANAQUE CITY', 'CITY_PASAY CITY', 'CITY_PASIG CITY',
                   'CITY_PATEROS', 'CITY_QUEZON CITY', 'CITY_SAN JUAN CITY',
                   'CITY_TAGUIG CITY', 'CITY_VALENZUELA CITY'] # This needs to be updated based on your final X_classification columns

# Define the target column names for display
target_names = ['RISK_LEVEL_Low', 'RISK_LEVEL_Moderate', 'RISK_LEVEL_High', 'RISK_LEVEL_Very High']


st.title("Dengue Risk Level Prediction")

st.write("Enter the feature values below to predict the dengue risk level.")

# Create input fields for each feature
input_data = {}
for feature in feature_columns:
    # You might want to customize the input type based on the feature (e.g., number_input, text_input)
    # For simplicity, using number_input for all numerical features
    if feature.startswith('CITY_'):
        # Handle city as a selectbox
        city_options = [col.replace('CITY_', '') for col in feature_columns if col.startswith('CITY_')]
        selected_city = st.selectbox("Select City", city_options)
        # One-hot encode the selected city
        for city_col in [col for col in feature_columns if col.startswith('CITY_')]:
            input_data[city_col] = 1 if city_col == f'CITY_{selected_city}' else 0
    else:
        input_data[feature] = st.number_input(f"Enter {feature}", value=0.0)

if st.button("Predict Risk Level"):
    # Create a DataFrame from the input data
    input_df = pd.DataFrame([input_data])

    # Ensure the columns are in the correct order
    input_df = input_df[feature_columns]

    # Preprocess the input data (scaling)
    # Make sure the scaler is fitted on the training data before saving
    try:
        input_scaled = scaler.transform(input_df)
    except Exception as e:
        st.error(f"Error during scaling: {e}")
        st.stop()


    # Reshape the input for the model (samples, timesteps, features)
    input_reshaped = input_scaled.reshape((input_scaled.shape[0], 1, input_scaled.shape[1]))

    # Make prediction
    prediction = model.predict(input_reshaped)

    # Convert predicted probabilities to binary labels using a threshold (e.g., 0.5)
    predicted_labels = (prediction > 0.5).astype(int)[0] # Get the first (and only) sample's predictions

    st.subheader("Predicted Risk Levels:")
    for i, target_name in enumerate(target_names):
        status = "Predicted" if predicted_labels[i] == 1 else "Not Predicted"
        st.write(f"- {target_name.replace('RISK_LEVEL_', '').replace('_', ' ')}: {status}")