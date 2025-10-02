import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
# import json # No longer needed if not loading feature_names.json
# import joblib # No longer needed if not loading scaler.pkl
# import requests # Uncomment if you will call a weather API directly

# Load the trained classification model
@st.cache_resource
def load_model():
    # Use the correct path to your saved model file
    model = tf.keras.models.load_model('Model/dengue_risk_level_classification_model.h5')
    return model

model = load_model()

# --- Scaler and Feature Names (Placeholders - NOT FOR REAL USE) ---
# This section uses placeholders as requested. In a real application,
# you MUST load the fitted scaler and the exact list of feature names
# used during model training.

# Determine the number of features from your trained model's input shape
# Use the actual input shape of your trained model
num_features = 62 # Replace with the actual number of features your model expects
feature_names = [f'feature_{i}' for i in range(num_features)] # Placeholder feature names

# Dummy scaler - REPLACE WITH YOUR LOADED FITTED SCALER IN A REAL APP
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# This dummy scaler is NOT fitted and will not correctly scale your data.
# In a real application, you need to load the scaler fitted on your training data.


# --- Streamlit App Layout ---
st.title('Dengue Risk Level Prediction with Weather Data')

st.write("Enter feature values, including weather data, to predict the dengue risk level.")

# Create input fields for features
input_data = {}
st.subheader("Input Features")

# You need to create input widgets for ALL num_features features
# (62 features in this case) and ensure they are in the correct order and data type.
# This is a simplified example.

# Example input fields for some features - YOU NEED TO ADD ALL 62 features
# Grouping inputs logically is recommended.

st.subheader("Weather Inputs")
# Example numerical weather inputs - replace with actual feature names
# These feature names must match the order and names used in your model's training data
# For this example, using generic names
if 'feature_0' in feature_names: # Replace 'feature_0' with the actual name of your first weather feature
    input_data['feature_0'] = st.number_input('Current Week Rainfall', value=0.0)
if 'feature_1' in feature_names: # Replace 'feature_1' with the actual name of your second weather feature
    input_data['feature_1'] = st.number_input('Current Week Max Temperature', value=30.0)
# Add inputs for all other weather features (current, lagged, rolled)...

st.subheader("Location")
# Example for one-hot encoded city - replace with actual city names and feature names
city_names = ['CALOOCAN CITY', 'LAS PINAS CITY', 'MAKATI CITY', 'MALABON CITY', 'MANDALUYONG CITY', 'MANILA CITY', 'MARIKINA CITY', 'MUNTINLUPA CITY', 'NAVOTAS CITY', 'PARANAQUE CITY', 'PASAY CITY', 'PASIG CITY', 'PATEROS', 'QUEZON CITY', 'SAN JUAN CITY', 'TAGUIG CITY', 'VALENZUELA CITY'] # Replace with your actual city names
selected_city = st.selectbox("Select City", city_names)

# You will need to map the selected_city to the correct one-hot encoded column name and set its value to 1.
# This requires knowing the exact column names from your training data.
# Example:
# if 'CITY_MAKATI_CITY' in feature_names: input_data['CITY_MAKATI_CITY'] = 1.0 # Example for Makati City


st.subheader("Other Features")
# Example other features - replace with actual feature names
if 'feature_20' in feature_names: # Replace 'feature_20' with the actual name of one of your other features
    input_data['feature_20'] = st.number_input('Population', value=1600000)
# Add inputs for all other features (lagged cases, population density, year_week, etc.)...


# --- API Call Integration (Example - Uncomment and modify) ---
# If you want to fetch data from an API directly in the app:
# api_key = "YOUR_WEATHER_API_KEY" # Get this securely, e.g., from Streamlit secrets
# city_for_api = selected_city # Or map selected_city to an API-compatible location name
# api_url = f"YOUR_WEATHER_API_ENDPOINT?location={city_for_api}&apikey={api_key}"
#
# if st.button('Fetch Weather Data from API'):
#     try:
#         response = requests.get(api_url)
#         response.raise_for_status() # Raise an exception for bad status codes
#         weather_data_from_api = response.json()
#
#         # Process weather_data_from_api to extract relevant features and update input_data
#         # This is where you map API response keys to your model's feature names.
#         # Example:
#         # input_data['feature_0'] = weather_data_from_api.get('current_rainfall', 0.0) # Map API data to placeholder feature names
#         # input_data['feature_1'] = weather_data_from_api.get('current_temp', 0.0)
#         # ... and calculate lagged/rolled features using historical data if needed.
#
#         st.success("Weather data fetched successfully from API (example). Please fill in other features and predict.")
#     except requests.exceptions.RequestException as e:
#         st.error(f"Error fetching weather data from API: {e}")
#         st.warning("Using manually entered weather data.")


if st.button('Predict Risk Level'):
    # --- Prepare Input Data for Prediction ---
    # Create a DataFrame with all expected features, initialized to 0.0
    # Use the placeholder feature_names for column names
    input_df = pd.DataFrame(0.0, index=[0], columns=feature_names)

    # Populate the DataFrame with user inputs
    # Map input_data from widgets to the correct placeholder feature_names columns
    for feature_name, value in input_data.items():
        if feature_name in input_df.columns:
             input_df[feature_name] = value

    # Handle one-hot encoded city: set the value to 1 for the selected city's column
    # This requires knowing the exact column names from your training data.
    # Example (REPLACE WITH ACTUAL COLUMN NAME LOGIC):
    city_col_name = f'CITY_{selected_city.replace(" ", "_").upper()}' # Example - adjust if needed
    if city_col_name in input_df.columns: # Check if the generated column name exists in placeholder features
         input_df[city_col_name] = 1.0
    # You would need to iterate through feature_names to find the correct city column
    # For example:
    # for col in feature_names:
    #     if col.startswith('CITY_') and selected_city.replace(" ", "_").upper() in col:
    #         input_df[col] = 1.0
    #         break # Assuming only one city can be selected


    # Ensure correct data types (e.g., float for numerical, int/float for binary)
    # input_df = input_df.astype(X_train_classification.dtypes) # Load and use actual dtypes


    # Scale numerical features using the dummy scaler
    # This will NOT produce correct results as the scaler is not fitted.
    # In a real app, load the fitted scaler and apply it correctly.
    numerical_cols_in_input = input_df.select_dtypes(include=np.number).columns.tolist()
    # Filter numerical_cols_in_input to exclude binary/one-hot encoded columns if necessary.
    # Example: numerical_cols_to_scale = [col for col in numerical_cols_in_input if not col.startswith('CITY_')]
    try:
         input_df[numerical_cols_in_input] = scaler.transform(input_df[numerical_cols_in_input])
    except Exception as e:
         st.error(f"Error scaling input features (using dummy scaler): {e}.")
         st.info("Please ensure you load and use the actual fitted scaler from training in a real application.")
         # Continue without scaling or stop depending on desired behavior


    # Ensure the final input_df has the EXACT same columns in the EXACT same order as the model expects
    # This is CRITICAL for the model prediction.
    # You must ensure the placeholder feature_names list matches the training features exactly.
    # If input_df is missing any columns from feature_names, add them with a default value (e.g., 0).
    # If input_df has extra columns, drop them.
    # Then, reindex input_df to match the order of feature_names.

    # Example: Reindex to match training features
    # if list(input_df.columns) != feature_names:
    #     st.warning("Input DataFrame columns do not exactly match the expected feature names or order. Attempting to reindex.")
    #     try:
    #         input_df = input_df.reindex(columns=feature_names, fill_value=0.0)
    #         st.info("Attempted to reindex input features.")
    #     except Exception as e:
    #         st.error(f"Failed to reindex input features: {e}. Cannot proceed with prediction.")
    #         st.stop() # Stop execution if reindexing fails
    # else:
    #      st.info("Input features match expected features and order.")


    # Reshape the input data for the model (samples, timesteps, features)
    # Assuming a single timestep as in our training data
    input_reshaped = input_df.values.reshape((input_df.shape[0], 1, input_df.shape[1]))


    # Make prediction
    try:
        prediction = model.predict(input_reshaped)
    except Exception as e:
        st.error(f"Error during model prediction: {e}. Ensure input shape matches model input shape.")
        st.stop()


    # Get the predicted class index
    predicted_class_index = np.argmax(prediction, axis=1)[0]

    # Map the predicted class index back to the original risk level label
    # This mapping MUST match the order of your one-hot encoded target columns (y_classification)
    # after preprocessing with drop_first=False.
    # Assuming the order is ['RISK_LEVEL_Low', 'RISK_LEVEL_Moderate', 'RISK_LEVEL_High', 'RISK_LEVEL_Very High']
    risk_level_map = {
        0: 'Low',
        1: 'Moderate',
        2: 'High',
        3: 'Very High'
    }
    predicted_risk_level = risk_level_map.get(predicted_class_index, 'Unknown')


    st.subheader('Prediction Result:')
    st.write(f'The predicted risk level is: **{predicted_risk_level}**')

    st.subheader('Prediction Probabilities:')
    # Display probabilities for each class
    # Ensure these column names match your model's output order
    col_names = ['Low', 'Moderate', 'High', 'Very High']
    prob_df = pd.DataFrame(prediction, columns=col_names)
    st.dataframe(prob_df)

# Add instructions on how to run the Streamlit app
st.markdown("---")
st.markdown("To run this Streamlit app:")
st.markdown("1. Save the code above as `streamlit_app.py` (which `%%writefile` does).")
st.markdown("2. Open a terminal in your Colab environment or local machine where the file is saved.")
st.markdown("3. Run the command: `streamlit run streamlit_app.py`")
st.markdown("4. If running in Colab, a public URL will be provided to access the app.")