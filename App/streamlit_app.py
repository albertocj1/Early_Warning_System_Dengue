import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import json
# import pickle # Uncomment if you save and load the scaler

# Load the trained classification model
@st.cache_resource
def load_model():
    # Use the correct path to your saved model file
    model = tf.keras.models.load_model('Model/dengue_risk_level_classification_model.h5')
    return model

model = load_model()

# --- Load Scaler and Feature Names (Essential for real application) ---
# In a real application, you MUST save the fitted scaler and the list of
# feature names used during training and load them here.
# Example:
# with open('scaler.pkl', 'rb') as f:
#     scaler = pickle.load(f)
#
# with open('feature_names.json', 'r') as f:
#     feature_names = json.load(f)

# For demonstration purposes, we'll create dummy feature names and a dummy scaler.
# REPLACE THESE WITH YOUR ACTUAL SAVED SCALER AND FEATURE NAMES!
# Determine the number of features from your trained model's input shape
num_features = model.input_shape[2]
feature_names = [f'feature_{i}' for i in range(num_features)]

# Dummy scaler - replace with loading your fitted scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# You would fit this scaler on your training data and save it.
# For this example, we'll just create an unfitted one, which is NOT correct for real use.
# You need to load the scaler fitted on X_train_classification.

# --- Streamlit App Layout ---
st.title('Dengue Risk Level Prediction')

st.write("Enter the feature values to predict the dengue risk level.")

# Create input fields for features
# This is a simplified example. You need to create appropriate input widgets
# (e.g., st.number_input, st.selectbox) for ALL your features based on their
# data types, ranges, and whether they are numerical or one-hot encoded categorical.
# You might group inputs or use different layouts depending on the number of features.
input_data = {}
st.subheader("Input Features")
# Example for numerical features:
num_numerical_features = min(10, num_features) # Show input for a few features as example
for i in range(num_numerical_features):
    input_data[feature_names[i]] = st.number_input(f'{feature_names[i]}', value=0.0, key=f'input_{i}')

# Add a note about full feature input in a real app
if num_features > num_numerical_features:
    st.info(f"This is a simplified input for the first {num_numerical_features} features. In a real app, you need inputs for all {num_features} features.")

# Add input for one-hot encoded city (example)
# You need to handle all your one-hot encoded features similarly
city_names = ['CALOOCAN CITY', 'LAS PINAS CITY', 'MAKATI CITY', 'MALABON CITY', 'MANDALUYONG CITY', 'MANILA CITY', 'MARIKINA CITY', 'MUNTINLUPA CITY', 'NAVOTAS CITY', 'PARANAQUE CITY', 'PASAY CITY', 'PASIG CITY', 'PATEROS', 'QUEZON CITY', 'SAN JUAN CITY', 'TAGUIG CITY', 'VALENZUELA CITY'] # Replace with your actual city names after one-hot encoding drop_first=True
selected_city = st.selectbox("Select City", city_names)

# Handle other one-hot encoded features like RISK_LEVEL (if predicting other features) or ALERT/EPIDEMIC flags

if st.button('Predict Risk Level'):
    # --- Prepare Input Data for Prediction ---
    # Create a DataFrame with all features, ensuring correct column order and dtypes
    # Initialize all feature columns to a default value (e.g., 0)
    input_df = pd.DataFrame(0.0, index=[0], columns=feature_names)

    # Populate the DataFrame with user inputs
    for feature, value in input_data.items():
        input_df[feature] = value

    # Handle one-hot encoded city input (example)
    # Find the column corresponding to the selected city and set its value to 1
    city_col_name = f'CITY_{selected_city}' # Adjust based on your one-hot encoding column names
    if city_col_name in input_df.columns:
        input_df[city_col_name] = 1.0
    else:
         st.warning(f"Column for selected city '{city_col_name}' not found in model features. Prediction may be incorrect.")
         # Fallback: if city column not found, model might still predict based on other features,
         # but it's best to ensure all expected features are present.


    # Ensure correct data types (e.g., float for numerical, int/float for binary)
    # This step is important if your input widgets don't guarantee the correct type.
    # input_df = input_df.astype(X_train_classification.dtypes) # Load and use actual dtypes

    # Scale numerical features using the loaded scaler
    # You need to identify which columns are numerical and apply the scaler only to them.
    # This requires knowing the numerical columns from your training data.
    # Example (assuming first 'num_numerical_features' are numerical - REPLACE WITH REAL LOGIC):
    # numerical_input_cols = feature_names[:num_numerical_features] # Replace with actual numerical column names
    # input_df[numerical_input_cols] = scaler.transform(input_df[numerical_input_cols]) # Use the fitted scaler


    # Reshape the input data for the model (samples, timesteps, features)
    # Assuming a single timestep as in our training data
    input_reshaped = input_df.values.reshape((input_df.shape[0], 1, input_df.shape[1]))


    # Make prediction
    prediction = model.predict(input_reshaped)

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
    col_names = ['Low', 'Moderate', 'High', 'Very High'] # Ensure this order matches your model's output
    prob_df = pd.DataFrame(prediction, columns=col_names)
    st.dataframe(prob_df)
