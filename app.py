import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load('final_gradient_boosting_model_v3.pkl')

# Title of the app
st.title('Airbnb Price Predictor')

# Subtitle
st.subheader('Predict the price of an Airbnb listing in San Francisco')

# Input features
bedrooms = st.number_input('Number of Bedrooms', min_value=0, max_value=10, value=1)
accommodates = st.number_input('Accommodates', min_value=1, max_value=20, value=1)
review_scores_rating = st.number_input('Average Review Rating', min_value=0.0, max_value=100.0, value=75.0)
bathrooms = st.number_input('Number of Bathrooms', min_value=0.0, max_value=10.0, value=1.0, step=0.5, format="%.1f")
room_type = st.selectbox('Room Type', ['Entire home/apt', 'Private room', 'Shared room'])

# Prepare the input data
room_type_private = 1 if room_type == 'Private room' else 0
room_type_shared = 1 if room_type == 'Shared room' else 0

input_data = pd.DataFrame({
    'bedrooms': [bedrooms],
    'accommodates': [accommodates],
    'review_scores_rating': [review_scores_rating],
    'room_type_Private room': [room_type_private],
    'bathrooms': [bathrooms],
    'room_type_Shared room': [room_type_shared],
})

# Prediction
if st.button('Predict Price'):
    log_price_pred = model.predict(input_data)[0]
    price_pred = np.exp(log_price_pred)  # Inverse of log transformation
    st.write(f'Predicted Price: ${price_pred:.2f}')
    
# Add a download button for the project report
st.write("\n\n")
st.subheader('Download Project Report')
report_content = """
# Predicting Airbnb Prices - A Machine Learning Approach

This report provides an overview of the development of the Airbnb Price Prediction model.

## Model Details
- **Model Type**: Gradient Boosting Regressor
- **Features Used**: 
  - Number of Bedrooms
  - Accommodates
  - Average Review Rating
  - Number of Bathrooms
  - Room Type (Entire home/apt, Private room, Shared room)
- **Target Variable**: Log-transformed price of Airbnb listings in San Francisco

## How to Use the App
1. Input the number of bedrooms, accommodates, average review rating, number of bathrooms, and room type.
2. Click the 'Predict Price' button to get the predicted price for the Airbnb listing.
"""

st.download_button(
    label="Download Report",
    data=report_content,
    file_name='Predicting Airbnb Prices - A Machine Learning Approach.pdf',
    mime='text/plain'
)
