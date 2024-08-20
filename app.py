import streamlit as st
import joblib
import numpy as np

# Load your trained model
model = joblib.load('final_gradient_boosting_model.pkl')

# Title of the app
st.title("Airbnb Price Prediction in San Francisco")

# Instructions for the user
st.write("Enter the details of the property below to predict the price:")

# Input fields for user data
bedrooms = st.number_input("Number of Bedrooms", min_value=0, max_value=10, value=1)
accommodates = st.number_input("Accommodates (Number of Guests)", min_value=0, max_value=20, value=1)
review_scores_rating = st.number_input("Review Scores Rating", min_value=0.0, max_value=100.0, value=50.0)
bathrooms = st.number_input("Number of Bathrooms", min_value=0.0, max_value=10.0, value=1.0)
room_type_private = st.checkbox("Private Room?")
room_type_shared = st.checkbox("Shared Room?")

# Prepare the input data for prediction
data = np.array([
    [
        bedrooms,
        accommodates,
        review_scores_rating,
        1 if room_type_private else 0,
        bathrooms,
        1 if room_type_shared else 0,
    ]
])

# Predict price using the model
if st.button("Predict Price"):
    predicted_log_price = model.predict(data)[0]
    predicted_price = np.exp(predicted_log_price)
    st.write(f"The predicted price is ${predicted_price:.2f}")
