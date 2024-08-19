import streamlit as st
import numpy as np
import joblib

# Load your trained model
model = joblib.load('final_gradient_boosting_model.pkl')

st.title("Airbnb Price Prediction in San Francisco")

# Sidebar for user input
st.sidebar.header('User Input Parameters')

def user_input_features():
    bedrooms = st.sidebar.slider('Bedrooms', 1, 10, 1)
    accommodates = st.sidebar.slider('Accommodates', 1, 10, 2)
    bathrooms = st.sidebar.slider('Bathrooms', 1, 5, 1)
    review_scores_rating = st.sidebar.slider('Review Scores Rating', 0.0, 100.0, 90.0)
    room_type_private = st.sidebar.selectbox('Is it a Private Room?', ('Yes', 'No'))
    room_type_shared = st.sidebar.selectbox('Is it a Shared Room?', ('Yes', 'No'))
    
    data = {
        'bedrooms': bedrooms,
        'accommodates': accommodates,
        'review_scores_rating': review_scores_rating,
        'room_type_Private room': 1 if room_type_private == 'Yes' else 0,
        'bathrooms': bathrooms,
        'room_type_Shared room': 1 if room_type_shared == 'Yes' else 0
    }
    features = np.array([list(data.values())])
    return features

input_df = user_input_features()

# Prediction
prediction = model.predict(input_df)
predicted_price = np.exp(prediction)[0]  # Convert log price back to the actual price

st.subheader('Prediction')
st.write(f"The predicted price is: ${predicted_price:.2f}")

