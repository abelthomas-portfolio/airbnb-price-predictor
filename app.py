import streamlit as st
import numpy as np
import pickle
import sklearn

print(sklearn.__version__)


# Load your trained model using pickle
with open('final_gradient_boosting_model_pickle.pkl', 'rb') as file:
    model = pickle.load(file)

# Title of the app
st.title("Airbnb Price Prediction in San Francisco")

# Add input fields for user data
bedrooms = st.number_input('Number of Bedrooms', min_value=1, max_value=10, value=1)
accommodates = st.number_input('Number of Guests Accommodated', min_value=1, max_value=20, value=2)
review_scores_rating = st.slider('Review Scores Rating', min_value=0.0, max_value=100.0, value=95.0)
bathrooms = st.number_input('Number of Bathrooms', min_value=1.0, max_value=10.0, value=1.0)
room_type_private_room = st.selectbox('Private Room', ['Yes', 'No'])
room_type_shared_room = st.selectbox('Shared Room', ['Yes', 'No'])

# Convert inputs into a format suitable for the model
room_type_private_room = 1 if room_type_private_room == 'Yes' else 0
room_type_shared_room = 1 if room_type_shared_room == 'Yes' else 0

# Prepare input data for prediction
input_data = np.array([[bedrooms, accommodates, review_scores_rating, bathrooms, room_type_private_room, room_type_shared_room]])

# Make the prediction
prediction = model.predict(input_data)

# Display the prediction
st.write(f"Predicted Price: {np.exp(prediction[0]):.2f}")
