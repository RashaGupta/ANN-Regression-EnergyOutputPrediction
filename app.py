import streamlit as st
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("ann_model.h5")

# Title of the app
st.title("Energy Output Prediction")

# Sidebar with input fields
st.sidebar.header("Input Variables")
temperature = st.sidebar.number_input("Temperature (AT)", value=0.0)
pressure = st.sidebar.number_input("Pressure (V)", value=0.0)
humidity = st.sidebar.number_input("Humidity (AP)", value=0.0)
vacuum = st.sidebar.number_input("Vacuum (RH)", value=0.0)

# Button to trigger prediction
if st.button("Predict"):
    # Make predictions
    input_features = np.array([[temperature, pressure, humidity, vacuum]])
    prediction = model.predict(input_features)

    # Display prediction
    st.write("Predicted Energy Output (PE):", prediction[0][0])