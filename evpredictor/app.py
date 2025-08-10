

import streamlit as st
import pandas as pd
from utils.predict import load_model, predict_price
from utils.data_viz import show_data_summary, show_charts

st.set_page_config(page_title="EV Price Predictor", layout="wide")

st.title("⚡ Electric Vehicle Price Predictor")

menu = ["📊 Data Analysis", "💶 Predict Price"]
choice = st.sidebar.radio("Navigate", menu)

if choice == "📊 Data Analysis":
    st.subheader("Electric Vehicle Dataset Overview")
    df = pd.read_csv("data/ev_cleaned_data.csv")
    show_data_summary(df)
    show_charts(df)

elif choice == "💶 Predict Price":
    st.subheader("Enter Vehicle Specifications")
    
    battery_size = st.number_input("Battery Size (kWh)", 20, 200, 50)
    range_km = st.number_input("Range (km)", 100, 1000, 300)
    top_speed = st.number_input("Top Speed (km/h)", 80, 300, 150)
    acceleration = st.number_input("0-100 km/h (sec)", 2.0, 15.0, 7.0)
    
    if st.button("Predict Price"):
        model, preprocessor = load_model()
        prediction = predict_price(model, preprocessor, [[battery_size, range_km, top_speed, acceleration]])
        st.success(f"Estimated Price: €{prediction:,.2f}")

