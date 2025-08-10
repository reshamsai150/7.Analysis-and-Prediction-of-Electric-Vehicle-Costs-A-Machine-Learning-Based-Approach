import streamlit as st
from utils.predict import predict_price

st.title("💶 EV Price Prediction")

# Example features
input_data = {
    "TopSpeed_KmH": st.number_input("Top Speed (Km/H)", 100, 350, 200),
    "Range_Km": st.number_input("Range (Km)", 50, 800, 300),
    "Battery_Pack_KWh": st.number_input("Battery Pack (kWh)", 20, 150, 75),
    "Efficiency_WhKm": st.number_input("Efficiency (Wh/Km)", 100, 300, 180),
    "FastCharge_KmH": st.number_input("Fast Charge (Km/H)", 50, 1000, 400),
}

if st.button("Predict Price"):
    price = predict_price(input_data)
    st.success(f"Estimated Price: €{price}")
