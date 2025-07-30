# code/ev_stations/dashboard.py
import streamlit as st
import pandas as pd
from ev_stations.locator import get_nearby_stations
from ev_stations.utils import geocode_address

st.set_page_config(page_title="EV Charging Station Locator", layout="wide")

st.title("🔌 EV Charging Station Locator")

address = st.text_input("Enter your location (city or address):", "Delhi, India")
radius = st.slider("Search Radius (km):", 1, 50, 10)

if st.button("Find Stations"):
    lat, lon = geocode_address(address)
    if lat and lon:
        stations = get_nearby_stations(lat, lon, distance_km=radius)
        if stations:
            df = pd.DataFrame(stations)
            st.map(df[["lat", "lon"]])
            st.dataframe(df)
        else:
            st.warning("No stations found in this area.")
    else:
        st.error("Could not find the location. Try a different address.")
