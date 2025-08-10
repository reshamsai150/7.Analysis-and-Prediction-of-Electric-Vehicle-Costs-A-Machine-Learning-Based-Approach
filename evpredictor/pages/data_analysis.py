import streamlit as st
import pandas as pd
from utils.data_viz import show_summary, plot_price_distribution, plot_range_vs_price

st.title("📊 EV Data Analysis")

df = pd.read_csv("data/ev_cleaned_data.csv")

show_summary(df)
plot_price_distribution(df)
plot_range_vs_price(df)
