import streamlit as st
import plotly.express as px

def show_data_summary(df):
    st.write("### Dataset Preview", df.head())
    st.write("### Statistics", df.describe())

def show_charts(df):
    fig1 = px.histogram(df, x="Price", nbins=30, title="Price Distribution")
    st.plotly_chart(fig1, use_container_width=True)
    
    fig2 = px.scatter(df, x="Battery", y="Price", color="Brand", title="Battery vs Price")
    st.plotly_chart(fig2, use_container_width=True)



