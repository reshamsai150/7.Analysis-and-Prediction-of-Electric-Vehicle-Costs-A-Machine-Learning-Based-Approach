import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

st.title("🤖 Retrain the Model")

df = pd.read_csv("data/ev_cleaned_data.csv")
features = [col for col in df.columns if col not in ["PriceEuro"]]

if st.button("Train Model"):
    X = df[features]
    y = df["PriceEuro"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    
    joblib.dump(model, "model/model.pkl")
    st.success("✅ Model retrained and saved!")
