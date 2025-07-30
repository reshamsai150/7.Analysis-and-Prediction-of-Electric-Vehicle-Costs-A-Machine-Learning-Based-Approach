import streamlit as st
import pandas as pd
import pickle


st.set_page_config(page_title="EV Cost Predictor", layout="centered")

# Define DummyModel class for compatibility
class DummyModel:
    def predict(self, X):
        return [1650000]  # Always returns ₹16.5 Lakh

# Load the model
model = None
try:
    model = pickle.load(open("model.pkl", "rb"))  # adjust path if needed
except FileNotFoundError:
    st.warning("⚠️ Model not found. Please run 'python create_model.py' to create a dummy model.")
except Exception as e:
    st.warning(f"⚠️ Error loading model: {e}")

# Sidebar Style (Optional)
with st.sidebar:
    st.image("https://img.icons8.com/dusk/64/electric.png", width=60)
    st.title("🔌 EV Cost Predictor")
    st.markdown("Enter the EV specifications to get a cost estimate.")

st.markdown("<h2 style='text-align: center;'>💰 Predict Electric Vehicle Cost</h2>", unsafe_allow_html=True)
st.write("---")

# Input fields
brand = st.selectbox("EV Brand", ["Tata", "MG", "Hyundai", "Mahindra", "Others"])
battery_capacity = st.slider("Battery Capacity (kWh)", min_value=10, max_value=150, step=1)
range_km = st.slider("Driving Range (km)", min_value=50, max_value=700, step=10)
motor_power = st.slider("Motor Power (kW)", min_value=30, max_value=300, step=5)

# Encode Brand if needed (modify as per model training)
brand_encoded = {
    "Tata": 0,
    "MG": 1,
    "Hyundai": 2,
    "Mahindra": 3,
    "Others": 4
}.get(brand, 4)

if st.button("🔍 Predict Cost"):
    if model is None:
        st.error("❌ Model not loaded. Please create a model first using 'python create_model.py'")
    else:
        input_df = pd.DataFrame([[brand_encoded, battery_capacity, range_km, motor_power]],
                                columns=["Brand", "Battery", "Range", "Power"])

        try:
            prediction = model.predict(input_df)[0]
            st.success(f"💸 Estimated Cost: ₹{int(prediction):,}")
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")

st.write("---")
st.caption("⚡ Built with Streamlit · ML powered")

