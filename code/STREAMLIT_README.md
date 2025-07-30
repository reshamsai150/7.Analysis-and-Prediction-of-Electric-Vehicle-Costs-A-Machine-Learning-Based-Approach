# 🔌 Streamlit Interface for EV Cost Prediction

A simple Streamlit web interface for predicting electric vehicle costs using machine learning.

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- Streamlit installed (`pip install streamlit`)

### Running the App

**Option 1: Using the run script**
```bash
cd code
python run_streamlit.py
```

**Option 2: Direct Streamlit command**
```bash
cd code
streamlit run streamlit_app.py
```

The app will be available at: **http://localhost:8501**

## 📱 Features

### Input Fields
- **EV Brand**: Select from Tata, MG, Hyundai, Mahindra, or Others
- **Battery Capacity**: Slider for battery capacity (10-150 kWh)
- **Driving Range**: Slider for driving range (50-700 km)
- **Motor Power**: Slider for motor power (30-300 kW)

### Prediction
- **Predict Button**: Click to get cost estimate
- **Result Display**: Shows estimated cost in Indian Rupees (₹)
- **Error Handling**: Graceful error messages if model is not available

## 🎯 Model Setup

### Using Dummy Model (for testing)
```bash
cd code
python dummy_model.py
```

### Using Your Own Model
Replace the `model.pkl` file with your trained model that accepts the same input format:
- Input columns: ["Brand", "Battery", "Range", "Power"]
- Output: Price prediction in rupees

## 🛠️ Troubleshooting

### "Model not found" Error
```bash
# Create dummy model for testing
python dummy_model.py
```

### Port Already in Use
```bash
# Use different port
streamlit run streamlit_app.py --server.port 8502
```

### Dependencies Missing
```bash
# Install Streamlit
pip install streamlit pandas
```

## 📊 Expected Input Format

The model expects a DataFrame with these columns:
- **Brand**: Encoded brand (0-4)
- **Battery**: Battery capacity in kWh
- **Range**: Driving range in km
- **Power**: Motor power in kW

## 🎨 Customization

### Adding More Brands
Edit the brand selection in `streamlit_app.py`:
```python
brand = st.selectbox("EV Brand", ["Tata", "MG", "Hyundai", "Mahindra", "Others", "Your Brand"])
```

### Modifying Input Ranges
Adjust the slider ranges in `streamlit_app.py`:
```python
battery_capacity = st.slider("Battery Capacity (kWh)", min_value=10, max_value=200, step=1)
```

---

**Built with Streamlit · ML powered** 