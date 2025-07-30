#!/usr/bin/env python3
"""
Create dummy model for EV Cost Predictor
========================================

This script creates a dummy model.pkl file for testing the Streamlit app.
"""

import pickle
import os

class DummyModel:
    def predict(self, X):
        return [1650000]  # Always returns ₹16.5 Lakh

def create_dummy_model():
    """Create and save dummy model"""
    try:
        # Create dummy model
        model = DummyModel()
        
        # Save to model.pkl
        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)
        
        print("✅ Dummy model created successfully!")
        print("📍 File saved as: model.pkl")
        print("💰 Model will always predict: ₹16,50,000")
        
        return True
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return False

if __name__ == "__main__":
    print("🔌 Creating dummy model for EV Cost Predictor...")
    success = create_dummy_model()
    
    if success:
        print("\n🚀 Now you can run the Streamlit app:")
        print("   python run_streamlit.py")
        print("   or")
        print("   streamlit run streamlit_app.py")
    else:
        print("\n❌ Failed to create model. Please check permissions.") 