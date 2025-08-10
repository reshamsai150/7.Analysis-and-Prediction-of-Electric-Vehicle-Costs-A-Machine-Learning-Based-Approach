

import joblib
import numpy as np

def load_model():
    model = joblib.load("model/model.pkl")
    preprocessor = joblib.load("model/preprocessor.pkl")
    return model, preprocessor

def predict_price(model, preprocessor, input_data):
    processed = preprocessor.transform(np.array(input_data))
    return model.predict(processed)[0]


