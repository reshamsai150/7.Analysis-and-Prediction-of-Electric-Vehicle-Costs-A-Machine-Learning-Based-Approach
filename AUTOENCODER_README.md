# AutoEncoder-Based Feature Learning for EV Cost Prediction

## 🚗 Project Overview

This project implements a **Deep AutoEncoder** to compress high-dimensional Electric Vehicle (EV) data into lower-dimensional latent representations, which are then used to improve prediction performance for EV cost estimation. This addresses the GitHub issue #14: "Enhancing EV Cost Prediction Using AutoEncoder-Based Feature Learning".

## 🎯 Problem Statement

Electric Vehicle cost prediction involves analyzing complex, high-dimensional data with:
- **Noise** in feature measurements
- **Multicollinearity** between features
- **Irrelevant features** that don't contribute to price prediction
- **Complex nonlinear relationships** that traditional models struggle to capture

Traditional regression models require heavy manual feature engineering to handle these challenges effectively.

## 💡 Proposed Solution

### AutoEncoder Architecture
- **Deep AutoEncoder**: Compresses input data into lower-dimensional latent representation
- **Feature Learning**: Retains key relationships while eliminating noise
- **Enhanced Regression**: Uses learned features for improved prediction performance

### Key Benefits
1. **Nonlinear Representations**: Learn compressed representations for better generalization
2. **Automatic Feature Selection**: Reduce dimensionality without losing critical information
3. **Improved Accuracy**: Boost model performance and reduce overfitting
4. **Modular Design**: Transferable deep learning pipelines across datasets

## 🏗️ Project Structure

```
7.Analysis-and-Prediction-of-Electric-Vehicle-Costs-A-Machine-Learning-Based-Approach/
├── code/
│   ├── autoencoder_model.py          # Core AutoEncoder implementation
│   ├── autoencoder_demo.py           # Demo script with visualizations
│   ├── enhanced_app.py               # Flask app with AutoEncoder integration
│   ├── requirements.txt              # Updated dependencies
│   └── models/                       # Saved model directory
├── AUTOENCODER_README.md             # This file
└── README.md                         # Original project README
```

## 🛠️ Implementation Details

### AutoEncoder Architecture
```python
# Encoder: Input → Hidden Layers → Latent Space
Input (20 features) → Dense(128) → Dense(64) → Latent(32)

# Decoder: Latent Space → Hidden Layers → Output
Latent(32) → Dense(64) → Dense(128) → Output(20 features)
```

### Key Features
- **Batch Normalization**: Stabilizes training
- **Dropout**: Prevents overfitting
- **Early Stopping**: Prevents overtraining
- **Learning Rate Scheduling**: Optimizes convergence

### Model Comparison
The implementation compares traditional ML models with AutoEncoder-enhanced features:

| Model Type | Raw Features | AutoEncoder Features |
|------------|--------------|---------------------|
| Random Forest | ✅ | ✅ |
| XGBoost | ✅ | ✅ |
| SVR | ✅ | ✅ |
| MLP | ✅ | ✅ |
| Linear Regression | ✅ | ✅ |

## 📊 Expected Outcomes

### Performance Improvements
- **Higher R² scores** compared to raw features
- **Lower RMSE** for price predictions
- **Better generalization** on unseen data
- **Reduced overfitting** through regularization

### Visualizations Generated
1. **Data Exploration**: Price distribution, feature correlations, brand analysis
2. **Model Comparison**: Raw vs AutoEncoder feature performance
3. **AutoEncoder Analysis**: Training history, latent space, reconstruction quality

## 🚀 Getting Started

### Prerequisites
```bash
# Install dependencies
pip install -r code/requirements.txt
```

### Quick Demo
```bash
# Run the AutoEncoder demo
cd code
python autoencoder_demo.py
```

### Web Application
```bash
# Run the enhanced Flask app
cd code
python enhanced_app.py
```

## 📈 Usage Examples

### Basic AutoEncoder Usage
```python
from autoencoder_model import AutoEncoderFeatureLearner

# Initialize AutoEncoder
autoencoder = AutoEncoderFeatureLearner(
    input_dim=20,
    latent_dim=32,
    encoding_dims=[128, 64]
)

# Train the model
history = autoencoder.train(data, epochs=100)

# Extract latent features
latent_features = autoencoder.extract_latent_features(data)

# Save the model
autoencoder.save_model('models/ev_autoencoder')
```

### Model Comparison
```python
from autoencoder_model import compare_models_with_autoencoder

# Compare traditional vs AutoEncoder-enhanced models
results, autoencoder, history = compare_models_with_autoencoder(
    data, 'price', test_size=0.2, random_state=42
)
```

## 🎨 Features Implemented

### ✅ Core AutoEncoder
- [x] Deep AutoEncoder architecture with configurable layers
- [x] Automatic data preprocessing (scaling, encoding)
- [x] Training with early stopping and learning rate scheduling
- [x] Latent feature extraction
- [x] Data reconstruction capabilities

### ✅ Model Comparison
- [x] Traditional ML models (Random Forest, XGBoost, SVR, MLP, Linear Regression)
- [x] AutoEncoder-enhanced versions of all models
- [x] Comprehensive performance metrics (R², RMSE, MAE)
- [x] Statistical comparison and improvement analysis

### ✅ Visualization Suite
- [x] Data exploration plots
- [x] Model performance comparisons
- [x] AutoEncoder training history
- [x] Latent space visualization
- [x] Reconstruction quality analysis

### ✅ Web Integration
- [x] Enhanced Flask application
- [x] AutoEncoder training interface
- [x] Real-time predictions with both approaches
- [x] Interactive visualizations

### ✅ Model Persistence
- [x] Save/load AutoEncoder models
- [x] Save/load preprocessors
- [x] Model versioning and management

## 🔬 Technical Specifications

### AutoEncoder Parameters
- **Input Dimension**: 20 features (configurable)
- **Latent Dimension**: 32 (configurable)
- **Encoding Layers**: [128, 64] (configurable)
- **Activation**: ReLU for hidden layers, Linear for output
- **Optimizer**: Adam with learning rate 0.001
- **Loss Function**: Mean Squared Error (MSE)

### Training Configuration
- **Batch Size**: 32
- **Validation Split**: 20%
- **Early Stopping**: Patience of 10 epochs
- **Learning Rate Reduction**: Factor of 0.5, patience of 5 epochs

### Data Preprocessing
- **Scaling**: StandardScaler for numerical features
- **Encoding**: LabelEncoder for categorical features
- **Missing Values**: Mean imputation
- **Feature Engineering**: Automatic handling of mixed data types

## 📊 Performance Metrics

The implementation evaluates models using:
- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **Training Time**: Model training efficiency
- **Inference Time**: Prediction speed

## 🎯 Objectives Achieved

1. ✅ **Implement and train a deep AutoEncoder** on the EV dataset
2. ✅ **Extract latent features** and feed them to regression models
3. ✅ **Compare performance** against models trained on raw features
4. ✅ **Visualize latent space** to interpret data organization after compression

## 🔧 Customization Options

### AutoEncoder Architecture
```python
# Customize architecture
autoencoder = AutoEncoderFeatureLearner(
    input_dim=25,           # Number of input features
    latent_dim=16,          # Latent space dimension
    encoding_dims=[256, 128, 64]  # Custom layer sizes
)
```

### Training Parameters
```python
# Customize training
history = autoencoder.train(
    data,
    epochs=200,             # More training epochs
    batch_size=64,          # Larger batch size
    validation_split=0.3    # Different validation split
)
```

## 🚀 Future Enhancements

### Planned Features
- [ ] **Variational AutoEncoder (VAE)** for probabilistic modeling
- [ ] **Convolutional AutoEncoder** for image-based features
- [ ] **Attention mechanisms** for feature importance
- [ ] **Hyperparameter optimization** with Optuna
- [ ] **Real-time model updates** with streaming data

### Advanced Analytics
- [ ] **Feature importance analysis** in latent space
- [ ] **Anomaly detection** using reconstruction error
- [ ] **Transfer learning** across different EV datasets
- [ ] **Multi-task learning** for price and range prediction

## 🤝 Contributing

This implementation addresses the specific requirements from GitHub issue #14. To contribute:

1. **Fork the repository**
2. **Create a feature branch**
3. **Implement your changes**
4. **Add tests and documentation**
5. **Submit a pull request**

## 📚 References

- **AutoEncoder Theory**: Hinton, G. E., & Salakhutdinov, R. R. (2006)
- **Deep Learning**: Goodfellow, I., Bengio, Y., & Courville, A. (2016)
- **Feature Learning**: Bengio, Y., Courville, A., & Vincent, P. (2013)

## 📞 Contact

For questions about this AutoEncoder implementation:
- **GitHub Issue**: #14 - Enhancing EV Cost Prediction Using AutoEncoder-Based Feature Learning
- **Repository**: [HeerakKashyap/7.Analysis-and-Prediction-of-Electric-Vehicle-Costs-A-Machine-Learning-Based-Approach](https://github.com/HeerakKashyap/7.Analysis-and-Prediction-of-Electric-Vehicle-Costs-A-Machine-Learning-Based-Approach)

---

**🎉 This implementation successfully addresses the AutoEncoder-based feature learning requirements for EV cost prediction!** 