"""
Test script for AutoEncoder implementation
This script verifies that the AutoEncoder works correctly with sample data
"""

import pandas as pd
import numpy as np
from autoencoder_model import AutoEncoderFeatureLearner
import warnings
warnings.filterwarnings('ignore')

def create_test_data(n_samples=100):
    """Create small test dataset"""
    np.random.seed(42)
    
    data = {
        'brand': np.random.choice(['Tesla', 'Nissan', 'BMW'], n_samples),
        'battery_capacity_kwh': np.random.uniform(40, 100, n_samples),
        'range_miles': np.random.uniform(150, 350, n_samples),
        'max_speed_mph': np.random.uniform(100, 160, n_samples),
        'safety_rating': np.random.uniform(4.0, 5.0, n_samples),
        'price': np.random.uniform(30000, 80000, n_samples)
    }
    
    return pd.DataFrame(data)

def test_autoencoder():
    """Test the AutoEncoder implementation"""
    print("🧪 Testing AutoEncoder Implementation")
    print("=" * 40)
    
    # Create test data
    print("📊 Creating test dataset...")
    test_data = create_test_data(100)
    print(f"Test data shape: {test_data.shape}")
    
    # Prepare features (exclude price)
    X = test_data.drop(columns=['price'])
    print(f"Feature columns: {list(X.columns)}")
    
    # Initialize AutoEncoder
    print("\n🤖 Initializing AutoEncoder...")
    autoencoder = AutoEncoderFeatureLearner(
        input_dim=X.shape[1],
        latent_dim=8,  # Smaller for testing
        encoding_dims=[32, 16]  # Smaller layers for testing
    )
    
    # Build model
    print("🔨 Building AutoEncoder architecture...")
    model = autoencoder.build_autoencoder()
    print(f"AutoEncoder built successfully!")
    print(f"Model summary:")
    model.summary()
    
    # Train model
    print("\n🏋️ Training AutoEncoder...")
    history = autoencoder.train(X, epochs=10, verbose=1)
    print("Training completed!")
    
    # Test latent feature extraction
    print("\n🔍 Testing latent feature extraction...")
    latent_features = autoencoder.extract_latent_features(X)
    print(f"Latent features shape: {latent_features.shape}")
    print(f"Latent features sample:\n{latent_features[:5]}")
    
    # Test reconstruction
    print("\n🔄 Testing data reconstruction...")
    reconstructed = autoencoder.reconstruct_data(X)
    print(f"Reconstructed data shape: {reconstructed.shape}")
    
    # Evaluate reconstruction quality
    print("\n📊 Evaluating reconstruction quality...")
    evaluation = autoencoder.evaluate_reconstruction(X)
    print(f"Reconstruction metrics:")
    for metric, value in evaluation.items():
        print(f"  {metric}: {value:.6f}")
    
    # Test model saving and loading
    print("\n💾 Testing model persistence...")
    save_path = 'models/test_autoencoder'
    autoencoder.save_model(save_path)
    print("Model saved successfully!")
    
    # Create new instance and load model
    new_autoencoder = AutoEncoderFeatureLearner(
        input_dim=X.shape[1],
        latent_dim=8,
        encoding_dims=[32, 16]
    )
    new_autoencoder.load_model(save_path)
    print("Model loaded successfully!")
    
    # Verify loaded model works
    test_latent = new_autoencoder.extract_latent_features(X[:5])
    print(f"Loaded model test - latent features shape: {test_latent.shape}")
    
    print("\n✅ All tests passed successfully!")
    return autoencoder, test_data

if __name__ == "__main__":
    # Run the test
    autoencoder, test_data = test_autoencoder()
    
    print("\n🎉 AutoEncoder implementation is working correctly!")
    print("📁 You can now run the full demo with:")
    print("   python autoencoder_demo.py") 