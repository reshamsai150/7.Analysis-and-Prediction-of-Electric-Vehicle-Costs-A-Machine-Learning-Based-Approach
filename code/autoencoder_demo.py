"""
AutoEncoder Demo for EV Cost Prediction
This script demonstrates the AutoEncoder-based feature learning approach
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from autoencoder_model import AutoEncoderFeatureLearner, compare_models_with_autoencoder
import warnings
warnings.filterwarnings('ignore')

def create_sample_ev_data(n_samples=1000):
    """
    Create sample EV data for demonstration
    """
    np.random.seed(42)
    
    # Generate realistic EV data
    data = {
        'brand': np.random.choice(['Tesla', 'Nissan', 'Chevrolet', 'BMW', 'Audi', 'Ford'], n_samples),
        'model': np.random.choice(['Model 3', 'Leaf', 'Bolt', 'i3', 'e-tron', 'Mustang Mach-E'], n_samples),
        'year': np.random.randint(2018, 2024, n_samples),
        'battery_capacity_kwh': np.random.uniform(40, 100, n_samples),
        'range_miles': np.random.uniform(150, 350, n_samples),
        'charging_time_hours': np.random.uniform(4, 12, n_samples),
        'max_speed_mph': np.random.uniform(100, 160, n_samples),
        'acceleration_0_60_sec': np.random.uniform(3, 8, n_samples),
        'weight_lbs': np.random.uniform(3000, 5000, n_samples),
        'length_inches': np.random.uniform(170, 200, n_samples),
        'width_inches': np.random.uniform(70, 80, n_samples),
        'height_inches': np.random.uniform(55, 65, n_samples),
        'seating_capacity': np.random.randint(2, 8, n_samples),
        'cargo_volume_cu_ft': np.random.uniform(10, 30, n_samples),
        'safety_rating': np.random.uniform(4.0, 5.0, n_samples),
        'warranty_years': np.random.randint(3, 8, n_samples),
        'maintenance_cost_per_year': np.random.uniform(500, 2000, n_samples),
        'tax_credit_available': np.random.choice([0, 1], n_samples),
        'fast_charging_support': np.random.choice([0, 1], n_samples),
        'autonomous_driving_level': np.random.randint(0, 4, n_samples)
    }
    
    # Create price based on features (with some noise)
    base_price = (
        data['battery_capacity_kwh'] * 1000 +
        data['range_miles'] * 50 +
        data['max_speed_mph'] * 20 +
        (8 - data['acceleration_0_60_sec']) * 2000 +
        data['safety_rating'] * 5000 +
        data['autonomous_driving_level'] * 3000 +
        data['tax_credit_available'] * 7500
    )
    
    # Add brand premium
    brand_premium = {
        'Tesla': 15000,
        'BMW': 12000,
        'Audi': 10000,
        'Ford': 5000,
        'Nissan': 3000,
        'Chevrolet': 2000
    }
    
    for i in range(n_samples):
        base_price[i] += brand_premium[data['brand'][i]]
    
    # Add noise
    data['price'] = base_price + np.random.normal(0, 5000, n_samples)
    
    return pd.DataFrame(data)

def run_autoencoder_demo():
    """
    Run the complete AutoEncoder demonstration
    """
    print("🚗 EV Cost Prediction with AutoEncoder Feature Learning")
    print("=" * 60)
    
    # Create sample data
    print("📊 Creating sample EV dataset...")
    ev_data = create_sample_ev_data(1000)
    print(f"Dataset shape: {ev_data.shape}")
    print(f"Features: {list(ev_data.columns[:-1])}")  # Exclude price
    print(f"Target: price")
    
    # Display data info
    print("\n📈 Dataset Overview:")
    print(ev_data.describe())
    
    # Data visualization
    print("\n📊 Creating visualizations...")
    create_data_visualizations(ev_data)
    
    # Run AutoEncoder comparison
    print("\n🤖 Training AutoEncoder and comparing models...")
    results, autoencoder, history = compare_models_with_autoencoder(
        ev_data, 'price', test_size=0.2, random_state=42
    )
    
    # Display results
    print("\n📊 Model Comparison Results:")
    display_comparison_results(results)
    
    # Visualize results
    print("\n📈 Creating comparison visualizations...")
    create_comparison_visualizations(results)
    
    # AutoEncoder specific visualizations
    print("\n🎨 Creating AutoEncoder visualizations...")
    create_autoencoder_visualizations(autoencoder, ev_data, history)
    
    # Save model
    print("\n💾 Saving trained AutoEncoder model...")
    autoencoder.save_model('models/ev_autoencoder')
    
    print("\n✅ Demo completed successfully!")
    return results, autoencoder

def create_data_visualizations(data):
    """Create data exploration visualizations"""
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('EV Dataset Exploration', fontsize=16, fontweight='bold')
    
    # Price distribution
    axes[0, 0].hist(data['price'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Price Distribution')
    axes[0, 0].set_xlabel('Price ($)')
    axes[0, 0].set_ylabel('Frequency')
    
    # Battery capacity vs Price
    axes[0, 1].scatter(data['battery_capacity_kwh'], data['price'], alpha=0.6, color='green')
    axes[0, 1].set_title('Battery Capacity vs Price')
    axes[0, 1].set_xlabel('Battery Capacity (kWh)')
    axes[0, 1].set_ylabel('Price ($)')
    
    # Range vs Price
    axes[0, 2].scatter(data['range_miles'], data['price'], alpha=0.6, color='orange')
    axes[0, 2].set_title('Range vs Price')
    axes[0, 2].set_xlabel('Range (miles)')
    axes[0, 2].set_ylabel('Price ($)')
    
    # Brand distribution
    brand_counts = data['brand'].value_counts()
    axes[1, 0].bar(brand_counts.index, brand_counts.values, color='lightcoral')
    axes[1, 0].set_title('Brand Distribution')
    axes[1, 0].set_xlabel('Brand')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Correlation heatmap (top features)
    top_features = ['price', 'battery_capacity_kwh', 'range_miles', 'max_speed_mph', 
                   'acceleration_0_60_sec', 'safety_rating']
    correlation_matrix = data[top_features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                ax=axes[1, 1], square=True)
    axes[1, 1].set_title('Feature Correlation Matrix')
    
    # Year vs Price
    axes[1, 2].boxplot([data[data['year'] == year]['price'] for year in sorted(data['year'].unique())],
                       labels=sorted(data['year'].unique()))
    axes[1, 2].set_title('Price by Year')
    axes[1, 2].set_xlabel('Year')
    axes[1, 2].set_ylabel('Price ($)')
    
    plt.tight_layout()
    plt.savefig('ev_data_exploration.png', dpi=300, bbox_inches='tight')
    plt.show()

def display_comparison_results(results):
    """Display model comparison results in a formatted table"""
    
    # Create results DataFrame
    results_df = pd.DataFrame(results).T
    results_df = results_df.round(4)
    
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    print(results_df.to_string())
    
    # Find best models
    best_r2_raw = results_df[results_df.index.str.contains('Raw')]['R2'].max()
    best_r2_latent = results_df[results_df.index.str.contains('Latent')]['R2'].max()
    best_rmse_raw = results_df[results_df.index.str.contains('Raw')]['RMSE'].min()
    best_rmse_latent = results_df[results_df.index.str.contains('Latent')]['RMSE'].min()
    
    print(f"\n🏆 Best Raw Features Model: R² = {best_r2_raw:.4f}, RMSE = {best_rmse_raw:.2f}")
    print(f"🏆 Best Latent Features Model: R² = {best_r2_latent:.4f}, RMSE = {best_rmse_latent:.2f}")
    
    if best_r2_latent > best_r2_raw:
        improvement = ((best_r2_latent - best_r2_raw) / best_r2_raw) * 100
        print(f"🎉 AutoEncoder improved R² by {improvement:.2f}%!")
    else:
        print("⚠️  AutoEncoder didn't improve R² in this case.")

def create_comparison_visualizations(results):
    """Create visualizations comparing raw vs latent features"""
    
    # Prepare data for plotting
    results_df = pd.DataFrame(results).T
    
    # Split into raw and latent
    raw_models = results_df[results_df.index.str.contains('Raw')]
    latent_models = results_df[results_df.index.str.contains('Latent')]
    
    # Clean model names
    raw_models.index = raw_models.index.str.replace(' (Raw)', '')
    latent_models.index = latent_models.index.str.replace(' (Latent)', '')
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Raw Features vs AutoEncoder Latent Features Comparison', fontsize=16, fontweight='bold')
    
    # R² comparison
    x = np.arange(len(raw_models))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, raw_models['R2'], width, label='Raw Features', alpha=0.8, color='skyblue')
    axes[0, 0].bar(x + width/2, latent_models['R2'], width, label='Latent Features', alpha=0.8, color='lightcoral')
    axes[0, 0].set_xlabel('Models')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_title('R² Score Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(raw_models.index, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # RMSE comparison
    axes[0, 1].bar(x - width/2, raw_models['RMSE'], width, label='Raw Features', alpha=0.8, color='skyblue')
    axes[0, 1].bar(x + width/2, latent_models['RMSE'], width, label='Latent Features', alpha=0.8, color='lightcoral')
    axes[0, 1].set_xlabel('Models')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].set_title('RMSE Comparison')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(raw_models.index, rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # MAE comparison
    axes[1, 0].bar(x - width/2, raw_models['MAE'], width, label='Raw Features', alpha=0.8, color='skyblue')
    axes[1, 0].bar(x + width/2, latent_models['MAE'], width, label='Latent Features', alpha=0.8, color='lightcoral')
    axes[1, 0].set_xlabel('Models')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].set_title('MAE Comparison')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(raw_models.index, rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Improvement percentage
    improvement_r2 = ((latent_models['R2'] - raw_models['R2']) / raw_models['R2']) * 100
    axes[1, 1].bar(x, improvement_r2, color='gold', alpha=0.8)
    axes[1, 1].set_xlabel('Models')
    axes[1, 1].set_ylabel('R² Improvement (%)')
    axes[1, 1].set_title('R² Improvement with AutoEncoder')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(raw_models.index, rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_autoencoder_visualizations(autoencoder, data, history):
    """Create AutoEncoder-specific visualizations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('AutoEncoder Analysis', fontsize=16, fontweight='bold')
    
    # Training history
    axes[0, 0].plot(history.history['loss'], label='Training Loss', color='blue')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', color='red')
    axes[0, 0].set_title('AutoEncoder Training History')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Latent space visualization
    X = data.drop(columns=['price'])
    latent_features = autoencoder.extract_latent_features(X)
    
    if latent_features.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(latent_features)
    else:
        latent_2d = latent_features
    
    scatter = axes[0, 1].scatter(latent_2d[:, 0], latent_2d[:, 1], 
                                c=data['price'], cmap='viridis', alpha=0.6)
    axes[0, 1].set_title('Latent Space Visualization')
    axes[0, 1].set_xlabel('Latent Dimension 1')
    axes[0, 1].set_ylabel('Latent Dimension 2')
    plt.colorbar(scatter, ax=axes[0, 1], label='Price')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Reconstruction quality
    X_processed = autoencoder.preprocess_data(X)
    reconstructed = autoencoder.reconstruct_data(X)
    
    # Plot original vs reconstructed for a few features
    feature_indices = [0, 5, 10]  # Sample features
    feature_names = list(X.columns)[:3]
    
    for i, (idx, name) in enumerate(zip(feature_indices, feature_names)):
        axes[1, 0].scatter(X_processed[:100, idx], reconstructed[:100, idx], 
                          alpha=0.6, label=name)
    
    axes[1, 0].plot([X_processed[:100, :3].min(), X_processed[:100, :3].max()], 
                    [X_processed[:100, :3].min(), X_processed[:100, :3].max()], 
                    'r--', label='Perfect Reconstruction')
    axes[1, 0].set_title('Original vs Reconstructed Features')
    axes[1, 0].set_xlabel('Original Values')
    axes[1, 0].set_ylabel('Reconstructed Values')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Latent feature importance
    latent_importance = np.abs(latent_features).mean(axis=0)
    axes[1, 1].bar(range(len(latent_importance)), latent_importance, color='lightgreen', alpha=0.8)
    axes[1, 1].set_title('Latent Feature Importance')
    axes[1, 1].set_xlabel('Latent Feature Index')
    axes[1, 1].set_ylabel('Average Absolute Value')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('autoencoder_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Run the complete demo
    results, autoencoder = run_autoencoder_demo()
    
    print("\n🎉 AutoEncoder Demo Completed Successfully!")
    print("📁 Generated files:")
    print("   - ev_data_exploration.png")
    print("   - model_comparison.png") 
    print("   - autoencoder_analysis.png")
    print("   - models/ev_autoencoder_*.h5 (AutoEncoder models)")
    print("   - models/ev_autoencoder_*.pkl (Preprocessors)") 