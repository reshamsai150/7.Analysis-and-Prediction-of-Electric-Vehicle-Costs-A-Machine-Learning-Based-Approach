"""
AutoEncoder-based Feature Learning for EV Cost Prediction
Author: [Your Name]
Date: [Current Date]
Description: Deep AutoEncoder implementation to compress high-dimensional EV data
             into lower-dimensional latent representations for improved prediction
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

class AutoEncoderFeatureLearner:
    """
    Deep AutoEncoder for feature learning in EV cost prediction
    """
    
    def __init__(self, input_dim, latent_dim=32, encoding_dims=[128, 64]):
        """
        Initialize AutoEncoder
        
        Args:
            input_dim (int): Number of input features
            latent_dim (int): Dimension of latent space
            encoding_dims (list): List of hidden layer dimensions for encoder
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoding_dims = encoding_dims
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        
    def build_autoencoder(self):
        """Build the AutoEncoder architecture"""
        
        # Input layer
        input_layer = layers.Input(shape=(self.input_dim,))
        
        # Encoder layers
        encoded = input_layer
        for dim in self.encoding_dims:
            encoded = layers.Dense(dim, activation='relu')(encoded)
            encoded = layers.BatchNormalization()(encoded)
            encoded = layers.Dropout(0.2)(encoded)
        
        # Latent layer
        latent = layers.Dense(self.latent_dim, activation='relu', name='latent')(encoded)
        
        # Decoder layers (reverse of encoder)
        decoded = latent
        for dim in reversed(self.encoding_dims):
            decoded = layers.Dense(dim, activation='relu')(decoded)
            decoded = layers.BatchNormalization()(decoded)
            decoded = layers.Dropout(0.2)(decoded)
        
        # Output layer
        output_layer = layers.Dense(self.input_dim, activation='linear')(decoded)
        
        # Create models
        self.autoencoder = keras.Model(input_layer, output_layer, name='autoencoder')
        self.encoder = keras.Model(input_layer, latent, name='encoder')
        
        # Decoder (takes latent space as input)
        latent_input = layers.Input(shape=(self.latent_dim,))
        decoded_layers = []
        for layer in self.autoencoder.layers:
            if layer.name == 'latent':
                break
            decoded_layers.append(layer)
        
        decoded_output = latent_input
        for layer in self.autoencoder.layers[len(decoded_layers)+1:]:
            decoded_output = layer(decoded_output)
        
        self.decoder = keras.Model(latent_input, decoded_output, name='decoder')
        
        # Compile autoencoder
        self.autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return self.autoencoder
    
    def preprocess_data(self, data):
        """
        Preprocess the data for AutoEncoder training
        
        Args:
            data (pd.DataFrame): Raw EV data
            
        Returns:
            np.array: Preprocessed data
        """
        data_processed = data.copy()
        
        # Handle categorical variables
        categorical_columns = data_processed.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                data_processed[col] = self.label_encoders[col].fit_transform(data_processed[col])
            else:
                data_processed[col] = self.label_encoders[col].transform(data_processed[col])
        
        # Handle missing values
        data_processed = data_processed.fillna(data_processed.mean())
        
        # Scale the data
        data_scaled = self.scaler.fit_transform(data_processed)
        
        return data_scaled
    
    def train(self, data, epochs=100, batch_size=32, validation_split=0.2, verbose=1):
        """
        Train the AutoEncoder
        
        Args:
            data (pd.DataFrame): Training data
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Fraction of data for validation
            verbose (int): Verbosity level
            
        Returns:
            keras.callbacks.History: Training history
        """
        # Preprocess data
        X_processed = self.preprocess_data(data)
        
        # Build model if not already built
        if self.autoencoder is None:
            self.build_autoencoder()
        
        # Early stopping callback
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Reduce learning rate callback
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        
        # Train the model
        history = self.autoencoder.fit(
            X_processed, X_processed,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, reduce_lr],
            verbose=verbose
        )
        
        return history
    
    def extract_latent_features(self, data):
        """
        Extract latent features from the trained AutoEncoder
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            np.array: Latent features
        """
        if self.encoder is None:
            raise ValueError("AutoEncoder must be trained before extracting latent features")
        
        # Preprocess data
        X_processed = self.preprocess_data(data)
        
        # Extract latent features
        latent_features = self.encoder.predict(X_processed)
        
        return latent_features
    
    def reconstruct_data(self, data):
        """
        Reconstruct data using the AutoEncoder
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            np.array: Reconstructed data
        """
        if self.autoencoder is None:
            raise ValueError("AutoEncoder must be trained before reconstruction")
        
        # Preprocess data
        X_processed = self.preprocess_data(data)
        
        # Reconstruct data
        reconstructed = self.autoencoder.predict(X_processed)
        
        return reconstructed
    
    def evaluate_reconstruction(self, data):
        """
        Evaluate the reconstruction quality
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            dict: Evaluation metrics
        """
        X_processed = self.preprocess_data(data)
        reconstructed = self.reconstruct_data(data)
        
        mse = mean_squared_error(X_processed, reconstructed)
        mae = mean_absolute_error(X_processed, reconstructed)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': np.sqrt(mse)
        }
    
    def visualize_latent_space(self, data, target_column=None, save_path=None):
        """
        Visualize the latent space
        
        Args:
            data (pd.DataFrame): Input data
            target_column (str): Target column for coloring (optional)
            save_path (str): Path to save the plot (optional)
        """
        if self.latent_dim > 2:
            print("Latent dimension > 2, using PCA for visualization")
            from sklearn.decomposition import PCA
            latent_features = self.extract_latent_features(data)
            pca = PCA(n_components=2)
            latent_2d = pca.fit_transform(latent_features)
        else:
            latent_2d = self.extract_latent_features(data)
        
        plt.figure(figsize=(10, 8))
        
        if target_column and target_column in data.columns:
            scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], 
                                c=data[target_column], cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, label=target_column)
        else:
            plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.6)
        
        plt.title('Latent Space Visualization')
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_training_history(self, history, save_path=None):
        """
        Plot training history
        
        Args:
            history: Training history from model.fit()
            save_path (str): Path to save the plot (optional)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MAE plot
        ax2.plot(history.history['mae'], label='Training MAE')
        ax2.plot(history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_model(self, filepath):
        """
        Save the trained model and preprocessors
        
        Args:
            filepath (str): Base filepath for saving
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save AutoEncoder models
        self.autoencoder.save(f"{filepath}_autoencoder.keras")
        self.encoder.save(f"{filepath}_encoder.keras")
        self.decoder.save(f"{filepath}_decoder.keras")
        
        # Save preprocessors
        joblib.dump(self.scaler, f"{filepath}_scaler.pkl")
        joblib.dump(self.label_encoders, f"{filepath}_label_encoders.pkl")
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load the trained model and preprocessors
        
        Args:
            filepath (str): Base filepath for loading
        """
        # Load AutoEncoder models
        self.autoencoder = keras.models.load_model(f"{filepath}_autoencoder.keras")
        self.encoder = keras.models.load_model(f"{filepath}_encoder.keras")
        self.decoder = keras.models.load_model(f"{filepath}_decoder.keras")
        
        # Load preprocessors
        self.scaler = joblib.load(f"{filepath}_scaler.pkl")
        self.label_encoders = joblib.load(f"{filepath}_label_encoders.pkl")
        
        print(f"Model loaded from {filepath}")


def compare_models_with_autoencoder(data, target_column, test_size=0.2, random_state=42):
    """
    Compare traditional ML models with AutoEncoder-enhanced features
    
    Args:
        data (pd.DataFrame): Complete dataset
        target_column (str): Target column name
        test_size (float): Fraction for test set
        random_state (int): Random seed
        
    Returns:
        dict: Comparison results
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor
    from sklearn.linear_model import LinearRegression
    from xgboost import XGBRegressor
    
    # Prepare data
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Initialize AutoEncoder
    autoencoder = AutoEncoderFeatureLearner(
        input_dim=X.shape[1],
        latent_dim=32,
        encoding_dims=[128, 64]
    )
    
    # Train AutoEncoder
    print("Training AutoEncoder...")
    history = autoencoder.train(X_train, epochs=50, verbose=1)
    
    # Extract latent features
    print("Extracting latent features...")
    X_train_latent = autoencoder.extract_latent_features(X_train)
    X_test_latent = autoencoder.extract_latent_features(X_test)
    
    # Define models to compare
    models = {
        'Random Forest (Raw)': RandomForestRegressor(n_estimators=100, random_state=random_state),
        'Random Forest (Latent)': RandomForestRegressor(n_estimators=100, random_state=random_state),
        'XGBoost (Raw)': XGBRegressor(n_estimators=100, random_state=random_state),
        'XGBoost (Latent)': XGBRegressor(n_estimators=100, random_state=random_state),
        'SVR (Raw)': SVR(kernel='rbf'),
        'SVR (Latent)': SVR(kernel='rbf'),
        'MLP (Raw)': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=random_state),
        'MLP (Latent)': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=random_state),
        'Linear Regression (Raw)': LinearRegression(),
        'Linear Regression (Latent)': LinearRegression()
    }
    
    results = {}
    
    # Train and evaluate models
    for name, model in models.items():
        print(f"Training {name}...")
        
        if 'Latent' in name:
            # Use latent features
            model.fit(X_train_latent, y_train)
            y_pred = model.predict(X_test_latent)
        else:
            # Use raw features
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'MSE': mse,
            'MAE': mae,
            'R2': r2,
            'RMSE': np.sqrt(mse)
        }
        
        print(f"{name} - R²: {r2:.4f}, RMSE: {np.sqrt(mse):.4f}")
    
    return results, autoencoder, history


if __name__ == "__main__":
    # Example usage
    print("AutoEncoder Feature Learning for EV Cost Prediction")
    print("=" * 50)
    
    # Load your EV dataset here
    # data = pd.read_csv('your_ev_dataset.csv')
    # results, autoencoder, history = compare_models_with_autoencoder(data, 'price')
    
    print("Model implementation complete!") 