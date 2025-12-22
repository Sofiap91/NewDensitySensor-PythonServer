"""
Neural Network Training for VNA-based Shear Strength Prediction
================================================================

Trains neural network models to predict shear strength from VNA measurements.
Can be used standalone or integrated with VNA interface.
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, List


class NeuralNetworkTrainer:
    """Train neural networks for shear strength prediction from VNA data"""
    
    def __init__(self, depth_cm: int = 100, models_folder: str = "models/nn"):
        """
        Initialize trainer
        
        Args:
            depth_cm: Target depth in cm (e.g., 20, 50, 80, 100)
            models_folder: Folder to save trained models
        """
        self.depth_cm = depth_cm
        self.models_folder = Path(models_folder)
        self.models_folder.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.scaler = None
        self.metadata = {
            'depth_cm': depth_cm,
            'created_at': datetime.now().isoformat()
        }
    
    def prepare_features(self, vna_measurements: List[Dict]) -> np.ndarray:
        """
        Extract features from VNA measurements
        
        Args:
            vna_measurements: List of dicts with 'epsilon_real', 'epsilon_imag', 'frequencies'
            
        Returns:
            Feature matrix (n_samples, n_features)
        """
        features_list = []
        
        for measurement in vna_measurements:
            # Extract permittivity features
            epsilon_real = measurement.get('epsilon_real', [])
            epsilon_imag = measurement.get('epsilon_imag', [])
            loss_tangent = measurement.get('loss_tangent', [])
            frequencies = measurement.get('frequencies_ghz', [])
            
            # Flatten all features into a single vector
            feature_vector = np.concatenate([
                epsilon_real,
                epsilon_imag,
                loss_tangent
            ])
            
            features_list.append(feature_vector)
        
        return np.array(features_list)
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        hidden_layers: Tuple[int, ...] = (256, 128, 64),
        max_iter: int = 1000,
        learning_rate_init: float = 0.001,
        early_stopping: bool = True,
        validation_fraction: float = 0.1,
        verbose: bool = True
    ) -> Dict:
        """
        Train neural network model
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            test_size: Fraction of data for testing
            hidden_layers: Tuple of hidden layer sizes
            max_iter: Maximum training iterations
            learning_rate_init: Initial learning rate
            early_stopping: Enable early stopping
            validation_fraction: Fraction for validation (if early_stopping=True)
            verbose: Print training progress
            
        Returns:
            Dictionary with training metrics
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"Training Neural Network for {self.depth_cm}cm depth")
            print(f"{'='*70}")
            print(f"Data: {X.shape[0]} samples, {X.shape[1]} features")
            print(f"Architecture: Input({X.shape[1]}) → {hidden_layers} → Output(1)")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Build and train model
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=20,
            random_state=42,
            verbose=verbose
        )
        
        if verbose:
            print(f"\nTraining with {len(X_train)} samples...")
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        train_metrics = self._calculate_metrics(y_train, y_train_pred)
        test_metrics = self._calculate_metrics(y_test, y_test_pred)
        
        # Store metadata
        self.metadata.update({
            'n_samples': len(X),
            'n_features': X.shape[1],
            'hidden_layers': hidden_layers,
            'max_iter': max_iter,
            'n_iter_': self.model.n_iter_,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        })
        
        if verbose:
            print(f"\n{'='*70}")
            print("Training Results")
            print(f"{'='*70}")
            print("\nTraining Set:")
            self._print_metrics(train_metrics)
            print("\nTest Set:")
            self._print_metrics(test_metrics)
            print(f"\nTraining completed in {self.model.n_iter_} iterations")
        
        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        }
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate regression metrics"""
        return {
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'r2': float(r2_score(y_true, y_pred)),
            'mape': float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100)
        }
    
    def _print_metrics(self, metrics: Dict):
        """Print metrics in a formatted way"""
        print(f"  MAE:  {metrics['mae']:.3f}")
        print(f"  RMSE: {metrics['rmse']:.3f}")
        print(f"  R²:   {metrics['r2']:.4f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Predictions (n_samples,)
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save(self, name: Optional[str] = None):
        """
        Save model, scaler, and metadata
        
        Args:
            name: Optional custom name (default: depth-based name)
        """
        if self.model is None:
            raise ValueError("No model to save. Train first.")
        
        if name is None:
            name = f"nn_{self.depth_cm}cm"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = self.models_folder / f"{name}_{timestamp}"
        
        # Save model and scaler
        model_path = f"{base_path}_model.pkl"
        scaler_path = f"{base_path}_scaler.pkl"
        metadata_path = f"{base_path}_metadata.json"
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"\nModel saved:")
        print(f"  Model:    {model_path}")
        print(f"  Scaler:   {scaler_path}")
        print(f"  Metadata: {metadata_path}")
        
        return {
            'model': model_path,
            'scaler': scaler_path,
            'metadata': metadata_path
        }
    
    def load(self, model_path: str, scaler_path: str, metadata_path: Optional[str] = None):
        """
        Load a trained model
        
        Args:
            model_path: Path to model file
            scaler_path: Path to scaler file
            metadata_path: Optional path to metadata file
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        if metadata_path and Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        
        print(f"Model loaded from {model_path}")
        if 'test_metrics' in self.metadata:
            print("Test performance:")
            self._print_metrics(self.metadata['test_metrics'])


# Standalone training example
if __name__ == "__main__":
    print("Neural Network Trainer")
    print("To use this, provide training data (X, y) where:")
    print("  X: VNA features (epsilon_real, epsilon_imag, loss_tangent)")
    print("  y: Target shear strength values")
    print("\nExample:")
    print("  trainer = NeuralNetworkTrainer(depth_cm=100)")
    print("  trainer.train(X, y)")
    print("  trainer.save()")
