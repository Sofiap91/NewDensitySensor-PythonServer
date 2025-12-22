"""
Production Inference Module
============================

Loads trained models and makes real-time predictions on VNA measurements.
"""

from pathlib import Path
import joblib
import numpy as np
from typing import Dict, Optional, List
import json


class ModelPredictor:
    """Load and use trained models for real-time prediction"""
    
    def __init__(self, models_folder: str = "models/nn"):
        """
        Initialize predictor
        
        Args:
            models_folder: Folder containing trained models
        """
        self.models_folder = Path(models_folder)
        self.models = {}  # {depth_cm: {'model': model, 'scaler': scaler, 'metadata': dict}}
        self.loaded_depths = []
    
    def load_model(self, depth_cm: int, model_name: Optional[str] = None):
        """
        Load a trained model for specific depth
        
        Args:
            depth_cm: Target depth (e.g., 20, 50, 80, 100)
            model_name: Optional specific model name (otherwise loads latest)
        """
        if model_name:
            # Load specific model by name
            model_path = self.models_folder / f"{model_name}_model.pkl"
            scaler_path = self.models_folder / f"{model_name}_scaler.pkl"
            metadata_path = self.models_folder / f"{model_name}_metadata.json"
        else:
            # Find latest model for this depth
            pattern = f"nn_{depth_cm}cm_*_model.pkl"
            model_files = sorted(self.models_folder.glob(pattern))
            
            if not model_files:
                raise FileNotFoundError(f"No models found for {depth_cm}cm depth")
            
            # Get latest (last in sorted list)
            model_path = model_files[-1]
            base_name = model_path.stem.replace('_model', '')
            scaler_path = self.models_folder / f"{base_name}_scaler.pkl"
            metadata_path = self.models_folder / f"{base_name}_metadata.json"
        
        # Load files
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        self.models[depth_cm] = {
            'model': model,
            'scaler': scaler,
            'metadata': metadata,
            'path': str(model_path)
        }
        self.loaded_depths.append(depth_cm)
        
        print(f"Loaded model for {depth_cm}cm:")
        print(f"  Path: {model_path}")
        if 'test_metrics' in metadata:
            metrics = metadata['test_metrics']
            print(f"  Test R²: {metrics.get('r2', 'N/A'):.4f}")
            print(f"  Test MAE: {metrics.get('mae', 'N/A'):.3f}")
    
    def load_all_depths(self, depths: List[int] = [20, 50, 80, 100]):
        """
        Load models for all specified depths
        
        Args:
            depths: List of depths to load models for
        """
        for depth in depths:
            try:
                self.load_model(depth)
            except FileNotFoundError as e:
                print(f"Warning: {e}")
    
    def predict(self, vna_measurement: Dict, depth_cm: int) -> Dict:
        """
        Make prediction for a single VNA measurement
        
        Args:
            vna_measurement: Dict with 'epsilon_real', 'epsilon_imag', 'loss_tangent'
            depth_cm: Target depth
            
        Returns:
            Dict with prediction and metadata
        """
        if depth_cm not in self.models:
            raise ValueError(f"No model loaded for {depth_cm}cm. Call load_model() first.")
        
        # Extract features
        features = self._extract_features(vna_measurement)
        
        # Get model components
        model = self.models[depth_cm]['model']
        scaler = self.models[depth_cm]['scaler']
        
        # Scale and predict
        features_scaled = scaler.transform(features.reshape(1, -1))
        prediction = model.predict(features_scaled)[0]
        
        return {
            'depth_cm': depth_cm,
            'shear_strength_kPa': float(prediction),
            'model_path': self.models[depth_cm]['path'],
            'timestamp': vna_measurement.get('timestamp', None)
        }
    
    def predict_all_depths(self, vna_measurement: Dict) -> Dict[int, Dict]:
        """
        Make predictions for all loaded depths
        
        Args:
            vna_measurement: VNA measurement data
            
        Returns:
            Dict mapping depth to prediction results
        """
        results = {}
        for depth in self.loaded_depths:
            try:
                results[depth] = self.predict(vna_measurement, depth)
            except Exception as e:
                print(f"Error predicting for {depth}cm: {e}")
                results[depth] = {'error': str(e)}
        
        return results
    
    def _extract_features(self, vna_measurement: Dict) -> np.ndarray:
        """
        Extract feature vector from VNA measurement
        
        Args:
            vna_measurement: Dict with epsilon_real, epsilon_imag, loss_tangent
            
        Returns:
            Feature array
        """
        epsilon_real = np.array(vna_measurement['epsilon_real'])
        epsilon_imag = np.array(vna_measurement['epsilon_imag'])
        loss_tangent = np.array(vna_measurement['loss_tangent'])
        
        # Concatenate all features
        features = np.concatenate([epsilon_real, epsilon_imag, loss_tangent])
        
        return features
    
    def get_model_info(self, depth_cm: int) -> Dict:
        """Get information about loaded model"""
        if depth_cm not in self.models:
            return {'error': 'Model not loaded'}
        
        return {
            'depth_cm': depth_cm,
            'path': self.models[depth_cm]['path'],
            'metadata': self.models[depth_cm]['metadata']
        }


# Convenience function for quick predictions
def quick_predict(vna_measurement: Dict, depth_cm: int, model_name: Optional[str] = None) -> float:
    """
    Quick prediction without managing predictor instance
    
    Args:
        vna_measurement: VNA measurement data
        depth_cm: Target depth
        model_name: Optional specific model name
        
    Returns:
        Predicted shear strength (kPa)
    """
    predictor = ModelPredictor()
    predictor.load_model(depth_cm, model_name)
    result = predictor.predict(vna_measurement, depth_cm)
    return result['shear_strength_kPa']
