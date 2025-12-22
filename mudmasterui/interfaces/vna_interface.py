"""
VNA Interface
Provides a class to interact with the VNA Flask server and process measurements.
"""
import requests
import numpy as np
import os
import glob
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class VNAInterface:
    def __init__(self, vna_server_url="http://127.0.0.1:5000", cal_folder=None):
        """
        Initialize the VNA interface.
        @param vna_server_url: URL of the VNA Flask server
        @param cal_folder: Path to calibration .s1p files (optional)
        """
        self.vna_server_url = vna_server_url
        self.cal_data = {}
        
        if cal_folder:
            self.load_calibration(cal_folder)

    def get_new_data(self):
        """
        Trigger a new measurement from the VNA server.
        @return: Dictionary with VNA measurement data, or None if request fails
        """
        try:
            response = requests.get(f"{self.vna_server_url}/VNA/get-new-measurement", timeout=30)
            if response.status_code == 200:
                return response.json()
            print(f"Failed to get VNA data: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with VNA server: {e}")
        return None

    def load_calibration(self, cal_folder):
        """Load .s1p calibration files."""
        for file_path in glob.glob(os.path.join(cal_folder, "*.s1p")):
            height = float(os.path.basename(file_path).replace('.s1p', ''))
            freq, s11_real, s11_imag = [], [], []
            
            with open(file_path, 'r') as f:
                for line in f:
                    if line.startswith('!') or line.startswith('#') or not line.strip():
                        continue
                    parts = line.split()
                    if len(parts) >= 3:
                        freq.append(float(parts[0]))
                        s11_real.append(float(parts[1]))
                        s11_imag.append(float(parts[2]))
            
            self.cal_data[height] = {
                'freq': np.array(freq),
                's11': np.array(s11_real) + 1j * np.array(s11_imag)
            }
        
        print(f"Loaded calibration for heights: {sorted(self.cal_data.keys())}")

    def apply_calibration(self, s11_complex, actuator_height_mm, frequencies):
        """
        Subtract background S11 at closest height.
        @param s11_complex: Complex S11 parameters
        @param actuator_height_mm: Current actuator height in mm
        @param frequencies: Frequency array for interpolation
        @return: Calibrated S11 parameters
        """
        if not self.cal_data:
            return s11_complex
        
        # Find closest calibration height
        closest = min(self.cal_data.keys(), key=lambda h: abs(h - actuator_height_mm))
        cal_freq = self.cal_data[closest]['freq']
        cal_s11 = self.cal_data[closest]['s11']
        
        # Interpolate if frequencies don't match
        if len(cal_s11) != len(s11_complex):
            cal_s11_real = np.interp(frequencies, cal_freq, cal_s11.real)
            cal_s11_imag = np.interp(frequencies, cal_freq, cal_s11.imag)
            cal_s11 = cal_s11_real + 1j * cal_s11_imag
        
        return s11_complex - cal_s11

    def convert_to_complex_sparams(self, vna_data):
        """
        Convert VNA amplitude (dB) and phase (degrees) to complex S-parameters.
        @param vna_data: Raw VNA measurement data
        @return: Dictionary with frequencies and complex S-parameters
        """
        if not vna_data or 'port1' not in vna_data:
            print("Invalid VNA data format")
            return None
        
        frequencies = []
        s_params = []
        
        for point in vna_data['port1']:
            freq = point['frequency']
            amplitude_db = point['amplitude']
            phase_deg = point['phase']
            
            # Convert to complex: magnitude * e^(j*phase)
            magnitude = 10 ** (amplitude_db / 20)
            phase_rad = np.deg2rad(phase_deg)
            s_param = magnitude * np.exp(1j * phase_rad)
            
            frequencies.append(freq)
            s_params.append(s_param)
        
        return {
            'frequencies': np.array(frequencies),
            's_parameters': np.array(s_params),
            'measurement_count': vna_data.get('measurementCount', 0),
            'sweep_time': vna_data.get('sweepTime', 0)
        }

    def extract_permittivity_simple(self, s11, frequencies):
        """
        Extract relative permittivity from S11 using simplified reflection model.
        For open-ended coaxial probe: epsilon_r ≈ ((1 - S11) / (1 + S11))^2
        
        @param s11: Complex S11 parameters
        @param frequencies: Frequencies in Hz
        @return: Dictionary with complex permittivity
        """
        # Avoid division by zero
        denominator = 1 + s11
        epsilon = np.where(
            np.abs(denominator) > 1e-10,
            ((1 - s11) / denominator) ** 2,
            np.nan + 1j*np.nan
        )
        
        epsilon_real = np.real(epsilon)
        epsilon_imag = np.imag(epsilon)
        
        return {
            'frequencies': frequencies,
            'frequencies_ghz': frequencies / 1e9,
            'epsilon_real': epsilon_real,
            'epsilon_imag': epsilon_imag,
            'epsilon_complex': epsilon,
            'loss_tangent': epsilon_imag / (epsilon_real + 1e-10)
        }

    def process_measurement(self, vna_data, actuator_height_mm=None):
        """
        Complete pipeline: raw VNA data → complex S-params → calibrated → permittivity.
        
        @param vna_data: Raw VNA measurement data
        @param actuator_height_mm: Actuator height for calibration (optional)
        @return: Dictionary with permittivity data
        """
        # Convert to complex S-parameters
        s_data = self.convert_to_complex_sparams(vna_data)
        if not s_data:
            return None
        
        # Apply calibration if height provided
        s11 = s_data['s_parameters']
        if actuator_height_mm is not None:
            s11 = self.apply_calibration(s11, actuator_height_mm, s_data['frequencies'])
        
        # Extract permittivity
        result = self.extract_permittivity_simple(s11, s_data['frequencies'])
        
        # Add metadata
        result['measurement_count'] = s_data['measurement_count']
        result['sweep_time'] = s_data['sweep_time']
        
        return result

    def collect_training_data(
        self, 
        actuator_heights: List[float], 
        ground_truth_values: List[float],
        num_measurements: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collect training data by taking VNA measurements at different conditions.
        
        @param actuator_heights: List of actuator heights (mm) for each sample
        @param ground_truth_values: List of true shear strength values (kPa)
        @param num_measurements: Number of repeat measurements per sample
        @return: (X, y) where X is feature matrix, y is target values
        """
        if len(actuator_heights) != len(ground_truth_values):
            raise ValueError("actuator_heights and ground_truth_values must have same length")
        
        print(f"\nCollecting training data for {len(actuator_heights)} samples...")
        
        features_list = []
        targets_list = []
        
        for i, (height, target) in enumerate(zip(actuator_heights, ground_truth_values)):
            print(f"Sample {i+1}/{len(actuator_heights)}: height={height}mm, target={target}kPa")
            
            sample_features = []
            for rep in range(num_measurements):
                # Get new VNA measurement
                vna_data = self.get_new_data()
                if not vna_data:
                    print(f"  Warning: Failed to get measurement (rep {rep+1})")
                    continue
                
                # Process measurement
                result = self.process_measurement(vna_data, actuator_height_mm=height)
                if not result:
                    print(f"  Warning: Failed to process measurement (rep {rep+1})")
                    continue
                
                # Extract features
                features = np.concatenate([
                    result['epsilon_real'],
                    result['epsilon_imag'],
                    result['loss_tangent']
                ])
                sample_features.append(features)
            
            if sample_features:
                # Average repeated measurements
                avg_features = np.mean(sample_features, axis=0)
                features_list.append(avg_features)
                targets_list.append(target)
                print(f"  ✓ Collected {len(sample_features)} measurements")
        
        X = np.array(features_list)
        y = np.array(targets_list)
        
        print(f"\nCollected {len(X)} valid samples")
        print(f"Feature shape: {X.shape}")
        
        return X, y

    def train_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth_cm: int = 100,
        hidden_layers: Tuple[int, ...] = (256, 128, 64),
        save_model: bool = True,
        model_name: Optional[str] = None
    ) -> Dict:
        """
        Train a neural network model from collected data.
        
        @param X: Feature matrix (n_samples, n_features)
        @param y: Target values (n_samples,)
        @param depth_cm: Target depth in cm
        @param hidden_layers: Neural network architecture
        @param save_model: Whether to save the trained model
        @param model_name: Optional custom model name
        @return: Training results dictionary
        """
        # Import here to avoid dependency if not used
        from models.nn_trainer import NeuralNetworkTrainer
        
        trainer = NeuralNetworkTrainer(depth_cm=depth_cm)
        
        results = trainer.train(
            X, y,
            hidden_layers=hidden_layers,
            verbose=True
        )
        
        if save_model:
            paths = trainer.save(name=model_name)
            results['saved_paths'] = paths
        
        return results

