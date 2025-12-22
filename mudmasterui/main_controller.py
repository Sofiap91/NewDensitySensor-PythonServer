"""
Main Controller.
Manages measurement workflow, Arduino deployment, and VNA data retrieval
"""
import threading
import time
import requests
import json
from datetime import datetime
from interfaces.teltonika_interface import TeltonikaInterface
from interfaces.arduino_interface import ArduinoInterface
from interfaces.vna_interface import VNAInterface
from models.predictor import ModelPredictor


class MainController:
    def __init__(self, vna_server_url="http://127.0.0.1:5000", cal_folder="Data/vna_cal", 
                 enable_predictions=True, target_depths=[20, 50, 80, 100]):
        """
        Initialize the main controller.

        @param vna_server_url: URL of the VNA server (default: http://127.0.0.1:5000)
        @param cal_folder: Path to VNA calibration files
        @param enable_predictions: Whether to load models and make predictions
        @param target_depths: List of depths (cm) to predict for
        """
        self.vna_server_url = vna_server_url
        self.teltonika = TeltonikaInterface()
        self.is_measuring = False
        self.measurement_thread = None
        self.arduino_deployed = False
        self.enable_predictions = enable_predictions
        self.target_depths = target_depths

        self.arduino = ArduinoInterface()
        self.vna = VNAInterface(vna_server_url=vna_server_url, cal_folder=cal_folder)
        
        # Initialize predictor if enabled
        self.predictor = None
        if enable_predictions:
            try:
                self.predictor = ModelPredictor()
                self.predictor.load_all_depths(target_depths)
                print("✓ Models loaded and ready for predictions")
            except Exception as e:
                print(f"Warning: Could not load prediction models: {e}")
                print("Running in data collection mode only")


    def start_measurements(self):
        """
        Start the measurement process.
        Triggers Arduino deployment and starts VNA data collection thread.
        """
        if self.is_measuring:
            print("Measurements already in progress.")
            return {"status": "error", "message": "Measurements already in progress"}

        print("Starting measurement process...")

        # Step 1: Deploy Arduino (placeholder - implement Arduino communication)
        if not self.deploy_arduino():
            return {"status": "error", "message": "Failed to deploy Arduino"}

        # Step 2: Start VNA data collection in a separate thread
        self.is_measuring = True
        self.measurement_thread = threading.Thread(target=self._collect_vna_data)
        self.measurement_thread.start()

        return {"status": "success", "message": "Measurements started successfully"}


    def stop_measurements(self):
        """Stop the measurement process."""
        if not self.is_measuring:
            print("No measurements in progress.")
            return {"status": "error", "message": "No measurements in progress"}
    
        print("Stopping measurement process...")
        self.is_measuring = False

        if self.measurement_thread:
            self.measurement_thread.join(timeout=10)

        # Retract Arduino (placeholder - implement Arduino communication)
        self.retract_arduino()

        return {"status": "success", "message": "Measurements stopped successfully"}


    def deploy_arduino(self):
        """
        Deploy Arduino to measurement position.
        """
        print("Deploying Arduino...")
        try:
            if self.arduino.connect():
                print("Connected to Arduino. Extending actuator...")
                result = self.arduino.fully_extend()
                print(f"Arduino extend result: {result}")
                self.arduino_deployed = (result == "success")
                return self.arduino_deployed
            else:
                print("Failed to connect to Arduino.")
                return False
        except Exception as e:
            print(f"Error during Arduino deployment: {e}")
            return False


    def retract_arduino(self):
        """
        Retract Arduino from measurement position.
        """
        print("Retracting Arduino...")
        try:
            result = self.arduino.fully_retract()
            print(f"Arduino retract result: {result}")
            self.arduino.disconnect()
            self.arduino_deployed = False
            return result == "success"
        except Exception as e:
            print(f"Error during Arduino retraction: {e}")
            return False


    def _collect_vna_data(self):
        """
        Collect VNA data in a loop until stopped.
        Runs in a separate thread.
        """
        print("VNA data collection thread started.")
    
        while self.is_measuring:
            try:
                # Get new measurement from VNA server
                response = requests.get(
                    f"{self.vna_server_url}/VNA/get-new-measurement",
                    timeout=30
                )

                if response.status_code == 200:
                    vna_data = response.json()
                    self._process_vna_data(vna_data)
                else:
                    print(f"Failed to get VNA data: {response.status_code}")

            except requests.exceptions.RequestException as e:
                print(f"Error collecting VNA data: {e}")

            # Wait before next measurement
            time.sleep(5)

        print("VNA data collection thread stopped.")


    def _process_vna_data(self, vna_data):
        """
        Process VNA data retrieved from the server.

        @param vna_data: Raw VNA data from the server
        """
        print(f"Processing VNA data (measurement count: {vna_data.get('measurementCount', 0)})")

        # Get GPS coordinates from Teltonika
        gps_coords = self.teltonika.get_gps_coordinates()
        
        # Get actuator height from Arduino
        actuator_height = None
        if self.arduino_deployed:
            try:
                actuator_height = self.arduino.get_distance_to_ground()
                print(f"Actuator height: {actuator_height}mm")
            except Exception as e:
                print(f"Warning: Could not get actuator height: {e}")
        
        # Process VNA measurement through interface
        try:
            vna_result = self.vna.process_measurement(vna_data, actuator_height_mm=actuator_height)
            
            if vna_result and self.predictor:
                # Make predictions for all depths
                predictions = self.predictor.predict_all_depths(vna_result)
                
                # Log results
                measurement_data = {
                    'timestamp': datetime.now().isoformat(),
                    'gps': gps_coords,
                    'actuator_height_mm': actuator_height,
                    'predictions': predictions,
                    'frequencies_ghz': vna_result['frequencies_ghz'].tolist()[:5],  # Sample of frequencies
                }
                
                self._log_measurement(measurement_data)
                
                # Print summary
                print("\nPredictions:")
                for depth, pred in predictions.items():
                    if 'shear_strength_kPa' in pred:
                        print(f"  {depth}cm: {pred['shear_strength_kPa']:.2f} kPa")
                
            elif vna_result:
                # No predictions, just log raw data
                print("VNA data processed (no predictions available)")
                
        except Exception as e:
            print(f"Error processing VNA data: {e}")

        # Log GPS if available
        if gps_coords:
            print(f"GPS: Lat={gps_coords['latitude']}, Lon={gps_coords['longitude']}")
        else:
            print("GPS: No coordinates available")
    
    def _log_measurement(self, measurement_data):
        """
        Log measurement data to file
        
        @param measurement_data: Dictionary containing measurement info
        """
        from pathlib import Path
        
        log_folder = Path("Data/measurements")
        log_folder.mkdir(parents=True, exist_ok=True)
        
        # Create daily log file
        date_str = datetime.now().strftime("%Y%m%d")
        log_file = log_folder / f"measurements_{date_str}.jsonl"
        
        # Append to log file (JSON Lines format)
        with open(log_file, 'a') as f:
            f.write(json.dumps(measurement_data) + '\n')


    def get_status(self):
        """Get the current status of the controller."""
        status = {
            "is_measuring": self.is_measuring,
            "arduino_deployed": self.arduino_deployed,
            "vna_server": self.vna_server_url,
            "predictions_enabled": self.enable_predictions,
            "target_depths": self.target_depths
        }
        
        if self.predictor:
            status["loaded_models"] = {
                depth: self.predictor.get_model_info(depth)
                for depth in self.predictor.loaded_depths
            }
        
        return status
