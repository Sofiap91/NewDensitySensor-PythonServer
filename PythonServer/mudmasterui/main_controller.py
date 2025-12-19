"""
Main Controller for VNA Data Collection System
Manages measurement workflow, Arduino deployment, and VNA data retrieval
"""
import threading
import time
import requests
import json
from datetime import datetime
from teltonika_interface import TeltonikaInterface


class MainController:
    def __init__(self, vna_server_url="http://127.0.0.1:5000"):
        """
        Initialize the main controller.

        @param vna_server_url: URL of the VNA server (default: http://127.0.0.1:5000)
        """
        self.vna_server_url = vna_server_url
        self.teltonika = TeltonikaInterface()
        self.is_measuring = False
        self.measurement_thread = None
        self.arduino_deployed = False


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
        # Placeholder for Arduino deployment logic
        # This should communicate with the Arduino to extend the actuator
        self.arduino_deployed = True
        return True


    def retract_arduino(self):
        """
        Retract Arduino from measurement position.
        """
        print("Retracting Arduino...")
        # Placeholder for Arduino retraction logic
        # This should communicate with the Arduino to retract the actuator
        self.arduino_deployed = False
        return True


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

        # For now, just print basic info
        if gps_coords:
            print(f"GPS: Lat={gps_coords['latitude']}, Lon={gps_coords['longitude']}")
        else:
            print("GPS: No coordinates available")


    def get_status(self):
        """Get the current status of the controller."""
        return {
            "is_measuring": self.is_measuring,
            "arduino_deployed": self.arduino_deployed,
            "vna_server": self.vna_server_url
        }
