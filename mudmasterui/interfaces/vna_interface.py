"""
Simplified VNA Interface
Provides a class to trigger and receive new VNA measurement data from the VNA Flask server.
"""
import requests
import constants as const


class VNAInterface:
    def __init__(self, vna_server_url="http://127.0.0.1:5000"):
        """
        Initialize the VNA interface.
        @param vna_server_url: URL of the VNA Flask server (default: http://127.0.0.1:5000)
        """
        self.vna_server_url = vna_server_url

    def get_new_data(self):
        """
        Trigger a new measurement and receive the data from the VNA server.
        @return: Dictionary with VNA measurement data, or None if request fails
        """
        try:
            response = requests.get(f"{self.vna_server_url}/VNA/get-new-measurement", timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Failed to get new VNA data: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with VNA server: {e}")
        return None
