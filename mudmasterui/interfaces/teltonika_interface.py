
import json
import requests
import urllib3
import constants as const

# Disable SSL warnings when using verify=False
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class TeltonikaInterface:
    def __init__(self):
        self.device_ip = const.TeltonikaEndpoints.ip_address.value
        self.user = const.TeltonikaEndpoints.username.value
        self.password = const.TeltonikaEndpoints.password.value
        self.token = None

    def login(self):
        """Logs in to the Teltonika router and retrieves an authentication token."""
        url = f'https://{self.device_ip}/api/login'
        payload = {
            "username": self.user,
            "password": self.password
        }
        try:
            response = requests.post(url, json=payload, verify=False, timeout=5)
            if response.ok:
                data = response.json()
                token = data.get('data', {}).get('token')
                if token:
                    print("Login successful.")
                    self.token = token
                    return True
            print("Failed to log in to the Teltonika router.")
        except requests.exceptions.RequestException as e:
            print(f"Error during login: {e}")
        self.token = None
        return False

    def get_gps_coordinates(self):
        """Retrieve GPS coordinates (latitude and longitude) from the Teltonika router."""
        # Only login if we don't have a token
        if not self.token:
            if not self.login():
                print("Unable to retrieve GPS data: Login failed.")
                return None

        url = f'https://{self.device_ip}/ubus'
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "call",
            "params": [
                self.token,
                "file",
                "exec",
                {"command": "gpsctl", "params": ["-ix"]}
            ]
        }

        try:
            response = requests.post(url, json=payload, verify=False, timeout=5)
            if response.ok:
                result = response.json().get('result', [])
                if len(result) > 1:
                    stdout = result[1].get('stdout', '')
                    latitude, longitude = map(float, stdout.strip().split())
                    return {"latitude": latitude, "longitude": longitude}
            print("Failed to retrieve GPS data or unexpected response structure.")
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Teltonika router: {e}")

        return None

if __name__ == "__main__":
    teltonika = TeltonikaInterface()
    coords = teltonika.get_gps_coordinates()
    if coords:
        print(f"Latitude: {coords['latitude']}, Longitude: {coords['longitude']}")
    else:
        print("Could not retrieve GPS coordinates.")




