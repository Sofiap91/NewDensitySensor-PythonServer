"""
*******************************************************************************
@file   teltonikaSupportFunctions.py
@author Joshua Paterson
@date   16 8 2024
@brief  Support functions for getting gps data from teltonika
"""
import json
import time
import threading

import requests
import json
import urllib3

# Disable SSL warnings when using verify=False
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning) 

from MudMasterUI.config import Config
from MudMasterUI import globalErrorVar

""" Defines
*******************************************************************************
"""
token = None
DEVICE_IP = Config.CONFIG_SYSTEM['teltonika']['DEVICE_IP']

""" Functions
*******************************************************************************
"""


def login_endpoint():
    """Logs in to the Teltonika device and retrieves an authentication token."""
    global token
    url = f'https://{DEVICE_IP}/api/login'
    headers = {
        'Content-Type': 'application/json'
    }
    data = {
        'username': Config.CONFIG_SYSTEM['teltonika']['username'],
        'password': Config.CONFIG_SYSTEM['teltonika']['password']
    }

    try:
        # Convert the data dictionary to JSON format
        json_data = json.dumps(data)

        # Make the POST request (verify=False to bypass SSL certificate verification for self-signed certs)
        response = requests.post(url, headers=headers, data=json_data, verify=False, timeout=5)

        # Check if the request was successful (status code 2xx)
        if response.ok:
            #print(f'Response Status Code: {response.status_code}')
            #print('Response Content:')
            #print(response.text)
            if(response.status_code == 200):
                    #saves token as variable 
                    # Convert response content to JSON
                    data = response.json()
                    token = data.get('data', {}).get('token')
            #print(token)
            #print("end of login")
            return token  # Return JSON response as Python dictionary
        else:
            globalErrorVar.ErrorFromTeltonika = True
            response.raise_for_status()  # Raise an HTTPError for non-2xx status codes

    except requests.exceptions.RequestException as e:
        globalErrorVar.ErrorFromTeltonika = True
        print(f'Error making request: {e}')  # Handle connection errors, timeouts, etc.
    
    return None  # Return None if request fails

def get_GPS_data_endpoint():
    """Retrieves GPS data from the Teltonika device using the provided token."""
    global token
    url = f'https://{DEVICE_IP}/ubus'  # Adjust the URL as per your device's API endpoint
    headers = {

    }
    
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "call",
        "params": [
            f'{token}',
            "file",
            "exec",
            {
                "command": "gpsctl",
                "params": ["-ix"]
            }
        ]
    }
    
    #print(payload)
    
    try:
        # Make the POST request with JSON payload (verify=False to bypass SSL certificate verification)
        response = requests.post(url, headers=headers, json=payload, verify=False, timeout=5)

        # Check if the request was successful (status code 2xx)
        if response.ok:
            #print(f'Response Status Code: {response.status_code}')
            #print('Response Content:')
            #print(response.json())  # Print the entire JSON response
            
            # Parse the response to get latitude and longitude
            result = response.json().get('result', [])
            if len(result) > 1:
                stdout = result[1].get('stdout', '')
                latitude, longitude = map(float, stdout.strip().split())
                #print(f'Latitude: {latitude}, Longitude: {longitude}')
                if(latitude == 0 and longitude == 0):
                    print("gps values are 0 and 0 likely no gps")
                    globalErrorVar.ErrorFromTeltonika = True
                    pass
                # Return the GPS data as a dictionary
                globalErrorVar.ErrorFromTeltonika = False
                return {"latitude": latitude, "longitude": longitude}
            else:
                globalErrorVar.ErrorFromTeltonika = True
                print('Unexpected response structure')
                return {"Response Structure Error"}
        else:
            globalErrorVar.ErrorFromTeltonika = True
            response.raise_for_status()  # Raise an HTTPError for non-2xx status codes
            return {"Response Not ok"}

    except requests.exceptions.RequestException as e:
        globalErrorVar.ErrorFromTeltonika = True
        print(f'Error making request: {e}')  # Handle connection errors, timeouts, etc.
        return {"Request Exception": str(e)}

    return {"Error from teltonika"}  # Return None if request fails




