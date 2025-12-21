import constants as const
from interfaces.teltonika_interface import TeltonikaInterface
from interfaces.vna_interface import VNAInterface
from interfaces.arduino_interface import ArduinoInterface

# DEBUG = 'router'
DEBUG = 'arduino'
# DEBUG = 'vna'
# DEBUG = 'all'


if __name__ == "__main__":
    if DEBUG == 'router' or DEBUG == 'all':
        # Initialize interfaces
        teltonika = TeltonikaInterface()
        coords = teltonika.get_gps_coordinates()
        if coords:
            print(f"Latitude: {coords['latitude']}, Longitude: {coords['longitude']}")
        else:
            print("Could not retrieve GPS coordinates.")

    if DEBUG == 'arduino' or DEBUG == 'all':
        arduino = ArduinoInterface()
        if arduino.connect():
            print("Connected to Arduino. Getting distance to ground...")
            distance = arduino.get_distance_to_ground()
            print(f"Distance to ground: {distance}")
            print("Retracting actuator...")
            retract_result = arduino.fullyfrom interfaces.arduino_interface import ArduinoInterface_retract()
            print(f"Retract result: {retract_result}")
            arduino.disconnect()
            print("Disconnected from Arduino.")
        else:
            print("Failed to connect to Arduino.")
