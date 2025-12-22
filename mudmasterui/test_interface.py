import constants as const
from interfaces.teltonika_interface import TeltonikaInterface
from interfaces.vna_interface import VNAInterface
from interfaces.arduino_interface import ArduinoInterface
import json

# DEBUG = 'router'
# DEBUG = 'arduino'
DEBUG = 'vna'
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
            retract_result = arduino.fully_retract()
            print(f"Retract result: {retract_result}")
            arduino.disconnect()
            print("Disconnected from Arduino.")
        else:
            print("Failed to connect to Arduino.")

    if DEBUG == 'vna' or DEBUG == 'all':
        # Initialize VNA with calibration
        vna = VNAInterface(
            vna_server_url="http://127.0.0.1:5000",
            cal_folder="Data/vna_cal"
        )

        with open('Data/get_new_measurement.json', 'r') as f:
            vna_data = json.load(f)

        if vna_data:
            print(f"Measurement count: {vna_data.get('measurementCount', 0)}")
            print(f"Sweep time: {vna_data.get('sweepTime', 0):.2f} ms")
            
            # Convert to complex S-parameters
            s_params = vna.convert_to_complex_sparams(vna_data)
            
            if s_params:
                print(f"Number of frequency points: {len(s_params['frequencies'])}")
                print(f"Frequency range: {s_params['frequencies'][0]/1e9:.2f} - {s_params['frequencies'][-1]/1e9:.2f} GHz")
                
                # Apply calibration at 100mm height
                actuator_height = 100  # mm
                s11_calibrated = vna.apply_calibration(
                    s_params['s_parameters'], 
                    actuator_height, 
                    frequencies=s_params['frequencies']
                )
                
                # Calculate permittivity
                epsilon_data = vna.extract_permittivity_simple(
                    s11_calibrated,
                    s_params['frequencies']
                )
                
                if epsilon_data:
                    # Show results at a few frequencies
                    indices = [0, len(epsilon_data['frequencies'])//2, -1]
                    print("\nPermittivity results:")
                    for i in indices:
                        freq_ghz = epsilon_data['frequencies_ghz'][i]
                        eps_real = epsilon_data['epsilon_real'][i]
                        eps_imag = epsilon_data['epsilon_imag'][i]
                        loss_tan = epsilon_data['loss_tangent'][i]
                        print(f"  {freq_ghz:.2f} GHz: ε' = {eps_real:.2f}, ε'' = {eps_imag:.2f}, tan δ = {loss_tan:.3f}")
        else:
            print("Failed to get VNA data")
