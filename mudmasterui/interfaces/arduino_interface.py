"""
Arduino Interface for Mounting System
Controls the actuator that adjusts the sensor height
Based on legacy interface_mountingSystem.py
"""

import serial
import serial.tools.list_ports
import time
import threading

# Device descriptions to look for when scanning ports
DEVICE_DESCRIPTIONS = ['Arduino']


class ArduinoInterface:
    def __init__(self, port=None, baudrate=9600, timeout=1.0, config_commands=None):
        """
        Initialize the Arduino interface with specified port and baudrate.

        @param port: Serial port to connect to (auto-detected if None)
        @param baudrate: Communication speed (default: 9600)
        @param timeout: Serial timeout in seconds (default: 1.0)
        @param config_commands: Dictionary of command strings for actuator control
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.connection = None
        self.serial_number = None
        self._status = -1  # 0 = good, 1 = bad, -1 = not connected
        self._lock = threading.Lock()

        # Default command configuration (can be overridden)
        self.config_commands = config_commands or {
            'RSI PRO Extend': '1\n',
            'RSI PRO Retract': '2\n',
            'RSI PRO Stop': '3\n',
            'RSI PRO Connection Check': '4\n',
            'Get Distance to Ground': '5\n'
        }


    def connect(self):
        """
        Establish a connection to the Arduino.
        Auto-detects Arduino port if not specified.
        """
        try:
            if self.connection is None:
                if self.port is None:
                    available_ports = self.find_arduino_ports()
                    if len(available_ports) >= 1:
                        self.port = available_ports[0].device
                        self.serial_number = available_ports[0].serial_number
                        print(f'Found Arduino on {self.port}')
                        print(f'Serial No.: {self.serial_number}')
                    else:
                        raise Exception("Arduino not found on any serial port.")

                self.connection = serial.Serial(
                    port=self.port,
                    baudrate=self.baudrate,
                    timeout=self.timeout,
                    bytesize=serial.EIGHTBITS
                )

                self._status = 0
                print(f"Connected to Arduino on port {self.port}")
                return True

        except Exception as e:
            print(f'Communication Error: {e}')
            self._status = -1
            return False


    def find_arduino_ports(self):
        """
        Scan available serial ports to find Arduino devices.

        @retval List of serial ports matching Arduino descriptions
        """
        port_list = []
        comports = list(serial.tools.list_ports.comports())
        for p in comports:
            if any(desc in p.description for desc in DEVICE_DESCRIPTIONS):
                print(f'{p} <description:> {p.description} <manufacturer:> {p.manufacturer}')
                port_list.append(p)
        return port_list


    def disconnect(self):
        """Close the connection to the Arduino."""
        try:
            if self.connection is not None:
                if self.connection.is_open:
                    self.connection.close()
                    print('Serial port closed.')
                self.connection = None
                self.serial_number = None
                self._status = -1
        except Exception as e:
            print(f'Error closing serial port: {e}')
            self._status = 1


    def write_read(self, command):
        """
        Send a command to Arduino and read the response.

        @param command: Command string to send
        @retval Response data from Arduino
        """
        with self._lock:
            if self.connection is None or not self.connection.is_open:
                raise Exception("Not connected to Arduino.")

            self.connection.write(bytes(command, 'utf-8'))
            time.sleep(0.05)
            data = self.connection.readline()
            return data


    def fully_extend(self):
        """
        Extends the actuator to its full extent.
        
        @retval str - Returns "success" if successful; otherwise "fail"
        """
        try:
            command = self.config_commands['RSI PRO Extend']
            print(f"Extending actuator with command: {command.strip()}")
            value = self.write_read(command)
            print(f"Response: {value.decode().strip() if value else 'No response'}")
            return "success"
        except Exception as e:
            print(f"Error at fully_extend: {e}")
            self.disconnect()
            self.connect()
            return "fail"


    def fully_retract(self):
        """
        Retracts the actuator to its initial position.

        @retval str - Returns "success" if successful; otherwise "fail"
        """
        try:
            command = self.config_commands['RSI PRO Retract']
            print(f"Retracting actuator with command: {command.strip()}")
            value = self.write_read(command)
            print(f"Response: {value.decode().strip() if value else 'No response'}")
            return "success"
        except Exception as e:
            print(f"Error at fully_retract: {e}")
            self.disconnect()
            self.connect()
            return "fail"


    def apply_brake(self):
        """
        Applies the brake to the actuator (stops movement).

        @retval str - Returns "success" if successful; otherwise "fail"
        """
        try:
            command = self.config_commands['RSI PRO Stop']
            print(f"Applying brake with command: {command.strip()}")
            value = self.write_read(command)
            print(f"Response: {value.decode().strip() if value else 'No response'}")
            return "success"
        except Exception as e:
            print(f"Error at apply_brake: {e}")
            self.disconnect()
            self.connect()
            return "fail"


    def connection_check(self):
        """
        Checks the connection to the actuator.

        @retval str - Returns "success" if connection is good; otherwise "fail"
        """
        try:
            command = self.config_commands['RSI PRO Connection Check']
            value = self.write_read(command)
            self._status = 0
            return "success"
        except Exception as e:
            print(f"Error at connection_check: {e}")
            self.disconnect()
            self.connect()
            return "fail"


    def get_distance_to_ground(self):
        """
        Gets the distance to ground from the actuator.

        @retval str - Returns distance value if successful; otherwise "fail"
        """
        try:
            command = self.config_commands['Get Distance to Ground']
            value = self.write_read(command)

            if value:
                response = value.decode().strip()
                print(f"Distance response: {response}")

                # Parse the response to extract distance
                if "Distance to Ground" in response:
                    distance = response.replace("Distance to Ground: ", "")
                    return distance

                # If response doesn't contain expected text, return raw response
                return response

            return "fail"


        except Exception as e:
            print(f"Error getting distance to ground: {e}")
            self.disconnect()
            self.connect()
            return "fail"


    def get_status(self):
        """
        Gets the current connection status.

        @retval int - Status code: 0 = good, 1 = bad, -1 = not connected
        """
        self.connection_check()
        return self._status


    def send_command(self, command):
        """
        Send a custom command to the Arduino.

        @param command: Command string to send
        @retval Response from Arduino
        """
        try:
            return self.write_read(command)
        except Exception as e:
            print(f"Error sending command: {e}")
            return None


    def read_response(self):
        """
        Read a response from the Arduino.

        @retval Response string from Arduino
        """
        if self.connection is None:
            raise Exception("Not connected to Arduino.")

        response = self.connection.readline().decode().strip()
        print(f"Received response: {response}")
        return response
