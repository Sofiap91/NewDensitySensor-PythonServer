"""
*******************************************************************************
@file   interface_MountingSystem.py
@author Scott Thomason, Joshua Paterson
@date   30 Sep 2021
@brief  Interface for the mounting system. This controls the actuator that 
        adjusts the height. It keeps track of the current system height and 
        adjusts it to ensure it is at the desired height.

REFERENCE:

*******************************************************************************
    Functions
*******************************************************************************
    Currently, there are no additional functions defined beyond those already
    documented.
*******************************************************************************
"""

""" Imports 
*******************************************************************************
"""

import os
import requests
import time
import threading
import json
import serial
import serial.tools.list_ports
from MudMasterUI import globalErrorVar

""" Defines
*******************************************************************************
"""
deviceDescriptions = ['Arduino']

""" Functions
*******************************************************************************
"""

def findSerialPorts():
    """
    Finds all serial devices connected to the system and makes a list of the ports.

    @param  None
    @retval List of serial ports - A list of available serial ports that match the 
            descriptions in `deviceDescriptions`.

    """
    portList = []
    comports = list(serial.tools.list_ports.comports())
    for p in comports:
        if any(desc in p.description for desc in deviceDescriptions):
            print(p, '<description:>', p.description, '<manufacturer:>', p.manufacturer, '<product:>', p.product, '<description length:>', len(p.description))
            portList.append(p)

    return portList

""" Classes
*******************************************************************************
"""

class MountingSystem_Manager(object):
    def __init__(self, app=None):
        """
        Initialization function for the mounting system manager.

        @param app: Reference to the application (optional). If provided, initializes 
                    the application.
        @retval None - Initialization function.

        """
        self._app = None
        self._serialPort = None
        self._status = -1  # 0 = good, 1 = bad, -1 = not connected
        self._calibrated = False
        self._currentHeight = None
        self._lastMessageTime = None
        self._config_commands = None
        self._current_actuator = 0

        self._heightRange = {'min': None, 'max': None}

        self._commandQueue = []

        self._managementThreadRun = False
        
        self._lock = threading.Lock()  # Lock used to protect data when it is being changed.
        # Start the management thread
        if app is not None:
            self.init_app(app)

    def init_app(self, app):
        """
        Initializes the application with configuration settings.

        @param app: Reference to the application to initialize.
        @retval None - Initializes the application and loads configuration settings.
        
        This function sets the application and configuration commands, and 
        opens the serial port if it is not already open.
        """
        self._app = app
        self._config_commands = self._app.config['CONFIG_SYSTEM'].get('mountingSystemCommands', {})
        self._config_limits = self._app.config['CONFIG_SYSTEM'].get('mountingSystem', {'actuator_max': 210, 'actuator_min': 0})  # defaults to 130 if there is no config
        
        if self._serialPort is None:
            self.openSerial()

    """ Serial Communications
    ***********************************************************************
    """
    def openSerial(self):
        """
        Attempts to find and open the serial port.

        @param  None
        @retval None - Opens the serial port if available and sets the port status.
        
        This function searches for available serial ports, connects to the first 
        matching port, and updates the connection status.
        """
        try:
            if self._serialPort is None:
                availablePorts = findSerialPorts()
                if len(availablePorts) >= 1:
                    # Found the Arduino
                    print('Found Arduino')
                    print(availablePorts)
                    self._serialPort = serial.Serial(port=availablePorts[0].device, 
                                                    baudrate=9600, 
                                                    timeout=1.0, 
                                                    bytesize=serial.EIGHTBITS)

                    self._serialNo = availablePorts[0].serial_number

                    print('Serial No.:', self._serialNo)
                    print('Connected')
                    self._status = 0

        except Exception as e:
            print('Communication Error: permission. {}'.format(e))
            self._status = -1
            
    def closeSerial(self):
        """
        Closes the serial port if it is open.

        @param  None
        @retval None - Closes the serial port and updates the port status.
        
        This function safely closes the serial port if it is open and updates
        the connection status.
        """
        try:
            if self._serialPort is not None:
                if self._serialPort.is_open:
                    self._serialPort.close()
                    print('Serial port closed.')
                self._serialPort = None
                self._serialNo = None
                self._status = -1
        except Exception as e:
            print('Error closing serial port: {}'.format(e))
            self._status = 1
    
    """ General Control Functions
    ***********************************************************************
    """

    """ General Query Functions
    ***********************************************************************
    """

    def get_actuator_position(self):
        """
        Gets the current actuator position.

        @param  None
        @retval Current actuator position - Returns the actuator position (hardcoded to 210 mm).

        This function provides the current position of the actuator. For testing purposes, 
        it returns a hardcoded value of 210 mm.
        """
        try:
            return 210
        except Exception as e:
            print(e)
            return 210

    def get_status(self):
        """
        Gets the current status of the mounting system.

        @param  None
        @retval Current status - Returns the current status of the mounting system.
        
        This function checks the connection and returns the status of the mounting system.
        """
        self.ConnectionCheck()
        return self._status
    
    """ Functions added by Joshua Paterson for new actuator 
    ***********************************************************************
    """
    def fullyExtend(self):
        """
        Extends the actuator to its full extent.

        @param  None
        @retval str - Returns "success" if the extension is successful; otherwise, returns "fail".
        
        This function sends a command to fully extend the actuator and handles any 
        exceptions that occur during the operation.
        """
        try:
            def write_read(x):
                self._serialPort.write(bytes(x, 'utf-8'))
                time.sleep(0.05)
                data = self._serialPort.readline()
                return data

            num = self._config_commands['RSI PRO Extend']
            print(num)
            value = write_read(num)
            globalErrorVar.ErrorFromActuatorReadWrite = False
            return "success"
        except Exception as e:
            print(f"An error occurred: {e}")
            globalErrorVar.ErrorFromActuatorReadWrite = True
            return "fail"
            self.closeSerial()
            self.openSerial()

    def fullyRetact(self):
        """
        Retracts the actuator to its initial position.

        @param  None
        @retval str - Returns "success" if the retraction is successful; otherwise, returns "fail".
        
        This function sends a command to fully retract the actuator and handles any 
        exceptions that occur during the operation.
        """
        try:
            def write_read(x):
                self._serialPort.write(bytes(x, 'utf-8'))
                time.sleep(0.05)
                data = self._serialPort.readline()
                return data

            num = self._config_commands['RSI PRO Retract']
            value = write_read(num)
            globalErrorVar.ErrorFromActuatorReadWrite = False
            return "success"
        except Exception as e:
            print(f"An error occurred at fullyRetact: {e}")
            globalErrorVar.ErrorFromActuatorReadWrite = True
            self.closeSerial()
            self.openSerial()
            return "fail"

    def ApplyBrake(self):
        """
        Applies the brake to the actuator.

        @param  None
        @retval str - Returns "success" if the brake is applied successfully; otherwise, returns "fail".
        
        This function sends a command to apply the brake to the actuator and handles 
        any exceptions that occur during the operation.
        """
        try:
            def write_read(x):
                self._serialPort.write(bytes(x, 'utf-8'))
                time.sleep(0.05)
                data = self._serialPort.readline()
                return data

            num = self._config_commands['RSI PRO Stop']
            value = write_read(num)
            globalErrorVar.ErrorFromActuatorReadWrite = False
            return "success"
        except Exception as e:
            print(f"An error occurred: {e}")
            globalErrorVar.ErrorFromActuatorReadWrite = True
            self.closeSerial()
            self.openSerial()
            return "fail"
        
    def ConnectionCheck(self):
        """
        Checks the connection to the actuator.

        @param  None
        @retval str - Returns "success" if the connection check is successful; otherwise, returns "fail".
        
        This function sends a command to check the connection to the actuator and 
        handles any exceptions that occur during the operation.
        """
        try:
            def write_read(x):
                self._serialPort.write(bytes(x, 'utf-8'))
                time.sleep(0.05)
                data = self._serialPort.readline()
                return data

            num = self._config_commands['RSI PRO Connection Check']
            value = write_read(num)
            globalErrorVar.ErrorFromActuatorReadWrite = False
            self._status = 0
            return "success"
        except Exception as e:
            print(f"An error occurred: {e}")
            globalErrorVar.ErrorFromActuatorReadWrite = True
            self.closeSerial()
            self.openSerial()
            return "fail"
        
    def GetDistanceToGround(self):
        """
        still testing function incomplete 
        
        
        Sends command "5" to the actuator and handles the response to get distance to the ground.

        @param  None
        @retval str - Returns "success" if the command is sent successfully; 
                  otherwise, returns "fail".
        """
        try:
            def write_read(x):
                """
                Sends a command to the actuator and reads the response.

                @param x: Command to be sent to the actuator.
                @retval data: Response from the actuator.
                """
                self._serialPort.write(bytes(x, 'utf-8'))
                time.sleep(0.05)
                data = self._serialPort.readline()
                return data

            # Command "5" should be mapped in your config commands
            num = self._config_commands['Get Distance to Ground']
            #print(f"Sending command: {num}")
            
            # Send the command and get the response
            value = write_read(num)
            #print(f"Response from actuator: {value.decode().strip()}")

            # If the response contains distance information, we can process it
            if "Distance to Ground" in value.decode():
                #print("Distance to ground successfully received from Arduino.")
                response = value.decode().strip()
                print (response)
                # Remove the "Distance to Ground" text from the response
                response = response.replace("Distance to Ground: ", "")
                return response
            
            # If the response is unexpected or empty
            print("Unexpected response:", value.decode())
            #globalErrorVar.ErrorFromActuatorReadWrite = True
            return "fail"
        
        except Exception as e:
            print(f"An error occurred while sending command 5: {e}")
            #globalErrorVar.ErrorFromActuatorReadWrite = True
            self.closeSerial()
            self.openSerial()
            return "fail"