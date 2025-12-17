"""
*******************************************************************************
@file   globalErrorVar.py
@author Joshua Paterson
@date   30 Sep 2024
@brief  Stores variables for error logging and status indicators for the UI.
REFERENCE:
    This module provides global variables used for error logging and status 
    indicators related to the Mounting System interface. These variables are 
    designed to help in displaying real-time status and errors on the user 
    interface (UI).

*******************************************************************************
    Functions
*******************************************************************************
    Currently, there are no functions defined in this module. The module 
    primarily contains global variables for error tracking and status updates.
*******************************************************************************
"""

# Error indicators
ErrorFromMeasurementManager = False  # Indicates a timeout or error when reading from the VNA (Vector Network Analyzer)
ErrorFromActuatorReadWrite = False  # Indicates errors related to read/write operations when sending commands to the actuator
ErrorFromTeltonika = False  # Indicates errors related to network issues with Teltonika (e.g., device not found or receiving non-200 HTTP responses)

# Status indicators for UI
CurrentlyExtending = False  # Indicates whether the actuator is currently extending; used for UI updates
CurrentlyRetracting = False  # Indicates whether the actuator is currently retracting; used for UI updates
CurrentlyCalibrating = False  # Indicates whether the actuator is currently calibrating; used for UI updates
CurrentlyLogging = False  # Indicates whether the actuator is currently logging data; used for UI updates

# TODO: Add additional error indicators and status flags as needed
# For example, an error for VNA connection refusal might be added
