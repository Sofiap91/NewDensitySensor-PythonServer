"""
*******************************************************************************
@file   module_measurement_manager.py
@author Scott Thomason, Joshua Paterson 
@date   03 May 2022
@brief  Module used to manage the measurement process. This will interface with 
        the VNA manager to get the data and then process it. Also uses a finite 
        state machine to switch between the idle, calibration and measurement 
        states.

REFERENCE:

*******************************************************************************
    Functions
*******************************************************************************

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
import numpy as np
import scipy.io
import cmath
import skrf.io
import skrf as rf
import datetime
from copy import deepcopy
from MudMasterUI.supportFunctions import *
from MudMasterUI.teltonikaSupportFunctions import *
from MudMasterUI import globalErrorVar


import random



""" Defines
*******************************************************************************
"""
state_list = ['idle', 'calibration', 'measurement']

testing = False



""" Classes
*******************************************************************************
"""
class Measurement_Manager(object):
    def __init__(self, app=None):
        """ @brief  Initialisation function for the measurement manager.
            @param  None
            @retval None - Initialisation function.

        """
        self._current_state = 'idle'
        self._current_measurement_data = {
            'measurement_date': None,
            'vna_data': None, 
            'actuator_extension': None,
            'permittivity': None, 
            'water_percentage': None, 
            'density': None,
            'vna_filename': None,
            'shear_vain_20cm': None,
            'shear_vain_50cm': None,
            'shear_vain_80cm': None,
            'Shear_Vain_20': None,
            'Shear_Vain_50': None,
            'Shear_Vain_80': None,
            'Shear Vain A': None,
            'Shear Vain B': None,
            'Shear Vain C': None,
            'Raw Sensor Data': None,
            'DistanceToGround': None,
            }

        self._next_measurement_time = 0

        self._current_calibration_position = -1  # set the current calibration position to -1 to indicate that the calibration is not complete
        self._save_data_directory = {
            'main': None,
            'calData': None, 
            'vnaData': None
            }

        self._measurement_file = None

        self._managementThreadRun = True
        
        if(app != None):
            self.init_app(app)


    def init_app(self, app, mounting_system, vna, dielectric_manager):
        """ @brief  Initialises the VNA manager. 
            @param  app - the flask app object
            @retval None

        """
        self._app = app
        self._mounting_system = mounting_system
        self._vna = vna
        self._dielectric_manager = dielectric_manager
        self._calibration_positions = list(range(self._app.config['CONFIG_SYSTEM']['mountingSystem']['actuator_min'], self._app.config['CONFIG_SYSTEM']['mountingSystem']['actuator_max'] + self._app.config['CONFIG_SYSTEM']['calibrationStep'], self._app.config['CONFIG_SYSTEM']['calibrationStep']))


        self._current_measurement_status = 0  # 0 - waiting, 1 - measuring

        # load the current site from the app.config 
        self._current_measurement_data['site'] = "Default"

        self.start_manager_thread()

    
    def start_manager_thread(self):
        """ @brief  Starts the fitting handler thread if the a reference to the application and database exist.
            @param  None
            @retval None

        """
        if(self._app != None):

            self._processThread = threading.Thread(target = self.managementThread)
            self._processThread.start()


    """ Functions
    *******************************************************************************
    """

    def get_measurement_display_data(self):
        """ @brief  This function will return the current measurement display data.
            @param  None
            @retval None
        """
        data_dict = {}
        data_dict['measurement_date'] = self._current_measurement_data['measurement_date']
        data_dict['actuator_extension'] = self._current_measurement_data['actuator_extension'] or 0
        data_dict['permittivity'] = self._current_measurement_data['permittivity'] or 0
        data_dict['water_percentage'] = self._current_measurement_data['water_percentage'] or 0
        data_dict['density'] = self._current_measurement_data['density'] or 0
        data_dict['status'] = self._current_measurement_status
        data_dict['next_measurement_seconds'] = round(self._next_measurement_time - time.monotonic())  # seconds until the next measurement

        # Add shear vain data
        data_dict['shear_vain_20cm'] = self._current_measurement_data['Shear_Vain_20'] or 0
        data_dict['shear_vain_50cm'] = self._current_measurement_data['Shear_Vain_50'] or 0
        data_dict['shear_vain_80cm'] = self._current_measurement_data['Shear_Vain_80'] or 0

        return data_dict



    """ State Control Functions
    *******************************************************************************
    """
    def set_idle(self):
        """ @brief  This function will set the state to idle.
            @param  None
            @retval True - the state was set to idle. False - the state was not set to idle.

        """
        return self.set_state('idle')

    def start_measurement(self):
        """ @brief  This function will start the measurement process.
            @param  None
            @retval True - the measurement process was started. False - the calibration hasn't been performed.

        """
        # create the measurement file
        self._measurement_file = os.path.join(self._app.config['MACHINE_DIRECTORY'], self._app.config['CONFIG_SYSTEM']['dataLogger']['baseFileName_measurement'].format(datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")))
        print('saving measurement data to: ', self._measurement_file)

        # add the headers to the measurement file
        with open(self._measurement_file, 'w') as f:
            f.write(','.join(self._app.config['CONFIG_SYSTEM']['dataLogger']['headings_measurement']) + '\n')

        return 


    """ State Functions
    *******************************************************************************
    """
    def idle_state(self):
        """ @brief  This state is used to wait for a command to start the 
                    measurement.
            @param  None
            @retval None

        """
        time.sleep(self._app.config['CONFIG_RUN']['measurement_manager']['idle_sleep_time'])

    

    """ Module Thread that joshua paterson added for RSI PRO mount 
    ***********************************************************************
    """
    def measurement_state_MV2(self, testing=False):
        """ @brief  This state is used to take a measurement from the VNA when using the button from the home page.
            @param  None
            @retval None

        """
        try:
            now = time.monotonic()
            
            if(now >= self._next_measurement_time):
                
                print("measurement_state_MV2")
                
                self._current_measurement_data['DistanceToGround'] = self._mounting_system.GetDistanceToGround()
                
                if (self._current_measurement_data['DistanceToGround'] == "fail"):
                    self._current_measurement_data['DistanceToGround'] = -1
                    
                
                print(self._current_measurement_data['DistanceToGround'])
                
                self._next_measurement_time = now + self._app.config['CONFIG_RUN']['measurement_manager']['measurement_delay']#the reason it take 30second before nex measurement 
                # get the measurement from the VNA
                try:
                    self._current_measurement_data['vna_data'] = self._vna.get_nextData(forNN=True)
                except Exception as e:
                    # Handle other exceptions
                    print(f"An error occurred in MV2 at self._vna.get_nextData(forNN=True): {e}")
                    return
                
                # Access the first (and only) dictionary in the list
                data_dict = self._current_measurement_data['vna_data']#work with data_dict in nn formatter
                
               # Load the trained model
                print("before load")
                self._dielectric_manager.load_model()
                # Run the model on live data
                DNN = self._dielectric_manager.run_model_on_live_data_elasticNet(data_dict)
                self._current_measurement_data['Shear_Vain_20'] = DNN[0]
                self._current_measurement_data['Shear_Vain_50'] = DNN[1]
                self._current_measurement_data['Shear_Vain_80'] = DNN[2]
                
                # Try to get GPS coordinates from Teltonika, use defaults if fails
                try:
                    login_endpoint()  # get gps coords from teltonika 
                    latLong = get_GPS_data_endpoint()
                    if latLong and 'latitude' in latLong and 'longitude' in latLong:
                        self._current_measurement_data['latitude'] = latLong['latitude']
                        self._current_measurement_data['longitude'] = latLong['longitude']
                    else:
                        print("GPS data structure invalid, using default coordinates (0,0)")
                        self._current_measurement_data['latitude'] = 0
                        self._current_measurement_data['longitude'] = 0
                except Exception as gps_error:
                    print(f"GPS retrieval failed: {gps_error}. Using default coordinates (0,0)")
                    self._current_measurement_data['latitude'] = 0
                    self._current_measurement_data['longitude'] = 0

                    # set the current datetime for the measurement
                self._current_measurement_data['measurement_date'] = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")  # current universal coordinated time

                ### convert raw data to the same structure as it is save in s1p files to save in same format in csv 
                
                # Extract frequency, S11R, and S11I from the datadict
                frequency = self._current_measurement_data['vna_data']['frequency']
                S11R = self._current_measurement_data['vna_data']['S11R']
                S11I = self._current_measurement_data['vna_data']['S11I']

                # Prepare the output data in the required format
                Raw_Sensor_data = "!Created with skrf (http://scikit-rf.org)\n"
                Raw_Sensor_data += "# Hz S RI R 50.0\n"
                Raw_Sensor_data += "!freq ReS11 ImS11\n"
                # Format the sensor data
                for i in range(len(frequency)):
                    # Format the frequency to 1 decimal place, and append the corresponding S11R and S11I values
                    Raw_Sensor_data += f"{frequency[i]:.1f} {S11R[i]} {S11I[i]}\n"
                # Replace the newline characters with spaces to ensure it's treated as a single cell in CSV
                # Enclose the entire block in quotes so that the newlines are preserved as part of a single cell
                Raw_Sensor_data = f'"{Raw_Sensor_data}"'
                # Save the data as a CSV cell
                self._current_measurement_data['Raw Sensor Data'] = Raw_Sensor_data                           
                ###
                # save the measurement to csv 
                # now save the remaining data as a csv file in the main directory
                #measurement_data_filename = self._app.config['CONFIG_SYSTEM']['dataLogger']['baseFileName_measurement'].format(self._current_measurement_data['measurement_date'].replace(':', '-') + '_' + str(self._current_measurement_data['actuator_extension']))
                save_measurement_data(self._measurement_file, self._current_measurement_data, self._app.config['CONFIG_SYSTEM']['dataLogger']['headings_measurement'])
                
                print('---------Measurements taken---------')
                
                
            time.sleep(1)
        except Exception as e:
            # Handle other exceptions
            print(f"An error occurred in MV2: {e}")
            
    def measurement_state_MV2_temp(self, testing=False):
        
        self._current_measurement_data['measurement_date'] = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")  # current universal coordinated time
        self._current_measurement_data['latitude'] = 0
        self._current_measurement_data['longitude'] = 0
        self._current_measurement_data['Shear_Vain_20'] = 0
        self._current_measurement_data['Shear_Vain_50'] = 0
        self._current_measurement_data['Shear_Vain_80'] = 0
        
        self._current_measurement_data['Raw Sensor Data'] = """
        1 2 3 
        3 4 5 
        7 8 9 
        """
        
        self._current_measurement_data['Raw Sensor Data'] = f'"{self._current_measurement_data["Raw Sensor Data"]}"'
        
        
        save_measurement_data(self._measurement_file, self._current_measurement_data, self._app.config['CONFIG_SYSTEM']['dataLogger']['headings_measurement'])
        print('---------Measurements taken---------')    
            
            
    def measurement_state_MV3(self, testing=False):
        """ @brief  This state is used to take a measurement from the VNA when using the button from the nn collection page.
            @param  None
            @retval None

        """
        try:
            #now = time.monotonic()
            print("measurement_state_MV3")    
            self._current_measurement_data['vna_data'] = self._vna.get_nextData(forNN=True)
            # set the current datetime for the measurement
            self._current_measurement_data['measurement_date'] = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")  # current universal coordinated time

            # save the measurement 
            # vna data needs to be saved as a touchstone file
            # everything else should be saved as a csv file
            vna_filename = self._app.config['CONFIG_SYSTEM']['dataLogger']['baseFileName_vna'].format(self._current_measurement_data['measurement_date'].replace(':', '-') + '_' + str(self._current_measurement_data['actuator_extension']))
            save_s11_data(vna_filename, globalErrorVar.NN_Data_Collection_File_path, self._current_measurement_data['vna_data'])
            self.start_measurement()
            print('---------Measurements taken MV3---------')
            time.sleep(1)
        except Exception as e:
            # Handle other exceptions
            print(f"An error occurred: {e}")



    """ Module Thread
    ***********************************************************************
    """

    def managementThread(self):
        """ @brief  This thread runs the mounting system. It will send 
                    commands to set the height of the module and receive any 
                    data that has been sent back. 
            @param  None
            @retval None

        """
        # initial setup

        print('measurement system  thread start')

        # while loop
        while(self._managementThreadRun):
            # check the current state
            if(self._current_state == 'idle'):
                self.idle_state()
            elif(self._current_state == 'measurement_MV2'):#added by me //will prevent below if else statment from running and will also cause the measuremnt to dely on  self.measurement_state(testing=testing) as well  
                self.measurement_state_MV2()
                #self.measurement_state_MV2_temp()
                print("manager loop")
                self._current_state = 'idle'
                # don't need the sleep, this is handled by the function
            else:
                print('measurement system  thread error')
                time.sleep(0.1)