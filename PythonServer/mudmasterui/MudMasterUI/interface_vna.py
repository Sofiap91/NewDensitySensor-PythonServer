"""
*******************************************************************************
@file   interface_VNA.py
@author Scott Thomason, Joshus Paterson 
@date   30 Sep 2021
@brief  VNA interfacing class that uses the Python requests module to connect to 
        the VNA server and request data from it. This is run in a thread that 
        will periodically request data from the VNA. This is made available 
        through class functions.

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
import numpy
import cmath
import numpy as np
from copy import deepcopy
import time
import psutil
from MudMasterUI import globalErrorVar

import ctypes
import subprocess

""" Defines
*******************************************************************************
"""
update_config = False


""" Classes
*******************************************************************************
"""

class VNA_Manager(object):
    def __init__(self, app=None):
        """ @brief  Initialisation function for the VNA manager.
            @param  config - dictionary containing configuration parameters for 
                        the VNA manager. If any parameter is not in the 
                        dictionary, the default will be used.
            @retval None - Initialisation function.

        """

        self._measurementData = {}
        self._currentData = {'port1': {}, 'port2': {}, 'measurementCount': -1, 'sweepTime': 0} # each will have a dictionary with frequency as the key, sweeptime is miliseconds
        self._status = 0 # 0 = good.
        self._vnaStatus = {}  # stores the VNA status dictionary
        self._vnaStatusTime = 0 # stores the time the VNA status was last updated
        self._vnaInfo = {} # stores the VNA info dictionary
        self._mode = 0 # 0 = configuration, 1 = continuous operation
        self._lastRequestedData = 0
        self._measuringFlag = False
        self._vnaProgram = None

        self._vnaServerActive = False

        self._managementThreadRun = True
        # start the management thread

        self._lock = threading.Lock() # lock used to protect data when it is being changed.

        if(app != None):
            self.init_app(app)


    def init_app(self, app):
        """ @brief  Initialises the VNA manager. 
            @param  app - the flask app object
            @retval None

        """
        self._app = app

        self._vnaIP = self._app.config['CONFIG_SYSTEM'].get('vnaIP', '127.0.0.1')
        self._vnaPort = self._app.config['CONFIG_SYSTEM'].get('vnaPort', 5000)
        self._updateInterval = self._app.config['CONFIG_SYSTEM'].get('vnaUpdateInterval', 0.5)
        self._statusInterval = self._app.config['CONFIG_SYSTEM'].get('vnaStatusInterval', 5)

        self._vnaRoutes = self._app.config['CONFIG_SYSTEM'].get('vnaRoutes', {})

        # current configuration for the VNA
        self._config_current = {
            'frequency': {
                'min': None, 
                'max': None, 
                'step': None
                },
            'power': None,
            }

        # set the new configuration data that from the config file, or all None
        self._config_new = self._app.config['CONFIG_SYSTEM'].get('vnaSettings', {'frequency': {'min': None, 'max': None, 'step': None}, 'power': None})
        
        self._vnaServerActive = self._app.config['CONFIG_SYSTEM'].get('vnaServerProcessName', 'MegiqVnaServer.exe') in (p.name() for p in psutil.process_iter())
        # if the server isn't active, then start the executable
        if(self._app.config['CONFIG_MACHINE'].get('vnaServerProcessPath', None) == None):
            print('WARNING: config file does not contain path to VNA server executable')

        if(self._vnaServerActive == False and self._app.config['CONFIG_MACHINE'].get('vnaServerProcessPath', None) != None):
            # check if there are currently any instances of the MiQVNA software running
            for p in psutil.process_iter():
                if p.name() == self._app.config['CONFIG_SYSTEM'].get('vnaProgramName', 'MiQVNA.exe'):
                    p.terminate()
            
            # Run the server executable as administrator
            self._run_as_admin(self._app.config['CONFIG_MACHINE'].get('vnaServerProcessPath'))
            #os.startfile(self._app.config['CONFIG_MACHINE'].get('vnaServerProcessPath')) # run as not admin

            # now wait until the server and vna program have both started
            foundProgram = False
            self._vnaProgram = None
            while foundProgram == False:
                for p in psutil.process_iter():
                    if p.name() == self._app.config['CONFIG_SYSTEM'].get('vnaProgramName', 'MiQVNA.exe'):
                        foundProgram = True
                        self._vnaProgram = p
                        break

            print(self._vnaProgram)
            print('PID: ', self._vnaProgram.pid)
            print('NAME: ', self._vnaProgram.name())
            print('CMDLINE: ', self._vnaProgram.cmdline())
            print('PPID: ', self._vnaProgram.ppid())
            print('PGROUP: ', self._vnaProgram.parent())



        self.start_manager_thread()

    
    def start_manager_thread(self):
        """ @brief  Starts the fitting handler thread if the a reference to the application and database exist.
            @param  None
            @retval None

        """
        if(self._app != None):

            self._processThread = threading.Thread(target = self.deviceManagementThread)
            self._processThread.start()


    """ VNA Communication Functions
    ***********************************************************************
    """
    def vna_webRequest(self, routeKey, data=None):
        """ @brief  HTTP request - GET or POST. 
            @param  data - request data, if needed
            @retval Dictionary of the JSON response

        """
        if(self._vnaServerActive == False):
            # check if it's running now
            self._vnaServerActive = self._app.config['CONFIG_SYSTEM'].get('vnaServerProcessName', 'MegiqVnaServer.exe') in (p.name() for p in psutil.process_iter())

        # if the server is active, then send the request
        if(self._vnaServerActive == True):
            urlData = 'http://{}:{}/{}'.format(self._vnaIP, self._vnaPort, self._vnaRoutes[routeKey]['endpoint'])

            try:
                if(self._vnaRoutes[routeKey]['method'] == 'GET'):
                    msg = requests.get(url = urlData, data = data)

                else:
                    msg = requests.post(url = urlData, data = data)

                if(msg.status_code == 200):
                    self._vnaServerActive = True
                    return msg.json()

            except Exception as e:
                print(e)
                print("-print to check if this is where the refused connection to vna server is from-")
                self._vnaServerActive = False
           
        return {}


    """ Data Access Functions
    ***********************************************************************
    """
    def get_currentData(self, asJson=False, magPhase=False, forNN=False):
        """ @brief  Returns the most recent data. 
            @param  asJson - whether to return the data as a json object
                    magPhase - whether to return the data as a dictionary of magnitude and phase - only when using asJson as well
            @retval current data

        """
        # acquire the lock before creating a copy of the data. A copy is 
        # used to ensure the requesting code doesn't lose access to the 
        # data when a new set of data is processed.
        self._lock.acquire()

        dataCopy = deepcopy(self._currentData)

        self._lastRequestedData = self._currentData['measurementCount']

        self._lock.release()

        if asJson:
            # iterate through the port1 data and convert the complex numbers to dictionaries of real and immaginary parts
            for key in dataCopy['port1']:
                if magPhase == False:
                    dataCopy['port1'][key] = {'real': dataCopy['port1'][key].real, 'imag': dataCopy['port1'][key].imag}

                else:
                    dataCopy['port1'][key] = {'mag': 10*np.log10(abs(dataCopy['port1'][key])), 'phase': cmath.phase(dataCopy['port1'][key])}

        if forNN:
            nnData = {}
            # formats the data for use in the neural network as {'port1': {'frequency': np.array[freq1, freq2, ..., freqN], 'S11R': np.array[S11R1, S11R2, ..., S11RN], 'S11I': np.array[S11I1, S11I2, ..., S11IN]}}
            nnData['frequency'] = []
            nnData['S11R'] = []
            nnData['S11I'] = []

            # now iterate through the dataCopy and add the data to the arrays
            for key in dataCopy['port1']:
                nnData['frequency'].append(key)
                nnData['S11R'].append(dataCopy['port1'][key].real)
                nnData['S11I'].append(dataCopy['port1'][key].imag)

            # convert the arrays to numpy arrays
            nnData['frequency'] = np.array(nnData['frequency'])
            nnData['S11R'] = np.array(nnData['S11R'])
            nnData['S11I'] = np.array(nnData['S11I'])

            return nnData

        return dataCopy


    def get_nextData(self, asJson=False, magPhase=False, forNN=False):
        """ @brief  Waits for and returns the next set of data. 
            @param  asJson - whether to return the data as a json object
                    magPhase - whether to return the data as a dictionary of magnitude and phase - only when using asJson as well
                    forNN - whether to return the data as a dictionary of frequency, S11R, and S11I - used for the Neural Network
            @retval current data

        """
        # acquire the lock before creating a copy of the data. A copy is 
        # used to ensure the requesting code doesn't lose access to the 
        # data when a new set of data is processed.
        startingMeasurement = self._currentData['measurementCount']
            
        start_time = time.time()
        timeout = 20  # 10 seconds timeout

        while (startingMeasurement == self._currentData['measurementCount'] and
           time.time() - start_time < timeout):
            time.sleep(0.1)

        if time.time() - start_time >= timeout:
            print("get nextData timed out")
            globalErrorVar.ErrorFromMeasurementManager = True
            raise TimeoutError("Timeout occurred: 10 seconds elapsed while waiting for new data.")
            #raise Exception("Timeout occurred: 10 seconds elapsed while waiting for new data.")
            
        # now there is new data, copy it and give it

        self._lock.acquire()

        dataCopy = deepcopy(self._currentData)

        self._lastRequestedData = self._currentData['measurementCount']

        self._lock.release()
        
        #print("here 3")

        if asJson:
            # iterate through the port1 data and convert the complex numbers to dictionaries of real and immaginary parts
            for key in dataCopy['port1']:
                if magPhase == False:
                    dataCopy['port1'][key] = {'real': dataCopy['port1'][key].real, 'imag': dataCopy['port1'][key].imag}

                else:
                    dataCopy['port1'][key] = {'mag': 10*np.log10(abs(dataCopy['port1'][key])), 'phase': np.angle(dataCopy['port1'][key])}

        if forNN:
            nnData = {}
            # formats the data for use in the neural network as {'port1': {'frequency': np.array[freq1, freq2, ..., freqN], 'S11R': np.array[S11R1, S11R2, ..., S11RN], 'S11I': np.array[S11I1, S11I2, ..., S11IN]}}
            nnData['frequency'] = []
            nnData['S11R'] = []
            nnData['S11I'] = []

            # now iterate through the dataCopy and add the data to the arrays
            for key in dataCopy['port1']:
                nnData['frequency'].append(key)
                nnData['S11R'].append(dataCopy['port1'][key].real)
                nnData['S11I'].append(dataCopy['port1'][key].imag)

            # convert the arrays to numpy arrays
            nnData['frequency'] = np.array(nnData['frequency'])
            nnData['S11R'] = np.array(nnData['S11R'])
            nnData['S11I'] = np.array(nnData['S11I'])

            return nnData

        return dataCopy


    def get_nextData_timedomain(self, asJson=False, asMag=False):
        """ @brief  Waits for and returns the next set of data. The data is converted from frequency domain to time domain.
            @param  asJson - whether to return the data as a json object
            @retval current data

        """
        print('asMag: {}'.format(asMag))
        # acquire the lock before creating a copy of the data. A copy is 
        # used to ensure the requesting code doesn't lose access to the 
        # data when a new set of data is processed.
        startingMeasurement = self._currentData['measurementCount']
        
        while(startingMeasurement == self._currentData['measurementCount']):
            time.sleep(1)

        # now there is new data, copy it and give it

        self._lock.acquire()

        dataCopy = deepcopy(self._currentData)

        self._lastRequestedData = self._currentData['measurementCount']

        self._lock.release()

        data_list = []

        # iterate through the port1 data and convert the complex frequency domain data to time domain data
        for key in dataCopy['port1']:
            # add the data to the data_list
            data_list.append(dataCopy['port1'][key])

        # convert the data_list to a numpy array
        data_array = np.array(data_list)
        # convert the frequency domain data in data_array to time domain and convert to db
        data_array = 10*np.log10(abs(np.fft.fft(data_array)))

        return data_array


    def get_statusData(self):
        """ @brief  Returns whether there is any new data. 
            @param  None
            @retval True - yes, False - no

        """
        status = False

        self._lock.acquire()

        status = self._measurementData['measurementCount'] > self._lastRequestedData

        self._lock.release()

        return status


    def has_data(self):
        """ @brief  Returns whether there is any data. 
            @param  None
            @retval True - yes, False - no

        """
        if(len(self._currentData['port1']) > 0):
            return True

        return False


    def get_statusVNA(self):
        """ @brief  Returns the current VNA status. 
            @param  None
            @retval dict

        """

        return self._vnaStatus.get('miqVNAStatus', {})


    def get_VNAInfo(self):
        """ @brief  Returns the VNA device information. 
            @param  None
            @retval dict

        """

        return self._vnaInfo


    def get_currentlyMeasuring(self):
        """ @brief  Returns whether the VNA is currently performing a measurement. 
            @param  None
            @retval True - yes, False - no

        """

        return self._measuringFlag


    def set_killVNA(self):
        """ @brief  Instructs the server to kill its VNA instance. 
            @param  None
            @retval None

        """

        self.vna_webRequest('set_kill')
        time.sleep(5)
        self._managementThreadRun = False

    
    """ Module Thread
    ***********************************************************************
    """

    def deviceManagementThread(self):
        """ @brief  This thread runs the VNA connection. It will handle 
                    the requesting and processing of data. This data will 
                    then be made available for external methods. 
            @param  None
            @retval None

        """
        # initial setup
        print('VNA control thread start')
        # first request the status from the VNA server to check whether it is ready to go
        self._vnaStatus = self.vna_webRequest('get_status')
        self._vnaStatusTime = time.monotonic()
        # check whether the VNA is ready to go
        while self._vnaServerActive == False:
            self._vnaStatus = self.vna_webRequest('get_status')
            self._vnaStatusTime = time.monotonic()

            if self._vnaServerActive == False:
                # if it's still inactive, wait and then try again
                time.sleep(self._app.config['CONFIG_SYSTEM'].get('vnaServerCheckInterval', 5))

        #TODO: only get the VNA info after the server has been verified to be online
        self._vnaInfo = self.vna_webRequest('get_vna_info')
        print(self._vnaStatus)

        # while loop
        while(self._managementThreadRun):
            now = time.monotonic()
            if(now > self._vnaStatusTime + self._statusInterval):
                self._vnaStatus = self.vna_webRequest('get_status')
                self._vnaStatusTime = now

            # check whether the VNA is in a state which is ready to interact:
            if(len(self._vnaStatus) > 0 and self._vnaStatus['miqVNAStatus']['status'] > 1):
                # Check for configuration changes.
                if(update_config == True and self._config_new != self._config_current):
                    # a change to the config
                    if(self._config_new['frequency'] != self._config_current['frequency']):
                        # frequency has changed. Check that there is enough parameters to configure the device
                        frequency_min = self._config_new['frequency']['min'] or self._config_current['frequency']['min']
                        frequency_max = self._config_new['frequency']['max'] or self._config_current['frequency']['max']
                        frequency_step = self._config_new['frequency']['step'] or self._config_current['frequency']['step']

                        if(frequency_min is not None and frequency_max is not None and frequency_step is not None):
                            # there is enough parameters to set the frequency
                            requestData = {'frequencyMin': frequency_min,
                                            'frequencyMax': frequency_max,
                                            'stepSize': frequency_step}

                            resp = self.vna_webRequest('set_frequency', requestData)

                            # update the current one
                            self._config_current['frequency']['min'] = frequency_min
                            self._config_current['frequency']['max'] = frequency_max
                            self._config_current['frequency']['step'] = frequency_step
                            
                    if(self._config_new['power'] != self._config_current['power']):
                        # frequency has changed. Check that there is enough parameters to configure the device
                        power = self._config_new['power'] or self._config_current['power']

                        if(power is not None):
                            # there is enough parameters to set the frequency
                            requestData = {'power': power}

                            resp = self.vna_webRequest('set_frequency', requestData)

                            # update current one
                            self._config_current['power'] = power


                # Now continue with normal measurement operation.
                # First, request the data from the server
                # Second, extract the data that is needed and store in the measurement 
                # data variable. If the new data is different to the previous, then set 
                # a flag to indicate that it is new

                self._measuringFlag = True
                start = time.monotonic()
                newData = self.vna_webRequest('get_new_measurement')
                end = time.monotonic()
                self._measuringFlag = False
                # the data has the following structure:
                # {'status': int, 'sweepTime': double, 'measurementCount': int, port1: list, port2: list}

                if('status' in newData):
                    if(newData['status'] == 0):
                        # data was good.
                        if(newData['measurementCount'] != self._measurementData.get('measurementCount', None)):
                            # the new data is different, so update the currently stored stuff
                            # acquire the lock
                            self._lock.acquire()

                            self._measurementData = newData

                            # reset the current data
                            self._currentData['port1'] = {}
                            self._currentData['port2'] = {}

                            # extract out the measurement data and prepare it for usage by other methods.
                            for measurement in newData['port1']:
                                # convert the amplitude back to linear
                                self._currentData['port1'][measurement['frequency']] = cmath.rect(10 ** (measurement['amplitude'] / 20), measurement['phase'])
                        
                            # for measurement in newData['port2']:
                            #     # convert the amplitude back to linear
                            #     self._currentData['port2'][measurement['frequency']] = cmath.rect(10 ** (measurement['amplitude'] / 20), measurement['phase'])
                        
                            # print('Measurement count: ' + str(newData['measurementCount']) + ' time: ' + str(end - start))
                            self._currentData['measurementCount'] = newData['measurementCount']
                            self._currentData['sweepTime'] = newData['sweepTime']
                            # release the lock so the data can be accessed
                            self._lock.release()

                time.sleep(self._updateInterval)

            else:
                # check the status again
                self._vnaStatus = self.vna_webRequest('get_status')
                time.sleep(5)
     
     
     
    def _run_as_admin(self, exe_path):
        """ @brief  Runs the executable with administrative privileges.
            @param  exe_path - Path to the executable
            @retval None

        """
        try:
            if not ctypes.windll.shell32.IsUserAnAdmin():
                ctypes.windll.shell32.ShellExecuteW(None, "runas", exe_path, None, None, 1)
            else:

                subprocess.run([exe_path], check=True)
        except Exception as e:
            print(f"Failed to start {exe_path} as admin: {e}")           
                




if __name__ == '__main__': 
    # library test code
    from config import config_system
    vna = VNA_Manager(config_system)

