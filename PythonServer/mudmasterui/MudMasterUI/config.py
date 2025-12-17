import os

basedir = os.path.abspath(os.path.dirname(__file__))


class Config(object):

    Usersfolder = "MMBOT"
    Userscloudfolder = "MMBOT"
    SECRET_KEY = '7f2GCO8mOE6EaSgNt2MSrkBIqdya3j-BW7MInjh75B1H5GRJtnT6dkDXaN39KalMjE1f2nhxT6w2IH-5Hm0adwlXY5eF6rx4A8wy'
    VERSION = '2.0.0'
    MACHINE_DIRECTORY = fr'C:\Users\{Usersfolder}\Documents\RTDS Local Storage'
    TRANSFER_DIRECTORY = fr'C:\Users\{Usersfolder}\Phibion Pty Ltd\Realtime Density Sensor - {Userscloudfolder}'
    AUTO_SHUT_DOWN_TIME = '19:00'
    ELASTICNET_MODEL_LOCATION_TWENTYCM = fr'C:\Users\{Usersfolder}\Phibion Pty Ltd\Realtime Density Sensor - {Userscloudfolder}\Megiq VNA Server\Python Server\mudmasterui\MudMasterUI\nnData\20_EN_MT_WELD.pkl'
    ELASTICNET_MODEL_LOCATION_FIFITYCM = fr'C:\Users\{Usersfolder}\Phibion Pty Ltd\Realtime Density Sensor - {Userscloudfolder}\Megiq VNA Server\Python Server\mudmasterui\MudMasterUI\nnData\50_EN_MT_WELD.pkl'
    ELASTICNET_MODEL_LOCATION_EIGHTYCM = fr'C:\Users\{Usersfolder}\Phibion Pty Ltd\Realtime Density Sensor - {Userscloudfolder}\Megiq VNA Server\Python Server\mudmasterui\MudMasterUI\nnData\80_EN_MT_WELD.pkl'
    #
    
    TRANSFER_DELAY = 10
    CONFIG_DIRECTORY = os.path.join(basedir, 'configFiles')

    # stores the runtime information for the specific machine
    CONFIG_FILE_RUN = os.path.join(CONFIG_DIRECTORY, 'config_run.json')

    # config variable that is used by the machine and updated by the user interface, this is loaded from the machine when it is started
    CONFIG_RUN = {
        'site': 'default',
        'measurement_manager': {  # time.sleep values used by the measurement manager
            'idle_sleep_time': 1,  # seconds
            'measurement_delay': 60,  # seconds TODO: set to 60 seconds?
        },
    }

    CONFIG_SYSTEM = {
        'vnaIP': '127.0.0.1',
        'vnaPort': 5000,
        'vnaRoutes': {
            'set_frequency': {
                'endpoint': 'VNA/settings/frequency',
                'method': 'POST'
            },
            'set_power': {
                'endpoint': 'VNA/settings/power',
                'method': 'POST'
            },
            'set_connection': {
                'endpoint': 'VNA/set-connection',
                'method': 'POST'
            },
            'set_sweep': {
                'endpoint': 'VNA/set-sweep',
                'method': 'POST'
            },
            'set_kill': {
                'endpoint': 'VNA/kill',
                'method': 'POST'
            },
            'get_status': {
                'endpoint': 'VNA/status',
                'method': 'GET'
            },
            'get_vna_info': {
                'endpoint': 'VNA/vna-info',
                'method': 'GET'
            },
            'get_current_measurement': {
                'endpoint': 'VNA/get-current-measurement',
                'method': 'GET'
            },
            'get_new_measurement': {
                'endpoint': 'VNA/get-new-measurement',
                'method': 'GET'
            },
        },
        'vnaServerProcessName': 'MegiqVnaServer.exe',
        'vnaServerCheckInterval': 5, # in seconds - time to wait between checking if the vna server is running
        'vnaProgramName': 'MiQVNA.exe',
        'vnaSettings': {
            'frequency': { #this may also do something 
                'min': 400e6,
                'max': 3e9,
                'step': 6.5e6  # no. of steps = 400
            },
            'power': -10,
            'averaging': 1,
        },
        'vnaUpdateInterval': 10,# was 5
        'vnaUpdateInterval_display': 5,
        'vnaStatusInterval': 10,
        'mountingSystemUpdateInterval': 1,
        'calibrationUpdateInterval': 1,
        'calibrationStep': 10,  # mm
        'measurementUpdateInterval': 1,
        'mountingSystemCommands': { 
            'RSI PRO Extend' : "1\n",
            'RSI PRO Retract' : "2\n",
            'RSI PRO Stop' : "3\n",
            'RSI PRO Connection Check' : "4\n",
            'Get Distance to Ground' : "5\n",
        },
        'mountingSystem': {
            'actuator_max': 220, # mm - actual length is 140 mm, but limiting by 10 mm to avoid overextension # change by me to 220 for testing
            'actuator_min': 0,
            'actuator_tolerance': 0.2, # mm - tolerance for the position control
            'extend_distance': 10, # mm
            'retract_distance': 10, # mm
        },
        'dataLogger': {
            'baseFileName_measurement': '{}_measurementData.csv',
            #'baseFileName_vna': '{}_vnaData.s1p',  # saved as a s1p touchstone file - using the skrf library
            'baseDirectory_cal_data': '{}_cal_data',
            'headings_measurement': [ 'measurement_date','latitude','longitude','Shear_Vain_20','Shear_Vain_50','Shear_Vain_80','Raw Sensor Data','DistanceToGround'],#added 'latitude','longitude'
        },
        'measurement_manager': {
            'measurement_delay_list': [10, 20, 30, 60, 120, 300]
        },
        'nn': {
            'theta_file': os.path.join(basedir, r'nnData/nn_training.npy'),
            'permittivity_scale': {
                'min': 1.0, 
                'max': 60.0,
                },
            'frequency_scale': {# may need to change this 
                'min': 0.9e9, # orginally 0.9e9  # when i add more frequencys then permittity stays at 50ish 
                'max': 3.67e9, # orginally 1.8e9 / i set to 3.67e9
            },
        },
        'teltonika':{
            'DEVICE_IP' : '192.168.1.1',
            'username' : 'admin',
            'password' : 'MudM45t3r'
            
        },
    }
