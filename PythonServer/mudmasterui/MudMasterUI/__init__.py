"""
The flask application package.
"""
import threading

from flask import Flask
import requests
from MudMasterUI.config import Config
from MudMasterUI.supportFunctions import *

from MudMasterUI.config_machine import Config_Machine

from MudMasterUI.interface_vna import VNA_Manager # import  vna controller class 
from MudMasterUI.interface_mountingSystem import MountingSystem_Manager # import  Actuator controller class 
from MudMasterUI.module_dielectricCalibration import VNA_Cal # import  neural network predictor class 
from MudMasterUI.module_measurement_manager import Measurement_Manager # import measurement controller class 

from MudMasterUI.fileTransfer import File_Transfer # import File Transfer class 


#initaliseing and creating 4 controller objects
controller_vna = VNA_Manager()
controller_mountingSystem = MountingSystem_Manager()
module_dielectric_manager = VNA_Cal()
measurement_manager = Measurement_Manager()

def start_measurement():
    """Function to send a GET request to start the measurement. used to start measurement on startup of server """
    url = 'http://localhost:80/mounting-system-v2/Measure'  # Adjust to your actual server URL
    try:
        response = requests.get(url)  # Send GET request to trigger Measure()
        print(f"HTTP request sent. Response: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error sending HTTP request: {e}")


def createApp(config_main=Config, config_machine=Config_Machine):
    """
    Initializes and deploys a Flask application with specified configurations and modules.

    @param config_main: Main configuration object for the Flask app. Default is `Config`.
    @param config_machine: Configuration object for machine-specific settings. Default is `Config_Machine`.
    @param site_config: Configuration object for site-specific settings. Default is `Site_Config`.
    
    @retval app: The initialized Flask app object.
    """
    # Create a new Flask application instance
    app = Flask(__name__)

    # Load the configuration settings into the app from the provided configuration objects
    app.config.from_object(config_main)       # Load main configuration
    app.config.from_object(config_machine)    # Load machine-specific configuration

    # Format and set the MACHINE_DIRECTORY in the app configuration based on machine number
    app.config['MACHINE_DIRECTORY'] = app.config['MACHINE_DIRECTORY'].format(
        app.config['CONFIG_MACHINE']['machineNumber'],
        app.config['CONFIG_MACHINE']['machineNumber']
    )
    
    # Load and apply the runtime configuration settings
    load_run_config(app)

    # Initialize various controllers with the Flask app instance
    controller_vna.init_app(app)  # Initialize VNA controller
    controller_mountingSystem.init_app(app)  # Initialize mounting system controller
    module_dielectric_manager.init_app(app)  # Initialize dielectric manager module
    
    # Initialize the measurement manager with the Flask app instance and controllers
    measurement_manager.init_app(
        app,
        controller_mountingSystem,
        controller_vna,
        module_dielectric_manager
    )
    #create File_transfer opject that will transer files from one folder to another 
    File_Transfer(app)
    # Register blueprints for different parts of the application / basically sets up webpages at certain web addresses 
    from MudMasterUI.main import bp as bp_main
    app.register_blueprint(bp_main)  # Register main blueprint

    from MudMasterUI.eventStreams import bp as bp_eventStreams
    app.register_blueprint(bp_eventStreams)  # Register event streams blueprint

    from MudMasterUI.vna import bp as bp_vna
    app.register_blueprint(bp_vna)  # Register VNA blueprint

    # Added by Joshua: Register blueprint for the updated mounting system
    from MudMasterUI.mountingSystemV2 import bp as bp_mountingSystemV2
    app.register_blueprint(bp_mountingSystemV2)  # Register updated mounting system blueprint

    
    #start taking measurments every x seconds - removed if you dont want to start measureing on server start up 
    threading.Thread(target=start_measurement, daemon=True).start()# start automated measureing start 
    
    return app