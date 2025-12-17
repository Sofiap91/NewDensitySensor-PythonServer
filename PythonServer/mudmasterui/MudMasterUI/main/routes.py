"""
*******************************************************************************
@file   MudMasterUI.main.routes.py
@author Scott Thomason
@date   02 Feb 2022
@brief  Routes for the main section of the app.

REFERENCE:

*******************************************************************************
    Functions
*******************************************************************************

*******************************************************************************
"""

""" Imports
*******************************************************************************
"""
import json
import time
import os

from datetime import datetime
from flask import render_template, flash, redirect, url_for, request, send_from_directory, jsonify, make_response, abort, Blueprint, current_app, session

from MudMasterUI.main import bp  # Import the Blueprint instance named 'bp' from MudMasterUI.main
from MudMasterUI.supportFunctions import save_run_config  # Import function to save run configuration
from MudMasterUI import controller_vna, controller_mountingSystem, module_dielectric_manager, measurement_manager  # Import various controllers and managers

""" Defines
*******************************************************************************
"""


""" Routes
*******************************************************************************
"""

@bp.route('/settings', methods=['POST'])
def settings_post():
    """
    Route to handle POST requests for updating settings. Receives updates and 
    saves them to the configuration file.

    The route updates the site and measurement delay settings if provided in 
    the request. It also saves the updated configuration.
    """
    # Receive and decode the settings updates sent as POST requests
    received_data = json.loads(request.get_data().decode())

    # Update the settings in the config file if provided
    if 'site' in received_data:
        # Check if the site is valid
        if received_data['site'] in current_app.config['SITE_CONFIG'].keys():
            # Update the site in the current run configuration
            current_app.config['CONFIG_RUN']['site'] = received_data['site']
            
            # Save the updated run configuration
            save_run_config(current_app)

            # Update the site in the measurement manager
            measurement_manager.set_site(current_app.config['CONFIG_RUN']['site'])

    if 'measurement_delay' in received_data:
        # Update the measurement delay setting
        current_app.config['CONFIG_RUN']['measurement_manager']['measurement_delay'] = int(received_data.get('measurement_delay', 60))

        # Save the updated run configuration
        save_run_config(current_app)

    # Return a JSON response indicating success
    return jsonify({'success': True})

