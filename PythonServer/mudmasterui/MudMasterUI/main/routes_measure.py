"""
*******************************************************************************
@file   MudMasterUI.main.routes_measure.py
@author Scott Thomason
@date   04 May 2022
@brief  Routes for the main section of the app. This file specifically handles 
        the calibration section of the application.

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
from MudMasterUI import controller_vna, controller_mountingSystem, module_dielectric_manager, measurement_manager  # Import various controllers and managers

""" Defines
*******************************************************************************
"""


""" Routes
*******************************************************************************
"""
@bp.route('/measure')
def measure():
    """
    Route to start the measurement process and render the measurement page.
    """
    #measurement_manager.start_measurement()  # Start the measurement process

    # Render the 'measure.html' template with the current year and other context variables
    return render_template(
        'main/measure.html',
        title='Tailings Measurement',
        year=datetime.now().year,
        showFooter=True,  # Show the footer on the page
    )


@bp.route('/measure/run', methods=['GET', 'POST'])
def measure_start():
    """
    Route to start the calibration process and return the current calibration progress.
    """
    print('starting calibration')  # Log a message indicating calibration has started
    measurement_manager.run_calibration()  # Initiate the calibration process
    # Return the current calibration progress as JSON
    return jsonify(measurement_manager.get_calibration_progress())


@bp.route('/measure/stop', methods=['GET'])
def measure_progress():
    """
    Route to get and return the current calibration progress.
    """
    # Return the current calibration progress as JSON
    return jsonify(measurement_manager.get_calibration_progress())