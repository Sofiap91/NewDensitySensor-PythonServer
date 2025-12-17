"""
*******************************************************************************
@file   MudMasterUI.eventStreams.routes.py
@author Scott Thomason, Joshua Paterson 
@date   28 Mar 2022
@brief  Routes for the app's event streams.

REFERENCE:
*******************************************************************************
    Functions
*******************************************************************************
"""

""" Imports
*******************************************************************************
"""
import json
import time
import os
import random
import numpy as np
from datetime import datetime
from copy import deepcopy
from flask import render_template, flash, redirect, url_for, request, send_from_directory, jsonify, make_response, abort, Blueprint, current_app, session, Response
import requests

from MudMasterUI import controller_vna, controller_mountingSystem, module_dielectric_manager, measurement_manager
from MudMasterUI.eventStreams import bp
from MudMasterUI.config import Config
from MudMasterUI import globalErrorVar

""" Routes
*******************************************************************************
"""

@bp.route('/stream/status-indicators', methods=['GET'])
def systemStatusIndicators():
    """
    Stream system status indicators as a server-sent event (SSE).

    Checks the status of various components and sends updates periodically.
    """
    DEVICE_IP = Config.CONFIG_SYSTEM['teltonika']['DEVICE_IP']

    def bool_to_int(value):
        """Convert boolean to integer (1 for True, 0 for False)."""
        return 1 if value else 0

    def is_teltonika_connected():
        """Check if Teltonika device is connected by sending a GET request."""
        try:
            response = requests.get(f"http://{DEVICE_IP}", timeout=3)
            if response.status_code == 200:
                return 1  # Teltonika is connected
            else:
                return 0  # Teltonika is not connected
        except requests.ConnectionError:
            return 0  # Teltonika is not connected (connection error)

    def stream_systemStatusIndicators(configDict):
        """Generate system status indicators for streaming."""
        vna_status = controller_vna.get_statusVNA().get('status', 0)

        previousVal = {
            'errorIndicator': 0,
            'vnaConnection': vna_status,
            'mountingSystemConection': controller_mountingSystem.get_status(),
            'teltonikaConnection': 0,
            'errorMeasureing': 0,
            'errorActuatorReadWrite': 0,
            'errorTeltonika': 0,
            'CurrentlyRetracting': 0,
            'CurrentlyExtending': 0,
            'CurrentlyCalibrating': 0,
            'CurrentlyLogging': 0
        }

        yield 'data: ' + json.dumps(previousVal) + '\n\n'

        while True:
            vna_status = controller_vna.get_statusVNA()

            newVal = {
                'errorIndicator': 0,
                'vnaConnection': vna_status.get('status', 0),
                'mountingSystemConection': controller_mountingSystem.get_status(),
                'teltonikaConnection': is_teltonika_connected(),
                'errorMeasureing': bool_to_int(globalErrorVar.ErrorFromMeasurementManager),
                'errorActuatorReadWrite': bool_to_int(globalErrorVar.ErrorFromActuatorReadWrite),
                'errorTeltonika': bool_to_int(globalErrorVar.ErrorFromTeltonika),
                'CurrentlyRetracting': bool_to_int(globalErrorVar.CurrentlyRetracting),
                'CurrentlyExtending': bool_to_int(globalErrorVar.CurrentlyExtending),
                'CurrentlyCalibrating': bool_to_int(globalErrorVar.CurrentlyCalibrating),
                'CurrentlyLogging': bool_to_int(globalErrorVar.CurrentlyLogging)
            }

            if newVal != previousVal:
                yield 'data: ' + json.dumps(newVal) + '\n\n'

            previousVal = newVal
            time.sleep(5)  # Sleep for 5 seconds before sending the next update

    return Response(stream_systemStatusIndicators(current_app.config), mimetype="text/event-stream")


@bp.route('/stream/vna-data', methods=['GET'])
def streamVnaData():
    """
    Stream VNA data as a server-sent event (SSE).

    Sends VNA data updates periodically, including stability and magnitude-phase information.
    """
    def stream_vnaData(configDict):
        """Generate VNA data for streaming."""
        if controller_vna.has_data():
            previous_data = controller_vna.get_currentData()['port1']
        else:
            previous_data = controller_vna.get_nextData()['port1']

        current_data = controller_vna.get_nextData()['port1']

        while True:
            data_diff = {freq: 10 * np.log10(np.abs(current_data[freq] - previous_data[freq])) for freq in current_data}

            data_dict = {
                'vna_data': controller_vna.get_currentData(asJson=True)['port1'],
                'vna_status': controller_vna.get_statusVNA(),
                'vna_stability': data_diff,
                'vna_mag_phase': controller_vna.get_currentData(asJson=True, magPhase=True)['port1'],
            }

            previous_data = current_data
            current_data = controller_vna.get_nextData()['port1']

            yield 'data: ' + json.dumps(data_dict) + '\n\n'

            time.sleep(configDict['CONFIG_SYSTEM']['vnaUpdateInterval_display'])

    return Response(stream_vnaData(current_app.config), mimetype="text/event-stream")


@bp.route('/stream/vna-data/s11', methods=['GET'])
def vnaData_s11():
    """
    Stream VNA S11 data as a server-sent event (SSE).

    Sends updates of the VNA S11 data periodically.
    """
    def stream_vnaData(configDict):
        """Generate VNA S11 data for streaming."""
        while True:
            data_dict = {
                'vna_data': controller_vna.get_currentData(asJson=True),
                'vna_status': controller_vna.get_statusVNA(),
            }

            yield 'data: ' + json.dumps(data_dict) + '\n\n'

            time.sleep(configDict['CONFIG_SYSTEM']['vnaUpdateInterval_display'])

    return Response(stream_vnaData(current_app.config), mimetype="text/event-stream")


@bp.route('/stream/vna-data/s11/mag-phase', methods=['GET'])
def vnaData_s11_mag_phase():
    """
    Stream VNA S11 magnitude-phase data as a server-sent event (SSE).

    Sends updates of the VNA S11 magnitude and phase data periodically.
    """
    def stream_vnaData(configDict):
        """Generate VNA S11 magnitude-phase data for streaming."""
        while True:
            yield 'data: ' + json.dumps(controller_vna.get_currentData(asJson=True, magPhase=True)) + '\n\n'

            time.sleep(configDict['CONFIG_SYSTEM']['vnaUpdateInterval_display'])

    return Response(stream_vnaData(current_app.config), mimetype="text/event-stream")


@bp.route('/stream/vna-data/delta', methods=['GET'])
def vnaData_delta():
    """
    Stream VNA data delta (difference) as a server-sent event (SSE).

    Sends updates of the delta between current and previous VNA data periodically.
    """
    def stream_vnaData(configDict):
        """Generate VNA data delta for streaming."""
        previous_data = controller_vna.get_currentData()['port1']
        new_data = controller_vna.get_nextData()['port1']

        while True:
            data_diff = {freq: 10 * np.log10(np.abs(new_data[freq] - previous_data[freq])) for freq in new_data}
            yield 'data: ' + json.dumps(data_diff) + '\n\n'
            previous_data = deepcopy(new_data)
            new_data = controller_vna.get_nextData()['port1']

            time.sleep(configDict['CONFIG_SYSTEM']['vnaUpdateInterval_display'])

    return Response(stream_vnaData(current_app.config), mimetype="text/event-stream")


@bp.route('/stream/mounting-system/actuator/position', methods=['GET'])#can maybe remove
def mountingSystem_actuator_position():
    """
    Stream mounting system actuator position as a server-sent event (SSE).

    Sends updates of the actuator position periodically.
    """
    def stream_mountingSystem_actuator_position(configDict):
        """Generate mounting system actuator position for streaming."""
        while True:
            yield 'data: ' + json.dumps({'position': controller_mountingSystem.get_actuator_position()}) + '\n\n'

            time.sleep(configDict['CONFIG_SYSTEM']['mountingSystemUpdateInterval'])

    return Response(stream_mountingSystem_actuator_position(current_app.config), mimetype="text/event-stream")


@bp.route('/stream/calibration-progress', methods=['GET'])
def calibration_progress():
    """
    Stream calibration progress as a server-sent event (SSE).

    Sends updates of the calibration progress periodically.
    """
    def stream_calibration_progress(configDict):
        """Generate calibration progress for streaming."""
        previous_data = measurement_manager.get_calibration_progress()
        yield 'data: ' + json.dumps(previous_data) + '\n\n'
        while True:
            new_data = measurement_manager.get_calibration_progress()
            if new_data != previous_data:
                yield 'data: ' + json.dumps(new_data) + '\n\n'
            
            previous_data = deepcopy(new_data)
            time.sleep(configDict['CONFIG_SYSTEM']['calibrationUpdateInterval'])

    return Response(stream_calibration_progress(current_app.config), mimetype="text/event-stream")


@bp.route('/stream/measurement-progress', methods=['GET'])
def measurement_progress():
    """
    Stream measurement progress as a server-sent event (SSE).

    Sends updates of the measurement progress periodically.
    """
    def stream_measurement_progress(configDict):
        """Generate measurement progress for streaming."""
        previous_data = measurement_manager.get_measurement_display_data()
        yield 'data: ' + json.dumps(previous_data) + '\n\n'
        while True:
            new_data = measurement_manager.get_measurement_display_data()
            if new_data != previous_data:
                yield 'data: ' + json.dumps(new_data) + '\n\n'

            previous_data = deepcopy(new_data)
            time.sleep(configDict['CONFIG_SYSTEM']['measurementUpdateInterval'])

    return Response(stream_measurement_progress(current_app.config), mimetype="text/event-stream")