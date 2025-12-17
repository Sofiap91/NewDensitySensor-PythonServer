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

import numpy as np

from datetime import datetime
from flask import render_template, flash, redirect, url_for, request, send_from_directory, jsonify, make_response, abort, Blueprint, current_app, session

from MudMasterUI.vna import bp
from MudMasterUI import controller_vna

""" Defines
*******************************************************************************
"""


""" Routes
*******************************************************************************
"""
@bp.route('/vna')
def vna():
    """Renders the home page."""
    current_data = controller_vna.get_currentData()['port1']

    frequencies = list(current_data.keys())
    
    # print('time domain vna data')
    # for dat in controller_vna.get_nextData_timedomain(asMag=True):
    #     print(dat)

    return render_template(
        'vna/vna_test.html',
        title='VNA Interface',
        year=datetime.now().year,
        showFooter = True,
        vna_status = controller_vna.get_statusVNA(),
        vna_data = controller_vna.get_currentData(asJson=True),
        frequencies = frequencies,
        vna_info = controller_vna.get_VNAInfo(),
    )